"""Cached evaluation statistics for the config-driven pipeline (Tier 2).

Entry point for ``python run.py evaluate --config <yaml>``.

Each statistic is computed once per source (agora test split / gaussian
baseline / ddpm samples) and cached to ``<run>/stats/<statistic>__<source>.npz``
together with a JSON metadata record of the parameters used.  Re-running the
stage loads the caches and only regenerates the figures, so plot formatting
can be iterated for free; a parameter change in the config invalidates the
affected caches automatically.

Sources are compared in physical map units (µK): the z-scored patch files,
the Gaussian baseline (generated in z-score space by notebook 03), and DDPM
samples are all denormalised with the same ``norm_params`` before any
statistic is computed.

ILC residual noise (Prabhu et al. §4.6) is added where the paper does so —
the summed-channel moments and the cross-moments — one ``cl2map`` realisation
per patch from ``total_ilc_residuals[<tier>]['mv']``, the same realisation in
both channels (matching the original HPC analysis).  Tiers are configured per
statistic via ``noise_tiers``; ``none`` computes the noiseless variant.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np

from foregrounds_diffusion.flatmaps import cl2map, get_lpf_hpf, radial_profile
from foregrounds_diffusion.moments import (
    compute_cross_moments,
    compute_summed_moments,
    mean_cls,
    mean_cross_cls,
)
from foregrounds_diffusion.peak_counts import count_minima_binned, count_peaks_binned
from foregrounds_diffusion.preprocessing import (
    apply_maxmin_normalization,
    denormalize_dm_maps,
)
from foregrounds_diffusion.stacking import extract_cutouts, select_snr_pixels

SOURCE_COLORS = {"agora": "black", "ddpm": "steelblue", "gaussian": "orangered"}
SOURCE_ORDER = ["agora", "ddpm", "gaussian"]


def _mapparams(cfg):
    """[nx, ny, dx, dy] with dx/dy in arcminutes, from the data config."""
    res = cfg.data.res
    dx = cfg.data.patch_deg * 60.0 / res
    return [res, res, dx, dx]


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------


def _test_split_indices(n, cfg):
    """Test-set indices of the seeded permutation used by pipeline/train.py."""
    rng = np.random.default_rng(seed=cfg.data.seed)
    indices = rng.permutation(n)
    start = int((cfg.data.train_size + cfg.data.val_size) * n)
    return indices[start:]


def load_sources(cfg, run):
    """Load all available map sources in physical units.

    Returns
    -------
    sources : dict
        ``name -> (cib, tsz)`` with each array of shape (N, H, W).
    norm_params : ndarray
        ``[cib_mean, cib_std, tsz_mean, tsz_std]`` from patch extraction.
    test_idx : ndarray
        Indices of the agora/gaussian test split.
    """
    ptsrc = cfg.preprocessing.point_source_mjy
    res = cfg.data.res
    patches_dir = Path(cfg.data.patches_dir) if cfg.data.patches_dir else run.patches

    cib_file = patches_dir / f"CIB_map_150GHz_{res}_st6_zscore_{ptsrc}mJy_lp.npy"
    tsz_file = patches_dir / f"tSZ3_map_150GHz_{res}_st6_zscore_{ptsrc}mJy_lp.npy"
    if not cib_file.exists():
        raise FileNotFoundError(
            f"patch file not found: {cib_file} — set data.patches_dir to the "
            "directory holding the notebook-03 outputs"
        )
    norm_params = np.load(patches_dir / f"norm_params_{ptsrc}mJy.npy")
    cib_mean, cib_std, tsz_mean, tsz_std = norm_params

    cib = np.load(cib_file)[:, :, :, 0]
    tsz = np.load(tsz_file)[:, :, :, 0]
    test_idx = _test_split_indices(len(cib), cfg)
    sources = {
        "agora": (
            cib[test_idx] * cib_std + cib_mean,
            tsz[test_idx] * tsz_std + tsz_mean,
        )
    }
    print(f"[evaluate] agora: {len(test_idx)} test patches (of {len(cib)})")
    del cib, tsz

    sample_files = sorted(run.samples.glob("*.npy")) if run.samples.exists() else []
    if sample_files:
        ddpm = np.concatenate([np.load(f) for f in sample_files])
        ddpm = denormalize_dm_maps(ddpm, cib_mean, cib_std, tsz_mean, tsz_std)
        sources["ddpm"] = (ddpm[:, 0], ddpm[:, 1])
        print(f"[evaluate] ddpm: {len(ddpm)} samples from {len(sample_files)} file(s)")
    else:
        print(f"[evaluate] ddpm: no samples in {run.samples} — skipping this source")

    gauss_file = patches_dir / f"gaussian_cib_tsz_{ptsrc}mJy_lp.npy"
    if gauss_file.exists():
        gauss = np.load(gauss_file)
        if gauss.ndim == 4 and gauss.shape[1] == 2:  # (N, 2, H, W), z-score space
            gauss = denormalize_dm_maps(gauss, cib_mean, cib_std, tsz_mean, tsz_std)
            gauss_cib, gauss_tsz = gauss[:, 0], gauss[:, 1]
        else:  # (N, H, W, 2) channels-last
            gauss_cib = gauss[:, :, :, 0] * cib_std + cib_mean
            gauss_tsz = gauss[:, :, :, 1] * tsz_std + tsz_mean
        sources["gaussian"] = (gauss_cib[test_idx], gauss_tsz[test_idx])
        print(f"[evaluate] gaussian: {len(test_idx)} baseline maps")
    else:
        print(f"[evaluate] gaussian: {gauss_file} not found — skipping this source")

    return sources, norm_params, test_idx


# ---------------------------------------------------------------------------
# ILC residual noise
# ---------------------------------------------------------------------------


class NoiseModel:
    """Generates flat-sky ILC residual-noise realisations per experiment tier."""

    def __init__(self, ilc_file, mapparams, base_seed=42):
        ilc = np.load(ilc_file, allow_pickle=True).item()
        self.residuals = ilc["total_ilc_residuals"]
        self.mapparams = mapparams
        self.base_seed = base_seed

    def realisations(self, tier, n, context=""):
        """Return (n, H, W) noise maps for *tier*, seeded by (tier, context)."""
        if tier not in self.residuals:
            raise KeyError(
                f"noise tier {tier!r} not in ILC file — available: {sorted(self.residuals)}"
            )
        ell, nl = self.residuals[tier]["mv"]
        # cl2map draws from the global numpy RNG; seed it deterministically so
        # cached results are reproducible per (tier, source) combination.
        np.random.seed(self.base_seed + zlib.crc32(f"{tier}|{context}".encode()) % 10_000)
        return np.array([cl2map(self.mapparams, nl, ell) for _ in range(n)])


# ---------------------------------------------------------------------------
# Statistic base class
# ---------------------------------------------------------------------------


class Statistic:
    """One evaluation statistic with npz caching and quick-look plotting."""

    name = "base"

    def __init__(self, params, n_jobs, mapparams, noise=None, norm_params=None):
        self.params = dict(params)
        self.n_jobs = n_jobs
        self.mapparams = mapparams
        self.noise = noise
        self.norm_params = norm_params

    # -- caching ------------------------------------------------------------

    def _meta(self, n_used):
        return json.dumps({**self.params, "n_maps_used": int(n_used)}, sort_keys=True)

    def cache_file(self, stats_dir, source):
        return Path(stats_dir) / f"{self.name}__{source}.npz"

    def compute_or_load(self, stats_dir, source, cib, tsz, force=False):
        """Load the cached result if its parameters match, else compute and save."""
        n_used = min(self.params.get("n_maps", len(cib)), len(cib))
        path = self.cache_file(stats_dir, source)
        if path.exists() and not force:
            with np.load(path, allow_pickle=False) as f:
                cached = {k: f[k] for k in f.files}
            if str(cached.pop("__meta__", "")) == self._meta(n_used):
                print(f"[evaluate] {self.name}/{source}: cached")
                return cached
            print(f"[evaluate] {self.name}/{source}: parameters changed — recomputing")
        else:
            print(f"[evaluate] {self.name}/{source}: computing ({n_used} maps)")
        result = self.compute(cib[:n_used], tsz[:n_used], source)
        np.savez_compressed(path, __meta__=self._meta(n_used), **result)
        return result

    # -- interface ----------------------------------------------------------

    def compute(self, cib, tsz, source):
        """Return a dict of arrays for one source. Maps are (N, H, W), physical units."""
        raise NotImplementedError

    def plot(self, results, plot_path):
        """Write a quick-look figure from cached results {source: dict}."""
        raise NotImplementedError

    def summarise(self, results):
        """Return list of one-line summary strings."""
        return []

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _ordered(results):
        return [(s, results[s]) for s in SOURCE_ORDER if s in results]


def _subplots(ncols, nrows=1, width=4.6, height=3.8):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows))
    return plt, fig, np.atleast_1d(axes).ravel()


# ---------------------------------------------------------------------------
# Two-point statistics (notebook 06)
# ---------------------------------------------------------------------------


class PowerSpectrum(Statistic):
    name = "power_spectrum"

    def compute(self, cib, tsz, source):
        p = self.params
        el, cl_cib, err_cib = mean_cls(
            cib, self.mapparams, p["lmin"], p["lmax"], p["binsize"], n_jobs=self.n_jobs
        )
        _, cl_tsz, err_tsz = mean_cls(
            tsz, self.mapparams, p["lmin"], p["lmax"], p["binsize"], n_jobs=self.n_jobs
        )
        return {
            "el": el,
            "cl_cib": cl_cib,
            "err_cib": err_cib,
            "cl_tsz": cl_tsz,
            "err_tsz": err_tsz,
        }

    def plot(self, results, plot_path):
        plt, fig, axes = _subplots(3)
        for ax, key, title in zip(axes[:2], ["cib", "tsz"], ["CIB", "tSZ"]):
            for src, r in self._ordered(results):
                ax.plot(r["el"], r[f"cl_{key}"], color=SOURCE_COLORS[src], label=src)
                ax.fill_between(
                    r["el"],
                    r[f"cl_{key}"] - r[f"err_{key}"],
                    r[f"cl_{key}"] + r[f"err_{key}"],
                    color=SOURCE_COLORS[src],
                    alpha=0.2,
                    lw=0,
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(r"$\ell$")
            ax.set_ylabel(r"$C_\ell$")
            ax.set_title(title)
            ax.legend()
        ax = axes[2]
        ax.axhline(0, color="k", lw=0.8, ls="--")
        if "agora" in results and "ddpm" in results:
            a, d = results["agora"], results["ddpm"]
            for key, label in [("cib", "CIB"), ("tsz", "tSZ")]:
                resid = (a[f"cl_{key}"] - d[f"cl_{key}"]) / (a[f"err_{key}"] + 1e-30)
                ax.plot(a["el"], resid, label=label)
            ax.legend()
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"$(C_\ell^{\rm Agora} - C_\ell^{\rm DDPM})/\sigma$")
        ax.set_title("residuals")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

    def summarise(self, results):
        lines = []
        if "agora" in results and "ddpm" in results:
            a, d = results["agora"], results["ddpm"]
            for key in ["cib", "tsz"]:
                resid = np.abs(a[f"cl_{key}"] - d[f"cl_{key}"]) / (a[f"err_{key}"] + 1e-30)
                lines.append(
                    f"power_spectrum[{key}]: max |Agora-DDPM| residual "
                    f"{resid.max():.2f}σ (mean {resid.mean():.2f}σ)"
                )
        return lines


class CrossSpectrum(Statistic):
    name = "cross_spectrum"

    def compute(self, cib, tsz, source):
        p = self.params
        el, cl_x, err_x = mean_cross_cls(
            cib, tsz, self.mapparams, p["lmin"], p["lmax"], p["binsize"], n_jobs=self.n_jobs
        )
        return {"el": el, "cl_cross": cl_x, "err_cross": err_x}

    def plot(self, results, plot_path):
        plt, fig, axes = _subplots(2)
        for src, r in self._ordered(results):
            axes[0].plot(r["el"], r["cl_cross"], color=SOURCE_COLORS[src], label=src)
            axes[0].fill_between(
                r["el"],
                r["cl_cross"] - r["err_cross"],
                r["cl_cross"] + r["err_cross"],
                color=SOURCE_COLORS[src],
                alpha=0.2,
                lw=0,
            )
        axes[0].set_xscale("log")
        axes[0].set_yscale("symlog")
        axes[0].set_xlabel(r"$\ell$")
        axes[0].set_ylabel(r"$C_\ell^{\rm CIB \times tSZ}$")
        axes[0].legend()
        axes[1].axhline(0, color="k", lw=0.8, ls="--")
        if "agora" in results and "ddpm" in results:
            a, d = results["agora"], results["ddpm"]
            resid = (a["cl_cross"] - d["cl_cross"]) / (a["err_cross"] + 1e-30)
            axes[1].plot(a["el"], resid)
        axes[1].set_xlabel(r"$\ell$")
        axes[1].set_ylabel(r"$\Delta C_\ell^{\times}/\sigma$")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Higher-order moments (notebook 07)
# ---------------------------------------------------------------------------


class Moments(Statistic):
    """S2/S3/S4 of the summed CIB+tSZ(+noise) field per ℓ-band."""

    name = "moments"
    moment_fn = staticmethod(compute_summed_moments)
    prefix = "summed"

    def _bands(self):
        p = self.params
        width = (p["lmax"] - p["lmin"]) // p["n_bands"]
        edges = [(p["lmin"] + i * width, p["lmin"] + (i + 1) * width) for i in range(p["n_bands"])]
        centers = np.array([0.5 * (lo + hi) for lo, hi in edges])
        filters = [get_lpf_hpf(self.mapparams, e, filter_type=2) for e in edges]
        return centers, filters

    def _one_tier(self, cib, tsz, filters):
        out = self.moment_fn(cib, tsz, filters, n_jobs=self.n_jobs)
        return out[0] if isinstance(out, tuple) else out

    def compute(self, cib, tsz, source):
        centers, filters = self._bands()
        result = {"band_centers": centers}
        for tier in self.params.get("noise_tiers", ["none"]):
            if tier == "none":
                noisy_cib = cib
            else:
                # One realisation per patch, added to a single channel so the
                # summed field CIB+tSZ+noise contains the noise exactly once.
                noise = self.noise.realisations(tier, len(cib), context=f"{self.name}|{source}")
                noisy_cib = cib + noise
            result[f"{self.prefix}_{tier}"] = self._one_tier(noisy_cib, tsz, filters)
        return result

    def _labels(self):
        return ["S2", "S3", "S4"]

    def plot(self, results, plot_path):
        labels = self._labels()
        tiers = self.params.get("noise_tiers", ["none"])
        plt, fig, axes = _subplots(len(labels), nrows=len(tiers))
        for t_i, tier in enumerate(tiers):
            for m_i, label in enumerate(labels):
                ax = axes[t_i * len(labels) + m_i]
                for src, r in self._ordered(results):
                    key = f"{self.prefix}_{tier}"
                    if key not in r:
                        continue
                    arr = r[key][:, :, m_i]  # (N, B)
                    ax.errorbar(
                        r["band_centers"],
                        arr.mean(axis=0),
                        yerr=arr.std(axis=0),
                        color=SOURCE_COLORS[src],
                        label=src,
                        marker=".",
                        ls="-",
                        capsize=2,
                    )
                if m_i == 0:
                    ax.set_yscale("log")
                ax.set_xlabel(r"$\ell$ band centre")
                ax.set_title(f"{label} ({tier})")
                if t_i == 0 and m_i == 0:
                    ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)


class CrossMoments(Moments):
    """All 12 CIB×tSZ cross-moment combinations per ℓ-band (Appendix C)."""

    name = "cross_moments"
    prefix = "cross"

    CROSS_LABELS = [
        "S2aa",
        "S2bb",
        "S2ab",
        "S3aaa",
        "S3bbb",
        "S3aab",
        "S3abb",
        "S4aaaa",
        "S4bbbb",
        "S4aaab",
        "S4aabb",
        "S4abbb",
    ]

    def _labels(self):
        return self.CROSS_LABELS

    def _one_tier(self, cib, tsz, filters):
        out, _labels = compute_cross_moments(cib, tsz, filters, n_jobs=self.n_jobs)
        return out

    def compute(self, cib, tsz, source):
        centers, filters = self._bands()
        result = {"band_centers": centers}
        for tier in self.params.get("noise_tiers", ["none"]):
            if tier == "none":
                a, b = cib, tsz
            else:
                # Same realisation in both channels — the ILC residual of one
                # observed sky enters every channel combination identically
                # (matches the original HPC cross-moment analysis).
                noise = self.noise.realisations(tier, len(cib), context=f"{self.name}|{source}")
                a, b = cib + noise, tsz + noise
            result[f"{self.prefix}_{tier}"] = self._one_tier(a, b, filters)
        return result

    def plot(self, results, plot_path):
        tiers = self.params.get("noise_tiers", ["none"])
        # 12 panels per tier is unwieldy — plot the first tier with noise if
        # present, else the noiseless tier; the full grid is for notebook 14.
        tier = next((t for t in tiers if t != "none"), tiers[0])
        labels = self._labels()
        plt, fig, axes = _subplots(4, nrows=3)
        for m_i, label in enumerate(labels):
            ax = axes[m_i]
            for src, r in self._ordered(results):
                key = f"{self.prefix}_{tier}"
                if key not in r:
                    continue
                arr = r[key][:, :, m_i]
                ax.errorbar(
                    r["band_centers"],
                    arr.mean(axis=0),
                    yerr=arr.std(axis=0),
                    color=SOURCE_COLORS[src],
                    label=src,
                    marker=".",
                    ls="-",
                    capsize=2,
                )
            ax.set_title(f"{label} ({tier})")
            if m_i >= 8:
                ax.set_xlabel(r"$\ell$ band centre")
            if m_i == 0:
                ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# One-point and morphological statistics (notebook 08)
# ---------------------------------------------------------------------------


class PixelHistograms(Statistic):
    """Pixel-intensity histograms in z-score units (common scale across sources).

    The raw (unsmoothed) density histogram is cached; Gaussian smoothing is
    applied at plot time since it is presentation, not measurement.
    """

    name = "pixel_histograms"

    def compute(self, cib, tsz, source):
        p = self.params
        cib_mean, cib_std, tsz_mean, tsz_std = self.norm_params
        bins_cib = np.linspace(*p["cib_range"], p["n_bins"] + 1)
        bins_tsz = np.linspace(*p["tsz_range"], p["n_bins"] + 1)
        h_cib, _ = np.histogram((cib - cib_mean) / cib_std, bins=bins_cib, density=True)
        h_tsz, _ = np.histogram((tsz - tsz_mean) / tsz_std, bins=bins_tsz, density=True)
        return {
            "bins_cib": 0.5 * (bins_cib[:-1] + bins_cib[1:]),
            "bins_tsz": 0.5 * (bins_tsz[:-1] + bins_tsz[1:]),
            "hist_cib": h_cib,
            "hist_tsz": h_tsz,
        }

    def plot(self, results, plot_path):
        from scipy.ndimage import gaussian_filter1d

        sigma = self.params.get("smooth_sigma", 1.0)
        plt, fig, axes = _subplots(2)
        for ax, key, title in zip(axes, ["cib", "tsz"], ["CIB", "tSZ"]):
            for src, r in self._ordered(results):
                ax.plot(
                    r[f"bins_{key}"],
                    gaussian_filter1d(r[f"hist_{key}"], sigma=sigma),
                    color=SOURCE_COLORS[src],
                    label=src,
                )
            ax.set_yscale("log")
            ax.set_xlabel(f"{title} pixel value (z-score units)")
            ax.set_ylabel("PDF")
            ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)


class MinkowskiFunctionals(Statistic):
    name = "minkowski_functionals"

    def compute(self, cib, tsz, source):
        from foregrounds_diffusion.morphology import compute_mfs

        p = self.params
        thresholds = np.linspace(p["threshold_min"], p["threshold_max"], p["n_thresholds"])
        result = {"thresholds": thresholds}
        for key, maps in [("cib", cib), ("tsz", tsz)]:
            m0, m1, m2 = compute_mfs(
                maps, apply_maxmin_normalization, thresholds, n_jobs=self.n_jobs
            )
            result[f"M0_{key}"], result[f"M1_{key}"], result[f"M2_{key}"] = m0, m1, m2
        return result

    def plot(self, results, plot_path):
        plt, fig, axes = _subplots(2, nrows=3)
        for row, mf in enumerate(["M0", "M1", "M2"]):
            for col, key in enumerate(["cib", "tsz"]):
                ax = axes[row * 2 + col]
                for src, r in self._ordered(results):
                    arr = r[f"{mf}_{key}"]
                    ax.plot(r["thresholds"], arr.mean(axis=0), color=SOURCE_COLORS[src], label=src)
                    ax.fill_between(
                        r["thresholds"],
                        arr.mean(axis=0) - arr.std(axis=0),
                        arr.mean(axis=0) + arr.std(axis=0),
                        color=SOURCE_COLORS[src],
                        alpha=0.2,
                        lw=0,
                    )
                ax.set_title(f"{mf} — {key.upper()}")
                if row == 2:
                    ax.set_xlabel(r"threshold $\nu$")
                if row == 0 and col == 0:
                    ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)


class MinkowskiTensors(Statistic):
    """Rank-2 Minkowski tensor anisotropy β(ν) (notebook 12 extension)."""

    name = "minkowski_tensors"

    def compute(self, cib, tsz, source):
        from foregrounds_diffusion.morphology import compute_minkowski_tensors

        p = self.params
        thresholds = np.linspace(p["threshold_min"], p["threshold_max"], p["n_thresholds"])
        tensor_types = tuple(p.get("tensor_types", ["W012"]))
        result = {"thresholds": thresholds}
        for key, maps in [("cib", cib), ("tsz", tsz)]:
            tensors = compute_minkowski_tensors(
                maps,
                apply_maxmin_normalization,
                thresholds,
                tensor_types=tensor_types,
                n_jobs=self.n_jobs,
            )
            for ttype, td in tensors.items():
                result[f"beta_{key}_{ttype}"] = td["beta"]
                result[f"theta_{key}_{ttype}"] = td["theta"]
        return result

    def plot(self, results, plot_path):
        tensor_types = list(self.params.get("tensor_types", ["W012"]))
        plt, fig, axes = _subplots(len(tensor_types), nrows=2)
        for row, key in enumerate(["cib", "tsz"]):
            for col, ttype in enumerate(tensor_types):
                ax = axes[row * len(tensor_types) + col]
                for src, r in self._ordered(results):
                    arr = r[f"beta_{key}_{ttype}"]
                    ax.plot(r["thresholds"], arr.mean(axis=0), color=SOURCE_COLORS[src], label=src)
                    ax.fill_between(
                        r["thresholds"],
                        arr.mean(axis=0) - arr.std(axis=0),
                        arr.mean(axis=0) + arr.std(axis=0),
                        color=SOURCE_COLORS[src],
                        alpha=0.2,
                        lw=0,
                    )
                ax.set_ylim(0, 1)
                ax.set_title(f"β — {key.upper()} ({ttype})")
                if row == 1:
                    ax.set_xlabel(r"threshold $\nu$")
                if row == 0 and col == 0:
                    ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# tSZ stacking (notebook 09)
# ---------------------------------------------------------------------------


class TszStacking(Statistic):
    """Stacked tSZ cluster profiles in SNR bins.

    tSZ at 150 GHz is a decrement (clusters are negative); peaks are selected
    and stacked on the sign-flipped map so the stacked amplitude is positive.
    The ``sign`` entry in the cache records the flip.
    """

    name = "tsz_stacking"

    @staticmethod
    def _bin_label(smin, smax):
        return f"{smin:g}-{smax:g}" if smax is not None else f"gt{smin:g}"

    def compute(self, cib, tsz, source):
        p = self.params
        cutout = p["cutout_pix"]
        dx_arcmin = self.mapparams[2]
        # Detect the decrement convention from the data: a dominant negative
        # tail means clusters are minima and the map must be sign-flipped.
        sign = -1.0 if np.abs(tsz.min()) > np.abs(tsz.max()) else 1.0
        maps = sign * (tsz - tsz.mean())
        result = {"sign": np.array(sign), "cutout_pix": np.array(cutout)}
        half = cutout // 2
        idx = np.indices((cutout, cutout)).astype(float)
        xy = ((idx[0] - half) * dx_arcmin / 60.0, (idx[1] - half) * dx_arcmin / 60.0)
        for smin, smax in p["snr_bins"]:
            label = self._bin_label(smin, smax)
            coords = select_snr_pixels(maps, smin, smax)
            cuts = extract_cutouts(maps, coords, cutout, max_cutouts=len(coords) or 1)
            if cuts is None:
                result[f"n_{label}"] = np.array(0)
                continue
            stack = cuts.mean(axis=0)
            result[f"stack_{label}"] = stack
            result[f"n_{label}"] = np.array(len(cuts))
            result[f"profile_{label}"] = radial_profile(
                stack, xy=xy, bin_size=1.0, minbin=0.0, maxbin=10.0, to_arcmins=1
            )
        return result

    def plot(self, results, plot_path):
        labels = [self._bin_label(smin, smax) for smin, smax in self.params["snr_bins"]]
        plt, fig, axes = _subplots(len(labels), nrows=2, height=3.4)
        for col, label in enumerate(labels):
            ax_img, ax_prof = axes[col], axes[len(labels) + col]
            agora = results.get("agora", {})
            if f"stack_{label}" in agora:
                ax_img.imshow(agora[f"stack_{label}"], cmap="RdBu_r")
            ax_img.set_title(f"Agora stack, SNR {label}")
            ax_img.axis("off")
            for src, r in self._ordered(results):
                if f"profile_{label}" not in r:
                    continue
                prof = r[f"profile_{label}"]
                ax_prof.errorbar(
                    prof[:, 0],
                    prof[:, 1],
                    yerr=prof[:, 2],
                    color=SOURCE_COLORS[src],
                    label=f"{src} (n={int(r[f'n_{label}'])})",
                    marker=".",
                )
            ax_prof.set_xlabel("radius [arcmin]")
            ax_prof.set_ylabel("stacked |ΔT|")
            ax_prof.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

    def summarise(self, results):
        lines = []
        for smin, smax in self.params["snr_bins"]:
            label = self._bin_label(smin, smax)
            counts = {src: int(r[f"n_{label}"]) for src, r in results.items() if f"n_{label}" in r}
            lines.append(f"tsz_stacking[{label}]: stacked peaks {counts}")
        return lines


# ---------------------------------------------------------------------------
# Peak / minima counts (notebook 10 extension)
# ---------------------------------------------------------------------------


class PeakCounts(Statistic):
    name = "peak_counts"
    count_fn = staticmethod(count_peaks_binned)

    def compute(self, cib, tsz, source):
        p = self.params
        thresholds = np.linspace(p["threshold_min"], p["threshold_max"], p["n_thresholds"])
        dx_arcmin = self.mapparams[2]
        result = {"thresholds": thresholds}
        for key, maps in [("cib", cib), ("tsz", tsz)]:
            for fwhm in p["smoothing_fwhm_arcmin"]:
                result[f"{key}_fwhm{fwhm:g}"] = self.count_fn(
                    maps, thresholds, fwhm, pixel_res_arcmin=dx_arcmin
                )
        return result

    def plot(self, results, plot_path):
        scales = self.params["smoothing_fwhm_arcmin"]
        plt, fig, axes = _subplots(len(scales), nrows=2, height=3.4)
        for row, key in enumerate(["cib", "tsz"]):
            for col, fwhm in enumerate(scales):
                ax = axes[row * len(scales) + col]
                for src, r in self._ordered(results):
                    arr = r[f"{key}_fwhm{fwhm:g}"]
                    ax.errorbar(
                        r["thresholds"],
                        arr.mean(axis=0),
                        yerr=arr.std(axis=0),
                        color=SOURCE_COLORS[src],
                        label=src,
                        marker=".",
                        ms=3,
                    )
                ax.set_yscale("log")
                ax.set_title(f"{key.upper()}, FWHM {fwhm:g}'")
                if row == 1:
                    ax.set_xlabel(r"$\nu = T/\sigma$")
                if row == 0 and col == 0:
                    ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)


class MinimaCounts(PeakCounts):
    name = "minima_counts"
    count_fn = staticmethod(count_minima_binned)


# ---------------------------------------------------------------------------
# Scattering transforms (notebook 11 extension)
# ---------------------------------------------------------------------------


class ScatteringTransforms(Statistic):
    name = "scattering_transforms"

    def compute(self, cib, tsz, source):
        from foregrounds_diffusion.scattering_stats import (
            compute_scattering_coefficients,
            scattering_summary,
        )

        p = self.params
        result = {}
        for key, maps in [("cib", cib), ("tsz", tsz)]:
            coeffs = compute_scattering_coefficients(
                np.ascontiguousarray(maps, dtype=np.float32), J=p["J"], L=p["L"]
            )
            result[f"S1_{key}"] = coeffs["S1"]
            result[f"S2_{key}"] = coeffs["S2"]
            result[f"summary_{key}"] = scattering_summary(coeffs)
        return result

    def plot(self, results, plot_path):
        plt, fig, axes = _subplots(2, nrows=2, height=3.4)
        for col, key in enumerate(["cib", "tsz"]):
            ax = axes[col]
            for src, r in self._ordered(results):
                s1 = r[f"S1_{key}"]
                ax.errorbar(
                    np.arange(s1.shape[1]),
                    s1.mean(axis=0),
                    yerr=s1.std(axis=0),
                    color=SOURCE_COLORS[src],
                    label=src,
                    marker="o",
                    ms=3,
                )
            ax.set_yscale("log")
            ax.set_xlabel("scale $j$")
            ax.set_ylabel("$S_1$")
            ax.set_title(key.upper())
            if col == 0:
                ax.legend()
            ax = axes[2 + col]
            if "agora" in results and "ddpm" in results:
                a = results["agora"][f"summary_{key}"]
                d = results["ddpm"][f"summary_{key}"]
                resid = (a.mean(axis=0) - d.mean(axis=0)) / (a.std(axis=0) + 1e-30)
                ax.bar(np.arange(len(resid)), resid, color="steelblue")
                ax.axhline(0, color="k", lw=0.8)
            ax.set_xlabel("feature index (S1 ⊕ S2)")
            ax.set_ylabel(r"(Agora − DDPM)/$\sigma$")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

STATISTIC_REGISTRY = {
    cls.name: cls
    for cls in [
        PowerSpectrum,
        CrossSpectrum,
        Moments,
        CrossMoments,
        PixelHistograms,
        MinkowskiFunctionals,
        MinkowskiTensors,
        TszStacking,
        PeakCounts,
        MinimaCounts,
        ScatteringTransforms,
    ]
}


def main(cfg, run, dry_run=False):
    """Compute (or load cached) statistics for every source and write figures."""
    mapparams = _mapparams(cfg)
    stat_names = [s for s in cfg.evaluation.statistics if s in STATISTIC_REGISTRY]
    unknown = set(cfg.evaluation.statistics) - set(stat_names)
    if unknown:
        print(f"[evaluate] skipping unknown statistics: {sorted(unknown)}")

    if dry_run:
        print(f"[evaluate] dry run — would compute: {stat_names}")
        return

    run.stats.mkdir(parents=True, exist_ok=True)
    run.plots.mkdir(parents=True, exist_ok=True)

    sources, norm_params, test_idx = load_sources(cfg, run)
    np.savez(
        run.stats / "test_split.npz",
        test_idx=test_idx,
        seed=cfg.data.seed,
        train_size=cfg.data.train_size,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
    )

    needs_noise = any(
        any(t != "none" for t in cfg.evaluation.params.get(n, {}).get("noise_tiers", []))
        for n in stat_names
    )
    noise = None
    if needs_noise:
        # relative paths resolve against the repo root, like every other
        # config path; absolute paths pass through untouched
        ilc_file = Path(__file__).resolve().parent.parent / cfg.evaluation.ilc_noise_file
        noise = NoiseModel(ilc_file, mapparams, base_seed=cfg.evaluation.noise_seed)
        print(f"[evaluate] ILC noise loaded from {ilc_file}")

    summary_lines = []
    for name in stat_names:
        stat = STATISTIC_REGISTRY[name](
            cfg.evaluation.params.get(name, {}),
            cfg.evaluation.n_jobs,
            mapparams,
            noise=noise,
            norm_params=norm_params,
        )
        results = {}
        for src, (cib, tsz) in sources.items():
            try:
                results[src] = stat.compute_or_load(run.stats, src, cib, tsz)
            except Exception as exc:  # keep going: one bad statistic must not
                # sink the whole overnight precompute run
                print(f"[evaluate] {name}/{src} FAILED: {exc!r}")
                summary_lines.append(f"{name}/{src}: FAILED — {exc!r}")
        if results:
            plot_path = run.plots / f"{name}.png"
            try:
                stat.plot(results, plot_path)
                print(f"[evaluate] wrote {plot_path}")
            except Exception as exc:
                print(f"[evaluate] {name} plot FAILED: {exc!r}")
            summary_lines.extend(stat.summarise(results))

    summary = run.stats / "summary.md"
    with open(summary, "w") as f:
        f.write(f"# Evaluation summary — {cfg.run_name}\n\n")
        f.write(
            f"Sources: {', '.join(f'{s} ({len(c)} maps)' for s, (c, _) in sources.items())}\n\n"
        )
        for line in summary_lines:
            f.write(f"- {line}\n")
    print(f"[evaluate] summary written to {summary}")
