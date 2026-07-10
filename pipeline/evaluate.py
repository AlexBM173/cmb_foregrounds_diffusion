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
from foregrounds_diffusion.preprocessing import apply_maxmin_normalization
from foregrounds_diffusion.stacking import extract_cutouts, select_snr_pixels

# Validated colourblind-safe (Okabe-Ito blue/vermillion + Tol wine).  Worst
# separation over ALL THREE pairs and every CVD type is ΔE 31 (floor 12):
# Agora↔DDPM 92 (protan), Agora↔Gaussian 32 (protan), DDPM↔Gaussian 67 (deutan).
# A green Gaussian fell to ΔE 17 against Agora under tritanopia.  Line style is a
# redundant (non-colour) encoding of the same identity.
SOURCE_COLORS = {"agora": "#0072B2", "ddpm": "#D55E00", "gaussian": "#882255"}
SOURCE_LINESTYLES = {"agora": "-", "ddpm": "--", "gaussian": ":"}
SOURCE_LABELS = {"agora": "Agora", "ddpm": "DDPM", "gaussian": "Gaussian"}
SOURCE_ORDER = ["agora", "ddpm", "gaussian"]

PANEL_FACECOLOR = "white"
GRID_COLOR = "#c9c9c4"


def _src_label(src):
    """Display name for a source: 'Agora', 'DDPM', 'Gaussian'."""
    return SOURCE_LABELS.get(src, src)


def _line_kw(src):
    """Colour + linestyle identity for one source (colour is never the only cue)."""
    return {"color": SOURCE_COLORS[src], "ls": SOURCE_LINESTYLES[src]}


def _apply_style():
    """Serif text, LaTeX-style mathtext — applied to every figure."""
    import matplotlib

    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.facecolor": PANEL_FACECOLOR,
            "axes.edgecolor": "#55555a",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.facecolor": "white",
            "legend.framealpha": 0.9,
            "legend.fontsize": 8,
        }
    )


# Per-channel metadata, ordered; sliced to the run's channel count
# (cfg.model.channels). Fields: (key, display label, base unit for axis labels,
# patch-file template). kappa (CMB lensing convergence) is dimensionless and
# achromatic — no frequency tag in its filename.
CHANNEL_META = [
    ("cib", "CIB", "uK", "CIB_map_150GHz_{res}_st6_zscore_{ptsrc}mJy_lp.npy"),
    ("tsz", "tSZ", "uK", "tSZ3_map_150GHz_{res}_st6_zscore_{ptsrc}mJy_lp.npy"),
    ("ksz", "kSZ", "uK", "kSZ_map_150GHz_{res}_st6_zscore_{ptsrc}mJy_lp.npy"),
    ("kappa", r"$\kappa$", "", "kappa_map_{res}_st6_zscore_{ptsrc}mJy_lp.npy"),
]
_CHANNEL_BY_KEY = {m[0]: m for m in CHANNEL_META}
# Canonical plotting order: CIB, tSZ, kSZ, kappa. Anything that draws one panel
# (row, column, curve) per channel follows this, regardless of registry or
# config ordering.
CHANNEL_ORDER = [m[0] for m in CHANNEL_META]


def _channel_display(key):
    return _CHANNEL_BY_KEY[key][1]


def _field_prefix(n_channels):
    """Figure-filename prefix identifying the run's field count: '2f_', '4f_'."""
    return f"{n_channels}f_"


def _to_dl(el, arr):
    """Convert C_ell (or its error) to D_ell = l(l+1)C_l / 2pi for plotting."""
    return el * (el + 1.0) * arr / (2.0 * np.pi)


def _dl_ylabel(base_i, base_j):
    """LaTeX D_ell y-axis label from two channels' base units ('uK' or '')."""
    n_uk = (base_i == "uK") + (base_j == "uK")
    if n_uk == 0:
        return r"$\mathcal{D}_\ell$ [dimensionless]"
    unit = r"\mu\mathrm{K}^2" if n_uk == 2 else r"\mu\mathrm{K}"
    return rf"$\mathcal{{D}}_\ell\ [{unit}]$"


def _grid(ax, zero=False, minor=False):
    """Faint reference gridlines; ``zero`` emphasises the y=0 line so the
    Gaussian zero-consistency checks (odd/standardised moments) are easy to read."""
    ax.set_facecolor(PANEL_FACECOLOR)
    ax.grid(True, which="major", color=GRID_COLOR, lw=0.6, alpha=0.9)
    if minor:
        ax.grid(True, which="minor", color=GRID_COLOR, lw=0.3, alpha=0.5)
    ax.set_axisbelow(True)
    if zero:
        ax.axhline(0.0, color="#55555a", lw=0.9, ls="--", zorder=1)


def _legend(ax, **kw):
    """Legend with the canonical source display names already applied."""
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, **kw)


def _summed_moment_tex(label):
    """'S2' -> '$S_2$' (standardised summed-field moment notation, report §7)."""
    return rf"$S_{{{label[1:]}}}$"


def _cross_moment_tex(label):
    """'S2aa' -> '$M_2^{aa}$', 'S3aab' -> '$M_3^{aab}$' (raw cross-moment notation)."""
    return rf"$M_{{{label[1]}}}^{{{label[2:]}}}$"


def _mapparams(cfg):
    """[nx, ny, dx, dy] with dx/dy in arcminutes, from the data config."""
    res = cfg.data.res
    dx = cfg.data.patch_deg * 60.0 / res
    return [res, res, dx, dx]


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------


def _test_split_indices(n, cfg):
    """Indices of every patch withheld from training, in train.py's permutation.

    ``pipeline/train.py`` trains on ``indices[:int(train_size * n)]`` and never
    carves out a validation set, so the whole remainder — ``val_size`` and
    ``test_size`` together — is unseen by the model and usable for evaluation.
    Starting at ``train_size + val_size`` would discard the ``val_size`` share
    of held-out maps for no benefit (for 701 patches: 71 maps instead of 141),
    needlessly inflating the Agora sample variance.
    """
    rng = np.random.default_rng(seed=cfg.data.seed)
    indices = rng.permutation(n)
    return indices[int(cfg.data.train_size * n) :]


def load_sources(cfg, run):
    """Load all available map sources in physical units, as C-channel stacks.

    Returns
    -------
    sources : dict
        ``name -> maps`` with ``maps`` of shape ``(C, N, H, W)`` (channel-first,
        physical units), for C = ``cfg.model.channels``.
    norm_params : ndarray
        The ``2C``-entry ``[mean_0, std_0, mean_1, std_1, ...]`` from extraction.
    channel_labels : list of str
        Ordered channel keys (e.g. ``["cib", "tsz", "ksz", "kappa"]``).
    test_idx : ndarray
        Indices of the agora/gaussian test split.
    """
    ptsrc = int(cfg.preprocessing.point_source_mjy)
    res = cfg.data.res
    C = cfg.model.channels
    patches_dir = Path(cfg.data.patches_dir) if cfg.data.patches_dir else run.patches
    meta = CHANNEL_META[:C]
    channel_labels = [m[0] for m in meta]

    files = [patches_dir / m[3].format(res=res, ptsrc=ptsrc) for m in meta]
    for f in files:
        if not f.exists():
            raise FileNotFoundError(
                f"patch file not found: {f} — set data.patches_dir to the directory "
                "holding the notebook-03 / nb03b outputs for all channels"
            )
    norm_params = np.load(patches_dir / f"norm_params_{ptsrc}mJy.npy")
    if len(norm_params) != 2 * C:
        raise ValueError(
            f"norm_params has {len(norm_params)} entries, expected {2 * C} for {C} channels"
        )
    means, stds = norm_params[0::2], norm_params[1::2]

    def _denorm_stack(zstack):
        """(C, N, H, W) z-score stack -> physical units, per channel."""
        return np.stack([zstack[i] * stds[i] + means[i] for i in range(C)], axis=0)

    # agora test split (each patch file is (N, H, W, 1) channels-last, z-score)
    ch_z = [np.load(f)[:, :, :, 0] for f in files]
    n_total = len(ch_z[0])
    test_idx = _test_split_indices(n_total, cfg)
    sources = {"agora": _denorm_stack(np.stack([c[test_idx] for c in ch_z], axis=0))}
    print(f"[evaluate] agora: {len(test_idx)} test patches (of {n_total}), {C} channels")
    del ch_z

    # ddpm samples are (N, C, H, W) z-score -> (C, N, H, W) physical
    sample_files = sorted(run.samples.glob("*.npy")) if run.samples.exists() else []
    if sample_files:
        ddpm = np.concatenate([np.load(f) for f in sample_files])  # (N, C, H, W)
        if ddpm.shape[1] != C:
            raise ValueError(f"ddpm samples have {ddpm.shape[1]} channels, expected {C}")
        sources["ddpm"] = _denorm_stack(np.moveaxis(ddpm, 1, 0))
        print(f"[evaluate] ddpm: {len(ddpm)} samples from {len(sample_files)} file(s)")
    else:
        print(f"[evaluate] ddpm: no samples in {run.samples} — skipping this source")

    # gaussian baseline: gaussian_<C>field_...; fall back to the legacy 2-field
    # gaussian_cib_tsz name for C == 2 so v4 runs still load their baseline.
    gauss_file = patches_dir / f"gaussian_{C}field_{ptsrc}mJy_lp.npy"
    if not gauss_file.exists() and C == 2:
        gauss_file = patches_dir / f"gaussian_cib_tsz_{ptsrc}mJy_lp.npy"
    if gauss_file.exists():
        gauss = np.load(gauss_file)
        if gauss.ndim == 4 and gauss.shape[1] == C:  # (N, C, H, W) z-score
            gz = np.moveaxis(gauss, 1, 0)
        elif gauss.ndim == 4 and gauss.shape[-1] == C:  # (N, H, W, C) channels-last
            gz = np.moveaxis(gauss, 3, 0)
        else:
            raise ValueError(
                f"gaussian baseline {gauss_file.name} has unexpected shape {gauss.shape}"
            )
        sources["gaussian"] = _denorm_stack(gz[:, test_idx])
        print(f"[evaluate] gaussian: {len(test_idx)} baseline maps ({gauss_file.name})")
    else:
        print(f"[evaluate] gaussian: {gauss_file} not found — skipping this source")

    return sources, norm_params, channel_labels, test_idx


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
    # n_field statistics receive the full (C, N, H, W) stack via compute(maps,
    # source); the rest keep the 2-field compute(cib, tsz, source) signature and
    # are handed channels 0 and 1 (CIB, tSZ) — bit-identical to the C=2 pipeline.
    n_field = False

    def __init__(
        self, params, n_jobs, mapparams, noise=None, norm_params=None, channel_labels=None
    ):
        self.params = dict(params)
        self.n_jobs = n_jobs
        self.mapparams = mapparams
        self.noise = noise
        self.norm_params = norm_params
        self.channel_labels = list(channel_labels) if channel_labels is not None else ["cib", "tsz"]

    # -- caching ------------------------------------------------------------

    def _meta(self, n_used):
        return json.dumps({**self.params, "n_maps_used": int(n_used)}, sort_keys=True)

    def cache_file(self, stats_dir, source):
        return Path(stats_dir) / f"{self.name}__{source}.npz"

    def compute_or_load(self, stats_dir, source, maps, force=False):
        """Load the cached result if its parameters match, else compute and save.

        ``maps`` is the ``(C, N, H, W)`` channel-first stack for one source.
        """
        n_used = min(self.params.get("n_maps", maps.shape[1]), maps.shape[1])
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
        sub = maps[:, :n_used]
        result = self.compute(sub, source) if self.n_field else self.compute(sub[0], sub[1], source)
        np.savez_compressed(path, __meta__=self._meta(n_used), **result)
        return result

    # -- interface ----------------------------------------------------------

    def compute(self, *args):
        """Return a dict of arrays for one source (physical-unit maps).

        n_field statistics override ``compute(self, maps, source)`` with
        ``maps`` of shape (C, N, H, W); the rest override
        ``compute(self, cib, tsz, source)`` with each map (N, H, W).
        """
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
    _apply_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows))
    return plt, fig, np.atleast_1d(axes).ravel()


def _finish(fig, plot_path, suptitle=None, tight=True):
    """Common figure close-out: suptitle, layout, save, close.

    A figure with a single titled panel gets ONE heading, not a title above a
    near-identical subtitle: the panel title is folded into the suptitle.

    ``tight=False`` for figures already using matplotlib's constrained layout.
    """
    import matplotlib.pyplot as plt

    if suptitle:
        titled = [ax for ax in fig.axes if ax.get_visible() and ax.get_title()]
        if len(titled) == 1:
            suptitle = f"{suptitle} — {titled[0].get_title()}"
            titled[0].set_title("")
        fig.suptitle(suptitle, fontsize=13)
    if tight:
        fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def _trim_ylim(ax, values, log=False, pad=0.06):
    """Set y-limits from the data actually drawn, so panels waste no space."""
    v = np.concatenate([np.asarray(a, dtype=float).ravel() for a in values if np.size(a)])
    v = v[np.isfinite(v)]
    if v.size == 0:
        return
    if log:
        v = v[v > 0]
        if v.size == 0:
            return
        lo, hi = np.log10(v.min()), np.log10(v.max())
        span = max(hi - lo, 0.15)
        ax.set_ylim(10 ** (lo - pad * span), 10 ** (hi + pad * span))
    else:
        lo, hi = v.min(), v.max()
        span = max(hi - lo, abs(hi) * 1e-3 or 1e-12)
        ax.set_ylim(lo - pad * span, hi + pad * span)


# ---------------------------------------------------------------------------
# Two-point statistics (notebook 06)
# ---------------------------------------------------------------------------


class PowerSpectrum(Statistic):
    name = "power_spectrum"
    n_field = True

    def compute(self, maps, source):
        p = self.params
        result = {}
        el = None
        for i, key in enumerate(self.channel_labels):
            el, cl, err = mean_cls(
                maps[i], self.mapparams, p["lmin"], p["lmax"], p["binsize"], n_jobs=self.n_jobs
            )
            result[f"cl_{key}"], result[f"err_{key}"] = cl, err
        result["el"] = el
        return result

    def plot(self, results, plot_path):
        """Log-log auto-spectra, each with its own residual box beneath
        (Prabhu et al. figure convention)."""
        import matplotlib

        matplotlib.use("Agg")
        _apply_style()
        import matplotlib.pyplot as plt

        keys = self.channel_labels
        fig = plt.figure(figsize=(4.8 * len(keys), 5.2), layout="constrained")
        gs = fig.add_gridspec(2, len(keys), height_ratios=[3, 1], hspace=0.05)
        for col, key in enumerate(keys):
            base = _CHANNEL_BY_KEY[key][2]
            ax = fig.add_subplot(gs[0, col])
            ax_r = fig.add_subplot(gs[1, col], sharex=ax)
            drawn = []
            for src, r in self._ordered(results):
                el = r["el"]
                dl, derr = _to_dl(el, r[f"cl_{key}"]), _to_dl(el, r[f"err_{key}"])
                ax.plot(el, dl, label=_src_label(src), lw=1.6, **_line_kw(src))
                ax.fill_between(
                    el, dl - derr, dl + derr, color=SOURCE_COLORS[src], alpha=0.18, lw=0
                )
                drawn += [dl - derr, dl + derr]
            ax.set_xscale("log")
            # linear y: D_ell spans well under a decade per channel, and a log
            # axis cannot show the bands where they cross zero
            _trim_ylim(ax, drawn)
            ax.set_ylabel(_dl_ylabel(base, base))
            ax.set_title(_channel_display(key))
            _grid(ax, minor=True)
            ax.tick_params(labelbottom=False)
            if col == 0:
                _legend(ax)

            # residual box: (Agora - DDPM) / sigma_Agora. Already in units of
            # sigma, so no sigma bands are drawn — the gridlines carry the scale.
            if "agora" in results and "ddpm" in results:
                a, d = results["agora"], results["ddpm"]
                resid = (a[f"cl_{key}"] - d[f"cl_{key}"]) / (a[f"err_{key}"] + 1e-30)
                ax_r.plot(a["el"], resid, color=SOURCE_COLORS["ddpm"], lw=1.3)
                lim = max(1.0, 1.15 * float(np.abs(resid).max()))
                ax_r.set_ylim(-lim, lim)
            ax_r.set_xscale("log")
            _grid(ax_r, zero=True, minor=True)
            ax_r.set_xlabel(r"$\ell$")
            ax_r.set_ylabel(r"$\Delta_\ell / \sigma$")
        _finish(fig, plot_path, r"Auto-power spectra $\mathcal{D}_\ell$", tight=False)

    def summarise(self, results):
        lines = []
        if "agora" in results and "ddpm" in results:
            a, d = results["agora"], results["ddpm"]
            for key in self.channel_labels:
                resid = np.abs(a[f"cl_{key}"] - d[f"cl_{key}"]) / (a[f"err_{key}"] + 1e-30)
                lines.append(
                    f"power_spectrum[{key}]: max |Agora-DDPM| residual "
                    f"{resid.max():.2f}σ (mean {resid.mean():.2f}σ)"
                )
        return lines


class CrossSpectrum(Statistic):
    name = "cross_spectrum"
    n_field = True

    def _pairs(self):
        c = len(self.channel_labels)
        return [(i, j) for i in range(c) for j in range(i + 1, c)]

    def compute(self, maps, source):
        p = self.params
        keys = self.channel_labels
        result = {}
        el = None
        for i, j in self._pairs():
            el, cl, err = mean_cross_cls(
                maps[i],
                maps[j],
                self.mapparams,
                p["lmin"],
                p["lmax"],
                p["binsize"],
                n_jobs=self.n_jobs,
            )
            result[f"cl_{keys[i]}_{keys[j]}"] = cl
            result[f"err_{keys[i]}_{keys[j]}"] = err
        result["el"] = el
        return result

    def plot(self, results, plot_path):
        """Cross-spectra, each with its own residual box beneath — same panel
        layout as the auto-spectra."""
        import matplotlib

        matplotlib.use("Agg")
        _apply_style()
        import matplotlib.pyplot as plt

        keys = self.channel_labels
        pairs = self._pairs()
        ncols = min(3, len(pairs))
        nrows = int(np.ceil(len(pairs) / ncols))
        fig = plt.figure(figsize=(4.8 * ncols, 5.2 * nrows), layout="constrained")
        # two gridspec rows per panel row: the spectrum (3) over its residual (1)
        gs = fig.add_gridspec(2 * nrows, ncols, height_ratios=[3, 1] * nrows, hspace=0.05)
        for idx, (i, j) in enumerate(pairs):
            row, col = divmod(idx, ncols)
            ki, kj = keys[i], keys[j]
            base_i, base_j = _CHANNEL_BY_KEY[ki][2], _CHANNEL_BY_KEY[kj][2]
            ax = fig.add_subplot(gs[2 * row, col])
            ax_r = fig.add_subplot(gs[2 * row + 1, col], sharex=ax)
            drawn = []
            for src, r in self._ordered(results):
                el = r["el"]
                dl = _to_dl(el, r[f"cl_{ki}_{kj}"])
                derr = _to_dl(el, r[f"err_{ki}_{kj}"])
                ax.plot(el, dl, label=_src_label(src), lw=1.6, **_line_kw(src))
                ax.fill_between(
                    el, dl - derr, dl + derr, color=SOURCE_COLORS[src], alpha=0.18, lw=0
                )
                drawn += [dl - derr, dl + derr]
            ax.set_xscale("log")
            # linear y: cross-spectra change sign, so no log axis can hold them
            _trim_ylim(ax, drawn)
            ax.set_ylabel(_dl_ylabel(base_i, base_j))
            ax.set_title(f"{_channel_display(ki)} $\\times$ {_channel_display(kj)}")
            _grid(ax, zero=True, minor=True)
            ax.tick_params(labelbottom=False)
            if idx == 0:
                _legend(ax)

            # residual box: (Agora - DDPM) / sigma_Agora, as for the auto-spectra
            if "agora" in results and "ddpm" in results:
                a, d = results["agora"], results["ddpm"]
                resid = (a[f"cl_{ki}_{kj}"] - d[f"cl_{ki}_{kj}"]) / (a[f"err_{ki}_{kj}"] + 1e-30)
                ax_r.plot(a["el"], resid, color=SOURCE_COLORS["ddpm"], lw=1.3)
                lim = max(1.0, 1.15 * float(np.abs(resid).max()))
                ax_r.set_ylim(-lim, lim)
            ax_r.set_xscale("log")
            _grid(ax_r, zero=True, minor=True)
            ax_r.set_xlabel(r"$\ell$")
            ax_r.set_ylabel(r"$\Delta_\ell / \sigma$")
        _finish(fig, plot_path, r"Cross-power spectra $\mathcal{D}_\ell^{\,ab}$", tight=False)

    def summarise(self, results):
        lines = []
        keys = self.channel_labels
        if "agora" in results and "ddpm" in results:
            a, d = results["agora"], results["ddpm"]
            for i, j in self._pairs():
                ki, kj = keys[i], keys[j]
                resid = np.abs(a[f"cl_{ki}_{kj}"] - d[f"cl_{ki}_{kj}"]) / (
                    a[f"err_{ki}_{kj}"] + 1e-30
                )
                lines.append(
                    f"cross_spectrum[{ki}×{kj}]: max |Agora-DDPM| residual "
                    f"{resid.max():.2f}σ (mean {resid.mean():.2f}σ)"
                )
        return lines


# ---------------------------------------------------------------------------
# Higher-order moments (notebook 07)
# ---------------------------------------------------------------------------


class Moments(Statistic):
    """S2/S3/S4 of the summed temperature field per ℓ-band, plus per-field
    moments of any non-temperature channel (κ).

    The summed field adds the *temperature* channels only (CIB+tSZ+kSZ = total
    150 GHz foreground temperature); κ is dimensionless and cannot be summed
    with µK fields, so its moments are computed on its own (``field_kappa``) and
    always noiseless — ILC residual noise is a temperature quantity that does
    not apply to lensing convergence.

    Noise convention (report §Summary statistics): mean curves always come
    from the noiseless maps, so no noise debiasing is required; the noisy
    tier is computed only to provide the error bars — its patch-to-patch
    scatter is the statistical precision a future experiment would achieve.
    """

    name = "moments"
    n_field = True
    moment_fn = staticmethod(compute_summed_moments)
    prefix = "summed"

    def _temp_other(self):
        """(temperature channel keys, non-temperature channel keys)."""
        temp = [k for k in self.channel_labels if _CHANNEL_BY_KEY[k][2] == "uK"]
        other = [k for k in self.channel_labels if _CHANNEL_BY_KEY[k][2] != "uK"]
        return temp, other

    def _variance_tier(self):
        """Tier whose scatter supplies the error bars ('none' if no noisy tier)."""
        noisy = [t for t in self.params.get("noise_tiers", ["none"]) if t != "none"]
        return noisy[-1] if noisy else "none"

    def _mean_and_err(self, r, m_i):
        """(mean over maps, std over maps) — noiseless mean, variance-tier std."""
        tiers = self.params.get("noise_tiers", ["none"])
        mean_tier = "none" if "none" in tiers else tiers[0]
        mean = r[f"{self.prefix}_{mean_tier}"][:, :, m_i].mean(axis=0)
        err = r[f"{self.prefix}_{self._variance_tier()}"][:, :, m_i].std(axis=0)
        return mean, err

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

    def compute(self, maps, source):
        centers, filters = self._bands()
        temp_keys, other_keys = self._temp_other()
        li = {k: i for i, k in enumerate(self.channel_labels)}
        result = {"band_centers": centers}

        # Summed temperature field: base + rest; ILC noise added once so the
        # summed field contains it exactly once (matches the 2-field pipeline).
        base = maps[li[temp_keys[0]]]
        rest = sum((maps[li[k]] for k in temp_keys[1:]), np.zeros_like(base))
        for tier in self.params.get("noise_tiers", ["none"]):
            if tier == "none":
                a = base
            else:
                noise = self.noise.realisations(tier, len(base), context=f"{self.name}|{source}")
                a = base + noise
            result[f"{self.prefix}_{tier}"] = self._one_tier(a, rest, filters)

        # Per-field moments of non-temperature channels (κ), always noiseless.
        for k in other_keys:
            fld = maps[li[k]]
            result[f"field_{k}"] = self._one_tier(fld, np.zeros_like(fld), filters)
        return result

    def _labels(self):
        return ["S2", "S3", "S4"]

    def _mean_err_for(self, r, base_key, m_i):
        """(mean, err) for moment m_i. ``base_key == prefix`` uses the tier
        convention (noiseless mean, variance-tier err); a ``field_<k>`` key is
        noiseless so mean and err both come from that single array."""
        if base_key == self.prefix:
            return self._mean_and_err(r, m_i)
        arr = r[base_key]
        return arr[:, :, m_i].mean(axis=0), arr[:, :, m_i].std(axis=0)

    def _row_specs(self):
        """(cache-key, row title) per plot row: summed temperature then each κ."""
        temp_keys, other_keys = self._temp_other()
        summed_title = "+".join(_channel_display(k) for k in temp_keys)
        return [(self.prefix, summed_title)] + [
            (f"field_{k}", _channel_display(k)) for k in other_keys
        ]

    def plot(self, results, plot_path):
        labels = self._labels()
        rows = self._row_specs()
        plt, fig, axes = _subplots(len(labels), nrows=len(rows))
        axes = np.atleast_1d(axes).reshape(len(rows), len(labels))
        for r_i, (base_key, row_title) in enumerate(rows):
            noiseless = base_key != self.prefix
            row_base = (
                "uK"
                if base_key == self.prefix
                else _CHANNEL_BY_KEY[base_key.replace("field_", "")][2]
            )
            for m_i, label in enumerate(labels):
                ax = axes[r_i, m_i]
                for src, r in self._ordered(results):
                    if noiseless and base_key not in r:
                        continue
                    mean, err = self._mean_err_for(r, base_key, m_i)
                    ax.errorbar(
                        r["band_centers"],
                        mean,
                        yerr=err,
                        label=_src_label(src),
                        marker=".",
                        capsize=2,
                        lw=1.4,
                        **_line_kw(src),
                    )
                # S2 = variance (carries units); S3/S4 are standardised, hence
                # dimensionless, and -> 0 for a Gaussian (zero reference line)
                is_variance = m_i == 0
                _grid(ax, zero=not is_variance, minor=is_variance)
                if is_variance:
                    ax.set_yscale("log")
                    unit = r" [$\mu$K$^2$]" if row_base == "uK" else " [dimensionless]"
                else:
                    unit = " [dimensionless]"
                ax.set_title(_summed_moment_tex(label) + unit)
                ax.set_ylabel(f"{row_title}\n{_summed_moment_tex(label)}")
                ax.set_xlabel(r"$\ell$")
                if r_i == 0 and m_i == 0:
                    _legend(ax)
        _finish(fig, plot_path, "Band-power moments")


class CrossMoments(Moments):
    """The 12 cross-moment combinations per ℓ-band for every channel pair
    (Appendix C generalised to C fields).

    All C(C−1)/2 pairs × 12 combinations are computed and cached
    (``cross_<tier>_<ki>_<kj>``); the quick-look figure shows the primary
    temperature pair (CIB×tSZ), with the rest available in the cache for
    targeted analysis. ILC noise is applied only to temperature-temperature
    pairs (same realisation in both channels); pairs involving κ are computed
    noiseless (κ has no ILC noise model)."""

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

    def _one_tier(self, a, b, filters):
        out, _labels = compute_cross_moments(a, b, filters, n_jobs=self.n_jobs)
        return out

    def _pairs(self):
        keys = self.channel_labels
        return [(keys[i], keys[j]) for i in range(len(keys)) for j in range(i + 1, len(keys))]

    @staticmethod
    def _both_temp(ki, kj):
        return _CHANNEL_BY_KEY[ki][2] == "uK" and _CHANNEL_BY_KEY[kj][2] == "uK"

    def _primary_pair(self):
        for ki, kj in self._pairs():
            if self._both_temp(ki, kj):
                return ki, kj
        return self._pairs()[0]

    def compute(self, maps, source):
        centers, filters = self._bands()
        li = {k: i for i, k in enumerate(self.channel_labels)}
        result = {"band_centers": centers}
        for tier in self.params.get("noise_tiers", ["none"]):
            for ki, kj in self._pairs():
                if tier != "none" and not self._both_temp(ki, kj):
                    continue  # κ pairs: noiseless only
                a, b = maps[li[ki]], maps[li[kj]]
                if tier != "none":
                    noise = self.noise.realisations(
                        tier, len(a), context=f"{self.name}|{source}|{ki}_{kj}"
                    )
                    a, b = a + noise, b + noise
                result[f"{self.prefix}_{tier}_{ki}_{kj}"] = self._one_tier(a, b, filters)
        return result

    def _cross_mean_err(self, r, ki, kj, m_i):
        tiers = self.params.get("noise_tiers", ["none"])
        mean_key = f"{self.prefix}_{'none' if 'none' in tiers else tiers[0]}_{ki}_{kj}"
        var_key = f"{self.prefix}_{self._variance_tier()}_{ki}_{kj}"
        if var_key not in r:  # κ pair — only the noiseless tier exists
            var_key = mean_key
        return r[mean_key][:, :, m_i].mean(axis=0), r[var_key][:, :, m_i].std(axis=0)

    @staticmethod
    def _cross_unit(label, base_a, base_b):
        r"""µK exponent of a raw cross-moment: one power per field occurrence
        in the superscript, counting only the fields carrying µK units."""
        suffix = label[2:]
        n = suffix.count("a") * (base_a == "uK") + suffix.count("b") * (base_b == "uK")
        if n == 0:
            return " [dimensionless]"
        return rf" [$\mu$K$^{{{n}}}$]" if n > 1 else r" [$\mu$K]"

    def plot(self, results, plot_path):
        ki, kj = self._primary_pair()
        base_a, base_b = _CHANNEL_BY_KEY[ki][2], _CHANNEL_BY_KEY[kj][2]
        labels = self._labels()
        plt, fig, axes = _subplots(4, nrows=3)
        for m_i, label in enumerate(labels):
            ax = axes[m_i]
            for src, r in self._ordered(results):
                if f"{self.prefix}_none_{ki}_{kj}" not in r:
                    continue
                mean, err = self._cross_mean_err(r, ki, kj, m_i)
                ax.errorbar(
                    r["band_centers"],
                    mean,
                    yerr=err,
                    label=_src_label(src),
                    marker=".",
                    capsize=2,
                    lw=1.4,
                    **_line_kw(src),
                )
            # raw moments: odd orders -> 0 for a Gaussian (zero check); even
            # orders sit at the non-zero Wick (disconnected) baseline
            ax.set_title(_cross_moment_tex(label) + self._cross_unit(label, base_a, base_b))
            ax.set_ylabel(_cross_moment_tex(label))
            _grid(ax, zero=True)
            ax.set_xlabel(r"$\ell$")
            if m_i == 0:
                _legend(ax)
        _finish(
            fig,
            plot_path,
            f"Cross-moments $M_p^{{ab}}$  ($a = ${_channel_display(ki)},"
            f"  $b = ${_channel_display(kj)})",
        )


# ---------------------------------------------------------------------------
# One-point and morphological statistics (notebook 08)
# ---------------------------------------------------------------------------


class PixelHistograms(Statistic):
    """Pixel-intensity histograms in physical μK (report §Summary statistics).

    Bin ranges come straight from the report text: [0, 100] μK for CIB and
    [−100, 0] μK for tSZ, 1,000 bins.  The raw (unsmoothed) density histogram
    is cached; Gaussian smoothing is applied at plot time since it is
    presentation, not measurement.
    """

    name = "pixel_histograms"
    n_field = True

    def _range(self, key, data):
        """Per-channel histogram range: config ``<key>_range`` if given, else
        a data-driven [μ−5σ, μ+8σ] (wide enough for the skewed foreground tails)."""
        p = self.params
        if f"{key}_range" in p:
            return tuple(p[f"{key}_range"])
        mu, sd = float(data.mean()), float(data.std())
        return (mu - 5.0 * sd, mu + 8.0 * sd)

    def compute(self, maps, source):
        nb = self.params["n_bins"]
        result = {}
        for i, key in enumerate(self.channel_labels):
            lo, hi = self._range(key, maps[i])
            bins = np.linspace(lo, hi, nb + 1)
            h, _ = np.histogram(maps[i], bins=bins, density=True)
            result[f"bins_{key}"] = 0.5 * (bins[:-1] + bins[1:])
            result[f"hist_{key}"] = h
        return result

    def plot(self, results, plot_path):
        from scipy.ndimage import gaussian_filter1d

        sigma = self.params.get("smooth_sigma", 1.0)
        keys = self.channel_labels
        plt, fig, axes = _subplots(len(keys))
        for ax, key in zip(axes, keys):
            unit = r" [$\mu$K]" if _CHANNEL_BY_KEY[key][2] == "uK" else ""
            for src, r in self._ordered(results):
                ax.plot(
                    r[f"bins_{key}"],
                    gaussian_filter1d(r[f"hist_{key}"], sigma=sigma),
                    label=_src_label(src),
                    lw=1.6,
                    **_line_kw(src),
                )
            # trim only the truly-empty ends, keeping the full non-Gaussian
            # tail: widest support across the physical sources (Agora + DDPM)
            # down to ~1e-6 of the peak density
            srcs = [s for s in ("agora", "ddpm") if s in results] or list(results)
            b = results[srcs[0]][f"bins_{key}"]
            dens = np.maximum.reduce([results[s][f"hist_{key}"] for s in srcs])
            support = np.where(dens > dens.max() * 1e-6)[0]
            if support.size:
                lo = b[max(int(support[0]) - 1, 0)]
                hi = b[min(int(support[-1]) + 1, len(b) - 1)]
                ax.set_xlim(lo, hi)
            ax.set_yscale("log")
            _grid(ax, minor=True)
            ax.set_title(_channel_display(key))
            ax.set_xlabel(f"{_channel_display(key)} pixel value{unit}")
            ax.set_ylabel("PDF [1/pixel value]" if unit else "PDF [dimensionless]")
            _legend(ax)
        _finish(fig, plot_path, "Pixel-intensity distributions")


class MinkowskiFunctionals(Statistic):
    name = "minkowski_functionals"
    n_field = True

    def compute(self, maps, source):
        from foregrounds_diffusion.morphology import compute_mfs

        p = self.params
        thresholds = np.linspace(p["threshold_min"], p["threshold_max"], p["n_thresholds"])
        result = {"thresholds": thresholds}
        for i, key in enumerate(self.channel_labels):
            m0, m1, m2 = compute_mfs(
                maps[i], apply_maxmin_normalization, thresholds, n_jobs=self.n_jobs
            )
            result[f"M0_{key}"], result[f"M1_{key}"], result[f"M2_{key}"] = m0, m1, m2
        return result

    def plot(self, results, plot_path):
        keys = self.channel_labels
        plt, fig, axes = _subplots(len(keys), nrows=3)
        axes = np.atleast_1d(axes).reshape(3, len(keys))
        # V_k(nu) with the threshold argument written out: distinguishes the
        # Minkowski functionals from the M_p N-point moments of the same name.
        # nu here runs over the MIN-MAX normalised map (compute() passes
        # apply_maxmin_normalization), so it is NOT the f/sigma of the peak counts.
        units = [r" [pixel$^2$]", " [pixel]", " [dimensionless]"]  # area, perimeter, Euler
        for row, mf in enumerate(["M0", "M1", "M2"]):
            v_tex = rf"$V_{row}(\nu)$"
            for col, key in enumerate(keys):
                ax = axes[row, col]
                for src, r in self._ordered(results):
                    arr = r[f"{mf}_{key}"]
                    m, s = arr.mean(axis=0), arr.std(axis=0)
                    ax.plot(r["thresholds"], m, label=_src_label(src), lw=1.6, **_line_kw(src))
                    ax.fill_between(
                        r["thresholds"], m - s, m + s, color=SOURCE_COLORS[src], alpha=0.18, lw=0
                    )
                ax.set_title(f"{v_tex} — {_channel_display(key)}")
                ax.set_ylabel(v_tex + units[row])
                _grid(ax, zero=(row == 2))
                ax.set_xlabel(r"threshold $\nu$ [min--max normalised]")
                if row == 0 and col == 0:
                    _legend(ax)
        _finish(fig, plot_path, r"Minkowski functionals $V_k(\nu)$")


class MinkowskiTensors(Statistic):
    """Rank-2 Minkowski tensor anisotropy β(ν) (notebook 12 extension)."""

    name = "minkowski_tensors"

    def compute(self, cib, tsz, source):
        from foregrounds_diffusion.morphology import compute_minkowski_tensors

        p = self.params
        thresholds = np.linspace(p["threshold_min"], p["threshold_max"], p["n_thresholds"])
        tensor_types = tuple(p.get("tensor_types", ["W021"]))
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

    # representative thresholds for the orientation distributions
    # (report §Summary statistics: ν = 0.2, 0.5, 0.8)
    THETA_NUS = (0.2, 0.5, 0.8)
    # tensor whose orientation/residuals the report singles out
    PRIMARY_TENSOR = "W021"

    def plot(self, results, plot_path):
        """Two figures: ``minkowski_tensors.png`` (β̄ anisotropy, each panel
        with its own residual box) and ``*_orientation.png`` (θ distributions).
        The single combined grid of all three was too busy."""
        import matplotlib

        matplotlib.use("Agg")
        _apply_style()
        import matplotlib.pyplot as plt

        tensor_types = list(self.params.get("tensor_types", ["W021"]))
        primary = self.PRIMARY_TENSOR if self.PRIMARY_TENSOR in tensor_types else tensor_types[0]

        def _sibling(suffix):
            return plot_path.with_name(f"{plot_path.stem}_{suffix}{plot_path.suffix}")

        # --- figure 1: β̄(ν) with residual boxes, field × tensor type ---------
        keys = ["cib", "tsz"]
        ncols = len(tensor_types)
        fig = plt.figure(figsize=(4.8 * ncols, 5.2 * len(keys)), layout="constrained")
        # two gridspec rows per field: the β̄ curve (3) over its residual (1)
        gs = fig.add_gridspec(2 * len(keys), ncols, height_ratios=[3, 1] * len(keys), hspace=0.05)
        for row, key in enumerate(keys):
            for col, ttype in enumerate(tensor_types):
                ax = fig.add_subplot(gs[2 * row, col])
                ax_r = fig.add_subplot(gs[2 * row + 1, col], sharex=ax)
                for src, r in self._ordered(results):
                    arr = r[f"beta_{key}_{ttype}"]
                    m, s = arr.mean(axis=0), arr.std(axis=0)
                    ax.plot(r["thresholds"], m, label=_src_label(src), lw=1.5, **_line_kw(src))
                    ax.fill_between(
                        r["thresholds"], m - s, m + s, color=SOURCE_COLORS[src], alpha=0.2, lw=0
                    )
                ax.set_ylim(0, 1)
                _grid(ax)
                ax.set_title(rf"$\bar\beta(\nu)$ — {_channel_display(key)} ({ttype})")
                ax.set_ylabel(r"$\bar\beta(\nu)$ [dimensionless]")
                ax.tick_params(labelbottom=False)
                if row == 0 and col == 0:
                    _legend(ax)

                # residual box: r(ν) = (beta_Agora - beta_src) / sigma_Agora.
                # Already in units of sigma, so no sigma bands are drawn. Both
                # non-Agora sources appear (the standalone residual figure this
                # replaces showed both); colour + linestyle carry the identity.
                lim = 1.0
                if "agora" in results:
                    a = results["agora"][f"beta_{key}_{ttype}"]
                    for src in ["ddpm", "gaussian"]:
                        if src not in results:
                            continue
                        d = results[src][f"beta_{key}_{ttype}"]
                        r_nu = (a.mean(axis=0) - d.mean(axis=0)) / (a.std(axis=0) + 1e-30)
                        ax_r.plot(results["agora"]["thresholds"], r_nu, lw=1.3, **_line_kw(src))
                        lim = max(lim, 1.15 * float(np.abs(r_nu).max()))
                ax_r.set_ylim(-lim, lim)
                _grid(ax_r, zero=True)
                ax_r.set_xlabel(r"threshold $\nu$")
                ax_r.set_ylabel(r"$r(\nu)$")
        _finish(fig, plot_path, r"Minkowski tensor anisotropy $\bar\beta(\nu)$", tight=False)

        # --- figure 2: orientation θ distributions (-> *_orientation.png) ---
        theta_bins = np.linspace(-np.pi / 2, np.pi / 2, 25)
        centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])
        plt, fig, axes = _subplots(len(self.THETA_NUS), nrows=2, height=3.4)
        axes = np.atleast_1d(axes).reshape(2, len(self.THETA_NUS))
        for row, key in enumerate(["cib", "tsz"]):
            for col, nu in enumerate(self.THETA_NUS):
                ax = axes[row, col]
                for src, r in self._ordered(results):
                    t_i = int(np.argmin(np.abs(r["thresholds"] - nu)))
                    hist, _ = np.histogram(
                        r[f"theta_{key}_{primary}"][:, t_i], bins=theta_bins, density=True
                    )
                    ax.step(
                        centers,
                        hist,
                        where="mid",
                        color=SOURCE_COLORS[src],
                        ls=SOURCE_LINESTYLES[src],
                        label=_src_label(src),
                    )
                ax.axhline(1 / np.pi, color="grey", lw=0.8, ls=":")  # isotropic ref
                _grid(ax)
                ax.set_title(rf"$\theta$ — {_channel_display(key)} ({primary}, $\nu = {nu:g}$)")
                ax.set_ylabel(r"PDF [1/rad]")
                ax.set_xlabel(r"major-axis orientation $\theta$ [rad]")
                if row == 0 and col == 0:
                    _legend(ax)
        _finish(fig, _sibling("orientation"), r"Minkowski tensor major-axis orientation $\theta$")

    def summarise(self, results):
        # report eq: r(ν) = (β̄_Agora − β̄_DDPM) / σ_Agora, quoted at max |r|
        lines = []
        if "agora" in results and "ddpm" in results:
            for ttype in self.params.get("tensor_types", ["W021"]):
                for key in ["cib", "tsz"]:
                    a = results["agora"][f"beta_{key}_{ttype}"]
                    d = results["ddpm"][f"beta_{key}_{ttype}"]
                    resid = np.abs(a.mean(axis=0) - d.mean(axis=0)) / (a.std(axis=0) + 1e-30)
                    nu_max = results["agora"]["thresholds"][resid.argmax()]
                    lines.append(
                        f"minkowski_tensors[{key},{ttype}]: max |Agora-DDPM| β residual "
                        f"{resid.max():.2f}σ (at ν={nu_max:.2f})"
                    )
        return lines


# ---------------------------------------------------------------------------
# tSZ stacking (notebook 09)
# ---------------------------------------------------------------------------


def _peaks_per_map(results, label):
    """Stacked-peak counts normalised by each source's map count.

    Sources are evaluated over different numbers of maps (Agora is limited to
    the test split, the DDPM to ``n_maps`` samples), so raw counts are not
    comparable between them — a source with more maps looks like it has more
    clusters.  Raw ``n`` is kept alongside the rate for traceability.
    """
    rates = {}
    for src, r in results.items():
        if f"n_{label}" not in r:
            continue
        n = int(r[f"n_{label}"])
        n_maps = int(r["n_maps"]) if "n_maps" in r else 0
        rates[src] = f"{n / n_maps:.2f} ({n})" if n_maps else f"? ({n})"
    return rates


def _peak_rate_legend(src, r, label):
    """Legend entry quoting peaks *per map* only — raw counts are not comparable
    between sources evaluated over different numbers of maps."""
    n = int(r[f"n_{label}"])
    if "n_maps" not in r:
        return _src_label(src)
    return f"{_src_label(src)} ({n / int(r['n_maps']):.2f}/map)"


class TszStacking(Statistic):
    """Stacked tSZ cluster profiles in SNR bins.

    tSZ at 150 GHz is a decrement (clusters are negative); peaks are selected
    and stacked on the sign-flipped map so the stacked amplitude is positive.
    The ``sign`` entry in the cache records the flip.

    Report convention: T̄ and σ_T are the mean and standard deviation of the
    *simulated* (Agora) tSZ maps, applied identically to every source so the
    SNR bins correspond to the same physical depths.  ``run_evaluate`` injects
    them as ``snr_ref_mean``/``snr_ref_std`` params (which also keys the cache
    to the reference values); without them each source falls back to its own
    global mean and per-map std — not comparable across sources with different
    variance.
    """

    name = "tsz_stacking"
    grid_plot = True  # drawn by _plot_stacking_grids, not by its own plot()
    target_key = "tsz"  # the stacked channel — orders the rows of the grids
    target_label = r"stacked $|\Delta T|$ [$\mu$K]"
    row_title = "tSZ on tSZ clusters"

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
        ref_mean = p.get("snr_ref_mean", tsz.mean())
        ref_std = p.get("snr_ref_std")  # None → per-map std inside the selector
        maps = sign * (tsz - ref_mean)
        result = {
            "sign": np.array(sign),
            "cutout_pix": np.array(cutout),
            "n_maps": np.array(len(tsz)),
        }
        half = cutout // 2
        idx = np.indices((cutout, cutout)).astype(float)
        xy = ((idx[0] - half) * dx_arcmin / 60.0, (idx[1] - half) * dx_arcmin / 60.0)
        for smin, smax in p["snr_bins"]:
            label = self._bin_label(smin, smax)
            coords = select_snr_pixels(maps, smin, smax, noise=ref_std)
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

    def summarise(self, results):
        lines = []
        for smin, smax in self.params["snr_bins"]:
            label = self._bin_label(smin, smax)
            rates = _peaks_per_map(results, label)
            lines.append(f"tsz_stacking[{label}]: peaks/map {rates}")
        return lines


# ---------------------------------------------------------------------------
# Cross-field cluster stacking (4-channel extension): select cluster locations
# by tSZ SNR, then stack a *different* channel there.
# ---------------------------------------------------------------------------


class _ClusterStack(Statistic):
    """Stack a target channel on tSZ-SNR-selected cluster locations.

    Peaks are selected from the tSZ map (sign-flipped, since 150 GHz tSZ is a
    decrement) in SNR bins using the same reference mean/std as ``TszStacking``
    (injected as ``snr_ref_mean``/``snr_ref_std``), so the SNR depths match
    across sources and across the stacking statistics. Subclasses set
    ``target_key`` (channel to stack), an optional ``_target`` transform (e.g.
    square for kSZ²), and the profile y-axis label ``target_label``.
    """

    n_field = True
    grid_plot = True  # drawn by _plot_stacking_grids, not by its own plot()
    select_key = "tsz"
    target_key = None
    target_label = ""
    row_title = ""

    def _target(self, arr):
        return arr

    @staticmethod
    def _bin_label(smin, smax):
        return f"{smin:g}-{smax:g}" if smax is not None else f"gt{smin:g}"

    def _channel(self, key):
        if key not in self.channel_labels:
            raise KeyError(f"{self.name} needs the {key!r} channel; run has {self.channel_labels}")
        return self.channel_labels.index(key)

    def compute(self, maps, source):
        p = self.params
        cutout = p["cutout_pix"]
        dx_arcmin = self.mapparams[2]
        tsz = maps[self._channel(self.select_key)]
        target = self._target(maps[self._channel(self.target_key)])
        # Select on the tSZ map: sign-flip a decrement so clusters are maxima.
        sign = -1.0 if np.abs(tsz.min()) > np.abs(tsz.max()) else 1.0
        ref_mean = p.get("snr_ref_mean", tsz.mean())
        ref_std = p.get("snr_ref_std")
        sel = sign * (tsz - ref_mean)
        result = {
            "sign": np.array(sign),
            "cutout_pix": np.array(cutout),
            "n_maps": np.array(len(tsz)),
        }
        half = cutout // 2
        idx = np.indices((cutout, cutout)).astype(float)
        xy = ((idx[0] - half) * dx_arcmin / 60.0, (idx[1] - half) * dx_arcmin / 60.0)
        for smin, smax in p["snr_bins"]:
            label = self._bin_label(smin, smax)
            coords = select_snr_pixels(sel, smin, smax, noise=ref_std)
            cuts = extract_cutouts(target, coords, cutout, max_cutouts=len(coords) or 1)
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

    def summarise(self, results):
        lines = []
        for smin, smax in self.params["snr_bins"]:
            label = self._bin_label(smin, smax)
            peak = {
                src: round(float(r[f"stack_{label}"][r[f"stack_{label}"].shape[0] // 2].max()), 4)
                for src, r in results.items()
                if f"stack_{label}" in r
            }
            rates = _peaks_per_map(results, label)
            lines.append(f"{self.name}[{label}]: peaks/map {rates}, central peak≈{peak}")
        return lines


class KappaOnTszStacking(_ClusterStack):
    """CMB-lensing κ stacked on tSZ cluster peaks — the mean cluster
    convergence profile; a direct test of the κ–tSZ cross-morphology."""

    name = "kappa_on_tsz_stacking"
    target_key = "kappa"
    target_label = r"stacked $\kappa$ [dimensionless]"
    row_title = r"$\kappa$ on tSZ clusters"


class KszStacking(_ClusterStack):
    """kSZ² stacked on tSZ cluster peaks. kSZ is sign-symmetric (random
    line-of-sight velocity), so the *mean* kSZ vanishes at clusters; the
    squared field gives the non-zero kSZ power (variance) profile at halos."""

    name = "ksz_stacking"
    target_key = "ksz"
    target_label = r"stacked $k_{\mathrm{SZ}}^2$ [$\mu$K$^2$]"
    row_title = r"$k_{\mathrm{SZ}}^2$ on tSZ clusters"

    def _target(self, arr):
        return arr**2


def _snr_title(label):
    """'5-10' -> '$5 < \\nu_{SNR} \\leq 10$'; 'gt20' -> '$\\nu_{SNR} > 20$'."""
    if label.startswith("gt"):
        return rf"$\nu_{{\mathrm{{SNR}}}} > {label[2:]}$"
    lo, hi = label.split("-")
    return rf"${lo} < \nu_{{\mathrm{{SNR}}}} \leq {hi}$"


def _plot_stacking_grids(entries, plots_dir, dx_arcmin, prefix=""):
    """Two combined figures across every active cluster-stacking statistic:
    ``<prefix>stacking_profiles.png`` (radial profiles) and
    ``<prefix>stacking_maps.png`` (the Agora stacked cutouts).  Rows are
    statistics, columns are SNR bins — a 3x3 grid for the 4-channel run, 1x3
    for the 2-channel v4 run.

    ``entries`` is a list of ``(stat, results)``; rows are ordered by the
    stacked channel (CIB, tSZ, kSZ, kappa), not by registry order.
    """
    if not entries:
        return []
    entries = sorted(entries, key=lambda e: CHANNEL_ORDER.index(e[0].target_key))
    written = []
    n_row = len(entries)
    bins = [entries[0][0]._bin_label(a, b) for a, b in entries[0][0].params["snr_bins"]]
    n_col = len(bins)

    # --- figure 1: radial profiles, all sources overlaid -------------------
    plt, fig, axes = _subplots(n_col, nrows=n_row, height=3.4)
    axes = np.atleast_1d(axes).reshape(n_row, n_col)
    for r_i, (stat, results) in enumerate(entries):
        for c_i, label in enumerate(bins):
            ax = axes[r_i, c_i]
            for src, r in stat._ordered(results):
                if f"profile_{label}" not in r:
                    continue
                prof = r[f"profile_{label}"]
                ax.errorbar(
                    prof[:, 0],
                    prof[:, 1],
                    yerr=prof[:, 2],
                    label=_peak_rate_legend(src, r, label),
                    marker=".",
                    lw=1.4,
                    capsize=2,
                    **_line_kw(src),
                )
            _grid(ax)
            ax.set_title(f"{stat.row_title} — {_snr_title(label)}", fontsize=10)
            ax.set_xlabel(r"$\theta$ [arcmin]")
            ax.set_ylabel(stat.target_label)
            # peaks/map depends only on the SNR bin (all rows select the same
            # clusters on the same tSZ map), so one legend per column suffices
            if r_i == 0:
                _legend(ax)
    path = plots_dir / f"{prefix}stacking_profiles.png"
    _finish(fig, path, "Stacked cluster radial profiles")
    written.append(path)

    # --- figure 2: the Agora stacked cutouts -------------------------------
    plt, fig, axes = _subplots(n_col, nrows=n_row, height=3.4)
    axes = np.atleast_1d(axes).reshape(n_row, n_col)
    half = None
    for r_i, (stat, results) in enumerate(entries):
        agora = results.get("agora", {})
        for c_i, label in enumerate(bins):
            ax = axes[r_i, c_i]
            key = f"stack_{label}"
            if key not in agora:
                ax.set_axis_off()
                continue
            stack = agora[key]
            half = 0.5 * stack.shape[0] * dx_arcmin
            # stacked amplitudes are positive magnitudes -> sequential, CVD-safe
            # colormap (a diverging map would put its neutral midpoint at an
            # arbitrary, meaningless value)
            im = ax.imshow(stack, cmap="cividis", origin="lower", extent=[-half, half, -half, half])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label(
                stat.target_label, fontsize=8
            )
            ax.set_title(f"{stat.row_title} — {_snr_title(label)}", fontsize=10)
            ax.set_xlabel(r"$\theta_x$ [arcmin]")
            ax.set_ylabel(r"$\theta_y$ [arcmin]")
    path = plots_dir / f"{prefix}stacking_maps.png"
    _finish(fig, path, "Agora stacked cluster cutouts")
    written.append(path)
    return written


# ---------------------------------------------------------------------------
# Peak / minima counts (notebook 10 extension)
# ---------------------------------------------------------------------------


class PeakCounts(Statistic):
    name = "peak_counts"
    n_field = True
    count_fn = staticmethod(count_peaks_binned)
    FIGURE_TITLE = r"Peak counts vs threshold $\nu$"

    def compute(self, maps, source):
        p = self.params
        # n_thresholds values are histogram bin EDGES (report: linspace(-1, 5, 30))
        edges = np.linspace(p["threshold_min"], p["threshold_max"], p["n_thresholds"])
        dx_arcmin = self.mapparams[2]
        result = {"thresholds": edges, "bin_centers": 0.5 * (edges[:-1] + edges[1:])}
        for i, key in enumerate(self.channel_labels):
            for fwhm in p["smoothing_fwhm_arcmin"]:
                result[f"{key}_fwhm{fwhm:g}"] = self.count_fn(
                    maps[i], edges, fwhm, pixel_res_arcmin=dx_arcmin
                )
        return result

    def plot(self, results, plot_path):
        scales = self.params["smoothing_fwhm_arcmin"]
        keys = self.channel_labels
        plt, fig, axes = _subplots(len(scales), nrows=len(keys), height=3.4)
        axes = np.atleast_1d(axes).reshape(len(keys), len(scales))
        for row, key in enumerate(keys):
            for col, fwhm in enumerate(scales):
                ax = axes[row, col]
                peak = 0.0
                for src, r in self._ordered(results):
                    arr = r[f"{key}_fwhm{fwhm:g}"]
                    mean, std = arr.mean(axis=0), arr.std(axis=0)
                    ax.stairs(
                        mean,
                        r["thresholds"],
                        color=SOURCE_COLORS[src],
                        ls=SOURCE_LINESTYLES[src],
                        label=_src_label(src),
                        lw=1.4,
                    )
                    ax.fill_between(
                        r["bin_centers"],
                        mean - std,
                        mean + std,
                        color=SOURCE_COLORS[src],
                        alpha=0.15,
                        step="mid",
                        lw=0,
                    )
                    peak = max(peak, float(np.max(mean + std)) if mean.size else 0.0)
                # Each panel autoscales to its OWN counts, so a channel/scale
                # with few peaks is not squashed flat by a busier neighbour.
                # symlog spans the wide count range while still showing the
                # near-zero bins a log axis hides; but below the linear
                # threshold symlog degenerates, so a sparse panel goes linear.
                if peak > 5.0:
                    ax.set_yscale("symlog", linthresh=1.0)
                else:
                    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
                if peak > 0:
                    ax.set_ylim(0, peak * 1.35)
                _grid(ax)
                ax.set_title(f"{_channel_display(key)}, FWHM $= {fwhm:g}'$")
                ax.set_ylabel("counts per map")
                ax.set_xlabel(r"threshold $\nu = f/\sigma$")
                if row == 0 and col == 0:
                    _legend(ax)
        _finish(fig, plot_path, self.FIGURE_TITLE)


class MinimaCounts(PeakCounts):
    name = "minima_counts"
    count_fn = staticmethod(count_minima_binned)
    FIGURE_TITLE = r"Minima counts vs threshold $\nu$"


# ---------------------------------------------------------------------------
# Scattering transforms (notebook 11 extension)
# ---------------------------------------------------------------------------


class ScatteringTransforms(Statistic):
    """S1/S2 scattering coefficients, plus (``covariance: true``) the
    scattering covariance (C01, C11) and the two-field CIB×tSZ cross
    covariance — the four WST tests of report §Summary statistics.

    Covariance vectors are the Cheng et al. ``for_synthesis_iso`` summaries
    (631 single-field / 2262 two-field coefficients at J=5, L=4); complex
    entries are cached as-is and compared via their real part.
    """

    name = "scattering_transforms"

    def compute(self, cib, tsz, source):
        from foregrounds_diffusion.scattering_stats import (
            compute_scattering_coefficients,
            compute_scattering_covariance,
            compute_scattering_covariance_2fields,
            scattering_summary,
        )

        p = self.params
        # default to CPU: batch FFTs over the full map stack overflow small
        # laptop GPUs, and the Cheng backend is fast on CPU (~0.1 s/map)
        device = p.get("device", "cpu")
        # covariance intermediates are (batch, J, L, H, W) complex — chunk the
        # stack to bound RAM (exact: normalisation is per image)
        cov_batch = p.get("cov_batch", 8)
        result = {}
        cib32 = np.ascontiguousarray(cib, dtype=np.float32)
        tsz32 = np.ascontiguousarray(tsz, dtype=np.float32)
        for key, maps in [("cib", cib32), ("tsz", tsz32)]:
            coeffs = compute_scattering_coefficients(maps, J=p["J"], L=p["L"], device=device)
            result[f"S1_{key}"] = coeffs["S1"]
            result[f"S2_{key}"] = coeffs["S2"]
            result[f"summary_{key}"] = scattering_summary(coeffs)
        if p.get("covariance", False):
            for key, maps in [("cib", cib32), ("tsz", tsz32)]:
                cov = compute_scattering_covariance(
                    maps, J=p["J"], L=p["L"], device=device, batch_size=cov_batch
                )
                if cov is None:
                    break  # Cheng backend unavailable — S1/S2 already cached
                result[f"C01_iso_{key}"] = cov["C01_iso"]
                result[f"C11_iso_{key}"] = cov["C11_iso"]
                result[f"synth_iso_{key}"] = cov["for_synthesis_iso"]
            cross = compute_scattering_covariance_2fields(
                cib32, tsz32, J=p["J"], L=p["L"], device=device, batch_size=cov_batch
            )
            if cross is not None:
                result["C01_iso_cross"] = cross["C01_iso"]
                result["C11_iso_cross"] = cross["C11_iso"]
                result["synth_iso_cross"] = cross["for_synthesis_iso"]
        return result

    def summarise(self, results):
        lines = []
        if "agora" in results and "ddpm" in results:
            for key in ["cib", "tsz", "cross"]:
                k = f"synth_iso_{key}"
                if k not in results["agora"] or k not in results["ddpm"]:
                    continue
                a = np.real(results["agora"][k])
                d = np.real(results["ddpm"][k])
                resid = np.abs(a.mean(axis=0) - d.mean(axis=0)) / (a.std(axis=0) + 1e-30)
                lines.append(
                    f"scattering_cov[{key}]: max |Agora-DDPM| residual "
                    f"{resid.max():.2f}σ (mean {resid.mean():.2f}σ, "
                    f"{a.shape[1]} iso coefficients)"
                )
        return lines

    def plot(self, results, plot_path):
        have_ad = "agora" in results and "ddpm" in results
        cov_keys = [
            k
            for k in ["cib", "tsz", "cross"]
            if have_ad
            and f"synth_iso_{k}" in results["agora"]
            and f"synth_iso_{k}" in results["ddpm"]
        ]
        nrows = 2 + (2 if cov_keys else 0)
        plt, fig, axes = _subplots(2, nrows=nrows, height=3.4)
        for col, key in enumerate(["cib", "tsz"]):
            ax = axes[col]
            for src, r in self._ordered(results):
                s1 = r[f"S1_{key}"]
                ax.errorbar(
                    np.arange(s1.shape[1]),
                    s1.mean(axis=0),
                    yerr=s1.std(axis=0),
                    color=SOURCE_COLORS[src],
                    ls=SOURCE_LINESTYLES[src],
                    label=_src_label(src),
                    marker="o",
                    ms=3,
                )
            ax.set_yscale("log")
            _grid(ax, minor=True)
            ax.set_xlabel("scale $j$")
            ax.set_ylabel(r"$S_1$ [dimensionless]")
            ax.set_title(rf"$S_1$ coefficients — {_channel_display(key)}")
            if col == 0:
                _legend(ax)
            ax = axes[2 + col]
            if have_ad:
                a = results["agora"][f"summary_{key}"]
                d = results["ddpm"][f"summary_{key}"]
                resid = (a.mean(axis=0) - d.mean(axis=0)) / (a.std(axis=0) + 1e-30)
                ax.bar(np.arange(len(resid)), resid, color=SOURCE_COLORS["ddpm"])
            _grid(ax, zero=True)
            ax.set_xlabel(r"feature index ($S_1 \oplus S_2$)")
            ax.set_ylabel(r"(Agora $-$ DDPM)/$\sigma$")
            ax.set_title(rf"$S_1 \oplus S_2$ residuals — {_channel_display(key)}")
        # scattering covariance residuals (report WST tests 3 and 4)
        for i, key in enumerate(cov_keys):
            ax = axes[4 + i]
            a = np.real(results["agora"][f"synth_iso_{key}"])
            d = np.real(results["ddpm"][f"synth_iso_{key}"])
            resid = (a.mean(axis=0) - d.mean(axis=0)) / (a.std(axis=0) + 1e-30)
            # distribution of the per-coefficient residuals — the jagged line
            # over hundreds/thousands of iso coefficients was unreadable
            rr = resid[np.isfinite(resid)]
            ax.hist(
                np.clip(rr, -3, 3), bins=40, color=SOURCE_COLORS["ddpm"], edgecolor="white", lw=0.3
            )
            ax.axvline(0, color="#55555a", lw=0.9, ls="--")
            frac = float(np.mean(np.abs(rr) < 1.0))
            _grid(ax)
            ax.set_xlabel(r"(Agora $-$ DDPM)/$\sigma$ per coefficient")
            ax.set_ylabel("count [coefficients]")
            label = "cross" if key == "cross" else _channel_display(key)
            ax.set_title(rf"$C_{{11}}$ residuals — {label}", fontsize=10)
            ax.text(
                0.03,
                0.94,
                f"{resid.size} coeffs\n{frac:.0%} within $1\\sigma$",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
            )
        for j in range(4 + len(cov_keys), len(axes)):
            axes[j].axis("off")
        _finish(fig, plot_path, "Scattering transforms")


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
        KappaOnTszStacking,
        KszStacking,
        PeakCounts,
        MinimaCounts,
        ScatteringTransforms,
    ]
}

# Statistics that select cluster locations by tSZ SNR — all share the Agora tSZ
# reference mean/std so the SNR depths are identical across sources and stacks.
_TSZ_STACKING_STATS = ("tsz_stacking", "kappa_on_tsz_stacking", "ksz_stacking")


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

    sources, norm_params, channel_labels, test_idx = load_sources(cfg, run)
    np.savez(
        run.stats / "test_split.npz",
        test_idx=test_idx,
        seed=cfg.data.seed,
        train_size=cfg.data.train_size,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
    )

    # Report convention: tSZ-stacking SNR bins are defined by the mean and
    # std of the *simulated* (Agora) maps, applied identically to every
    # source. Injected as params so the cache meta keys on the reference.
    stacking_stats = [s for s in stat_names if s in _TSZ_STACKING_STATS]
    if stacking_stats and "agora" in sources and "tsz" in channel_labels:
        ref_tsz = sources["agora"][channel_labels.index("tsz")]
        ref_mean = round(float(ref_tsz.mean()), 6)
        ref_std = round(float(ref_tsz.std()), 6)
        for s in stacking_stats:
            sp = cfg.evaluation.params.setdefault(s, {})
            sp["snr_ref_mean"], sp["snr_ref_std"] = ref_mean, ref_std

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

    # figures are tagged by field count ('2f_', '4f_') so the 2- and 4-field
    # runs' plots stay distinguishable once collected into the report
    prefix = _field_prefix(len(channel_labels))

    summary_lines = []
    stacking_entries = []
    for name in stat_names:
        stat = STATISTIC_REGISTRY[name](
            cfg.evaluation.params.get(name, {}),
            cfg.evaluation.n_jobs,
            mapparams,
            noise=noise,
            norm_params=norm_params,
            channel_labels=channel_labels,
        )
        results = {}
        for src, maps in sources.items():
            try:
                results[src] = stat.compute_or_load(run.stats, src, maps)
            except Exception as exc:  # keep going: one bad statistic must not
                # sink the whole overnight precompute run
                print(f"[evaluate] {name}/{src} FAILED: {exc!r}")
                summary_lines.append(f"{name}/{src}: FAILED — {exc!r}")
        if results and getattr(stat, "grid_plot", False):
            # stacking statistics share two combined grids, drawn after the loop
            stacking_entries.append((stat, results))
        elif results:
            plot_path = run.plots / f"{prefix}{name}.png"
            try:
                stat.plot(results, plot_path)
                print(f"[evaluate] wrote {plot_path}")
            except Exception as exc:
                print(f"[evaluate] {name} plot FAILED: {exc!r}")
            try:
                summary_lines.extend(stat.summarise(results))
            except Exception as exc:
                print(f"[evaluate] {name} summarise FAILED: {exc!r}")
                summary_lines.append(f"{name}: summarise FAILED — {exc!r}")

    try:
        for p in _plot_stacking_grids(stacking_entries, run.plots, mapparams[2], prefix=prefix):
            print(f"[evaluate] wrote {p}")
    except Exception as exc:
        print(f"[evaluate] stacking grids plot FAILED: {exc!r}")

    summary = run.stats / "summary.md"
    with open(summary, "w") as f:
        f.write(f"# Evaluation summary — {cfg.run_name}\n\n")
        f.write(
            "Sources: "
            + ", ".join(f"{s} ({m.shape[1]} maps)" for s, m in sources.items())
            + f"\nChannels: {', '.join(channel_labels)}\n\n"
        )
        for line in summary_lines:
            f.write(f"- {line}\n")
    print(f"[evaluate] summary written to {summary}")
