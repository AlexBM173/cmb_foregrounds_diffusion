"""Tests for pipeline/evaluate.py — cached statistics on synthetic maps."""

import json

import numpy as np
import pytest

from config.validate import load_config
from pipeline.evaluate import (
    NoiseModel,
    PixelHistograms,
    PowerSpectrum,
    TszStacking,
    load_sources,
    main,
)
from pipeline.rundir import RunDir

RES = 64
N_PATCHES = 20  # -> 16 train, 4 held out for evaluation (val+test of the 80/10/10 split)


@pytest.fixture
def patches_dir(tmp_path):
    """Synthetic z-scored patch files with the notebook-03 naming scheme."""
    d = tmp_path / "patches"
    d.mkdir()
    rng = np.random.default_rng(7)
    cib = rng.standard_normal((N_PATCHES, RES, RES, 1)).astype(np.float32)
    # tSZ with a decrement (negative) tail, in z-score space
    tsz = (
        rng.standard_normal((N_PATCHES, RES, RES, 1))
        - 0.5 * rng.exponential(1.0, (N_PATCHES, RES, RES, 1))
    ).astype(np.float32)
    gauss = rng.standard_normal((N_PATCHES, 2, RES, RES)).astype(np.float32)
    np.save(d / f"CIB_map_150GHz_{RES}_st6_zscore_2mJy_lp.npy", cib)
    np.save(d / f"tSZ3_map_150GHz_{RES}_st6_zscore_2mJy_lp.npy", tsz)
    np.save(d / "gaussian_cib_tsz_2mJy_lp.npy", gauss)
    np.save(d / "norm_params_2mJy.npy", np.array([20.0, 5.0, -5.0, 3.5]))
    return d


@pytest.fixture
def ilc_file(tmp_path):
    ell = np.arange(2, 3000).astype(float)
    nl = 1e-6 * (ell / 1000.0) ** 2 + 1e-7
    ilc = {"total_ilc_residuals": {"spt3g": {"mv": (ell, nl)}}}
    path = tmp_path / "ilc.npy"
    np.save(path, ilc, allow_pickle=True)
    return path


@pytest.fixture
def cfg(tmp_path, patches_dir, ilc_file):
    yaml_text = f"""
run_name: eval_test
output:
  base_dir: {tmp_path / "runs"}
data:
  res: {RES}
  patches_dir: {patches_dir}
evaluation:
  n_jobs: 1
  ilc_noise_file: {ilc_file}
  noise_seed: 123
  statistics: [power_spectrum, cross_spectrum, moments, pixel_histograms,
               tsz_stacking, peak_counts]
  power_spectrum: {{lmin: 300, lmax: 1500, binsize: 200, n_maps: 10}}
  cross_spectrum: {{lmin: 300, lmax: 1500, binsize: 200, n_maps: 10}}
  moments: {{n_bands: 2, lmin: 300, lmax: 1700, n_maps: 10, noise_tiers: [none, spt3g]}}
  pixel_histograms: {{n_bins: 50, cib_range: [-5.0, 5.0], tsz_range: [-8.0, 4.0],
                     smooth_sigma: 1.0, n_maps: 10}}
  tsz_stacking: {{snr_bins: [[1, 2], [2, null]], cutout_pix: 16, n_maps: 10}}
  peak_counts: {{smoothing_fwhm_arcmin: [5.0], threshold_min: -1.0,
                threshold_max: 3.0, n_thresholds: 10, n_maps: 10}}
"""
    path = tmp_path / "eval_test.yaml"
    path.write_text(yaml_text)
    return load_config(path)


@pytest.fixture
def run(tmp_path):
    return RunDir(root=tmp_path / "runs" / "eval_test")


def test_load_sources_split_and_units(cfg, run):
    sources, norm_params, channel_labels, test_idx = load_sources(cfg, run)
    assert set(sources) == {"agora", "gaussian"}  # no ddpm samples yet
    assert channel_labels == ["cib", "tsz"]
    assert len(test_idx) == 4
    cib, tsz = sources["agora"]  # (C, N, H, W) unpacks to two (N, H, W) for C=2
    assert cib.shape == (4, RES, RES)
    # denormalised to physical units: mean should sit near cib_mean=20
    assert 10.0 < cib.mean() < 30.0
    # every patch train.py withheld: it trains on indices[:int(0.8*n)] and never
    # splits off a validation set, so val+test together are the held-out set
    expected = np.random.default_rng(seed=42).permutation(N_PATCHES)[16:]
    np.testing.assert_array_equal(test_idx, expected)


def test_load_sources_picks_up_ddpm_samples(cfg, run):
    run.samples.mkdir(parents=True)
    np.save(run.samples / "samples.npy", np.zeros((4, 2, RES, RES), dtype=np.float32))
    sources, _, _, _ = load_sources(cfg, run)
    assert "ddpm" in sources
    cib, tsz = sources["ddpm"]
    assert cib.shape == (4, RES, RES)
    # zeros in z-score space denormalise to the channel means
    assert np.allclose(cib, 20.0) and np.allclose(tsz, -5.0)


def test_noise_model_deterministic(ilc_file):
    mp = [RES, RES, 5.625, 5.625]
    nm = NoiseModel(ilc_file, mp, base_seed=1)
    a = nm.realisations("spt3g", 3, context="x")
    b = nm.realisations("spt3g", 3, context="x")
    c = nm.realisations("spt3g", 3, context="y")
    assert a.shape == (3, RES, RES)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    with pytest.raises(KeyError):
        nm.realisations("nonexistent_tier", 1)


def test_main_end_to_end_and_caching(cfg, run, capsys):
    from pipeline.evaluate import STATISTIC_REGISTRY

    main(cfg, run)
    for stat in cfg.evaluation.statistics:
        for src in ["agora", "gaussian"]:
            f = run.stats / f"{stat}__{src}.npz"
            assert f.exists(), f"missing cache {f}"
        # figures are tagged with the run's field count; cluster-stacking
        # statistics share two combined grid figures instead of one each
        if getattr(STATISTIC_REGISTRY[stat], "grid_plot", False):
            assert (run.plots / "2f_stacking_profiles.png").exists()
            assert (run.plots / "2f_stacking_maps.png").exists()
        else:
            assert (run.plots / f"2f_{stat}.png").exists()
    assert (run.stats / "test_split.npz").exists()
    assert (run.stats / "summary.md").exists()

    # noise tiers present in the moments cache
    with np.load(run.stats / "moments__agora.npz") as f:
        assert "summed_none" in f.files and "summed_spt3g" in f.files
        assert f["summed_none"].shape == (4, 2, 3)  # (N_test, bands, [S2,S3,S4])
        assert not np.allclose(f["summed_none"], f["summed_spt3g"])

    # second run must hit every cache
    mtimes = {f: f.stat().st_mtime_ns for f in run.stats.glob("*__*.npz")}
    capsys.readouterr()
    main(cfg, run)
    out = capsys.readouterr().out
    assert out.count("cached") == len(mtimes)
    assert mtimes == {f: f.stat().st_mtime_ns for f in run.stats.glob("*__*.npz")}


def test_parameter_change_invalidates_cache(cfg, run):
    stat = PowerSpectrum(
        {"lmin": 300, "lmax": 1500, "binsize": 200, "n_maps": 4}, 1, [RES, RES, 5.625, 5.625]
    )
    rng = np.random.default_rng(0)
    maps = rng.standard_normal((2, 4, RES, RES))  # (C, N, H, W)
    run.stats.mkdir(parents=True)
    r1 = stat.compute_or_load(run.stats, "agora", maps)
    stat2 = PowerSpectrum(
        {"lmin": 300, "lmax": 1500, "binsize": 100, "n_maps": 4}, 1, [RES, RES, 5.625, 5.625]
    )
    r2 = stat2.compute_or_load(run.stats, "agora", maps)
    assert len(r2["el"]) != len(r1["el"])
    # cache now holds the new parameters
    with np.load(stat2.cache_file(run.stats, "agora")) as f:
        assert json.loads(str(f["__meta__"]))["binsize"] == 100


def test_pixel_histograms_physical_units():
    # report convention: histograms of the physical-unit maps directly
    params = {
        "n_bins": 40,
        "cib_range": [0.0, 40.0],
        "tsz_range": [-20.0, 0.0],
        "smooth_sigma": 1.0,
        "n_maps": 8,
    }
    stat = PixelHistograms(params, 1, [RES, RES, 5.625, 5.625])
    rng = np.random.default_rng(3)
    cib = rng.standard_normal((8, RES, RES)) * 3.0 + 20.0
    tsz = rng.standard_normal((8, RES, RES)) * 2.0 - 8.0
    r = stat.compute(np.stack([cib, tsz]), "agora")
    for key, centre in [("cib", 20.0), ("tsz", -8.0)]:
        h, bc = r[f"hist_{key}"], r[f"bins_{key}"]
        assert np.all(h >= 0)
        # density integrates to ~1 over a range covering the distribution
        assert np.isclose(np.trapezoid(h, bc), 1.0, atol=0.05)
        # histogram peaks near the physical mean
        assert abs(bc[np.argmax(h)] - centre) < 2.0


def test_tsz_stacking_detects_decrement_sign():
    params = {"snr_bins": [[1, None]], "cutout_pix": 8, "n_maps": 4}
    stat = TszStacking(params, 1, [RES, RES, 5.625, 5.625])
    rng = np.random.default_rng(4)
    tsz = rng.standard_normal((4, RES, RES))
    tsz[:, 30, 30] = -25.0  # strong decrements -> clusters are minima
    r = stat.compute(None, tsz, "agora")
    assert r["sign"] == -1.0
    assert int(r["n_gt1"]) > 0
    # the stacked centre must be positive after the sign flip
    stack = r["stack_gt1"]
    c = stack.shape[0] // 2
    assert stack[c, c] > 0
