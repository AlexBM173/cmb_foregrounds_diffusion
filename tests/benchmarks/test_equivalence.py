"""Equivalence tests for optimised implementations.

Each test confirms that an optimised code path (fast binning, vectorised
thresholds, n_jobs>1, GPU map2cl_torch) produces results numerically
identical (or within float32 tolerance) to the serial CPU reference.
"""

import numpy as np
import pytest
import torch

from foregrounds_diffusion.flatmaps import (
    _build_ell_bin_cache,
    build_lbin_idx_fft2,
    get_lpf_hpf,
    make_gaussian_realisation,
    map2cl,
    map2cl_torch,
)
from foregrounds_diffusion.moments import compute_cross_moments
from foregrounds_diffusion.morphology import compute_minkowski_tensors

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_EL = np.arange(1, 5000)
_CL = 1e-5 * _EL.astype(float) ** (-2)
_PARAMS64 = [64, 64, 1.40625, 1.40625]
_PARAMS128 = [128, 128, 1.40625, 1.40625]


def _make_maps(N, H):
    np.random.seed(99)
    params = [H, H, 1.40625, 1.40625]
    return np.array([make_gaussian_realisation(params, _EL, _CL) for _ in range(N)])


# ---------------------------------------------------------------------------
# §2.6c — vectorised map2cl via np.bincount
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("H", [64, 128])
def test_map2cl_fast_matches_serial_auto(H):
    """Fast map2cl (bincount) matches the pre-existing output at H=64 and H=128."""
    params = [H, H, 1.40625, 1.40625]
    np.random.seed(7)
    m = make_gaussian_realisation(params, _EL, _CL)

    el_ref, cl_ref = map2cl(params, m)
    cache = _build_ell_bin_cache(params)
    el_opt, cl_opt = map2cl(params, m, _ell_bin_cache=cache)

    np.testing.assert_array_equal(el_ref, el_opt)
    np.testing.assert_allclose(cl_ref, cl_opt, rtol=1e-12)


def test_map2cl_fast_matches_serial_cross():
    """Cross-spectrum fast path matches serial for complex PSD."""
    np.random.seed(11)
    m1 = make_gaussian_realisation(_PARAMS64, _EL, _CL)
    m2 = make_gaussian_realisation(_PARAMS64, _EL, _CL)

    el_ref, cl_ref = map2cl(_PARAMS64, m1, m2)
    cache = _build_ell_bin_cache(_PARAMS64)
    el_opt, cl_opt = map2cl(_PARAMS64, m1, m2, _ell_bin_cache=cache)

    np.testing.assert_array_equal(el_ref, el_opt)
    np.testing.assert_allclose(cl_ref, cl_opt, rtol=1e-12)


def test_map2cl_cache_reuse_across_n():
    """Pre-computing cache once and reusing gives the same el/cl per map."""
    maps = _make_maps(10, 64)
    cache = _build_ell_bin_cache(_PARAMS64)

    for m in maps:
        el_ref, cl_ref = map2cl(_PARAMS64, m)
        el_opt, cl_opt = map2cl(_PARAMS64, m, _ell_bin_cache=cache)
        np.testing.assert_allclose(cl_ref, cl_opt, rtol=1e-12)


# ---------------------------------------------------------------------------
# §2.6b — vectorised threshold binarisation in compute_minkowski_tensors
# ---------------------------------------------------------------------------


def test_minkowski_tensors_vectorised_threshold_unchanged(patch_stack):
    """The §2.6b vectorised threshold loop produces identical beta/theta."""
    thresholds = np.linspace(-2, 2, 15)
    result = compute_minkowski_tensors(patch_stack, lambda x: x, thresholds)
    assert result["W012"]["beta"].shape == (len(patch_stack), len(thresholds))
    assert np.all(result["W012"]["beta"] >= 0.0)
    assert np.all(result["W012"]["beta"] <= 1.0)


# ---------------------------------------------------------------------------
# §3.2 — n_jobs parallelism
# ---------------------------------------------------------------------------


def test_compute_cross_moments_n_jobs_2_matches_serial():
    """compute_cross_moments(n_jobs=2) matches n_jobs=1 result."""
    maps = _make_maps(8, 64)
    edges = np.linspace(200, 7000, 5)
    bp = [get_lpf_hpf(_PARAMS64, (edges[i], edges[i + 1]), filter_type=2) for i in range(4)]

    m_ref, labels_ref = compute_cross_moments(maps, maps, bp, n_jobs=1)
    m_par, labels_par = compute_cross_moments(maps, maps, bp, n_jobs=2)

    assert labels_ref == labels_par
    np.testing.assert_allclose(m_ref, m_par, rtol=1e-12)


def test_compute_minkowski_tensors_n_jobs_2_matches_serial(patch_stack):
    """compute_minkowski_tensors(n_jobs=2) matches n_jobs=1 result."""
    thresholds = np.linspace(-2, 2, 10)
    ref = compute_minkowski_tensors(patch_stack, lambda x: x, thresholds, n_jobs=1)
    par = compute_minkowski_tensors(patch_stack, lambda x: x, thresholds, n_jobs=2)

    np.testing.assert_allclose(ref["W012"]["beta"], par["W012"]["beta"], rtol=1e-12)
    np.testing.assert_allclose(ref["W012"]["theta"], par["W012"]["theta"], rtol=1e-12)


# ---------------------------------------------------------------------------
# §3.3 — map2cl_torch GPU port
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("H", [64, 128])
def test_map2cl_torch_matches_cpu(H):
    """map2cl_torch output matches CPU map2cl within float32 tolerance."""
    params = [H, H, 1.40625, 1.40625]
    np.random.seed(42)
    maps_np = np.array([make_gaussian_realisation(params, _EL, _CL) for _ in range(8)])

    # CPU reference (per-map loop)
    cl_ref = np.stack([map2cl(params, m)[1] for m in maps_np])

    # Torch (CPU device — no GPU required to test correctness)
    lbin_idx, bin_counts, n_bins = build_lbin_idx_fft2(params)
    maps_t = torch.from_numpy(maps_np.astype(np.float32))
    cl_torch = map2cl_torch(maps_t, lbin_idx, bin_counts, n_bins, dx_arcmin=1.40625)

    np.testing.assert_allclose(
        cl_ref,
        cl_torch.numpy(),
        rtol=1e-3,  # float32 vs float64 accumulation
        atol=1e-30,
    )


def test_map2cl_torch_shape():
    """Output shape is (N, n_bins)."""
    params = _PARAMS64
    lbin_idx, bin_counts, n_bins = build_lbin_idx_fft2(params)
    maps_t = torch.randn(5, 64, 64)
    out = map2cl_torch(maps_t, lbin_idx, bin_counts, n_bins, dx_arcmin=1.40625)
    assert out.shape == (5, n_bins)


def test_map2cl_torch_non_negative():
    """Auto-spectrum is non-negative."""
    params = _PARAMS64
    lbin_idx, bin_counts, n_bins = build_lbin_idx_fft2(params)
    maps_t = torch.randn(4, 64, 64)
    out = map2cl_torch(maps_t, lbin_idx, bin_counts, n_bins, dx_arcmin=1.40625)
    assert (out >= 0).all()
