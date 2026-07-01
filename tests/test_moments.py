import numpy as np
import pytest

from foregrounds_diffusion.flatmaps import get_lpf_hpf, make_gaussian_realisation
from foregrounds_diffusion.moments import (
    compute_cross_moments,
    compute_summed_moments,
    mean_cls,
    mean_cross_cls,
)

_EL = np.arange(1, 5000)
_CL = 1e-5 * _EL.astype(float) ** (-2)

_EXPECTED_LABELS = [
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


@pytest.fixture
def bp_filters(flatskymapparams):
    bands = [(200, 500), (500, 1500), (1500, 4000)]
    return [get_lpf_hpf(flatskymapparams, band, filter_type=2) for band in bands]


# ---------------------------------------------------------------------------
# mean_cls
# ---------------------------------------------------------------------------


def test_mean_cls_return_shapes(patch_stack, flatskymapparams):
    el, mean_cl, std_cl = mean_cls(patch_stack, flatskymapparams, lmin=200, lmax=5000, binsize=200)
    assert el.ndim == 1
    assert mean_cl.shape == el.shape
    assert std_cl.shape == el.shape


def test_mean_cls_auto_positive(patch_stack, flatskymapparams):
    el, mean_cl, std_cl = mean_cls(patch_stack, flatskymapparams, lmin=200, lmax=5000, binsize=200)
    assert np.all(mean_cl >= 0)


# ---------------------------------------------------------------------------
# mean_cross_cls
# ---------------------------------------------------------------------------


def test_mean_cross_cls_independent_near_zero(flatskymapparams):
    np.random.seed(10)
    maps_a = np.array([make_gaussian_realisation(flatskymapparams, _EL, _CL) for _ in range(30)])
    np.random.seed(20)
    maps_b = np.array([make_gaussian_realisation(flatskymapparams, _EL, _CL) for _ in range(30)])
    el, mean_cl, std_cl = mean_cross_cls(
        maps_a, maps_b, flatskymapparams, lmin=200, lmax=5000, binsize=200
    )
    # Cross-spectrum of independent fields: mean should be small relative to std
    assert np.abs(mean_cl).mean() < 3 * std_cl.mean()


# ---------------------------------------------------------------------------
# compute_summed_moments
# ---------------------------------------------------------------------------


def test_compute_summed_moments_shape(patch_stack, flatskymapparams, bp_filters):
    N = len(patch_stack)
    result = compute_summed_moments(patch_stack, patch_stack, bp_filters)
    assert result.shape == (N, len(bp_filters), 3)


def test_compute_summed_moments_gaussian_skewness_near_zero(flatskymapparams, bp_filters):
    np.random.seed(30)
    N = 30
    maps = np.array([make_gaussian_realisation(flatskymapparams, _EL, _CL) for _ in range(N)])
    result = compute_summed_moments(maps, maps, bp_filters)
    mean_s3 = np.abs(result[:, :, 1]).mean()
    assert mean_s3 < 1.0  # S3 (skewness) near zero for Gaussian field


# ---------------------------------------------------------------------------
# compute_cross_moments
# ---------------------------------------------------------------------------


def test_compute_cross_moments_shape(patch_stack, flatskymapparams, bp_filters):
    N = len(patch_stack)
    moments_out, labels = compute_cross_moments(patch_stack, patch_stack, bp_filters)
    assert moments_out.shape == (N, len(bp_filters), 12)
    assert len(labels) == 12


def test_compute_cross_moments_labels(patch_stack, flatskymapparams, bp_filters):
    _, labels = compute_cross_moments(patch_stack, patch_stack, bp_filters)
    assert labels == _EXPECTED_LABELS


# ---------------------------------------------------------------------------
# n_jobs parallelism: serial == parallel
# ---------------------------------------------------------------------------


def test_mean_cls_parallel_matches_serial(patch_stack, flatskymapparams):
    el_s, cl_s, std_s = mean_cls(patch_stack, flatskymapparams, 300, 4000, 100, n_jobs=1)
    el_p, cl_p, std_p = mean_cls(patch_stack, flatskymapparams, 300, 4000, 100, n_jobs=2)
    np.testing.assert_array_equal(el_s, el_p)
    np.testing.assert_allclose(cl_s, cl_p, rtol=1e-10)
    np.testing.assert_allclose(std_s, std_p, rtol=1e-10)


def test_compute_summed_moments_parallel_matches_serial(patch_stack, bp_filters):
    serial = compute_summed_moments(patch_stack, patch_stack, bp_filters, n_jobs=1)
    parallel = compute_summed_moments(patch_stack, patch_stack, bp_filters, n_jobs=2)
    np.testing.assert_allclose(serial, parallel, rtol=1e-10)


def test_compute_cross_moments_parallel_matches_serial(patch_stack, bp_filters):
    s_out, s_labels = compute_cross_moments(patch_stack, patch_stack, bp_filters, n_jobs=1)
    p_out, p_labels = compute_cross_moments(patch_stack, patch_stack, bp_filters, n_jobs=2)
    np.testing.assert_allclose(s_out, p_out, rtol=1e-10)
    assert s_labels == p_labels


# ---------------------------------------------------------------------------
# Cross-moment correctness with DISTINCT channels.
#
# The tests above pass the same array as both CIB and tSZ, so S2aa == S2bb ==
# S2ab and S3aab == S3abb identically — a channel swap or a wrong cross-term
# exponent (e.g. a**2 * b coded as a * b**2) would pass silently.  The two
# tests below use genuinely distinct a and b to close that gap.
# ---------------------------------------------------------------------------


def _skewed_field(flatskymapparams, seed):
    """Non-Gaussian field with nonzero odd moments (chi-square-like)."""
    nx, ny = flatskymapparams[0], flatskymapparams[1]
    rng = np.random.default_rng(seed)
    return rng.standard_normal((nx, ny)) ** 2


def test_cross_moments_scaled_channel_ratios(flatskymapparams, bp_filters):
    # b = 2a exactly.  bandpass_filter is linear, so b_filtered = 2 a_filtered,
    # giving each of the 12 moments an exact, DISTINCT ratio to its a-only
    # counterpart.  Any channel swap or wrong cross-term exponent breaks one.
    a = _skewed_field(flatskymapparams, 100)
    cib = a[None]
    tsz = 2.0 * a[None]
    out, labels = compute_cross_moments(cib, tsz, bp_filters)
    m = {lab: out[0, :, i] for i, lab in enumerate(labels)}

    np.testing.assert_allclose(m["S2bb"], 4 * m["S2aa"], rtol=1e-6)
    np.testing.assert_allclose(m["S2ab"], 2 * m["S2aa"], rtol=1e-6)
    np.testing.assert_allclose(m["S3bbb"], 8 * m["S3aaa"], rtol=1e-6)
    np.testing.assert_allclose(m["S3aab"], 2 * m["S3aaa"], rtol=1e-6)
    np.testing.assert_allclose(m["S3abb"], 4 * m["S3aaa"], rtol=1e-6)
    np.testing.assert_allclose(m["S4bbbb"], 16 * m["S4aaaa"], rtol=1e-6)
    np.testing.assert_allclose(m["S4aaab"], 2 * m["S4aaaa"], rtol=1e-6)
    np.testing.assert_allclose(m["S4aabb"], 4 * m["S4aaaa"], rtol=1e-6)
    np.testing.assert_allclose(m["S4abbb"], 8 * m["S4aaaa"], rtol=1e-6)
    # non-triviality: the a-only odd moment must be genuinely nonzero
    assert np.max(np.abs(m["S3aaa"])) > 0


def test_cross_moments_analytic_values(flatskymapparams, bp_filters):
    # Pin the exact definition of each moment against a direct numpy
    # computation on two INDEPENDENT distinct fields (so a**2*b != a*b**2).
    from foregrounds_diffusion.flatmaps import bandpass_filter

    a = _skewed_field(flatskymapparams, 1)
    b = _skewed_field(flatskymapparams, 2)
    out, labels = compute_cross_moments(a[None], b[None], bp_filters)
    m = {lab: out[0, :, i] for i, lab in enumerate(labels)}
    for j, bp in enumerate(bp_filters):
        af = bandpass_filter(a, bp)
        bf = bandpass_filter(b, bp)
        np.testing.assert_allclose(m["S2aa"][j], np.mean(af**2), rtol=1e-6)
        np.testing.assert_allclose(m["S2bb"][j], np.mean(bf**2), rtol=1e-6)
        np.testing.assert_allclose(m["S2ab"][j], np.mean(af * bf), rtol=1e-6)
        np.testing.assert_allclose(m["S3aab"][j], np.mean(af**2 * bf), rtol=1e-6)
        np.testing.assert_allclose(m["S3abb"][j], np.mean(af * bf**2), rtol=1e-6)
        np.testing.assert_allclose(m["S4aabb"][j], np.mean(af**2 * bf**2), rtol=1e-6)
        np.testing.assert_allclose(m["S4abbb"][j], np.mean(af * bf**3), rtol=1e-6)


# ---------------------------------------------------------------------------
# Non-Gaussian regime: the summed-moment S3/S4 columns are near zero for a
# Gaussian field (tested above) but must respond to genuine non-Gaussianity.
# Uses the correlated lognormal fixture, whose skewness survives bandpass
# filtering (unlike per-pixel chi-square noise, which the CLT Gaussianises).
# ---------------------------------------------------------------------------


def test_skewed_patch_stack_is_nongaussian(skewed_patch_stack):
    from scipy.stats import kurtosis, skew

    flat = skewed_patch_stack.ravel()
    assert skew(flat) > 1.0  # empirically ≈ 5; Gaussian ≈ 0
    assert kurtosis(flat) > 1.0


def test_summed_moments_nongaussian_exceeds_gaussian(patch_stack, skewed_patch_stack, bp_filters):
    gauss = compute_summed_moments(patch_stack, patch_stack, bp_filters)
    skewed = compute_summed_moments(skewed_patch_stack, skewed_patch_stack, bp_filters)
    # Skewness (S3) and excess kurtosis (S4) magnitudes are substantially
    # larger for the non-Gaussian field even after bandpass filtering.
    assert np.abs(skewed[:, :, 1]).mean() > 1.5 * np.abs(gauss[:, :, 1]).mean()
    assert np.abs(skewed[:, :, 2]).mean() > 1.5 * np.abs(gauss[:, :, 2]).mean()
