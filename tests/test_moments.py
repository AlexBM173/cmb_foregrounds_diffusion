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
    'S2aa', 'S2bb', 'S2ab',
    'S3aaa', 'S3bbb', 'S3aab', 'S3abb',
    'S4aaaa', 'S4bbbb', 'S4aaab', 'S4aabb', 'S4abbb',
]


@pytest.fixture
def bp_filters(flatskymapparams):
    bands = [(200, 500), (500, 1500), (1500, 4000)]
    return [get_lpf_hpf(flatskymapparams, band, filter_type=2) for band in bands]


# ---------------------------------------------------------------------------
# mean_cls
# ---------------------------------------------------------------------------

def test_mean_cls_return_shapes(patch_stack, flatskymapparams):
    el, mean_cl, std_cl = mean_cls(patch_stack, flatskymapparams,
                                    lmin=200, lmax=5000, binsize=200)
    assert el.ndim == 1
    assert mean_cl.shape == el.shape
    assert std_cl.shape == el.shape


def test_mean_cls_auto_positive(patch_stack, flatskymapparams):
    el, mean_cl, std_cl = mean_cls(patch_stack, flatskymapparams,
                                    lmin=200, lmax=5000, binsize=200)
    assert np.all(mean_cl >= 0)


# ---------------------------------------------------------------------------
# mean_cross_cls
# ---------------------------------------------------------------------------

def test_mean_cross_cls_independent_near_zero(flatskymapparams):
    np.random.seed(10)
    maps_a = np.array([make_gaussian_realisation(flatskymapparams, _EL, _CL)
                       for _ in range(30)])
    np.random.seed(20)
    maps_b = np.array([make_gaussian_realisation(flatskymapparams, _EL, _CL)
                       for _ in range(30)])
    el, mean_cl, std_cl = mean_cross_cls(maps_a, maps_b, flatskymapparams,
                                          lmin=200, lmax=5000, binsize=200)
    # Cross-spectrum of independent fields: mean should be small relative to std
    assert np.abs(mean_cl).mean() < 3 * std_cl.mean()


# ---------------------------------------------------------------------------
# compute_summed_moments
# ---------------------------------------------------------------------------

def test_compute_summed_moments_shape(patch_stack, flatskymapparams, bp_filters):
    N = len(patch_stack)
    result = compute_summed_moments(patch_stack, patch_stack, bp_filters)
    assert result.shape == (N, len(bp_filters), 3)


def test_compute_summed_moments_gaussian_skewness_near_zero(
        flatskymapparams, bp_filters):
    np.random.seed(30)
    N = 30
    maps = np.array([make_gaussian_realisation(flatskymapparams, _EL, _CL)
                     for _ in range(N)])
    result = compute_summed_moments(maps, maps, bp_filters)
    mean_s3 = np.abs(result[:, :, 1]).mean()
    assert mean_s3 < 1.0   # S3 (skewness) near zero for Gaussian field


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
