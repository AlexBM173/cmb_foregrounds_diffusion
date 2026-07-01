import numpy as np
import pytest
from scipy.ndimage import maximum_filter, minimum_filter

from foregrounds_diffusion.peak_counts import (
    compute_peak_minima_counts,
    count_minima_binned,
    count_peaks_binned,
    find_minima,
    find_peaks,
    smooth_map,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(99)


@pytest.fixture
def patch(rng):
    return rng.standard_normal((64, 64))


@pytest.fixture
def patch_stack(rng):
    return rng.standard_normal((10, 64, 64))


# ---------------------------------------------------------------------------
# smooth_map
# ---------------------------------------------------------------------------

def test_smooth_map_preserves_shape(patch):
    out = smooth_map(patch, fwhm_arcmin=5.0)
    assert out.shape == patch.shape


def test_smooth_map_reduces_std(patch):
    out = smooth_map(patch, fwhm_arcmin=10.0)
    assert out.std() < patch.std()


def test_smooth_map_zero_fwhm_is_identity(patch):
    out = smooth_map(patch, fwhm_arcmin=0.0)
    np.testing.assert_allclose(out, patch, rtol=1e-6)


def test_smooth_map_large_fwhm_nearly_constant(patch):
    out = smooth_map(patch, fwhm_arcmin=200.0)
    assert out.std() < 0.01 * patch.std()


# ---------------------------------------------------------------------------
# find_peaks
# ---------------------------------------------------------------------------

def test_find_peaks_returns_1d(patch):
    peaks = find_peaks(patch)
    assert peaks.ndim == 1


def test_find_peaks_all_are_local_maxima(patch):
    smoothed = smooth_map(patch, fwhm_arcmin=5.0)
    peaks = find_peaks(smoothed, filter_size=3)
    # Every returned value should equal the local maximum in its neighbourhood.
    local_max = maximum_filter(smoothed, size=3)
    # Each returned value must appear at a position where smoothed == local_max.
    peak_positions = (local_max == smoothed)
    border = 1
    peak_positions[:border, :] = False
    peak_positions[-border:, :] = False
    peak_positions[:, :border] = False
    peak_positions[:, -border:] = False
    expected = smoothed[peak_positions]
    np.testing.assert_array_equal(np.sort(peaks), np.sort(expected))


def test_find_peaks_empty_for_constant_map():
    # Constant map: every pixel equals the local max, but after border removal
    # the inner pixels are all "peaks". Just check it runs.
    patch = np.ones((16, 16))
    peaks = find_peaks(patch)
    assert peaks.ndim == 1


def test_find_peaks_single_spike():
    patch = np.zeros((16, 16))
    patch[8, 8] = 1.0
    peaks = find_peaks(patch, filter_size=3)
    assert 1.0 in peaks


# ---------------------------------------------------------------------------
# find_minima
# ---------------------------------------------------------------------------

def test_find_minima_returns_1d(patch):
    minima = find_minima(patch)
    assert minima.ndim == 1


def test_find_minima_all_are_local_minima(patch):
    smoothed = smooth_map(patch, fwhm_arcmin=5.0)
    minima = find_minima(smoothed, filter_size=3)
    local_min = minimum_filter(smoothed, size=3)
    min_positions = (local_min == smoothed)
    border = 1
    min_positions[:border, :] = False
    min_positions[-border:, :] = False
    min_positions[:, :border] = False
    min_positions[:, -border:] = False
    expected = smoothed[min_positions]
    np.testing.assert_array_equal(np.sort(minima), np.sort(expected))


def test_find_minima_single_dip():
    patch = np.zeros((16, 16))
    patch[8, 8] = -1.0
    minima = find_minima(patch, filter_size=3)
    assert -1.0 in minima


# ---------------------------------------------------------------------------
# count_peaks_binned
# ---------------------------------------------------------------------------

def test_count_peaks_binned_shape(patch_stack):
    thresholds = np.linspace(-3, 3, 15)
    counts = count_peaks_binned(patch_stack, thresholds, fwhm_arcmin=5.0)
    assert counts.shape == (len(patch_stack), len(thresholds))


def test_count_peaks_binned_non_negative(patch_stack):
    thresholds = np.linspace(-2, 4, 10)
    counts = count_peaks_binned(patch_stack, thresholds, fwhm_arcmin=5.0)
    assert np.all(counts >= 0)


def test_count_peaks_binned_decreasing_with_threshold(patch_stack):
    # Higher ν threshold → fewer peaks (cumulative from above).
    thresholds = np.linspace(-1, 4, 20)
    counts = count_peaks_binned(patch_stack, thresholds, fwhm_arcmin=5.0)
    mean_counts = counts.mean(axis=0)
    assert np.all(np.diff(mean_counts) <= 0)


def test_count_peaks_binned_zero_std_map_handled():
    maps = np.zeros((3, 32, 32))
    thresholds = np.array([0.0, 1.0])
    counts = count_peaks_binned(maps, thresholds, fwhm_arcmin=5.0)
    assert counts.shape == (3, 2)


# ---------------------------------------------------------------------------
# count_minima_binned
# ---------------------------------------------------------------------------

def test_count_minima_binned_shape(patch_stack):
    thresholds = np.linspace(-4, 0, 10)
    counts = count_minima_binned(patch_stack, thresholds, fwhm_arcmin=5.0)
    assert counts.shape == (len(patch_stack), len(thresholds))


def test_count_minima_binned_non_negative(patch_stack):
    thresholds = np.linspace(-4, 0, 10)
    counts = count_minima_binned(patch_stack, thresholds, fwhm_arcmin=5.0)
    assert np.all(counts >= 0)


def test_count_minima_binned_increasing_with_threshold(patch_stack):
    # More negative ν threshold → fewer minima counted (fewer below a lower bar).
    thresholds = np.linspace(-4, -0.5, 15)
    counts = count_minima_binned(patch_stack, thresholds, fwhm_arcmin=5.0)
    mean_counts = counts.mean(axis=0)
    assert np.all(np.diff(mean_counts) >= 0)


# ---------------------------------------------------------------------------
# compute_peak_minima_counts
# ---------------------------------------------------------------------------

def test_compute_peak_minima_counts_structure(patch_stack):
    thresholds_p = np.linspace(-1, 4, 10)
    thresholds_m = np.linspace(-4, 1, 10)
    scales = (2.0, 5.0)
    results = compute_peak_minima_counts(
        patch_stack, thresholds_p, thresholds_m,
        smoothing_scales_arcmin=scales)
    assert set(results.keys()) == set(scales)
    for fwhm in scales:
        assert 'peaks' in results[fwhm]
        assert 'minima' in results[fwhm]


def test_compute_peak_minima_counts_shapes(patch_stack):
    thresholds_p = np.linspace(-1, 4, 12)
    thresholds_m = np.linspace(-4, 1, 8)
    N = len(patch_stack)
    results = compute_peak_minima_counts(
        patch_stack, thresholds_p, thresholds_m,
        smoothing_scales_arcmin=(3.0,))
    assert results[3.0]['peaks'].shape == (N, 12)
    assert results[3.0]['minima'].shape == (N, 8)
