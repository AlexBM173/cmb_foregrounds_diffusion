"""Tests for flat-sky masking utilities.

HEALPix functions (get_point_source_mask_in_healpix,
get_apodised_mdpl2_cluster_mask, get_mdpl2_halo_cat, …) require external
data files on the cluster and are not tested here.
"""
import numpy as np
import pytest

from foregrounds_diffusion.masking import (
    boundary_apod_mask,
    get_peak_masks,
    inpaint_masked_regions,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(7)


@pytest.fixture
def smooth_map(rng):
    # 32×32 Gaussian map; std≈1, no extreme peaks
    return rng.standard_normal((32, 32))


@pytest.fixture
def map_with_spike(smooth_map):
    m = smooth_map.copy()
    # Inject a 50σ spike so it's reliably above any threshold.
    m[16, 16] = 50.0 * m.std()
    return m


# ---------------------------------------------------------------------------
# get_peak_masks
# ---------------------------------------------------------------------------

def test_get_peak_masks_returns_two_arrays(smooth_map):
    peak_mask, mask = get_peak_masks(smooth_map, mask_threshold_sigma_units=10)
    assert peak_mask.shape == smooth_map.shape
    assert mask.shape == smooth_map.shape


def test_get_peak_masks_values_zero_or_one_no_apod(smooth_map):
    peak_mask, _ = get_peak_masks(smooth_map, mask_threshold_sigma_units=10,
                                   mask_radius_pixel_units=0)
    assert set(np.unique(peak_mask)) <= {0.0, 1.0}


def test_get_peak_masks_high_threshold_all_ones(smooth_map):
    # Threshold so high that nothing is masked.
    peak_mask, mask = get_peak_masks(smooth_map, mask_threshold_sigma_units=1000)
    assert np.all(peak_mask == 1.0)
    assert np.all(mask == 1.0)


def test_get_peak_masks_spike_is_masked(map_with_spike):
    peak_mask, _ = get_peak_masks(map_with_spike, mask_threshold_sigma_units=10)
    assert peak_mask[16, 16] == 0.0


def test_get_peak_masks_circle_radius_punches_hole(map_with_spike):
    _, mask = get_peak_masks(map_with_spike, mask_threshold_sigma_units=10,
                              mask_radius_pixel_units=3, mask_shape='circle',
                              perform_apod=0)
    # The spike pixel itself is masked.
    assert mask[16, 16] == 0.0


def test_get_peak_masks_square_radius_punches_hole(map_with_spike):
    _, mask = get_peak_masks(map_with_spike, mask_threshold_sigma_units=10,
                              mask_radius_pixel_units=3, mask_shape='square',
                              perform_apod=0)
    assert mask[16, 16] == 0.0


def test_get_peak_masks_invalid_shape_raises():
    with pytest.raises(AssertionError):
        get_peak_masks(np.ones((8, 8)), mask_shape='triangle')


def test_get_peak_masks_negative_radius_raises():
    with pytest.raises(AssertionError):
        get_peak_masks(np.ones((8, 8)), mask_radius_pixel_units=-1)


# ---------------------------------------------------------------------------
# inpaint_masked_regions
# ---------------------------------------------------------------------------

def test_inpaint_replaces_masked_pixels(rng):
    hmap = rng.standard_normal(64)
    mask = np.ones(64)
    mask[10:20] = 0.0
    original_vals = hmap[10:20].copy()
    result = inpaint_masked_regions(hmap, mask, rng=rng)
    # Masked pixels should be changed.
    assert not np.allclose(result[10:20], original_vals)


def test_inpaint_preserves_unmasked_pixels(rng):
    hmap = rng.standard_normal(64)
    mask = np.ones(64)
    mask[10:20] = 0.0
    result = inpaint_masked_regions(hmap, mask, rng=rng)
    np.testing.assert_array_equal(result[mask > 0.5], hmap[mask > 0.5])


def test_inpaint_replacement_matches_unmasked_stats(rng):
    # With many unmasked pixels, replacement mean/std should be close to unmasked.
    np.random.seed(0)
    hmap = np.random.randn(1000)
    mask = np.ones(1000)
    mask[:100] = 0.0
    result = inpaint_masked_regions(hmap, mask, rng=rng)
    unmasked_mean = hmap[mask > 0.5].mean()
    unmasked_std = hmap[mask > 0.5].std()
    replaced = result[mask < 0.5]
    assert replaced.mean() == pytest.approx(unmasked_mean, abs=0.3)
    assert replaced.std() == pytest.approx(unmasked_std, rel=0.5)


def test_inpaint_does_not_mutate_input(rng):
    hmap = rng.standard_normal(32)
    original = hmap.copy()
    mask = np.ones(32)
    mask[:5] = 0.0
    inpaint_masked_regions(hmap, mask, rng=rng)
    np.testing.assert_array_equal(hmap, original)


def test_inpaint_no_masked_pixels_is_identity(rng):
    hmap = rng.standard_normal(32)
    mask = np.ones(32)
    result = inpaint_masked_regions(hmap, mask, rng=rng)
    np.testing.assert_array_equal(result, hmap)


# ---------------------------------------------------------------------------
# boundary_apod_mask
# ---------------------------------------------------------------------------

@pytest.fixture
def xy_grid():
    x, y = np.meshgrid(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32))
    return x, y


def test_boundary_apod_mask_shape(xy_grid):
    x, y = xy_grid
    mask = boundary_apod_mask(x, y, mask_radius=0.3, perform_apod=False)
    assert mask.shape == x.shape


def test_boundary_apod_circle_centre_masked(xy_grid):
    x, y = xy_grid
    mask = boundary_apod_mask(x, y, mask_radius=0.3, mask_shape='circle',
                               perform_apod=False)
    # Centre of the grid (x≈0, y≈0) lies inside the circle: should be 0.
    cx, cy = mask.shape[0] // 2, mask.shape[1] // 2
    assert mask[cx, cy] == 0.0


def test_boundary_apod_circle_edge_unmasked(xy_grid):
    x, y = xy_grid
    mask = boundary_apod_mask(x, y, mask_radius=0.2, mask_shape='circle',
                               perform_apod=False)
    # Corner pixels (|r|>0.2) should be unmasked.
    assert mask[0, 0] == 1.0
    assert mask[-1, -1] == 1.0


def test_boundary_apod_square_centre_masked(xy_grid):
    x, y = xy_grid
    mask = boundary_apod_mask(x, y, mask_radius=0.3, mask_shape='square',
                               perform_apod=False)
    cx, cy = mask.shape[0] // 2, mask.shape[1] // 2
    assert mask[cx, cy] == 0.0


def test_boundary_apod_invalid_shape_raises(xy_grid):
    x, y = xy_grid
    with pytest.raises(AssertionError):
        boundary_apod_mask(x, y, mask_radius=0.3, mask_shape='hexagon')


def test_boundary_apod_mask_values_in_unit_interval(xy_grid):
    x, y = xy_grid
    mask = boundary_apod_mask(x, y, mask_radius=0.3, perform_apod=True)
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0 + 1e-10
