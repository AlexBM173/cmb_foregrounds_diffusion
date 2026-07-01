import numpy as np
import pytest

from foregrounds_diffusion.stacking import extract_cutouts, select_snr_pixels


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def noise_maps(rng):
    # 5 maps, 32×32, ~unit Gaussian noise
    return rng.standard_normal((5, 32, 32))


@pytest.fixture
def maps_with_peak(noise_maps):
    maps = noise_maps.copy()
    # Inject a strong peak (SNR ≈ 20) at (2, 16, 16)
    maps[2, 16, 16] = 20.0 * maps[2].std()
    return maps


# ---------------------------------------------------------------------------
# select_snr_pixels
# ---------------------------------------------------------------------------

def test_select_snr_pixels_returns_list_of_tuples(maps_with_peak):
    coords = select_snr_pixels(maps_with_peak, snr_min=3, snr_max=None)
    assert isinstance(coords, list)
    for item in coords:
        assert len(item) == 3


def test_select_snr_pixels_detects_injected_peak(maps_with_peak):
    coords = select_snr_pixels(maps_with_peak, snr_min=10, snr_max=None)
    patch_idxs = [c[0] for c in coords]
    assert 2 in patch_idxs


def test_select_snr_pixels_upper_bound_excludes_peak(maps_with_peak):
    # SNR window [3, 8): the injected peak at ~20 should not appear.
    coords = select_snr_pixels(maps_with_peak, snr_min=3, snr_max=8)
    for patch_idx, ri, rj in coords:
        if patch_idx == 2:
            assert not (ri == 16 and rj == 16)


def test_select_snr_pixels_empty_for_very_high_threshold(noise_maps):
    coords = select_snr_pixels(noise_maps, snr_min=100, snr_max=None)
    assert coords == []


def test_select_snr_pixels_zero_noise_map_skipped():
    maps = np.zeros((3, 16, 16))
    coords = select_snr_pixels(maps, snr_min=1, snr_max=None)
    assert coords == []


def test_select_snr_pixels_coords_in_bounds(maps_with_peak):
    N, H, W = maps_with_peak.shape
    coords = select_snr_pixels(maps_with_peak, snr_min=1, snr_max=None)
    for patch_idx, ri, rj in coords:
        assert 0 <= patch_idx < N
        assert 0 <= ri < H
        assert 0 <= rj < W


# ---------------------------------------------------------------------------
# extract_cutouts
# ---------------------------------------------------------------------------

def test_extract_cutouts_shape(maps_with_peak):
    coords = [(2, 16, 16)]
    cutouts = extract_cutouts(maps_with_peak, coords, cutout_size=8)
    assert cutouts is not None
    assert cutouts.shape == (1, 8, 8)


def test_extract_cutouts_dtype_float32(maps_with_peak):
    coords = [(2, 16, 16)]
    cutouts = extract_cutouts(maps_with_peak, coords, cutout_size=4)
    assert cutouts.dtype == np.float32


def test_extract_cutouts_respects_max_cutouts(maps_with_peak):
    # Use interior coords guaranteed to survive boundary clipping (half=3, map=32×32).
    coords = [(i, 16, 16) for i in range(len(maps_with_peak))]
    cutouts = extract_cutouts(maps_with_peak, coords, cutout_size=6, max_cutouts=2)
    assert cutouts is not None
    assert cutouts.shape[0] <= 2


def test_extract_cutouts_skips_boundary_coords(maps_with_peak):
    # Peak too close to edge: half=4, peak at (0, 2, 2) → ri0=2-4=-2 < 0, skip.
    coords = [(0, 2, 2)]
    cutouts = extract_cutouts(maps_with_peak, coords, cutout_size=8)
    assert cutouts is None


def test_extract_cutouts_returns_none_when_all_skip(maps_with_peak):
    # All coords near the boundary so nothing survives.
    coords = [(0, 0, 0), (1, 0, 0)]
    cutouts = extract_cutouts(maps_with_peak, coords, cutout_size=16)
    assert cutouts is None


def test_extract_cutouts_values_match_source(maps_with_peak):
    ri, rj, size = 16, 16, 8
    half = size // 2
    coords = [(2, ri, rj)]
    cutouts = extract_cutouts(maps_with_peak, coords, cutout_size=size)
    expected = maps_with_peak[2, ri - half:ri + half, rj - half:rj + half]
    np.testing.assert_allclose(cutouts[0], expected.astype(np.float32))
