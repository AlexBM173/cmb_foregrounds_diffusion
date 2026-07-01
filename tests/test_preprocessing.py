import numpy as np
import pytest
import torch

from foregrounds_diffusion.preprocessing import (
    apply_maxmin_normalization,
    apply_stdnorm,
    augment_images_unique,
    get_lpf_hpf,
    split_data_to_tensors,
)

# ---------------------------------------------------------------------------
# apply_maxmin_normalization
# ---------------------------------------------------------------------------


def test_maxmin_range():
    rng = np.random.default_rng(0)
    maps = rng.standard_normal((10, 64, 64))
    out = apply_maxmin_normalization(maps)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_maxmin_exactly_zero_and_one():
    maps = np.array([[[0.0, 2.0], [1.0, 4.0]]])
    out = apply_maxmin_normalization(maps)
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)


def test_maxmin_uniform_map_returns_zeros():
    maps = np.ones((5, 32, 32)) * 3.14
    out = apply_maxmin_normalization(maps)
    assert np.all(out == 0.0)


def test_maxmin_preserves_shape():
    maps = np.random.randn(7, 16, 16)
    assert apply_maxmin_normalization(maps).shape == maps.shape


# ---------------------------------------------------------------------------
# apply_stdnorm
# ---------------------------------------------------------------------------


def test_stdnorm_zero_mean_unit_std():
    rng = np.random.default_rng(1)
    # channels-last: (N, H, W, C)
    maps = rng.standard_normal((20, 32, 32, 2))
    out = apply_stdnorm(maps)
    for c in range(2):
        assert np.mean(out[..., c]) == pytest.approx(0.0, abs=1e-10)
        assert np.std(out[..., c]) == pytest.approx(1.0, abs=1e-10)


def test_stdnorm_zero_std_channel_returns_zeros():
    maps = np.zeros((10, 16, 16, 2))
    maps[..., 1] = np.random.randn(10, 16, 16)
    out = apply_stdnorm(maps)
    assert np.all(out[..., 0] == 0.0)


def test_stdnorm_preserves_shape():
    maps = np.random.randn(5, 8, 8, 3)
    assert apply_stdnorm(maps).shape == maps.shape


def test_stdnorm_does_not_mutate_input():
    maps = np.random.randn(4, 16, 16, 2)
    original = maps.copy()
    apply_stdnorm(maps)
    np.testing.assert_array_equal(maps, original)


# ---------------------------------------------------------------------------
# augment_images_unique
# ---------------------------------------------------------------------------


def test_augment_output_shape():
    N, C, H, W = 3, 2, 16, 16
    images = torch.randn(N, C, H, W)
    out = augment_images_unique(images)
    assert out.shape == (8 * N, C, H, W)


def test_augment_all_variants_distinct():
    # For a random patch, all 8 augmentations of a single image should be distinct.
    torch.manual_seed(0)
    images = torch.randn(1, 2, 16, 16)
    out = augment_images_unique(images)
    assert out.shape[0] == 8
    for i in range(8):
        for j in range(i + 1, 8):
            assert not torch.allclose(out[i], out[j])


def test_augment_preserves_pixel_statistics():
    # Rotations and flips preserve mean and std.
    torch.manual_seed(1)
    images = torch.randn(4, 2, 16, 16)
    out = augment_images_unique(images)
    assert out.mean().item() == pytest.approx(images.mean().item(), abs=1e-4)


# ---------------------------------------------------------------------------
# get_lpf_hpf
# ---------------------------------------------------------------------------


@pytest.fixture
def params():
    return [64, 64, 1.40625, 1.40625]


def test_lpf_shape(params):
    filt = get_lpf_hpf(params, 2000, filter_type=0)
    assert filt.shape == (64, 64)


def test_lpf_passes_low_ell(params):
    # Low-pass: centre pixel (ell≈0) should be 1.
    filt = get_lpf_hpf(params, 3000, filter_type=0)
    assert filt[0, 0] == 1.0


def test_hpf_zeros_low_ell(params):
    # High-pass: centre pixel (ell≈0) should be 0.
    filt = get_lpf_hpf(params, 500, filter_type=1)
    assert filt[0, 0] == 0.0


def test_bandpass_is_product_of_lpf_and_hpf(params):
    lmin, lmax = 500, 3000
    bp = get_lpf_hpf(params, (lmin, lmax), filter_type=2)
    lp = get_lpf_hpf(params, lmax, filter_type=0)
    hp = get_lpf_hpf(params, lmin, filter_type=1)
    np.testing.assert_array_equal(bp, lp * hp)


def test_lpf_binary_values(params):
    filt = get_lpf_hpf(params, 2000, filter_type=0)
    assert set(np.unique(filt)) <= {0.0, 1.0}


# ---------------------------------------------------------------------------
# split_data_to_tensors
# ---------------------------------------------------------------------------


def test_split_sizes_sum_to_n():
    data = np.random.randn(100, 16, 16, 2).astype(np.float32)
    tr, va, te = split_data_to_tensors(data, train_size=0.7, val_size=0.15, test_size=0.15)
    assert tr.shape[0] + va.shape[0] + te.shape[0] == 100


def test_split_dtype_float32():
    data = np.random.randn(50, 8, 8, 2).astype(np.float32)
    tr, va, te = split_data_to_tensors(data)
    assert tr.dtype == torch.float32


def test_split_channels_first():
    data = np.random.randn(40, 16, 16, 2).astype(np.float32)
    tr, _, _ = split_data_to_tensors(data)
    # channels-first: (N, C, H, W) where C=2
    assert tr.shape[1] == 2


def test_split_invalid_fractions_raises():
    data = np.random.randn(10, 8, 8, 1).astype(np.float32)
    with pytest.raises(ValueError):
        split_data_to_tensors(data, train_size=0.5, val_size=0.5, test_size=0.5)
