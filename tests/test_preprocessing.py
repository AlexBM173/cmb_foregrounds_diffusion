import numpy as np
import pytest
import torch

from foregrounds_diffusion.preprocessing import (
    apply_maxmin_normalization,
    apply_stdnorm,
    augment_images_unique,
    denormalize_dm_maps,
    get_lpf_hpf,
    load_all_moments,
    renormalize_dm_maps,
    split_data_to_tensors,
    wiener_filter,
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


# ---------------------------------------------------------------------------
# renormalize_dm_maps
# ---------------------------------------------------------------------------


def test_renormalize_dm_maps_shape():
    rng = np.random.default_rng(0)
    # channels-first DM samples
    dm = rng.standard_normal((8, 2, 16, 16)).astype(np.float32)
    # channels-last training maps
    train = rng.standard_normal((8, 16, 16, 2)).astype(np.float32)
    out = renormalize_dm_maps(dm, train, variance_scaling=False)
    assert out.shape == (8, 2, 16, 16)


def test_renormalize_dm_maps_range_matches_train():
    # Without variance_scaling the output range must match the training range.
    rng = np.random.default_rng(1)
    dm = rng.uniform(0, 1, (16, 2, 8, 8)).astype(np.float32)
    train = rng.uniform(3.0, 7.0, (16, 8, 8, 2)).astype(np.float32)
    out = renormalize_dm_maps(dm, train, variance_scaling=False)
    for c in range(2):
        tr_min = train[:, :, :, c].min()
        tr_max = train[:, :, :, c].max()
        assert out[:, c].min() >= tr_min - 1e-5
        assert out[:, c].max() <= tr_max + 1e-5


def test_renormalize_dm_maps_variance_scaling():
    rng = np.random.default_rng(2)
    dm = rng.standard_normal((32, 2, 8, 8)).astype(np.float32)
    train = rng.standard_normal((32, 8, 8, 2)).astype(np.float32) * 5 + 10
    out = renormalize_dm_maps(dm, train, variance_scaling=True)
    # Variance-scaled output std should be closer to training std than raw dm std.
    for c in range(2):
        tr_std = train[:, :, :, c].std()
        out_std = out[:, c].std()
        dm_std = dm[:, c].std()
        assert abs(out_std - tr_std) < abs(dm_std - tr_std)


def test_renormalize_dm_maps_does_not_mutate_input():
    rng = np.random.default_rng(3)
    dm = rng.standard_normal((4, 2, 8, 8)).astype(np.float32)
    train = rng.standard_normal((4, 8, 8, 2)).astype(np.float32)
    dm_orig = dm.copy()
    renormalize_dm_maps(dm, train)
    np.testing.assert_array_equal(dm, dm_orig)


# ---------------------------------------------------------------------------
# denormalize_dm_maps
# ---------------------------------------------------------------------------


def test_denormalize_dm_maps_shape():
    dm = np.random.randn(8, 2, 16, 16).astype(np.float32)
    out = denormalize_dm_maps(dm, cib_mean=1.0, cib_std=2.0, tsz_mean=0.5, tsz_std=0.1)
    assert out.shape == dm.shape


def test_denormalize_dm_maps_roundtrip():
    # Applying z-score then inverting should recover the original.
    rng = np.random.default_rng(4)
    original = rng.standard_normal((8, 2, 16, 16)).astype(np.float32)
    cib_mean, cib_std = float(original[:, 0].mean()), float(original[:, 0].std())
    tsz_mean, tsz_std = float(original[:, 1].mean()), float(original[:, 1].std())
    # Normalise (z-score)
    normed = original.copy()
    normed[:, 0] = (normed[:, 0] - cib_mean) / cib_std
    normed[:, 1] = (normed[:, 1] - tsz_mean) / tsz_std
    # Invert
    recovered = denormalize_dm_maps(normed, cib_mean, cib_std, tsz_mean, tsz_std)
    np.testing.assert_allclose(recovered, original, atol=1e-5)


def test_denormalize_dm_maps_does_not_mutate_input():
    dm = np.random.randn(4, 2, 8, 8).astype(np.float32)
    dm_orig = dm.copy()
    denormalize_dm_maps(dm, 0.0, 1.0, 0.0, 1.0)
    np.testing.assert_array_equal(dm, dm_orig)


# ---------------------------------------------------------------------------
# load_all_moments
# ---------------------------------------------------------------------------


def test_load_all_moments_keys(tmp_path):
    # Synthetic moments array: (N=5, L=3, 12)
    rng = np.random.default_rng(5)
    data = rng.standard_normal((5, 3, 12)).astype(np.float32)
    fname = tmp_path / "moments.npy"
    np.save(fname, data)
    bp = np.array([200.0, 500.0, 1000.0])
    result = load_all_moments(str(fname), bp)
    assert set(result.keys()) == {f"moment_{i:02d}" for i in range(12)}


def test_load_all_moments_shape(tmp_path):
    rng = np.random.default_rng(6)
    data = rng.standard_normal((10, 4, 12)).astype(np.float32)
    fname = tmp_path / "moments.npy"
    np.save(fname, data)
    bp = np.array([100.0, 300.0, 700.0, 1500.0])
    result = load_all_moments(str(fname), bp)
    # Each key holds a list of N=10 normalised arrays of length L=4
    assert len(result["moment_00"]) == 10
    assert len(result["moment_00"][0]) == 4


def test_load_all_moments_max_lines(tmp_path):
    rng = np.random.default_rng(7)
    data = rng.standard_normal((20, 2, 12)).astype(np.float32)
    fname = tmp_path / "moments.npy"
    np.save(fname, data)
    bp = np.array([300.0, 800.0])
    result = load_all_moments(str(fname), bp, max_lines=5)
    assert len(result["moment_00"]) == 5


# ---------------------------------------------------------------------------
# wiener_filter
# ---------------------------------------------------------------------------


def test_wiener_filter_shape():
    params = [64, 64, 1.40625, 1.40625]
    el = np.arange(0, 8001)
    cl_s = np.ones(len(el)) * 1e-6
    cl_n = np.ones(len(el)) * 1e-7
    wf = wiener_filter(params, cl_s, cl_n, el=el)
    assert wf.shape == (64, 64)


def test_wiener_filter_range():
    # Wiener filter values must be in (0, 1] for positive signal and noise.
    params = [64, 64, 1.40625, 1.40625]
    el = np.arange(0, 8001)
    cl_s = np.ones(len(el)) * 1e-6
    cl_n = np.ones(len(el)) * 1e-6
    wf = wiener_filter(params, cl_s, cl_n, el=el)
    assert np.all(wf >= 0)
    assert np.all(wf <= 1.0 + 1e-10)


def test_wiener_filter_el_default():
    # Without el, it defaults to np.arange(len(cl_signal)).
    params = [32, 32, 1.40625, 1.40625]
    cl_s = np.ones(5001) * 1e-5
    cl_n = np.ones(5001) * 1e-5
    wf = wiener_filter(params, cl_s, cl_n)
    assert wf.shape == (32, 32)


def test_wiener_filter_snr_ratio():
    # High SNR → filter ≈ 1; low SNR → filter ≈ 0.
    params = [64, 64, 1.40625, 1.40625]
    el = np.arange(0, 8001)
    # Strong signal, tiny noise → filter close to 1
    wf_high = wiener_filter(params, np.ones(len(el)) * 1e-4, np.ones(len(el)) * 1e-10, el=el)
    # Tiny signal, strong noise → filter close to 0
    wf_low = wiener_filter(params, np.ones(len(el)) * 1e-10, np.ones(len(el)) * 1e-4, el=el)
    assert wf_high.mean() > 0.99
    assert wf_low.mean() < 0.01
