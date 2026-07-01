import numpy as np
import pytest

from foregrounds_diffusion.flatmaps import (
    bandpass_filter,
    cl2map,
    get_lpf_hpf,
    get_lxly,
    make_gaussian_realisation,
    map2cl,
    radial_profile,
)

# ---------------------------------------------------------------------------
# get_lxly
# ---------------------------------------------------------------------------


def test_get_lxly_shape(flatskymapparams):
    lx, ly = get_lxly(flatskymapparams)
    nx, ny = flatskymapparams[0], flatskymapparams[1]
    assert lx.shape == (ny, nx)
    assert ly.shape == (ny, nx)


def test_get_lxly_dtype(flatskymapparams):
    lx, ly = get_lxly(flatskymapparams)
    assert np.issubdtype(lx.dtype, np.floating)
    assert np.issubdtype(ly.dtype, np.floating)


def test_get_lxly_dc_zero(flatskymapparams):
    lx, ly = get_lxly(flatskymapparams)
    assert lx[0, 0] == 0.0
    assert ly[0, 0] == 0.0


# ---------------------------------------------------------------------------
# map2cl
# ---------------------------------------------------------------------------


def test_map2cl_output_shape(flatskymapparams, gaussian_patch):
    el, cl = map2cl(flatskymapparams, gaussian_patch)
    assert el.ndim == 1
    assert cl.shape == el.shape


def test_map2cl_auto_nonnegative(flatskymapparams, gaussian_patch):
    el, cl = map2cl(flatskymapparams, gaussian_patch)
    assert np.all(np.isfinite(cl))
    assert np.all(cl >= 0)


def test_map2cl_symmetric_under_flip(flatskymapparams, gaussian_patch):
    _, cl1 = map2cl(flatskymapparams, gaussian_patch)
    _, cl2 = map2cl(flatskymapparams, gaussian_patch[::-1, :])
    np.testing.assert_allclose(cl1, cl2, rtol=1e-10)


# ---------------------------------------------------------------------------
# cl2map
# ---------------------------------------------------------------------------


def test_cl2map_roundtrip(flatskymapparams_256):
    # cl2map then map2cl: median fractional error within 50% (single realisation)
    np.random.seed(10)
    el_in = np.arange(1, 10001)
    cl_in = 1e-6 * el_in.astype(float) ** (-2)
    m = cl2map(flatskymapparams_256, cl_in, el_in)
    el_out, cl_out = map2cl(flatskymapparams_256, m, minbin=300, maxbin=5000)
    cl_ref = np.interp(el_out, el_in, cl_in)
    valid = cl_ref > 0
    ratio = cl_out[valid] / cl_ref[valid]
    assert pytest.approx(float(np.median(ratio)), abs=0.5) == 1.0


# ---------------------------------------------------------------------------
# make_gaussian_realisation
# ---------------------------------------------------------------------------


def test_make_gaussian_realisation_single_shape(flatskymapparams):
    np.random.seed(3)
    el = np.arange(1, 5000)
    cl = 1e-5 * np.ones(len(el))
    m = make_gaussian_realisation(flatskymapparams, el, cl)
    nx, ny = flatskymapparams[0], flatskymapparams[1]
    assert m.shape == (nx, ny)


def test_make_gaussian_realisation_single_zero_mean(flatskymapparams):
    np.random.seed(4)
    el = np.arange(1, 5000)
    cl = 1e-5 * np.ones(len(el))
    m = make_gaussian_realisation(flatskymapparams, el, cl)
    assert abs(np.mean(m)) < 1e-10


def test_make_gaussian_realisation_correlated_two_field(flatskymapparams_256):
    # Correlated pair: auto-spectra positive, cross-spectrum sign matches cl12
    np.random.seed(5)
    el = np.arange(1, 10000)
    cl1 = 1e-10 * np.ones(len(el))
    cl2 = 1e-10 * np.ones(len(el))
    cl12 = 0.5e-10 * np.ones(len(el))  # positive 50% correlation
    sim = make_gaussian_realisation(
        flatskymapparams_256, el, cl1, cl2=cl2, cl12=cl12, qu_or_eb="eb"
    )
    assert sim.shape[0] == 3
    _, cl_a = map2cl(flatskymapparams_256, sim[0])
    _, cl_b = map2cl(flatskymapparams_256, sim[1])
    assert np.all(cl_a >= 0)
    assert np.all(cl_b >= 0)
    _, cl_cross = map2cl(flatskymapparams_256, sim[0], sim[1])
    assert np.mean(cl_cross) > 0


# ---------------------------------------------------------------------------
# radial_profile
# ---------------------------------------------------------------------------


def test_radial_profile_shape(flatskymapparams, gaussian_patch):
    lx, ly = get_lxly(flatskymapparams)
    psd = np.abs(np.fft.fft2(gaussian_patch)) ** 2
    result = radial_profile(psd, (lx, ly), bin_size=200, minbin=100, maxbin=5000, to_arcmins=0)
    n_bins = len(np.arange(100, 5000, 200))
    assert result.shape == (n_bins, 3)


def test_radial_profile_bins_monotonic(flatskymapparams, gaussian_patch):
    lx, ly = get_lxly(flatskymapparams)
    psd = np.abs(np.fft.fft2(gaussian_patch)) ** 2
    result = radial_profile(psd, (lx, ly), bin_size=200, minbin=100, maxbin=5000, to_arcmins=0)
    assert np.all(np.diff(result[:, 0]) > 0)


# ---------------------------------------------------------------------------
# bandpass_filter / get_lpf_hpf
# ---------------------------------------------------------------------------


def test_bandpass_filter_low_pass_suppresses_high_ell(flatskymapparams, gaussian_patch):
    lp = get_lpf_hpf(flatskymapparams, 500, filter_type=0)
    filtered = bandpass_filter(gaussian_patch, lp)
    _, cl_orig = map2cl(flatskymapparams, gaussian_patch, minbin=1000, maxbin=8000)
    _, cl_filt = map2cl(flatskymapparams, filtered, minbin=1000, maxbin=8000)
    assert cl_filt.mean() < cl_orig.mean()
