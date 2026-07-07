import numpy as np
import pytest

from foregrounds_diffusion.flatmaps import (
    bandpass_filter,
    cl2map,
    convert_eb_qu,
    get_lpf_hpf,
    get_lxly,
    make_correlated_gaussian_fields,
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
# make_correlated_gaussian_fields (N-field generalisation)
# ---------------------------------------------------------------------------


def _three_field_matrix(el):
    """Red autos + a valid (PSD) target correlation matrix, shape (3, 3, len(el))."""
    amp = [3.0, 1.0, 0.5]
    auto = [a * (el / 1000.0) ** -2.0 for a in amp]
    R = np.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.4], [0.3, 0.4, 1.0]])
    clm = np.zeros((3, 3, len(el)))
    for i in range(3):
        for j in range(3):
            clm[i, j] = R[i, j] * np.sqrt(auto[i] * auto[j])
    return clm, R


def test_correlated_fields_shape_and_zero_mean(flatskymapparams_256):
    el = np.arange(1, 8000.0)
    clm, _ = _three_field_matrix(el)
    rng = np.random.default_rng(0)
    one = make_correlated_gaussian_fields(flatskymapparams_256, el, clm, rng=rng)
    assert one.shape == (3, flatskymapparams_256[1], flatskymapparams_256[0])
    assert np.allclose(one.mean(axis=(-2, -1)), 0.0, atol=1e-6)
    many = make_correlated_gaussian_fields(flatskymapparams_256, el, clm, n_realisations=4, rng=rng)
    assert many.shape == (4, 3, flatskymapparams_256[1], flatskymapparams_256[0])


def test_correlated_fields_recovers_spectra(flatskymapparams_256):
    # The full C×C spectrum matrix must be reproduced: autos and cross r's.
    el = np.arange(1, 8000.0)
    clm, R = _three_field_matrix(el)
    n = 80
    fields = make_correlated_gaussian_fields(
        flatskymapparams_256, el, clm, n_realisations=n, rng=np.random.default_rng(1)
    )
    binkw = dict(binsize=200, minbin=300, maxbin=6000)
    autos = {}
    for i in range(3):
        acc = 0.0
        for r in range(n):
            e, cl = map2cl(flatskymapparams_256, fields[r, i], **binkw)
            acc = acc + cl
        autos[i] = acc / n
        inp = np.interp(e, el, clm[i, i])
        ok = inp > 0
        assert np.median(autos[i][ok] / inp[ok]) == pytest.approx(1.0, abs=0.15)
    for i in range(3):
        for j in range(i + 1, 3):
            acc = 0.0
            for r in range(n):
                e, cx = map2cl(flatskymapparams_256, fields[r, i], fields[r, j], **binkw)
                acc = acc + cx
            r_meas = np.median((acc / n / np.sqrt(autos[i] * autos[j]))[2:-2])
            assert r_meas == pytest.approx(R[i, j], abs=0.07)


def test_correlated_fields_matches_two_field(flatskymapparams_256):
    # C=2 is statistically equivalent to make_gaussian_realisation (same spectra).
    el = np.arange(1, 8000.0)
    clm, R = _three_field_matrix(el)
    clm2 = clm[:2, :2]
    n = 80
    fields = make_correlated_gaussian_fields(
        flatskymapparams_256, el, clm2, n_realisations=n, rng=np.random.default_rng(2)
    )
    binkw = dict(binsize=200, minbin=300, maxbin=6000)
    a0 = a1 = ax = 0.0
    for r in range(n):
        _, c0 = map2cl(flatskymapparams_256, fields[r, 0], **binkw)
        _, c1 = map2cl(flatskymapparams_256, fields[r, 1], **binkw)
        e, cx = map2cl(flatskymapparams_256, fields[r, 0], fields[r, 1], **binkw)
        a0, a1, ax = a0 + c0, a1 + c1, ax + cx
    inp0 = np.interp(e, el, clm2[0, 0])
    assert np.median((a0 / n / inp0)[inp0 > 0]) == pytest.approx(1.0, abs=0.15)
    r_meas = np.median((ax / np.sqrt(a0 * a1))[2:-2])
    assert r_meas == pytest.approx(R[0, 1], abs=0.07)


def test_correlated_fields_regularises_non_psd(flatskymapparams_256):
    # An over-correlated (r > 1) matrix is non-PSD; eigenvalue clipping must keep
    # the output finite rather than raising.
    el = np.arange(1, 8000.0)
    clm, _ = _three_field_matrix(el)
    auto0 = clm[0, 0]
    auto1 = clm[1, 1]
    clm[0, 1] = clm[1, 0] = 1.3 * np.sqrt(auto0 * auto1)  # r = 1.3 → non-PSD
    out = make_correlated_gaussian_fields(
        flatskymapparams_256, el, clm, rng=np.random.default_rng(3)
    )
    assert out.shape[0] == 3
    assert np.all(np.isfinite(out))


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


# ---------------------------------------------------------------------------
# radial_profile — xy kwarg path
# ---------------------------------------------------------------------------


def test_radial_profile_no_xy_uses_pixel_indices(flatskymapparams, gaussian_patch):
    # When xy=None, radius is computed from pixel indices (not ell coordinates).
    psd = np.abs(np.fft.fft2(gaussian_patch)) ** 2
    result = radial_profile(psd, bin_size=5, minbin=0, maxbin=30, to_arcmins=0)
    assert result.shape[1] == 3
    # Bin centres should be monotonically increasing.
    assert np.all(np.diff(result[:, 0]) > 0)


def test_radial_profile_to_arcmins_scales_radius(flatskymapparams, gaussian_patch):
    # With to_arcmins=1, bin centres are 60× larger than with to_arcmins=0
    # when xy=None (pixel-index radii scaled to arcmin).
    psd = np.abs(np.fft.fft2(gaussian_patch)) ** 2
    r_pix = radial_profile(psd, bin_size=5, minbin=0, maxbin=30, to_arcmins=0)
    r_arcmin = radial_profile(psd, bin_size=5 * 60, minbin=0, maxbin=30 * 60, to_arcmins=1)
    # Bin centres in arcmin mode are 60× those in pixel mode.
    np.testing.assert_allclose(r_arcmin[:, 0], r_pix[:, 0] * 60.0, rtol=1e-10)


# ---------------------------------------------------------------------------
# convert_eb_qu
# ---------------------------------------------------------------------------


def test_convert_eb_qu_shape(flatskymapparams):
    rng = np.random.default_rng(10)
    nx, ny = flatskymapparams[0], flatskymapparams[1]
    e = rng.standard_normal((nx, ny))
    b = rng.standard_normal((nx, ny))
    q, u = convert_eb_qu(e, b, flatskymapparams, eb_to_qu=1)
    assert q.shape == (nx, ny)
    assert u.shape == (nx, ny)


def test_convert_eb_qu_preserves_total_energy(flatskymapparams):
    # A rotation in Fourier space is unitary: total power (E² + B²) is conserved.
    rng = np.random.default_rng(11)
    nx, ny = flatskymapparams[0], flatskymapparams[1]
    e = rng.standard_normal((nx, ny))
    b = rng.standard_normal((nx, ny))
    q, u = convert_eb_qu(e, b, flatskymapparams, eb_to_qu=1)
    energy_in = np.var(e) + np.var(b)
    energy_out = np.var(q) + np.var(u)
    assert energy_out == pytest.approx(energy_in, rel=0.05)


def test_convert_eb_qu_qu_to_eb_shape(flatskymapparams):
    rng = np.random.default_rng(12)
    nx, ny = flatskymapparams[0], flatskymapparams[1]
    q = rng.standard_normal((nx, ny))
    u = rng.standard_normal((nx, ny))
    e, b = convert_eb_qu(q, u, flatskymapparams, eb_to_qu=0)
    assert e.shape == (nx, ny)
    assert b.shape == (nx, ny)


# ---------------------------------------------------------------------------
# make_gaussian_realisation — additional code paths
# ---------------------------------------------------------------------------


def test_cl2map_el_none_path(flatskymapparams):
    # cl2map with el=None triggers the fallback el = np.arange(len(cl))
    np.random.seed(22)
    cl = np.ones(8001) * 1e-5
    m = cl2map(flatskymapparams, cl)  # no el argument
    assert m.shape == (flatskymapparams[0], flatskymapparams[1])


def test_make_gaussian_realisation_qu_path(flatskymapparams_256):
    # qu_or_eb='qu' applies an EB→QU rotation to the second field
    np.random.seed(20)
    el = np.arange(1, 10000)
    cl1 = 1e-10 * np.ones(len(el))
    cl2 = 1e-10 * np.ones(len(el))
    cl12 = 0.5e-10 * np.ones(len(el))
    sim = make_gaussian_realisation(
        flatskymapparams_256, el, cl1, cl2=cl2, cl12=cl12, qu_or_eb="qu"
    )
    assert sim.shape[0] == 3  # [T, Q, U]
    # The polarisation fields should have finite values.
    assert np.all(np.isfinite(sim))


def test_make_gaussian_realisation_beam_convolution(flatskymapparams):
    # Passing bl smoothes the map; high-ℓ power should be suppressed.
    np.random.seed(21)
    el = np.arange(1, 5000)
    cl = 1e-5 * np.ones(len(el))
    # Narrow beam: kills multipoles above ~500
    bl = np.exp(-el * (el + 1) * (np.radians(0.5) / (8 * np.log(2))) ** 2)
    m_beam = make_gaussian_realisation(flatskymapparams, el, cl, bl=bl)
    m_nobeam = make_gaussian_realisation(flatskymapparams, el, cl)
    _, cl_beam = map2cl(flatskymapparams, m_beam, minbin=1500, maxbin=4000)
    _, cl_nobeam = map2cl(flatskymapparams, m_nobeam, minbin=1500, maxbin=4000)
    assert cl_beam.mean() < cl_nobeam.mean()
