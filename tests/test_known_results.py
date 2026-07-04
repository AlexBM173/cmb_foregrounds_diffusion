"""Known-result validation of the statistics modules.

Each test compares a statistic computed on a synthetic field against a
closed-form expectation, so a silent normalisation or convention bug
fails loudly instead of skewing science results downstream:

1. ``map2cl`` absolute normalisation — white noise has C_ell = sigma^2 *
   Omega_pix exactly, and a power-law GRF must be recovered within sample
   variance (the existing roundtrip test cannot catch a self-inverse
   normalisation error).
2. Minkowski functionals — ratio form of the Tomita (1986) Gaussian
   formulas: V0 tracks erfc(nu/sqrt(2))/2, V1 ~ exp(-nu^2/2), V2 ~
   nu*exp(-nu^2/2) (ratios cancel the sigma_0/sigma_1 amplitudes and the
   quantimpy output conventions).
3. ``make_gaussian_realisation`` correlated pair — the recovered
   correlation coefficient r(ell) must match the input, not just its sign
   (a regression guard for the qu_or_eb="qu" default, which silently
   destroys the correlation).
4. Peak/minima counts — a deterministic map with K inserted blobs on a
   smooth background has exactly K + 1 local maxima.
5. Scattering transform — exact homogeneity and translation invariances,
   plus phase-randomisation consistency: a GRF and its phase-randomised
   surrogate are statistically identical, while a non-Gaussian field must
   separate from its own surrogate in the S2/S1 ratio.
"""

import numpy as np
import pytest
from scipy.special import erfc

from foregrounds_diffusion.flatmaps import cl2map, make_gaussian_realisation, map2cl
from foregrounds_diffusion.peak_counts import count_peaks_binned, find_minima, find_peaks

# ---------------------------------------------------------------------------
# 1. map2cl absolute normalisation
# ---------------------------------------------------------------------------


def test_map2cl_white_noise_flat_spectrum(flatskymapparams_256):
    # White noise with per-pixel variance sigma^2 has C_ell = sigma^2 * Omega_pix.
    nx, ny, dx, dy = flatskymapparams_256
    omega_pix = np.radians(dx / 60.0) * np.radians(dy / 60.0)
    sigma = 3.0

    rng = np.random.default_rng(11)
    cls = []
    for _ in range(10):
        noise = rng.normal(0.0, sigma, (ny, nx))
        el, cl = map2cl(flatskymapparams_256, noise, binsize=500, minbin=1000, maxbin=8000)
        cls.append(cl)
    cl_mean = np.mean(cls, axis=0)

    expected = sigma**2 * omega_pix
    assert np.allclose(cl_mean, expected, rtol=0.05)


def test_map2cl_recovers_input_power_law(flatskymapparams_256):
    # Generate from a known power law with cl2map and recover it with map2cl.
    el_in = np.arange(1, 10000).astype(float)
    cl_in = 1e-6 * (el_in / 1000.0) ** -2.0

    np.random.seed(12)
    cls = []
    for _ in range(10):
        m = cl2map(flatskymapparams_256, cl_in, el_in)
        el, cl = map2cl(flatskymapparams_256, m, binsize=500, minbin=1000, maxbin=6000)
        cls.append(cl)
    cl_mean = np.mean(cls, axis=0)

    cl_expected = 1e-6 * (el / 1000.0) ** -2.0
    assert np.allclose(cl_mean, cl_expected, rtol=0.10)


# ---------------------------------------------------------------------------
# 2. Minkowski functionals vs Tomita GRF formulas (ratio form)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def grf_mfs():
    """Minkowski functionals of a GRF stack at nu thresholds, averaged over maps.

    The spectrum must be a band-pass: the Tomita comparison needs the field
    to be both well-resolved (coherence length of many pixels, or the
    marching-squares perimeter/Euler estimates are biased) and ergodic
    (many coherence patches per map, or the per-map z-scored histogram of a
    single realisation deviates from the ensemble Gaussian).  A coherence
    length of ~13 px on a 256-px map satisfies both.
    """
    from foregrounds_diffusion.morphology import compute_mfs

    el = np.arange(1, 10000).astype(float)
    cl = np.exp(-(((el - 1200.0) / 500.0) ** 2))

    np.random.seed(13)
    params = [256, 256, 1.40625, 1.40625]
    maps = np.array([cl2map(params, cl, el) for _ in range(24)])

    nus = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    zscore = lambda m: (m - m.mean()) / m.std()  # noqa: E731 — thresholds in sigma units
    M0, M1, M2 = compute_mfs(maps, zscore, nus)
    return nus, M0.mean(axis=0), M1.mean(axis=0), M2.mean(axis=0)


def test_mf_area_matches_erfc(grf_mfs):
    pytest.importorskip("quantimpy")
    nus, M0, _, _ = grf_mfs
    expected = 0.5 * erfc(nus / np.sqrt(2.0))
    # M0 is proportional to the area fraction (quantimpy returns an
    # unnormalised vertex count); normalise via the nu = 0 value
    ratio = M0 / M0[nus == 0.0]
    # rtol accommodates the nu = 2 bin: a 2% area fraction estimated from
    # 24 maps carries a few-percent sampling error
    assert np.allclose(ratio, expected / 0.5, rtol=0.04)


def test_mf_perimeter_matches_gaussian_form(grf_mfs):
    pytest.importorskip("quantimpy")
    nus, _, M1, _ = grf_mfs
    # Tomita: V1(nu) proportional to exp(-nu^2/2) -> ratios to nu = 0 are known.
    # The small high-|nu| excursion islands carry a curvature-dependent
    # marching-squares bias, so |nu| = 2 gets a looser tolerance.
    ratio = M1 / M1[nus == 0.0]
    expected = np.exp(-(nus**2) / 2.0)
    inner = np.abs(nus) <= 1.0
    assert np.allclose(ratio[inner], expected[inner], rtol=0.03)
    assert np.allclose(ratio[~inner], expected[~inner], rtol=0.08)


def test_mf_euler_matches_gaussian_form(grf_mfs):
    pytest.importorskip("quantimpy")
    nus, _, _, M2 = grf_mfs
    chi_p1 = M2[nus == 1.0][0]
    chi_m1 = M2[nus == -1.0][0]
    chi_p2 = M2[nus == 2.0][0]
    chi_0 = M2[nus == 0.0][0]
    # V2(nu) proportional to nu * exp(-nu^2/2): antisymmetric, zero at nu = 0
    assert chi_p1 > 0 and chi_m1 < 0
    assert abs(chi_p1 + chi_m1) < 0.1 * abs(chi_p1)
    assert abs(chi_0) < 0.1 * abs(chi_p1)
    expected_21 = 2.0 * np.exp(-1.5)  # V2(2)/V2(1)
    assert np.isclose(chi_p2 / chi_p1, expected_21, rtol=0.10)


# ---------------------------------------------------------------------------
# 3. Correlated-pair recovery in make_gaussian_realisation
# ---------------------------------------------------------------------------


def test_correlated_realisation_recovers_r_and_amplitudes(flatskymapparams_256):
    # Constant spectra with r = 0.6: recovered auto-amplitudes and r(ell) must
    # match the inputs (guards the "qu" default, which zeroes the correlation).
    el = np.arange(1, 10000).astype(float)
    cl1 = 2e-10 * np.ones(len(el))
    cl2 = 8e-10 * np.ones(len(el))
    r_in = 0.6
    cl12 = r_in * np.sqrt(cl1 * cl2)

    np.random.seed(14)
    cl_a, cl_b, cl_x = [], [], []
    for _ in range(10):
        sim = make_gaussian_realisation(
            flatskymapparams_256, el, cl1, cl2=cl2, cl12=cl12, qu_or_eb="eb"
        )
        _, a = map2cl(flatskymapparams_256, sim[0], binsize=500, minbin=1000, maxbin=6000)
        _, b = map2cl(flatskymapparams_256, sim[1], binsize=500, minbin=1000, maxbin=6000)
        _, x = map2cl(flatskymapparams_256, sim[0], sim[1], binsize=500, minbin=1000, maxbin=6000)
        cl_a.append(a)
        cl_b.append(b)
        cl_x.append(x)
    cl_a, cl_b, cl_x = (np.mean(c, axis=0) for c in (cl_a, cl_b, cl_x))

    assert np.allclose(cl_a, 2e-10, rtol=0.10)
    assert np.allclose(cl_b, 8e-10, rtol=0.10)
    r_rec = cl_x / np.sqrt(cl_a * cl_b)
    assert np.allclose(r_rec, r_in, atol=0.05)


# ---------------------------------------------------------------------------
# 4. Deterministic peak / minima counts
# ---------------------------------------------------------------------------


def _blob_map():
    """A smooth background with one off-grid maximum plus 4 sharp blobs."""
    n = 64
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    background = 0.1 * np.exp(-((yy - 30.3) ** 2 + (xx - 33.7) ** 2) / (2 * 12.0**2))
    blob_centres = [(12, 12), (12, 50), (50, 12), (50, 50)]
    m = background.copy()
    for cy, cx in blob_centres:
        m += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.5**2))
    return m, len(blob_centres)


def test_find_peaks_deterministic_count_and_amplitudes():
    m, k = _blob_map()
    peaks = find_peaks(m, filter_size=3)
    assert len(peaks) == k + 1  # 4 blobs + 1 background maximum
    top = np.sort(peaks)[-k:]
    assert np.all(top > 0.9)  # blob amplitudes ~1
    assert np.sort(peaks)[0] < 0.2  # background maximum ~0.1


def test_find_minima_deterministic_count():
    m, k = _blob_map()
    minima = find_minima(-m, filter_size=3)
    assert len(minima) == k + 1


def test_count_peaks_binned_matches_direct_count():
    m, k = _blob_map()
    # Negligible smoothing (fwhm << pixel): the binned count at a very low
    # threshold must equal the direct peak count.
    counts = count_peaks_binned(
        m[None, :, :], thresholds=[-10.0, 3.0], fwhm_arcmin=0.1, pixel_res_arcmin=1.40625
    )
    assert counts.shape == (1, 2)
    assert counts[0, 0] == k + 1
    # Only the 4 sharp blobs survive a 3-sigma threshold
    assert counts[0, 1] == k
    # Cumulative counts are monotonically non-increasing in the threshold
    assert counts[0, 1] <= counts[0, 0]


# ---------------------------------------------------------------------------
# 5. Scattering transform: invariances + phase-randomisation consistency
# ---------------------------------------------------------------------------


def _has_scattering_backend():
    from foregrounds_diffusion.scattering_stats import _get_backend

    try:
        _get_backend()
        return True
    except ImportError:
        return False


needs_backend = pytest.mark.skipif(
    not _has_scattering_backend(), reason="no scattering backend installed"
)


def _phase_randomise(m, rng):
    """Gaussian surrogate with the identical power spectrum (random phases)."""
    f = np.fft.fft2(m)
    phases = np.exp(2j * np.pi * rng.random(f.shape))
    surrogate = np.fft.ifft2(np.abs(f) * phases).real
    return surrogate


@pytest.fixture(scope="module")
def scattering_grf_stack():
    el = np.arange(1, 10000).astype(float)
    cl = (el / 1000.0) ** -2.5
    np.random.seed(15)
    params = [64, 64, 1.40625, 1.40625]
    return np.array([cl2map(params, cl, el) for _ in range(8)])


@needs_backend
def test_scattering_homogeneity(scattering_grf_stack):
    # |a.x * psi| = a.|x * psi| for a > 0: S1 and S2 scale exactly linearly
    from foregrounds_diffusion.scattering_stats import compute_scattering_coefficients

    x = scattering_grf_stack[:2]
    c1 = compute_scattering_coefficients(x, J=3, L=4, device="cpu")
    c5 = compute_scattering_coefficients(5.0 * x, J=3, L=4, device="cpu")
    assert np.allclose(c5["S1"], 5.0 * c1["S1"], rtol=1e-4)
    assert np.allclose(c5["S2"], 5.0 * c1["S2"], rtol=1e-4)


@needs_backend
def test_scattering_translation_invariance(scattering_grf_stack):
    from foregrounds_diffusion.scattering_stats import compute_scattering_coefficients

    x = scattering_grf_stack[:2]
    # Invariance is approximate, not exact: kymatio reflection-pads
    # internally, so a periodic roll perturbs coefficients near the
    # boundary at the few-percent level.  A broken implementation (e.g. a
    # missing modulus) produces order-unity differences.
    x_roll = np.roll(x, shift=(16, -8), axis=(1, 2))
    c = compute_scattering_coefficients(x, J=3, L=4, device="cpu")
    c_roll = compute_scattering_coefficients(x_roll, J=3, L=4, device="cpu")
    assert np.allclose(c_roll["S1"], c["S1"], rtol=0.05)


@needs_backend
def test_scattering_separates_non_gaussian_from_surrogate(scattering_grf_stack):
    # A GRF matches its phase-randomised surrogate; a non-Gaussian field with
    # the same power spectrum must separate in the sparsity ratio S2/S1 —
    # exactly the property the DDPM-vs-Gaussian-baseline comparison relies on.
    from foregrounds_diffusion.scattering_stats import compute_scattering_coefficients

    rng = np.random.default_rng(16)
    grf = scattering_grf_stack
    grf_surr = np.array([_phase_randomise(m, rng) for m in grf])
    nongauss = np.sign(grf) * grf**2  # pointwise non-linearity: non-Gaussian
    nongauss_surr = np.array([_phase_randomise(m, rng) for m in nongauss])

    def mean_s2_over_s1(stack):
        c = compute_scattering_coefficients(stack, J=3, L=4, device="cpu")
        s1 = c["S1_mean"]  # (J,)
        s2 = c["S2_mean"]  # (J, J, L)
        j1, j2 = 0, 2  # a well-populated cross-scale pair
        return s2[j1, j2].mean() / s1[j1]

    ratio_grf = mean_s2_over_s1(grf)
    ratio_grf_surr = mean_s2_over_s1(grf_surr)
    ratio_ng = mean_s2_over_s1(nongauss)
    ratio_ng_surr = mean_s2_over_s1(nongauss_surr)

    # Gaussian input: surrogate is statistically the same field
    assert np.isclose(ratio_grf, ratio_grf_surr, rtol=0.15)
    # Non-Gaussian input: the ratio must move away from its Gaussian surrogate
    gauss_gap = abs(ratio_grf - ratio_grf_surr)
    ng_gap = abs(ratio_ng - ratio_ng_surr)
    assert ng_gap > 3.0 * gauss_gap
