import numpy as np
import pytest

from foregrounds_diffusion.statistics import fitgaussian, fitting_func, gaussian, moments, stats


def test_gaussian_callable():
    g = gaussian(height=1.0, center_x=5.0, center_y=5.0, width_x=2.0, width_y=2.0)
    assert callable(g)


def test_gaussian_peak_value():
    h = 3.0
    g = gaussian(height=h, center_x=4.0, center_y=4.0, width_x=1.0, width_y=1.0)
    assert g(4.0, 4.0) == pytest.approx(h)


def test_gaussian_decays_away_from_centre():
    g = gaussian(height=1.0, center_x=0.0, center_y=0.0, width_x=1.0, width_y=1.0)
    assert g(0.0, 0.0) > g(2.0, 0.0)


def test_moments_returns_correct_length():
    rng = np.random.default_rng(42)
    data = rng.random((32, 32))
    result = moments(data)
    assert len(result) == 5


def test_moments_centre_estimate():
    # Synthetic Gaussian centred at (16, 16) in a 32×32 image
    x = np.arange(32)
    y = np.arange(32)
    xx, yy = np.meshgrid(x, y)
    data = np.exp(-((xx - 16) ** 2 + (yy - 16) ** 2) / (2 * 3**2))
    _, cx, cy, _, _ = moments(data)
    assert abs(cx - 16) < 1.0
    assert abs(cy - 16) < 1.0


def test_fitgaussian_centre_within_one_pixel():
    # moments() / fitgaussian() return (height, row_centroid, col_centroid, ...).
    # np.meshgrid(x, y) gives xx[i,j]=j (col) and yy[i,j]=i (row), so the
    # Gaussian must be built with col offset applied to xx and row to yy.
    x = np.arange(32)
    y = np.arange(32)
    xx, yy = np.meshgrid(x, y)
    row_true, col_true = 17, 14
    data = np.exp(-((xx - col_true) ** 2 + (yy - row_true) ** 2) / (2 * 3**2))
    _, row_fit, col_fit, _, _ = fitgaussian(data)
    assert abs(row_fit - row_true) < 1.0
    assert abs(col_fit - col_true) < 1.0


def test_stats_values():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mn, mx, mean, std = stats(arr)
    assert mn == pytest.approx(1.0)
    assert mx == pytest.approx(5.0)
    assert mean == pytest.approx(3.0)
    assert std == pytest.approx(np.std(arr))


# ---------------------------------------------------------------------------
# fitting_func
# ---------------------------------------------------------------------------


def _make_grid(size=16):
    x = np.linspace(0, size - 1, size, dtype=float)
    xg, yg = np.meshgrid(x, x)
    return xg, yg


def test_fitting_func_return_fit_gives_image():
    xg, yg = _make_grid()
    p = np.array([0.0, 1.0, 7.0, 7.0, 3.0])
    tmap = np.zeros((16, 16))
    out = fitting_func(p, p.copy(), xg, yg, tmap, return_fit=1)
    assert out.shape == (16, 16)
    # Peak should be near centre (7, 7)
    assert out[7, 7] > out[0, 0]


def test_fitting_func_return_residual():
    xg, yg = _make_grid()
    p = np.array([0.0, 1.0, 7.0, 7.0, 3.0])
    tmap = np.ones((16, 16)) * 0.5
    resid = fitting_func(p, p.copy(), xg, yg, tmap, return_fit=0)
    assert resid.shape == (16 * 16,)


def test_fitting_func_fixed_parameters():
    xg, yg = _make_grid()
    p0 = np.array([0.0, 1.0, 8.0, 8.0, 2.0])
    p = np.array([0.1, 0.9, 5.0, 5.0, 3.0])
    tmap = np.zeros((16, 16))
    # Fix indices 2 and 3 (x_cen, y_cen): they must be restored to p0 values.
    out_p = fitting_func(p.copy(), p0, xg, yg, tmap, fixed=[2, 3], return_fit=1)
    out_ref = fitting_func(np.array([0.1, 0.9, 8.0, 8.0, 3.0]), p0, xg, yg, tmap, return_fit=1)
    np.testing.assert_allclose(out_p, out_ref, atol=1e-10)


def test_fitting_func_lbounds_violation_returns_tmap():
    xg, yg = _make_grid()
    tmap = np.ones((16, 16)) * 99.0
    p = np.array([0.0, 0.1, 7.0, 7.0, 0.5])  # width 0.5 < lbound 1.0
    lbounds = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    # When a parameter is below lbound, fitting_func returns tmap immediately.
    result = fitting_func(p, p.copy(), xg, yg, tmap, lbounds=lbounds, return_fit=1)
    np.testing.assert_array_equal(result, tmap)


def test_fitting_func_ubounds_violation_returns_tmap():
    xg, yg = _make_grid()
    tmap = np.ones((16, 16)) * 77.0
    p = np.array([0.0, 2.0, 7.0, 7.0, 5.0])  # amp 2.0 > ubound 1.5
    ubounds = np.array([1.0, 1.5, 20.0, 20.0, 10.0])
    result = fitting_func(p, p.copy(), xg, yg, tmap, ubounds=ubounds, return_fit=1)
    np.testing.assert_array_equal(result, tmap)


def test_fitting_func_7param_rotated_gaussian():
    xg, yg = _make_grid(32)
    # 7-parameter vector: [baseline, amp, x_cen, y_cen, wx, wy, rotation_deg]
    p = np.array([0.0, 1.0, 15.0, 15.0, 4.0, 2.0, 30.0])
    tmap = np.zeros((32, 32))
    out = fitting_func(p, p.copy(), xg, yg, tmap, return_fit=1)
    assert out.shape == (32, 32)
    assert out.max() == pytest.approx(1.0, abs=0.05)
