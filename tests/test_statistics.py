import numpy as np
import pytest

from foregrounds_diffusion.statistics import fitgaussian, gaussian, moments, stats


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
