"""Tests for scattering_stats.py.

compute_scattering_coefficients and compute_scattering_covariance require
an external backend (Cheng et al. scattering_transform or kymatio).  When
neither is installed, tests that need them are skipped.

scattering_summary is pure numpy and runs without any backend.
"""

import importlib
import sys
from unittest.mock import patch as mock_patch

import numpy as np
import pytest

import foregrounds_diffusion.scattering_stats as ss

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_backend():
    try:
        ss._get_backend()
        return True
    except ImportError:
        return False


def _fake_coeffs(N=5, J=3, L=4):
    return {
        "J": J,
        "L": L,
        "S0": np.ones((N, 1)),
        "S1": np.ones((N, J)),
        "S2": np.ones((N, J, J, L)),
        "S1_mean": np.ones(J),
        "S2_mean": np.ones((J, J, L)),
    }


# ---------------------------------------------------------------------------
# _get_backend — graceful failure
# ---------------------------------------------------------------------------


def test_get_backend_raises_when_both_absent():
    """When neither scattering nor kymatio is importable, ImportError is raised."""
    with mock_patch.dict(sys.modules, {"scattering": None, "kymatio": None}):
        # Force both to appear missing even if installed
        blocked = {"scattering": None, "kymatio": None}
        with mock_patch.dict(sys.modules, blocked):
            # Reload to bypass cached import inside the module
            with pytest.raises((ImportError, SystemError)):
                # Either ImportError or SystemError when module is None
                importlib.reload(ss)
                ss._get_backend()


def test_get_backend_import_error_message():
    """ImportError message should mention both backend options."""
    # A None entry in sys.modules makes `import <name>` raise ImportError,
    # simulating absence even when the package is actually installed.
    with mock_patch.dict(sys.modules, {"scattering": None, "kymatio": None}):
        with pytest.raises(ImportError) as exc_info:
            ss._get_backend()
    msg = str(exc_info.value)
    assert "scattering" in msg.lower() and "kymatio" in msg.lower()


# ---------------------------------------------------------------------------
# scattering_summary — pure numpy, no backend needed
# ---------------------------------------------------------------------------


def test_scattering_summary_shape_all_scales():
    """Output has N rows; features = J (S1) + J*(J-1)/2*L (S2 upper triangle)."""
    J, L, N = 3, 4, 5
    coeffs = _fake_coeffs(N=N, J=J, L=L)
    out = ss.scattering_summary(coeffs)
    # S1: J features; S2 upper triangle: (0,1),(0,2),(1,2) → 3 pairs × L = 12
    expected_features = J + (J * (J - 1) // 2) * L
    assert out.shape == (N, expected_features)


def test_scattering_summary_shape_subset_scales():
    J, L, N = 5, 4, 8
    coeffs = _fake_coeffs(N=N, J=J, L=L)
    scale_idx = [0, 2, 4]
    out = ss.scattering_summary(coeffs, scale_idx=scale_idx)
    # S1: 3 features; S2 upper tri among [0,2,4] in range(J):
    #   pairs with j1 in [0,2,4] and j2 in [j1+1..J): (0,1),(0,2),(0,3),(0,4),
    #   (2,3),(2,4),(4, ?) → j1=0:j2=1..4 (4 pairs), j1=2:j2=3,4 (2 pairs), j1=4: none
    # = 4+2 = 6 pairs × L=4 = 24; S1=3; total=27
    n_s1 = len(scale_idx)
    n_pairs = sum(1 for j1 in scale_idx for j2 in range(j1 + 1, J))
    expected = n_s1 + n_pairs * L
    assert out.shape == (N, expected)


def test_scattering_summary_values_ones_coeffs():
    """With all-ones coefficients, summary values should all be 1."""
    coeffs = _fake_coeffs(N=4, J=3, L=2)
    out = ss.scattering_summary(coeffs)
    np.testing.assert_array_equal(out, np.ones_like(out))


def test_scattering_summary_single_scale():
    """With J=1, S2 upper triangle is empty; output is just S1."""
    coeffs = _fake_coeffs(N=3, J=1, L=4)
    out = ss.scattering_summary(coeffs)
    assert out.shape == (3, 1)


# ---------------------------------------------------------------------------
# compute_scattering_coefficients — skip when no backend
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_backend(), reason="no scattering backend installed")
def test_compute_scattering_coefficients_output_keys():
    rng = np.random.default_rng(0)
    patches = rng.standard_normal((4, 32, 32)).astype(np.float32)
    coeffs = ss.compute_scattering_coefficients(patches, J=2, L=2)
    for key in ("S0", "S1", "S2", "S1_mean", "S2_mean", "J", "L"):
        assert key in coeffs


@pytest.mark.skipif(not _has_backend(), reason="no scattering backend installed")
def test_compute_scattering_coefficients_shapes():
    N, J, L = 6, 3, 2
    rng = np.random.default_rng(1)
    patches = rng.standard_normal((N, 32, 32)).astype(np.float32)
    coeffs = ss.compute_scattering_coefficients(patches, J=J, L=L)
    assert coeffs["S0"].shape == (N, 1)
    assert coeffs["S1"].shape == (N, J)
    assert coeffs["S2"].shape[0] == N


# ---------------------------------------------------------------------------
# compute_scattering_covariance — skip when no backend
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_backend(), reason="no scattering backend installed")
def test_compute_scattering_covariance_returns_dict_or_none():
    rng = np.random.default_rng(2)
    patches = rng.standard_normal((3, 32, 32)).astype(np.float32)
    result = ss.compute_scattering_covariance(patches, J=2, L=2)
    # Returns None when Cheng backend absent, dict otherwise.
    assert result is None or isinstance(result, dict)
