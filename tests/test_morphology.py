import numpy as np
import pytest

from foregrounds_diffusion.morphology import (
    _eigendecompose_2x2,
    _tensor_W012,
    _tensor_W200,
    compute_minkowski_tensors,
)

# ---------------------------------------------------------------------------
# _eigendecompose_2x2
# ---------------------------------------------------------------------------


def test_eigendecompose_identity_is_isotropic():
    beta, theta = _eigendecompose_2x2(np.eye(2))
    assert beta == pytest.approx(1.0)
    # theta is undefined when eigenvalues are equal; only beta is tested here


def test_eigendecompose_anisotropic_tensor():
    # Diagonal tensor with 4:1 eigenvalue ratio → β = 0.25
    W = np.diag([1.0, 4.0])
    beta, theta = _eigendecompose_2x2(W)
    assert beta == pytest.approx(0.25, abs=1e-10)


def test_eigendecompose_beta_in_unit_interval(binary_map):
    from foregrounds_diffusion.morphology import _tensor_W012

    W = _tensor_W012(binary_map)
    beta, _ = _eigendecompose_2x2(W)
    assert 0.0 <= beta <= 1.0


# ---------------------------------------------------------------------------
# _tensor_W012
# ---------------------------------------------------------------------------


def test_tensor_W012_all_ones_approximately_isotropic():
    # Square boundary → β close to 1
    binary = np.ones((64, 64), dtype=bool)
    W = _tensor_W012(binary)
    beta, _ = _eigendecompose_2x2(W)
    assert beta > 0.9


def test_tensor_W012_returns_2x2(binary_map):
    W = _tensor_W012(binary_map)
    assert W.shape == (2, 2)


# ---------------------------------------------------------------------------
# _tensor_W200
# ---------------------------------------------------------------------------


def test_tensor_W200_circular_is_isotropic():
    # Circular disk centred in a 64×64 map → β ≈ 1
    y, x = np.mgrid[-32:32, -32:32]
    binary = (x**2 + y**2) < 15**2
    W = _tensor_W200(binary)
    beta, _ = _eigendecompose_2x2(W)
    assert beta > 0.9


def test_tensor_W200_elongated_is_anisotropic():
    # Thin horizontal strip → β < 0.5 (highly elongated)
    binary = np.zeros((64, 64), dtype=bool)
    binary[30:34, 5:59] = True  # 4-pixel tall, 54-pixel wide
    W = _tensor_W200(binary)
    beta, _ = _eigendecompose_2x2(W)
    assert beta < 0.5


# ---------------------------------------------------------------------------
# compute_minkowski_tensors
# ---------------------------------------------------------------------------


def test_compute_minkowski_tensors_shape(patch_stack):
    thresholds = np.linspace(-2, 2, 10)
    result = compute_minkowski_tensors(patch_stack, lambda x: x, thresholds)
    N, T = len(patch_stack), len(thresholds)
    assert "W012" in result
    assert result["W012"]["beta"].shape == (N, T)
    assert result["W012"]["theta"].shape == (N, T)


def test_compute_minkowski_tensors_beta_in_unit_interval(patch_stack):
    thresholds = np.linspace(-2, 2, 10)
    result = compute_minkowski_tensors(patch_stack, lambda x: x, thresholds)
    beta = result["W012"]["beta"]
    assert np.all(beta >= 0.0)
    assert np.all(beta <= 1.0)


def test_compute_minkowski_tensors_multiple_types(patch_stack):
    thresholds = np.linspace(-1, 1, 5)
    result = compute_minkowski_tensors(
        patch_stack,
        lambda x: x,
        thresholds,
        tensor_types=("W012", "W200", "W201"),
    )
    for key in ("W012", "W200", "W201"):
        assert key in result
        assert result[key]["beta"].shape == (len(patch_stack), len(thresholds))


# ---------------------------------------------------------------------------
# compute_mfs (optional — requires quantimpy)
# ---------------------------------------------------------------------------


def _require_quantimpy():
    """Skip the test if quantimpy is missing or ABI-incompatible with numpy."""
    try:
        from quantimpy import minkowski  # noqa: F401
    except (ImportError, ValueError) as exc:
        pytest.skip(f"quantimpy not available: {exc}")


def test_compute_mfs_shape(patch_stack):
    _require_quantimpy()
    from foregrounds_diffusion.morphology import compute_mfs
    from foregrounds_diffusion.preprocessing import apply_maxmin_normalization

    thresholds = np.linspace(0.1, 0.9, 8)
    M0, M1, M2 = compute_mfs(patch_stack, apply_maxmin_normalization, thresholds)
    N, T = len(patch_stack), len(thresholds)
    assert M0.shape == (N, T)
    assert M1.shape == (N, T)
    assert M2.shape == (N, T)


def test_compute_mfs_area_decreasing_with_threshold(patch_stack):
    _require_quantimpy()
    from foregrounds_diffusion.morphology import compute_mfs
    from foregrounds_diffusion.preprocessing import apply_maxmin_normalization

    thresholds = np.linspace(0.1, 0.9, 8)
    M0, _, _ = compute_mfs(patch_stack, apply_maxmin_normalization, thresholds)
    # M0 (area fraction) should decrease as threshold increases
    assert np.all(np.diff(M0.mean(axis=0)) < 0)
