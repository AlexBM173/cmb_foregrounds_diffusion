"""FlatCutter patch-extraction tests.

Two regressions are guarded here:

1. ``rotate_to_pole_and_interpolate`` used to apply a spin-2 (Q, U) rotation to
   the last two maps whenever more than one map was passed, inferring "this is
   polarisation" from the list length.  Passing the independent scalar fields
   (CIB, tSZ, ...) together therefore rotated them into each other and corrupted
   both.

2. The fix makes ``spin2`` a *required, keyword-only* argument -- it can never
   be inferred -- and validates the map count against it, raising ``ValueError``
   on a conflict (e.g. ``spin2=True`` with one or four maps).
"""

import astropy.units as u
import healpy as hp
import numpy as np
import pytest

from foregrounds_diffusion.preprocessing import FlatCutter

NSIDE = 64
RES = 16


@pytest.fixture(scope="module")
def cutter():
    return FlatCutter(ang_x=6 * u.deg, ang_y=6 * u.deg, xres=RES, yres=RES)


@pytest.fixture(scope="module")
def maps():
    """Two smooth, sign-separated fields standing in for CIB (>=0) and tSZ (<=0)."""
    v = np.array(hp.pix2vec(NSIDE, np.arange(hp.nside2npix(NSIDE))))
    cib = np.abs(v[0] + 0.5 * v[1]) + 2.0
    tsz = -np.abs(0.3 * v[1] + 0.2 * v[2]) - 1.0
    return cib, tsz


# Includes the polar rings, where the historical corruption peaked.
LATS = [45.0, -45.0, 26.0, -87.0, 86.0, 90.0, -90.0]


# --------------------------------------------------------------------------
# Scalar extraction (spin2=False): fields must never leak into each other
# --------------------------------------------------------------------------
@pytest.mark.parametrize("lat", LATS)
def test_multi_map_extraction_matches_per_channel(cutter, maps, lat):
    """With spin2=False, passing maps together equals extracting each alone."""
    cib, tsz = maps
    joint = cutter.rotate_to_pole_and_interpolate(0 * u.deg, lat * u.deg, [cib, tsz], spin2=False)
    solo_cib = cutter.rotate_to_pole_and_interpolate(0 * u.deg, lat * u.deg, cib, spin2=False)
    solo_tsz = cutter.rotate_to_pole_and_interpolate(0 * u.deg, lat * u.deg, tsz, spin2=False)
    np.testing.assert_allclose(joint[:, :, 0], solo_cib[:, :, 0], rtol=0, atol=0)
    np.testing.assert_allclose(joint[:, :, 1], solo_tsz[:, :, 0], rtol=0, atol=0)


@pytest.mark.parametrize("lat", LATS)
def test_scalar_extraction_preserves_sign(cutter, maps, lat):
    """Bilinear interpolation is bounded by its inputs, so a one-sided field
    stays one-sided.  Sign flips were the visible symptom of the spin-2 bug."""
    cib, tsz = maps
    patch = cutter.rotate_to_pole_and_interpolate(0 * u.deg, lat * u.deg, [cib, tsz], spin2=False)
    assert patch[:, :, 0].min() >= cib.min()
    assert patch[:, :, 1].max() <= tsz.max()


def test_shape_and_single_map(cutter, maps):
    """Single-map calls keep their (xres, yres, 1) contract."""
    cib, _ = maps
    patch = cutter.rotate_to_pole_and_interpolate(0 * u.deg, 45 * u.deg, cib, spin2=False)
    assert patch.shape == (RES, RES, 1)


# --------------------------------------------------------------------------
# spin2 must be passed explicitly -- it is never inferred
# --------------------------------------------------------------------------
def test_spin2_is_required(cutter, maps):
    """Omitting spin2 is a TypeError (keyword-only, no default)."""
    cib, _ = maps
    with pytest.raises(TypeError):
        cutter.rotate_to_pole_and_interpolate(0 * u.deg, 45 * u.deg, cib)


def test_spin2_is_keyword_only(cutter, maps):
    """spin2 cannot be passed positionally."""
    cib, tsz = maps
    with pytest.raises(TypeError):
        cutter.rotate_to_pole_and_interpolate(0 * u.deg, 45 * u.deg, [cib, tsz], True)


# --------------------------------------------------------------------------
# spin2=True: real polarisation use, with a sensible map count
# --------------------------------------------------------------------------
def test_spin2_true_mixes_qu(cutter, maps):
    """spin2=True rotates the (Q, U) pair, so it differs from the scalar path."""
    cib, tsz = maps
    plain = cutter.rotate_to_pole_and_interpolate(0 * u.deg, -87 * u.deg, [cib, tsz], spin2=False)
    rotated = cutter.rotate_to_pole_and_interpolate(0 * u.deg, -87 * u.deg, [cib, tsz], spin2=True)
    assert not np.allclose(plain, rotated)


def test_spin2_true_accepts_iqu(cutter, maps):
    """Three maps [I, Q, U] are valid: I passes through, (Q, U) are rotated."""
    cib, tsz = maps
    intensity = cib + tsz  # a third, independent scalar to act as I
    out = cutter.rotate_to_pole_and_interpolate(
        0 * u.deg, -87 * u.deg, [intensity, cib, tsz], spin2=True
    )
    assert out.shape == (RES, RES, 3)
    # I (first channel) is untouched by the spin rotation -> identical to its
    # own scalar extraction; Q, U are mixed and so must differ.
    solo_i = cutter.rotate_to_pole_and_interpolate(0 * u.deg, -87 * u.deg, intensity, spin2=False)
    np.testing.assert_allclose(out[:, :, 0], solo_i[:, :, 0], rtol=0, atol=0)


# --------------------------------------------------------------------------
# Conflicts between the map count and spin2 raise ValueError
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n_maps", [1, 4, 5])
def test_spin2_true_rejects_bad_count(cutter, maps, n_maps):
    """spin2=True demands exactly 2 (Q,U) or 3 (I,Q,U) maps."""
    cib, tsz = maps
    stack = [cib, tsz, cib, tsz, cib][:n_maps]
    with pytest.raises(ValueError, match="spin2=True expects 2 maps"):
        cutter.rotate_to_pole_and_interpolate(0 * u.deg, 45 * u.deg, stack, spin2=True)


@pytest.mark.parametrize("spin2", [True, False])
def test_empty_map_list_rejected(cutter, spin2):
    with pytest.raises(ValueError, match="at least one map"):
        cutter.rotate_to_pole_and_interpolate(0 * u.deg, 45 * u.deg, [], spin2=spin2)


@pytest.mark.parametrize("n_maps", [1, 2, 3, 4])
def test_scalar_mode_accepts_any_count(cutter, maps, n_maps):
    """spin2=False imposes no upper bound -- scalars are unambiguous."""
    cib, tsz = maps
    stack = [cib, tsz, cib, tsz][:n_maps]
    out = cutter.rotate_to_pole_and_interpolate(0 * u.deg, 45 * u.deg, stack, spin2=False)
    assert out.shape == (RES, RES, n_maps)
