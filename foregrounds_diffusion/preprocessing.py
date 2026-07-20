"""Data preprocessing utilities: normalisation, patch extraction, and filtering.

This module covers everything between raw HEALPix FITS files and the
training-ready ``.npy`` arrays consumed by :mod:`foregrounds_diffusion.train`.
The full pipeline is documented in ``docs/tutorials/02_masking.ipynb`` and
``docs/tutorials/03_patch_extraction.ipynb``.

Normalisation
-------------
Two schemes are used, one per channel:

- **CIB** (``apply_maxmin_normalization``): min–max scaled to [0, 1].
  CIB intensities are non-negative by construction, so this is lossless.
- **tSZ** (``apply_stdnorm``): z-score normalised to zero mean and unit
  variance per channel across the full training set.

The inverse operations are :func:`renormalize_dm_maps` (for post-sampling
rescaling back to physical units) and :func:`denormalize_dm_maps` (for
z-score inversion).

Patch extraction
----------------
:class:`FlatCutter` projects HEALPix full-sky maps onto flat-sky patches
using a gnomonic (tangent-plane) projection.  :func:`get_patch_centers`
returns a grid of pointing centres that tiles the sky.

Filtering
---------
:func:`get_lpf_hpf` builds low-pass, high-pass, or band-pass 2D filters in
ℓ-space.  The training data uses a low-pass cut at ℓ = 7000 to avoid
aliasing artefacts from the HEALPix pixel window function.

:func:`wiener_filter` builds a Wiener filter given signal and noise spectra.

Dataset splitting
-----------------
:func:`split_data_to_tensors` performs an 80/10/10 train/val/test split,
transposes from channels-last ``(N, H, W, C)`` to channels-first
``(N, C, H, W)``, and returns ``torch.Tensor`` objects ready for the
``DataLoader``.
"""

import astropy.units as u
import healpy as hp
import numpy as np
import torch

from foregrounds_diffusion.flatmaps import cl_to_cl2d, get_lxly

# ---------------------------------------------------------------------------
# Normalisation utilities
# ---------------------------------------------------------------------------


def apply_maxmin_normalization(maps):
    """Normalise an array to [0, 1] using global min–max scaling.

    Parameters
    ----------
    maps : ndarray
        Input array of any shape.  NaN values are ignored when computing
        the range.

    Returns
    -------
    ndarray
        Normalised copy with values in [0, 1].  Returns an array of zeros
        if ``max − min == 0`` (constant input).
    """
    min_val = np.nanmin(maps)
    max_val = np.nanmax(maps)
    denom = max_val - min_val
    if denom == 0:
        return np.zeros_like(maps)
    return (maps - min_val) / denom


def apply_stdnorm(maps):
    """Z-score normalise each channel of a channels-last array independently.

    Parameters
    ----------
    maps : ndarray, shape (..., C)
        Array with channels in the last axis.

    Returns
    -------
    ndarray, shape (..., C)
        Copy with each channel shifted to zero mean and unit variance.
        Channels with zero standard deviation are set to zero.
    """
    maps = maps.copy()
    for c in range(maps.shape[-1]):
        channel = maps[..., c]
        std = np.std(channel)
        if std == 0:
            maps[..., c] = 0.0
        else:
            maps[..., c] = (channel - np.mean(channel)) / std
    return maps


def renormalize_dm_maps(dm_maps, train_maps, variance_scaling=True):
    """Rescale diffusion-model output maps to match training-set statistics.

    Parameters
    ----------
    dm_maps : ndarray, shape (N, C, H, W)
        Raw diffusion-model samples (channels-first).
    train_maps : ndarray, shape (N, H, W, C)
        Reference training maps (channels-last).
    variance_scaling : bool
        If *True*, also match the per-channel standard deviation.

    Returns
    -------
    ndarray, shape (N, C, H, W)
        Renormalised maps in channels-first layout.
    """
    dm_maps = np.transpose(dm_maps, (0, 2, 3, 1)).copy()
    num_channels = train_maps.shape[-1]

    for i in range(num_channels):
        tr_min = np.min(train_maps[:, :, :, i])
        tr_max = np.max(train_maps[:, :, :, i])
        dm_maps[:, :, :, i] = dm_maps[:, :, :, i] * (tr_max - tr_min) + tr_min

        if variance_scaling:
            dm_mean = np.mean(dm_maps[:, :, :, i])
            dm_std = np.std(dm_maps[:, :, :, i])
            tr_mean = np.mean(train_maps[:, :, :, i])
            tr_std = np.std(train_maps[:, :, :, i])
            if dm_std == 0:
                # Constant channel: shift the mean only, skip the (tr_std/dm_std)
                # scaling that would otherwise divide by zero and produce NaNs.
                dm_maps[:, :, :, i] = dm_maps[:, :, :, i] - dm_mean + tr_mean
            else:
                dm_maps[:, :, :, i] = (dm_maps[:, :, :, i] - dm_mean) * (tr_std / dm_std) + tr_mean

    return np.transpose(dm_maps, (0, 3, 1, 2))


def rescale_samples(samples, cib_factor=1.0, tsz_factor=1.0):
    """Apply per-channel scalar rescaling to DDPM samples (paper §3.2).

    Prabhu et al. correct the DDPM's slight under-dispersion by multiplying
    each generated channel by a single global factor: the ratio of the AGORA
    sample standard deviation to the generated sample standard deviation.
    This is a pure scalar multiply, distinct from the affine
    :func:`renormalize_dm_maps`.

    Parameters
    ----------
    samples : ndarray, shape (N, C, H, W)
        Raw DDPM samples, channels-first (channel 0 = CIB, channel 1 = tSZ).
    cib_factor, tsz_factor : float
        Multiplicative factors for the CIB (channel 0) and tSZ (channel 1)
        channels.  ``1.0`` leaves a channel unchanged.  Prabhu et al. cite
        1.0328 (CIB) and 1.1425 (tSZ) for their checkpoint; for a different
        checkpoint, measure ``std(AGORA) / std(generated)`` per channel.

    Returns
    -------
    ndarray, shape (N, C, H, W)
        Rescaled copy.  The input is not modified.
    """
    out = samples.copy()
    out[:, 0] *= cib_factor
    if out.shape[1] > 1:
        out[:, 1] *= tsz_factor
    return out


def denormalize_dm_maps(dm_maps, cib_mean, cib_std, tsz_mean, tsz_std):
    """Invert Z-score normalisation applied during patch extraction.

    Parameters
    ----------
    dm_maps : ndarray, shape (N, 2, H, W)
        Raw DDPM samples in Z-score space (channels-first).
    cib_mean, cib_std : float
        Z-score parameters for the CIB channel (channel 0).
    tsz_mean, tsz_std : float
        Z-score parameters for the tSZ channel (channel 1).

    Returns
    -------
    ndarray, shape (N, 2, H, W)
        Denormalised maps in the same physical units as the training patches.
    """
    dm_maps = dm_maps.copy()
    dm_maps[:, 0] = dm_maps[:, 0] * cib_std + cib_mean
    dm_maps[:, 1] = dm_maps[:, 1] * tsz_std + tsz_mean
    return dm_maps


# ---------------------------------------------------------------------------
# Moments loading
# ---------------------------------------------------------------------------


def load_all_moments(filename, bandpass_centers, max_lines=-1):
    """Load and normalise scattering moments from a .npy file.

    Parameters
    ----------
    filename : str
        Path to the .npy moments array with shape (N, L, 12).
    bandpass_centers : array_like
        Bandpass centre values used for normalisation.
    max_lines : int
        Number of realisations to load.  *-1* loads all.

    Returns
    -------
    dict
        Dictionary keyed ``"moment_00"`` … ``"moment_11"``, each value being
        a list of normalised moment arrays.
    """
    _raw = np.load(filename)
    moments_data = _raw if max_lines == -1 else _raw[:max_lines]
    norms = [
        bandpass_centers,  # S2aa
        bandpass_centers,  # S2bb
        bandpass_centers,  # S2ab
        bandpass_centers,  # S3aaa
        bandpass_centers,  # S3bbb
        bandpass_centers,  # S3aab
        bandpass_centers,  # S3abb
        bandpass_centers**2,  # S4aaaa
        bandpass_centers**2,  # S4bbbb
        bandpass_centers**2,  # S4aaab
        bandpass_centers**2,  # S4aabb
        bandpass_centers**2,  # S4abbb
    ]
    moments = {}
    for i in range(12):
        label = f"moment_{i:02d}"
        moments[label] = [m / norms[i] for m in moments_data[:, :, i]]
    return moments


# ---------------------------------------------------------------------------
# Patch-centre computation and HEALPix patch extraction
# ---------------------------------------------------------------------------


@u.quantity_input
def get_patch_centers(gal_cut: u.deg, step_size: u.deg, pole_cut: u.deg):
    """Compute patch centres on the sky, avoiding the Galactic plane.

    Parameters
    ----------
    gal_cut : `~astropy.units.Quantity`
        Half-width of the Galactic-plane exclusion zone in degrees.
    step_size : `~astropy.units.Quantity`
        Stepping distance in Galactic latitude in degrees.
    pole_cut : `~astropy.units.Quantity`
        Keeps latitude rings ``pole_cut / 2`` clear of either pole.  Note this
        bounds the *ring latitudes*, not the patch footprints: a patch centred
        at the limit still extends over the pole.  Because rings are quantised
        by ``step_size``, the bound often does not bind at all -- at
        ``gal_cut=20``, ``step_size=6`` it only shifts the southern rings off
        the pole (-90 -> -87) and leaves the northern rings untouched.

    Returns
    -------
    list of tuple
        Each element is ``(lon, lat)`` as `~astropy.units.Quantity` in degrees.
    """
    gal_cut = gal_cut.to(u.deg)
    step_size = step_size.to(u.deg)
    pole_cut = pole_cut.to(u.deg)
    southern = (
        np.arange(-90 + pole_cut.value / 2, (-gal_cut - step_size).value, step_size.value) * u.deg
    )
    northern = (
        np.arange((gal_cut + step_size).value, 90 - pole_cut.value / 2, step_size.value) * u.deg
    )
    lat_range = np.concatenate((southern, northern))

    centers = []
    for t in lat_range:
        step = step_size.value / np.cos(t.to(u.rad).value)
        for i in np.arange(0, 360, step):
            centers.append((i * u.deg, t))
    return centers


class FlatCutter:
    """Extract flat-sky patches from a HEALPix map by rotation and interpolation.

    Parameters
    ----------
    ang_x, ang_y : `~astropy.units.Quantity`
        Angular extent of the patch in the x and y directions.
    xres, yres : int
        Number of pixels in x and y.
    """

    @u.quantity_input
    def __init__(self, ang_x: u.deg, ang_y: u.deg, xres: int, yres: int):
        self.xres = xres
        self.yres = yres
        self.ang_x = ang_x
        self.ang_y = ang_y

        self.xarr = np.linspace(
            -self.ang_x.to(u.rad).value / 2.0, self.ang_x.to(u.rad).value / 2.0, xres
        )
        self.yarr = np.linspace(
            -self.ang_y.to(u.rad).value / 2.0, self.ang_y.to(u.rad).value / 2.0, yres
        )

        xgrid, ygrid = np.meshgrid(self.xarr, self.yarr)
        xgrid = xgrid.ravel()[None, :]
        ygrid = ygrid.ravel()[None, :]
        zgrid = np.ones_like(ygrid)

        self.vecs = np.concatenate((xgrid, ygrid, zgrid)).T
        self.lons, self.lats = hp.vec2ang(self.vecs, lonlat=True)
        self.lats *= u.deg
        self.lons *= u.deg

    @u.quantity_input
    def rotate_to_pole_and_interpolate(self, lon: u.deg, lat: u.deg, ma, *, spin2: bool):
        """Rotate the patch grid to *(lon, lat)* and sample the map.

        Parameters
        ----------
        lon, lat : `~astropy.units.Quantity`
            Sky position of the patch centre.
        ma : ndarray or list of ndarray
            HEALPix map(s) to sample.  The permitted count depends on *spin2*
            (see below); a conflict raises `ValueError`.
        spin2 : bool, keyword-only, required
            Whether *ma* holds a spin-2 (polarisation) field.  Must be passed
            explicitly -- it is never inferred from the number of maps.

            * ``False`` -- every entry is an independent scalar (I) field,
              sampled on its own.  Any count ``>= 1`` is allowed.
            * ``True`` -- the **last two** entries are the (Q, U) components of
              one spin-2 field and are rotated into the patch frame.  Exactly
              two maps ``[Q, U]`` or three ``[I, Q, U]`` are allowed.

        Returns
        -------
        ndarray, shape (xres, yres) or (xres, yres, nmaps)
            Interpolated flat-sky patch(es).

        Raises
        ------
        ValueError
            If the number of maps conflicts with *spin2*: zero maps in either
            mode, or a count outside ``{2, 3}`` when ``spin2=True``.

        Notes
        -----
        *spin2* is required precisely because inferring it from ``len(ma)``
        caused a real corruption: the rotation mixes the last two maps into
        each other, so silently applying it to unrelated scalar fields (CIB,
        tSZ, kSZ, κ) damages both.  The mixing angle grows towards the poles
        but is non-zero at every latitude.
        """
        if isinstance(ma, (list, tuple)) and len(ma) == 0:
            raise ValueError("rotate_to_pole_and_interpolate requires at least one map")
        if hp.pixelfunc.maptype(ma) == 0:
            ma = [ma]
        n = len(ma)
        if spin2 and n not in (2, 3):
            raise ValueError(
                f"spin2=True expects 2 maps [Q, U] or 3 maps [I, Q, U], got {n}. "
                "Pass spin2=False to sample independent scalar fields."
            )
        rotator = hp.Rotator(rot=[lon.to(u.deg).value, lat.to(u.deg).value - 90.0], deg=True)
        self.inv_lon_grid, self.inv_lat_grid = rotator.I(
            self.lons.to(u.deg).value, self.lats.to(u.deg).value, lonlat=True
        )
        m_rot = [
            hp.get_interp_val(each, self.inv_lon_grid, self.inv_lat_grid, lonlat=True)
            for each in ma
        ]

        if spin2:
            m_rot[-2], m_rot[-1] = _spin2rot(
                m_rot[-2],
                m_rot[-1],
                rotator.angle_ref(self.inv_lon_grid, self.inv_lat_grid, lonlat=True),
            )
            m_rot[-2], m_rot[-1] = _spin2rot(m_rot[-2], m_rot[-1], self.lons.to(u.rad).value)

        return np.moveaxis(np.array(m_rot).reshape(-1, self.xres, self.yres), 0, -1)


def _spin2rot(q, u, angle):
    """Rotate spin-2 field (Q, U) by *angle* (internal helper)."""
    c, s = np.cos(2 * angle), np.sin(2 * angle)
    return c * q - s * u, s * q + c * u


# ---------------------------------------------------------------------------
# HEALPix map utilities
# ---------------------------------------------------------------------------


def replace_zeros_with_neighbor_avg(hp_map):
    """Replace zero pixels in a HEALPix map with the average of non-zero neighbours.

    Parameters
    ----------
    hp_map : ndarray
        1D HEALPix map array.

    Returns
    -------
    ndarray
        Modified map with zero pixels filled.
    """
    nside = hp.get_nside(hp_map)
    zeros_indices = np.where(hp_map == 0)[0]
    for idx in zeros_indices:
        neighbors = hp.get_all_neighbours(nside, idx)
        valid = neighbors[(neighbors >= 0) & (hp_map[neighbors] != 0)]
        hp_map[idx] = np.mean(hp_map[valid]) if len(valid) > 0 else 0
    return hp_map


# ---------------------------------------------------------------------------
# Fourier-space filtering
# ---------------------------------------------------------------------------


def get_lpf_hpf(flatskymapparams, lmin_lmax, filter_type=0):
    """Build a 2D Fourier filter (low-pass, high-pass, or band-pass).

    Parameters
    ----------
    flatskymapparams : list
        [nx, ny, dx, dy] — see :func:`~foregrounds_diffusion.flatmaps.get_lxly`.
    lmin_lmax : float or tuple of float
        Cutoff multipole (scalar) or (lmin, lmax) for band-pass.
    filter_type : int
        0 → low-pass, 1 → high-pass, 2 → band-pass.

    Returns
    -------
    ndarray
        2D binary filter array.
    """
    lx, ly = get_lxly(flatskymapparams)
    ell = np.sqrt(lx**2.0 + ly**2.0)
    fft_filter = np.ones(ell.shape)
    if filter_type == 0:
        fft_filter[ell > lmin_lmax] = 0.0
    elif filter_type == 1:
        fft_filter[ell < lmin_lmax] = 0.0
    elif filter_type == 2:
        lmin, lmax = lmin_lmax
        fft_filter[ell < lmin] = 0.0
        fft_filter[ell > lmax] = 0.0
    return fft_filter


def wiener_filter(mapparams, cl_signal, cl_noise, el=None):
    """Compute a 2D Wiener filter from signal and noise power spectra.

    Parameters
    ----------
    mapparams : list
        [nx, ny, dx, dy] — see :func:`~foregrounds_diffusion.flatmaps.get_lxly`.
    cl_signal, cl_noise : array_like
        1D signal and noise power spectra.
    el : array_like, optional
        Multipoles.  Defaults to ``np.arange(len(cl_signal))``.

    Returns
    -------
    ndarray
        2D Wiener filter.
    """
    if el is None:
        el = np.arange(len(cl_signal))
    cl_signal2d = cl_to_cl2d(el, cl_signal, mapparams)
    cl_noise2d = cl_to_cl2d(el, cl_noise, mapparams)
    return cl_signal2d / (cl_signal2d + cl_noise2d)


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------


def split_data_to_tensors(data, train_size=0.7, val_size=0.15, test_size=0.15, seed=42):
    """Split a numpy array into train/val/test PyTorch tensors.

    Parameters
    ----------
    data : ndarray, shape (N, H, W, C)
        Input data in channels-last layout.
    train_size, val_size, test_size : float
        Fractional split sizes (must sum to 1).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_set, val_set, test_set : torch.Tensor
        Tensors in channels-first layout (N, C, H, W).
    """
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.")

    rng = np.random.default_rng(seed)
    indices = np.arange(data.shape[0])
    rng.shuffle(indices)

    train_end = int(train_size * len(indices))
    val_end = train_end + int(val_size * len(indices))

    train_set = torch.tensor(data[indices[:train_end]].transpose(0, 3, 1, 2), dtype=torch.float32)
    val_set = torch.tensor(
        data[indices[train_end:val_end]].transpose(0, 3, 1, 2), dtype=torch.float32
    )
    test_set = torch.tensor(data[indices[val_end:]].transpose(0, 3, 1, 2), dtype=torch.float32)

    return train_set, val_set, test_set


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------


def augment_images_unique(images):
    """Apply 8× augmentation: 4 rotations × horizontal flip.

    Parameters
    ----------
    images : torch.Tensor, shape (N, C, H, W)
        Training images in channels-first layout.

    Returns
    -------
    torch.Tensor, shape (8N, C, H, W)
        Augmented images (each original appears as 8 distinct variants).
    """
    augmented = []
    for img in images:
        for k in range(4):
            rotated = torch.rot90(img, k=k, dims=(1, 2))
            augmented.append(rotated)
            augmented.append(torch.flip(rotated, dims=[2]))
    return torch.stack(augmented)
