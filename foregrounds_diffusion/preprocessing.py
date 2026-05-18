import numpy as np
import torch
import healpy as hp
import astropy.units as u
from scipy import ndimage, optimize

from foregrounds_diffusion.flatmaps import get_lxly, cl_to_cl2d


# ---------------------------------------------------------------------------
# Normalisation utilities
# ---------------------------------------------------------------------------

def apply_maxmin_normalization(maps):
    """Min-max normalise an array to [0, 1].

    Parameters
    ----------
    maps : ndarray
        Input array of any shape.

    Returns
    -------
    ndarray
        Normalised array.
    """
    min_val = np.nanmin(maps)
    max_val = np.nanmax(maps)
    return (maps - min_val) / (max_val - min_val)


def apply_stdnorm(maps):
    """Standard-normalise an array channel-wise: (x − μ) / σ.

    Parameters
    ----------
    maps : ndarray, shape (..., C)
        Input array whose last axis is the channel dimension.

    Returns
    -------
    ndarray
        Normalised array with the same shape.
    """
    maps = maps.copy()
    for c in range(maps.shape[-1]):
        channel = maps[..., c]
        maps[..., c] = (channel - np.mean(channel)) / np.std(channel)
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
            dm_maps[:, :, :, i] = (
                (dm_maps[:, :, :, i] - dm_mean) * (tr_std / dm_std) + tr_mean
            )

    return np.transpose(dm_maps, (0, 3, 1, 2))


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
    moments_data = np.load(filename)[:max_lines]
    norms = [
        bandpass_centers,       # S2aa
        bandpass_centers,       # S2bb
        bandpass_centers,       # S2ab
        bandpass_centers,       # S3aaa
        bandpass_centers,       # S3bbb
        bandpass_centers,       # S3aab
        bandpass_centers,       # S3abb
        bandpass_centers ** 2,  # S4aaaa
        bandpass_centers ** 2,  # S4bbbb
        bandpass_centers ** 2,  # S4aaab
        bandpass_centers ** 2,  # S4aabb
        bandpass_centers ** 2,  # S4abbb
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
def get_patch_centers(gal_cut: u.deg, step_size: u.deg):
    """Compute patch centres on the sky, avoiding the Galactic plane.

    Parameters
    ----------
    gal_cut : `~astropy.units.Quantity`
        Half-width of the Galactic-plane exclusion zone in degrees.
    step_size : `~astropy.units.Quantity`
        Stepping distance in Galactic latitude in degrees.

    Returns
    -------
    list of tuple
        Each element is ``(lon, lat)`` as `~astropy.units.Quantity` in degrees.
    """
    gal_cut = gal_cut.to(u.deg)
    step_size = step_size.to(u.deg)
    southern = np.arange(-90, (-gal_cut - step_size).value, step_size.value) * u.deg
    northern = np.arange((gal_cut + step_size).value, 90, step_size.value) * u.deg
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

        self.xarr = np.linspace(-self.ang_x.to(u.rad).value / 2.,
                                 self.ang_x.to(u.rad).value / 2., xres)
        self.yarr = np.linspace(-self.ang_y.to(u.rad).value / 2.,
                                 self.ang_y.to(u.rad).value / 2., yres)

        xgrid, ygrid = np.meshgrid(self.xarr, self.yarr)
        xgrid = xgrid.ravel()[None, :]
        ygrid = ygrid.ravel()[None, :]
        zgrid = np.ones_like(ygrid)

        self.vecs = np.concatenate((xgrid, ygrid, zgrid)).T
        self.lons, self.lats = hp.vec2ang(self.vecs, lonlat=True)
        self.lats *= u.deg
        self.lons *= u.deg

    @u.quantity_input
    def rotate_to_pole_and_interpolate(self, lon: u.deg, lat: u.deg, ma):
        """Rotate the patch grid to *(lon, lat)* and sample the map.

        Parameters
        ----------
        lon, lat : `~astropy.units.Quantity`
            Sky position of the patch centre.
        ma : ndarray or list of ndarray
            HEALPix map(s) to sample.

        Returns
        -------
        ndarray, shape (xres, yres) or (xres, yres, nmaps)
            Interpolated flat-sky patch(es).
        """
        if hp.pixelfunc.maptype(ma) == 0:
            ma = [ma]
        rotator = hp.Rotator(rot=[lon.to(u.deg).value,
                                  lat.to(u.deg).value - 90.], deg=True)
        self.inv_lon_grid, self.inv_lat_grid = rotator.I(
            self.lons.to(u.deg).value,
            self.lats.to(u.deg).value,
            lonlat=True)
        m_rot = [hp.get_interp_val(each, self.inv_lon_grid,
                                   self.inv_lat_grid, lonlat=True)
                 for each in ma]

        if len(m_rot) > 1:
            m_rot[-2], m_rot[-1] = _spin2rot(
                m_rot[-2], m_rot[-1],
                rotator.angle_ref(self.inv_lon_grid, self.inv_lat_grid,
                                  lonlat=True))
            m_rot[-2], m_rot[-1] = _spin2rot(
                m_rot[-2], m_rot[-1], self.lons.to(u.rad).value)
        else:
            m_rot = m_rot[0]

        return np.moveaxis(
            np.array(m_rot).reshape(-1, self.xres, self.yres), 0, -1)


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
    ell = np.sqrt(lx ** 2. + ly ** 2.)
    fft_filter = np.ones(ell.shape)
    if filter_type == 0:
        fft_filter[ell > lmin_lmax] = 0.
    elif filter_type == 1:
        fft_filter[ell < lmin_lmax] = 0.
    elif filter_type == 2:
        lmin, lmax = lmin_lmax
        fft_filter[ell < lmin] = 0.
        fft_filter[ell > lmax] = 0.
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
# Masking utilities
# ---------------------------------------------------------------------------

def get_peak_masks(tmap, mask_threshold_sigma_units=10,
                   mask_radius_pixel_units=0, perform_apod=1,
                   mask_shape='circle', taper_radius_fac=2.):
    """Generate a sigma-clipping peak mask for a flat-sky map.

    Parameters
    ----------
    tmap : ndarray
        Input flat-sky map.
    mask_threshold_sigma_units : float
        Sigma threshold above which pixels are masked.
    mask_radius_pixel_units : float
        Radius (in pixels) of the mask hole punched around each peak.
        When 0, only the peak pixel itself is masked.
    perform_apod : bool
        Apodise the mask boundary.
    mask_shape : {'circle', 'square'}
        Shape of the mask hole.
    taper_radius_fac : float
        Apodisation taper radius as a multiple of *mask_radius_pixel_units*.

    Returns
    -------
    peak_mask, mask : ndarray
        Binary peak mask and (apodised) extended mask.
    """
    assert mask_radius_pixel_units >= 0
    assert mask_shape in ['circle', 'square']

    peak_mask = np.ones(tmap.shape)
    inds = np.where(abs(tmap) > abs(np.mean(tmap))
                    + mask_threshold_sigma_units * np.std(tmap))
    peak_mask[inds] = 0.

    if mask_radius_pixel_units > 0:
        x_grid, y_grid = np.indices(tmap.shape)
        mask = np.ones(y_grid.shape)
        for i, j in zip(inds[0], inds[1]):
            y, x = x_grid[j, i], y_grid[j, i]
            if mask_shape == 'circle':
                radius = np.sqrt((x - x_grid) ** 2. + (y - y_grid) ** 2.)
                inds_to_mask = np.where(radius <= mask_radius_pixel_units)
            else:
                inds_to_mask = np.where((abs(x - x_grid) < mask_radius_pixel_units)
                                        & (abs(y - y_grid) < mask_radius_pixel_units))
            mask[inds_to_mask[0], inds_to_mask[1]] = 0.

        taper_radius = mask_radius_pixel_units * taper_radius_fac
        if perform_apod:
            ker = np.hanning(taper_radius)
            ker2d = np.asarray(np.sqrt(np.outer(ker, ker)))
            mask = ndimage.convolve(mask, ker2d)
            mask /= mask.max()
    else:
        mask = peak_mask

    return peak_mask, mask


def boundary_apod_mask(x_grid, y_grid, mask_radius, perform_apod=True,
                       mask_shape='circle', taper_radius_fac=6.):
    """Create an apodised boundary mask on a 2D grid.

    Parameters
    ----------
    x_grid, y_grid : ndarray
        Coordinate grids (e.g. RA, Dec) of the map.
    mask_radius : float
        Mask radius in the same units as *x_grid* and *y_grid*.
    perform_apod : bool
        Apodise the mask boundary.
    mask_shape : {'circle', 'square'}
        Shape of the masked region.
    taper_radius_fac : float
        Apodisation taper radius as a multiple of *mask_radius*.

    Returns
    -------
    ndarray
        Binary or apodised mask with the same shape as *x_grid*.
    """
    assert mask_shape in ['circle', 'square']
    mask = np.ones(y_grid.shape)

    if mask_shape == 'circle':
        radius = np.sqrt(x_grid ** 2. + y_grid ** 2.)
        inds_to_mask = np.where(radius <= mask_radius)
    else:
        inds_to_mask = np.where((abs(x_grid) < mask_radius)
                                & (abs(y_grid) < mask_radius))

    mask[inds_to_mask[0], inds_to_mask[1]] = 0.

    if perform_apod:
        taper_radius = mask_radius * taper_radius_fac
        ker = np.hanning(taper_radius)
        ker2d = np.asarray(np.sqrt(np.outer(ker, ker)))
        mask = ndimage.convolve(mask, ker2d)
        mask /= mask.max()

    return mask


def get_mask_using_gaussian_fitting(nonpeak_mask, mul_width_by_factor=2,
                                    ini_height=0., ini_amp=1., ini_rot=0.,
                                    ini_blob_size_in_pixels=10.,
                                    use_elliptical_gaussian=False,
                                    perform_apod=True):
    """Fit Gaussians to blobs in a binary mask and create a smooth mask.

    Parameters
    ----------
    nonpeak_mask : ndarray
        Binary mask where 1 marks regions to be masked.
    mul_width_by_factor : float
        Multiply the fitted Gaussian width by this factor for the mask radius.
    ini_height, ini_amp, ini_rot : float
        Initial Gaussian parameters (baseline, amplitude, rotation).
    ini_blob_size_in_pixels : float
        Initial guess for the Gaussian width in pixels.
    use_elliptical_gaussian : bool
        Fit an elliptical (asymmetric) Gaussian if *True*.
    perform_apod : bool
        Apodise the final mask.

    Returns
    -------
    ndarray
        Final (possibly apodised) mask.
    """
    from foregrounds_diffusion.statistics import fitting_func

    ny, nx = nonpeak_mask.shape
    x = y = np.arange(nx)
    xgrid, ygrid = np.meshgrid(x, y)
    wx = wy = ini_blob_size_in_pixels

    non_zero_yinds, non_zero_xinds = np.where(nonpeak_mask == 1)
    howmany = len(non_zero_yinds)

    total_mask = np.zeros(nonpeak_mask.shape)
    for cntr, (bx, by) in enumerate(zip(non_zero_xinds, non_zero_yinds)):
        if not (cntr < 10 or cntr > howmany - 10):
            continue
        if use_elliptical_gaussian:
            p0 = [ini_height, ini_amp, bx, by, wx, wy, ini_rot]
        else:
            p0 = [ini_height, ini_amp, bx, by, wx]

        p1, _ = optimize.leastsq(fitting_func, p0[:],
                                  args=(p0, xgrid, ygrid, nonpeak_mask))
        x_fit, y_fit, x_width = p1[2], p1[3], p1[4]
        width_for_mask = x_width * mul_width_by_factor
        rad_grid = np.hypot(xgrid - x_fit, ygrid - y_fit)
        curr_mask = np.zeros(nonpeak_mask.shape)
        curr_mask[rad_grid <= width_for_mask] = 1.
        total_mask += curr_mask

    final_mask = np.ones(nonpeak_mask.shape)
    final_mask[total_mask != 0] = 0.

    if perform_apod:
        npix_cos = int((nx / 10.))
        ker = np.hanning(npix_cos)
        ker2d = np.asarray(np.sqrt(np.outer(ker, ker)))
        final_mask = ndimage.convolve(final_mask, ker2d)
        final_mask /= final_mask.max()

    return final_mask


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------

def split_data_to_tensors(data, train_size=0.7, val_size=0.15,
                          test_size=0.15, seed=42):
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

    train_set = torch.tensor(
        data[indices[:train_end]].transpose(0, 3, 1, 2), dtype=torch.float32)
    val_set = torch.tensor(
        data[indices[train_end:val_end]].transpose(0, 3, 1, 2), dtype=torch.float32)
    test_set = torch.tensor(
        data[indices[val_end:]].transpose(0, 3, 1, 2), dtype=torch.float32)

    return train_set, val_set, test_set