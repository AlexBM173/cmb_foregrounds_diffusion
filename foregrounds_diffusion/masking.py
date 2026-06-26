"""Masking utilities for flat-sky patches and full-sky HEALPix maps.

Flat-sky helpers (formerly in ``preprocessing``):
    get_peak_masks, inpaint_masked_regions, boundary_apod_mask,
    get_mask_using_gaussian_fitting

AGORA MDPL2 cluster/point-source masks (formerly ``get_cluster_source_mask_for_agora``):
    get_mdpl2_halo_cat, get_cluster_mask_radius,
    get_point_source_mask_in_healpix, get_mdpl2_conversion_factors_K_to_MjyperSr,
    apodize_binary_mask_prof, get_apodised_mdpl2_cluster_mask
"""

import numpy as np
import healpy as hp
from scipy import ndimage, optimize


# ---------------------------------------------------------------------------
# Flat-sky peak and boundary masking
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


def inpaint_masked_regions(hmap, mask, rng=None):
    """Replace masked pixels with Gaussian noise matching unmasked-region statistics.

    Parameters
    ----------
    hmap : ndarray, shape (npix,)
        HEALPix (or flat-sky) map.
    mask : ndarray, shape (npix,)
        Mask array (1 = keep, 0 = replace).
    rng : numpy.random.Generator, optional
        Random number generator.  A fresh ``default_rng()`` is used when *None*.

    Returns
    -------
    ndarray
        Copy of *hmap* with masked pixels replaced by Gaussian noise.
    """
    if rng is None:
        rng = np.random.default_rng()
    hmap = hmap.copy()
    unmasked = hmap[mask > 0.5]
    mean = unmasked.mean()
    std = unmasked.std()
    masked_idx = np.where(mask < 0.5)[0]
    hmap[masked_idx] = rng.normal(loc=mean, scale=std, size=len(masked_idx))
    return hmap


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
# AGORA MDPL2 halo catalogue
# ---------------------------------------------------------------------------

def get_mdpl2_halo_cat(halo_cat_fname, get_velocities=True):
    """Load the MDPL2 halo catalogue.

    Parameters
    ----------
    halo_cat_fname : str
        Path to the halo catalogue file (.npy or .npz).
    get_velocities : bool
        If *True*, also return the three velocity components.

    Returns
    -------
    tuple of ndarray
        ``(ra, dec, z, m200c, m500c)`` or
        ``(ra, dec, z, m200c, m500c, vlos, vtht, vphi)`` when
        ``get_velocities=True``.
    """
    loaded = np.load(halo_cat_fname, allow_pickle=True)

    if isinstance(loaded, np.lib.npyio.NpzFile):
        mdpl2_ra    = loaded['totra']
        mdpl2_dec   = loaded['totdec']
        mdpl2_z     = loaded['totz']
        mdpl2_m200c = loaded['totm200']
        mdpl2_m500c = loaded['totm500']
        mdpl2_vlos  = loaded['totvlos']
        mdpl2_vtht  = loaded['totvtht']
        mdpl2_vphi  = loaded['totvphi']
    else:
        # Legacy .npy format: columns packed as rows
        cols = loaded.T
        mdpl2_ra, mdpl2_dec, mdpl2_z, mdpl2_m200c, mdpl2_m500c, \
            mdpl2_vlos, mdpl2_vtht, mdpl2_vphi = cols

    if get_velocities:
        return mdpl2_ra, mdpl2_dec, mdpl2_z, mdpl2_m200c, mdpl2_m500c, \
               mdpl2_vlos, mdpl2_vtht, mdpl2_vphi
    return mdpl2_ra, mdpl2_dec, mdpl2_z, mdpl2_m200c, mdpl2_m500c


# ---------------------------------------------------------------------------
# Cluster mask radius
# ---------------------------------------------------------------------------

def get_cluster_mask_radius(m500c):
    """Return a mask radius in arcminutes based on cluster mass.

    Parameters
    ----------
    m500c : float
        Cluster mass M_500c in solar masses.

    Returns
    -------
    float
        Mask radius in arcminutes.
    """
    if m500c < 1e14:
        return 3.
    elif m500c < 3e14:
        return 5.
    elif m500c < 5e14:
        return 8.
    else:
        return 10.


# ---------------------------------------------------------------------------
# Point-source mask
# ---------------------------------------------------------------------------

def get_point_source_mask_in_healpix(freq, hmap_Mjy_per_sr,
                                     threshold_mjy_freq0,
                                     threshold2_mjy_freq0=None,
                                     freq0=150., spec_index=3.4,
                                     full_sky=True, ang_res_am=None,
                                     return_flux_map_in_mjy=False):
    """Identify pixels containing point sources above a flux threshold.

    Parameters
    ----------
    freq : float
        Observation frequency in GHz.
    hmap_Mjy_per_sr : ndarray
        HEALPix map in MJy/sr.
    threshold_mjy_freq0 : float
        Flux threshold in mJy at *freq0*.
    threshold2_mjy_freq0 : float, optional
        Upper flux threshold for band-pass masking.
    freq0 : float
        Reference frequency in GHz.
    spec_index : float
        Spectral index for frequency scaling.
    full_sky : bool
        If *True*, treat *hmap_Mjy_per_sr* as a full-sky HEALPix map and
        compute pixel area from NSIDE.  Otherwise supply *ang_res_am*.
    ang_res_am : float, optional
        Pixel angular resolution in arcminutes (required when ``full_sky=False``).
    return_flux_map_in_mjy : bool
        If *True*, also return the flux map in mJy.

    Returns
    -------
    mask_pixels : ndarray of int
        Pixel indices (or (row, col) index tuple for flat maps) to mask.
    hmap_mjy : ndarray
        Flux map in mJy (only returned when ``return_flux_map_in_mjy=True``).
    """
    if full_sky:
        nside = hp.get_nside(hmap_Mjy_per_sr)
        pix_area = hp.nside2resol(nside) ** 2.
    else:
        assert ang_res_am is not None
        pix_area = np.radians(ang_res_am / 60.) ** 2.

    hmap_mjy = np.copy(hmap_Mjy_per_sr) * pix_area * 1e9  # MJy/sr → mJy

    scaling = (freq / freq0) ** spec_index
    threshold_mjy = threshold_mjy_freq0 * scaling

    if threshold2_mjy_freq0 is None:
        mask_pixels = np.where(hmap_mjy >= threshold_mjy)
    else:
        threshold2 = threshold2_mjy_freq0 * scaling
        mask_pixels = np.where((hmap_mjy >= threshold_mjy)
                               & (hmap_mjy < threshold2))

    if full_sky:
        mask_pixels = mask_pixels[0]

    if return_flux_map_in_mjy:
        return mask_pixels, hmap_mjy
    return mask_pixels


# ---------------------------------------------------------------------------
# Unit-conversion factors
# ---------------------------------------------------------------------------

def get_mdpl2_conversion_factors_K_to_MjyperSr(expname, band):
    """Look up the K → MJy/sr conversion factor for a given experiment and band.

    Parameters
    ----------
    expname : str or None
        Experiment name: ``'planck'``, ``'spt3g'``/``'spt'``/``'spt4'``,
        ``'cmbs4'``/``'s4wide'``/``'s4deep'``, or *None* for a generic set.
    band : int
        Band centre frequency in GHz.

    Returns
    -------
    float
        Conversion factor.
    """
    planck = {100: 243.623, 143: 371.036, 217: 481.882,
              353: 287.281, 545: 57.6963, 857: 2.26476}
    spt = {95: 208.973, 150: 375.876, 220: 472.522,
           221: 473.332, 285: 414.977, 286: 414.977, 345: 310.827}
    s4 = {145: 379.391 * 0.976, 155: 403.379 * 0.975}
    generic = {95: 208.973, 150: 375.876, 220: 472.522,
               285: 414.977, 345: 310.827}

    if expname == 'planck':
        return planck[band]
    elif expname in ('spt3g', 'spt', 'spt4'):
        return spt[band]
    elif expname in ('cmbs4', 's4wide', 's4deep'):
        return s4[band]
    else:
        return generic[band]


# ---------------------------------------------------------------------------
# Apodisation
# ---------------------------------------------------------------------------

def apodize_binary_mask_prof(binary_mask, dist_smooth_angle,
                             apod_start_dist, apod_end_dist):
    """Apodise a binary HEALPix mask using a distance-based smooth profile.

    The apodisation profile is ``(x − sin x) / 2π`` on ``[0, 2π]``, the
    integral of a cosine-kernel cross-section.

    Parameters
    ----------
    binary_mask : ndarray
        Full-sky HEALPix binary mask (1 = unmasked, 0 = masked).
    dist_smooth_angle : float
        FWHM of the Gaussian kernel applied to the distance map, in radians.
    apod_start_dist : float
        Distance below which pixels are set to 0, in radians.
    apod_end_dist : float
        Distance above which the profile is not applied, in radians.

    Returns
    -------
    ndarray
        Apodised mask.
    """
    net_dist = apod_end_dist - apod_start_dist

    dist_map = hp.dist2holes(binary_mask)
    binary_mask[dist_map <= apod_start_dist] = 0.

    smooth_region = (dist_map > apod_start_dist) & (dist_map < apod_end_dist)
    nside = hp.get_nside(binary_mask)
    dist_map = hp.smoothing(dist_map, fwhm=dist_smooth_angle, lmax=nside)

    smooth_mask = np.array(binary_mask)
    x = (dist_map[smooth_region] - apod_start_dist) / net_dist * 2. * np.pi
    del dist_map
    smooth_mask[smooth_region] = (x - np.sin(x)) / (2. * np.pi)

    smooth_mask *= binary_mask
    smooth_mask = np.clip(smooth_mask, 0., 1.)
    del binary_mask
    return smooth_mask


# ---------------------------------------------------------------------------
# Cluster mask
# ---------------------------------------------------------------------------

def get_apodised_mdpl2_cluster_mask(nside, halo_cat_fname,
                                    m500c_threshold=5e13,
                                    cluster_lmz_dic=None,
                                    howmanythetaforclusters=-1,
                                    apodise=True,
                                    expname=None):
    """Build an apodised HEALPix cluster mask from the MDPL2 halo catalogue.

    Parameters
    ----------
    nside : int
        HEALPix NSIDE resolution.
    halo_cat_fname : str
        Path to the halo catalogue ``.npy`` file.
    m500c_threshold : float
        Minimum M_500c (in M_☉) for a cluster to be masked.
        Pass ``-1`` to use the redshift-dependent mass limit from
        *cluster_lmz_dic* instead.
    cluster_lmz_dic : dict, optional
        Redshift-dependent mass-limit dictionary (required when
        ``m500c_threshold=-1``).  Must contain keys ``'redshift'`` and
        ``'M500c'``.
    howmanythetaforclusters : float
        If ``> 0``, compute the mask radius as this multiple of θ_500c.
        Otherwise use :func:`get_cluster_mask_radius`.
    apodise : bool
        Apodise the binary mask.
    expname : str, optional
        Experiment name (used only when ``m500c_threshold=-1``).

    Returns
    -------
    ndarray
        Final apodised (or binary) full-sky cluster mask.
    """
    import gc
    from astropy.cosmology import FlatLambdaCDM

    (mdpl2_ra, mdpl2_dec, mdpl2_z,
     mdpl2_m200c, mdpl2_m500c,
     mdpl2_vlos, mdpl2_vtht, mdpl2_vphi) = get_mdpl2_halo_cat(halo_cat_fname)

    # --- select clusters to mask ---
    if m500c_threshold != -1:
        clus_inds = np.where(mdpl2_m500c >= m500c_threshold)[0]
    else:
        assert cluster_lmz_dic is not None and expname is not None
        redshifts = cluster_lmz_dic['redshift']
        dz = np.diff(redshifts)[0]
        lim_M500c = cluster_lmz_dic['M500c'] * 1e14
        clus_inds = []
        inds = np.where(mdpl2_z < redshifts[0])[0]
        clus_inds.extend(inds[mdpl2_m500c[inds] > lim_M500c[0]])
        for zcntr, zzz in enumerate(redshifts):
            inds = np.where((mdpl2_z >= zzz) & (mdpl2_z < zzz + dz))[0]
            passed = np.where(mdpl2_m500c[inds] > lim_M500c[zcntr])[0]
            clus_inds.extend(inds[passed])
        clus_inds = np.asarray(clus_inds)

    print(f'\tTotal clusters to mask: {len(clus_inds)}')

    # --- optionally compute θ_500c-based radii ---
    if howmanythetaforclusters > 0:
        try:
            from colossus.halo import concentration, mass_defs
            from colossus.cosmology import cosmology as colossus_cosmology
            colossus_cosmology.setCosmology('planck15')
        except ImportError:
            raise ImportError("colossus is required for θ_500c-based masking.")

        cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
        cluster_mask_radius_am_arr = []
        for cntr, iii in enumerate(clus_inds):
            if cntr % 1000 == 0:
                print(cntr)
            c500c = concentration.concentration(mdpl2_m500c[iii], '500c', mdpl2_z[iii])
            _, r500c, _ = mass_defs.changeMassDefinition(
                mdpl2_m500c[iii], c500c, mdpl2_z[iii], '500c', '500c', profile='nfw')
            r500c_mpc = r500c / 1e3
            ang_dia_dist = cosmo.comoving_distance(mdpl2_z[iii]) / (1. + mdpl2_z[iii])
            theta500c_am = np.degrees(r500c_mpc / ang_dia_dist.value) * 60.
            cluster_mask_radius_am_arr.append(
                int(theta500c_am * howmanythetaforclusters) + 1)

        arr = np.asarray(cluster_mask_radius_am_arr, dtype=float)
        arr_mod = np.zeros_like(arr)
        arr_mod[arr <= 5.] = 5.
        arr_mod[(arr > 5.) & (arr <= 10.)] = 8.
        arr_mod[(arr > 10.) & (arr <= 20.)] = 15.
        arr_mod[(arr > 20.) & (arr <= 50.)] = 35.
        arr_mod[(arr > 50.) & (arr <= 100.)] = 75.
        arr_mod[arr > 100.] = 100.
        cluster_mask_radius_am_arr = arr_mod

    # --- build combined binary mask ---
    npix = hp.nside2npix(nside)
    combined_mask = np.ones(npix, dtype=np.float32)

    for cntr, iii in enumerate(clus_inds):
        if cntr % 5000 == 0:
            print(cntr)
        ppp = hp.ang2pix(nside,
                         np.radians(90. - mdpl2_dec[iii]),
                         np.radians(mdpl2_ra[iii]))
        if howmanythetaforclusters > 0:
            r_am = cluster_mask_radius_am_arr[cntr]
        else:
            r_am = get_cluster_mask_radius(mdpl2_m500c[iii])

        ivec = hp.pix2vec(nside, ppp)
        disc = hp.query_disc(nside, ivec, np.deg2rad(r_am / 60.))
        combined_mask[disc] = 0.

    del mdpl2_ra, mdpl2_dec, mdpl2_z, mdpl2_m200c, mdpl2_m500c
    del mdpl2_vlos, mdpl2_vtht, mdpl2_vphi, clus_inds
    gc.collect()

    print("Starting apodisation")

    if apodise:
        apod_fwhm_rad = np.radians(15. / 60.)  # 15 arcmin apodisation
        final_hmask = hp.smoothing(combined_mask, fwhm=apod_fwhm_rad,
                                   lmax=2*nside).astype(np.float32)
        del combined_mask
        gc.collect()
        final_hmask = np.clip(final_hmask, 0., 1.)
        final_hmask /= final_hmask.max()
    else:
        final_hmask = combined_mask

    return final_hmask
