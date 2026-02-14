import numpy as np
import astropy.units as u
import healpy as hp
import os
import matplotlib.pyplot as plt
#from foregrounds_diffusion.flatmaps import FlatSkyMap, CIBMap, TSZMap, SimMap, DiffusionMap

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    plots_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    cib_map = hp.read_map(os.path.join(data_dir, 'agora_len_mag_cibmap_act_150ghz.fits'), memmap=True)

    hp.mollview(cib_map, title='CIB Map at 150 GHz', unit='K_CMB')
    plt.savefig(os.path.join(plots_dir, 'cib_map_150ghz.png'), dpi=300)

    tsz_map = hp.read_map(os.path.join(data_dir, 'agora_ltszNG_bahamas80_bnd_unb_1.0e+12_1.0e+18_lensed.fits'), memmap=True)

    hp.mollview(tsz_map, title='tSZ Map at 150 GHz', unit='K_CMB')
    plt.savefig(os.path.join(plots_dir, 'tsz_map_150ghz.png'), dpi=300)

    # Get pixel info:
    nside = hp.get_nside(cib_map)
    npix = hp.nside2npix(nside)
    pix_size = hp.nside2resol(nside, arcmin=True)
    print(f"NSIDE: {nside}, NPIX: {npix}, Pixel Size: {pix_size:.2f} arcmin")

    # Low pass filter both maps
    ell_cut = 7000
    ell_max = 13000
    cib_alms = hp.sphtfunc.map2alm(cib_map, lmax=ell_max)
    cib_alms_filtered = hp.sphtfunc.almxfl(cib_alms, np.where(np.arange(ell_max + 1) <= ell_cut, 1.0, 0.0))
    cib_map_filtered = hp.sphtfunc.alm2map(cib_alms_filtered, nside=nside)
    hp.fitsfunc.write_map(os.path.join(data_dir, 'cib_map_150ghz_filtered.fits'), cib_map_filtered, overwrite=True)
    tsz_alms = hp.sphtfunc.map2alm(tsz_map, lmax=ell_max)
    tsz_alms_filtered = hp.sphtfunc.almxfl(tsz_alms, np.where(np.arange(ell_max + 1) <= ell_cut, 1.0, 0.0))
    tsz_map_filtered = hp.sphtfunc.alm2map(tsz_alms_filtered, nside=nside)
    hp.fitsfunc.write_map(os.path.join(data_dir, 'tsz_map_150ghz_filtered.fits'), tsz_map_filtered, overwrite=True)

    # Save and visualise filtered maps
    hp.mollview(cib_map_filtered, title='Low-Pass Filtered CIB Map at 150 GHz', unit='K_CMB')
    plt.savefig(os.path.join(plots_dir, 'cib_map_150ghz_filtered.png'))
    hp.mollview(tsz_map_filtered, title='Low-Pass Filtered tSZ Map at 150 GHz', unit='K_CMB')
    plt.savefig(os.path.join(plots_dir, 'tsz_map_150ghz_filtered.png'))

    # Define a function to control masking the maps
    def get_peak_masks(tmap, mask_threshold_sigma_units = 10, mask_radius_pixel_units = 0, perform_apod = 1, mask_shape = 'circle', taper_radius_fac = 2. ):
    
    """
    Get masking using sigma clipping.

    Parameters
    ----------
    tmap: array
        flatsky map.
    mask_threshold_sigma_units: float
        Threshold for sigma clipping.
        mask = np.where( (abs(tmap) > np.mean(tmap) + sigma * np.std(tmap)) )
    mask_radius_pixel_units: float
        Punch bigger hole around each peak.
        Default is 0, in which case, only peak pixels satisfying the above condition are masked.
    perform_apod: boolean
        Apodise the binary mask.
        Default is True.
    mask_shape: str
        Circular or a square mask.
        Default is circle.
        Must be one of ['circle', 'square']
    taper_radius_fac: float
        Radius for apodisation.
        taper_radius = taper_radius_fac * mask_radius.
        Default is 2 (FIX ME: which works well but need to be explored further).

    Returns
    -------
    peak_mask: array, shape is tmap.shape.
        Binary peak mask.
    mask: array shape is tmap.shape.
        Binary or apodised mask with some finite radius around the peaks.
        Only computed when mask_radius_pixel_units>0.
        else, mask_radius_pixel_units == peak_mask
    """

    import scipy as sc
    import scipy.ndimage as ndimage

    assert mask_radius_pixel_units >=0

    peak_mask = np.ones( tmap.shape )
    inds = np.where( (abs(tmap) > abs(np.mean(tmap)) + mask_threshold_sigma_units * np.std(tmap)) )
    peak_mask[inds] = 0.


    if mask_radius_pixel_units > 0:
        x_grid, y_grid = np.indices( tmap.shape )

        assert mask_shape in ['circle', 'square']

        mask = np.ones( y_grid.shape )

        #imshow(mask); colorbar(); 
        #pick all the masked inds
        for (i,j) in zip(inds[0], inds[1]):

            y, x = x_grid[j, i], y_grid[j, i]
            
            #plot( y, x, 'ko', color = 'None', mec = 'black', ms = 10, mew = 1.)

            if mask_shape == 'circle':
                radius = np.sqrt( ((x-x_grid)**2. + (y-y_grid)**2.) )
                inds_to_mask = np.where((radius<=mask_radius_pixel_units))
            elif mask_shape == 'square':
                inds_to_mask = np.where( (abs(x-x_grid)<mask_radius_pixel_units) & (abs(y-y_grid)<mask_radius_pixel_units))

            mask[inds_to_mask[0], inds_to_mask[1]] = 0.
        #show(); sys.exit()

        taper_radius = mask_radius_pixel_units * taper_radius_fac #
        if perform_apod:
            ker=np.hanning(taper_radius)
            ker2d=np.asarray( np.sqrt(np.outer(ker,ker)) )
            mask=ndimage.convolve(mask, ker2d)
            mask/=mask.max()
    else:
        mask = peak_mask

    return peak_mask, mask

    def create_point_source_mask(map, masking_units="mJy", masking_threshold=2, map_frequency_ghz=150):
        # Input map is in muK_CMB, so we must convert the masking threshold
        nside = hp.pixelfunc.get_nside(map)
        mask = np.ones_like(map)
        if masking_units == "mJy":
            threshold_muK = masking_threshold * 1e-3 * hp.pixelfunc.nside2pixarea(nside, degrees=True) * hp.sphtfunc.thermodynamic_temperature_conversion(map_frequency_ghz * u.GHz, u.uK_CMB)
        elif masking_units == "muK_CMB":
            threshold_muK = masking_threshold
        else:
            raise ValueError("masking_units must be either 'mJy' or 'muK_CMB'")
        mask[map > threshold_muK] = 0
        return mask