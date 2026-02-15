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