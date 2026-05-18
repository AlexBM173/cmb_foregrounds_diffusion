import numpy as np


# ---------------------------------------------------------------------------
# Fourier-space grid helpers
# ---------------------------------------------------------------------------

def get_lxly(flatskymapparams):
    """Return 2D Fourier wavenumber grids lx and ly.

    Parameters
    ----------
    flatskymapparams : list
        [nx, ny, dx, dy] where ny, nx = flatskymap.shape and dx, dy are the
        pixel resolution in arcminutes.  Example: [100, 100, 0.5, 0.5] gives a
        50' x 50' map at 0.5' resolution.

    Returns
    -------
    lx, ly : ndarray
        2D arrays of Fourier wavenumbers.
    """
    nx, ny, dx, dy = flatskymapparams
    dx = np.radians(dx / 60.)
    dy = np.radians(dy / 60.)
    lx, ly = np.meshgrid(np.fft.fftfreq(nx, dx), np.fft.fftfreq(ny, dy))
    lx *= 2 * np.pi
    ly *= 2 * np.pi
    return lx, ly


def get_lxly_az_angle(lx, ly):
    """Return the azimuthal angle in Fourier space.

    Parameters
    ----------
    lx, ly : ndarray
        2D Fourier wavenumber arrays from :func:`get_lxly`.

    Returns
    -------
    ndarray
        Azimuthal angle array.
    """
    return 2 * np.arctan2(lx, -ly)


# ---------------------------------------------------------------------------
# Power-spectrum ↔ map conversion
# ---------------------------------------------------------------------------

def cl_to_cl2d(el, cl, flatskymapparams):
    """Interpolate a 1D power spectrum onto a 2D Fourier grid.

    Parameters
    ----------
    el : array_like
        Multipole values at which *cl* is defined.
    cl : array_like
        1D power spectrum C_ℓ.
    flatskymapparams : list
        [nx, ny, dx, dy] — see :func:`get_lxly`.

    Returns
    -------
    ndarray
        2D power spectrum on the Fourier grid.
    """
    lx, ly = get_lxly(flatskymapparams)
    ell = np.sqrt(lx ** 2. + ly ** 2.)
    cl2d = np.interp(ell.flatten(), el, cl).reshape(ell.shape)
    return cl2d


def map2cl(flatskymapparams, flatskymap1, flatskymap2=None,
           binsize=None, minbin=100, maxbin=10000):
    """Compute auto- or cross-power spectrum of flat-sky map(s).

    Parameters
    ----------
    flatskymapparams : list
        [nx, ny, dx, dy] — see :func:`get_lxly`.
    flatskymap1 : ndarray, shape (ny, nx)
        First map.
    flatskymap2 : ndarray, shape (ny, nx), optional
        Second map for cross-spectrum.  Auto-spectrum computed when *None*.
    binsize : float, optional
        Bin width in ℓ.  Computed automatically when *None*.
    minbin, maxbin : float
        Minimum and maximum ℓ bins.

    Returns
    -------
    el, cl : ndarray
        Binned multipoles and power spectrum.
    """
    nx, ny, dx, dy = flatskymapparams
    dx_rad = np.radians(dx / 60.)
    lx, ly = get_lxly(flatskymapparams)

    if binsize is None:
        binsize = lx.ravel()[1] - lx.ravel()[0]

    if flatskymap2 is None:
        flatskymap_psd = abs(np.fft.fft2(flatskymap1) * dx_rad) ** 2 / (nx * ny)
    else:
        assert flatskymap1.shape == flatskymap2.shape
        flatskymap_psd = (np.fft.fft2(flatskymap1) * dx_rad
                         * np.conj(np.fft.fft2(flatskymap2)) * dx_rad
                         / (nx * ny))

    rad_prf = radial_profile(flatskymap_psd, (lx, ly),
                             bin_size=binsize, minbin=minbin,
                             maxbin=maxbin, to_arcmins=0)
    el, cl = rad_prf[:, 0], rad_prf[:, 1]
    return el, cl


def cl2map(flatskymapparams, cl, el=None):
    """Generate a Gaussian realisation of a flat-sky map from a 1D C_ℓ.

    For correlated multi-field realisations see :func:`make_gaussian_realisation`.

    Parameters
    ----------
    flatskymapparams : list
        [nx, ny, dx, dy] — see :func:`get_lxly`.
    cl : array_like
        1D power spectrum.
    el : array_like, optional
        Multipoles.  Defaults to ``np.arange(len(cl))``.

    Returns
    -------
    ndarray
        Simulated flat-sky map.
    """
    if el is None:
        el = np.arange(len(cl))

    nx, ny, dx, dy = flatskymapparams
    cl2d = cl_to_cl2d(el, cl, flatskymapparams)

    dx_rad = np.radians(dx / 60.)
    pix_area_norm = np.sqrt(1. / dx_rad ** 2.)
    cl2d_sqrt_normed = np.sqrt(cl2d) * pix_area_norm

    gauss_reals = np.random.randn(nx, ny)
    flatskymap = np.fft.ifft2(np.fft.fft2(gauss_reals) * cl2d_sqrt_normed).real
    flatskymap -= np.mean(flatskymap)
    return flatskymap


def make_gaussian_realisation(mapparams, el, cl, cl2=None, cl12=None,
                              bl=None, qu_or_eb='qu'):
    """Generate a (possibly correlated two-field) Gaussian flat-sky realisation.

    Parameters
    ----------
    mapparams : list
        [nx, ny, dx, dy] — see :func:`get_lxly`.
    el : array_like
        Multipoles.
    cl : array_like
        Auto-spectrum of field 1 (or the only field when *cl2* is *None*).
    cl2 : array_like, optional
        Auto-spectrum of field 2.  Required together with *cl12*.
    cl12 : array_like, optional
        Cross-spectrum between field 1 and field 2.
    bl : array_like, optional
        Beam transfer function (1D or 2D).  Applied to the output if given.
    qu_or_eb : {'qu', 'eb'}
        Whether polarisation output should be in Q/U or E/B convention.

    Returns
    -------
    ndarray
        Simulated map (1D or 3-component array for polarisation).
    """
    nx, ny, dx, dy = mapparams
    dx = dx * np.radians(1 / 60.)
    dy = dy * np.radians(1 / 60.)
    norm = np.sqrt(1. / (dx * dy))

    cltwod = cl_to_cl2d(el, cl, mapparams)

    if cl2 is not None:
        assert cl12 is not None
        cltwod12 = cl_to_cl2d(el, cl12, mapparams)
        cltwod2 = cl_to_cl2d(el, cl2, mapparams)

    if cl2 is None:
        cltwod = cltwod ** 0.5 * norm
        cltwod[np.isnan(cltwod)] = 0.
        gauss_reals = np.random.standard_normal([nx, ny])
        SIM = np.fft.ifft2(np.copy(cltwod) * np.fft.fft2(gauss_reals)).real
    else:
        cltwod12[np.isnan(cltwod12)] = 0.
        cltwod2[np.isnan(cltwod2)] = 0.

        gauss_reals_1_fft = np.fft.fft2(np.random.standard_normal([nx, ny]))
        gauss_reals_2_fft = np.fft.fft2(np.random.standard_normal([nx, ny]))

        cltwod_tmp = np.copy(cltwod) ** 0.5 * norm
        SIM_FIELD_1 = np.fft.ifft2(cltwod_tmp * gauss_reals_1_fft).real

        t1 = np.copy(gauss_reals_1_fft) * cltwod12 / np.copy(cltwod) ** 0.5
        t2 = (np.copy(gauss_reals_2_fft)
              * (cltwod2 - cltwod12 ** 2. / np.copy(cltwod)) ** 0.5)
        SIM_FIELD_2_FFT = (t1 + t2) * norm
        SIM_FIELD_2_FFT[np.isnan(SIM_FIELD_2_FFT)] = 0.
        SIM_FIELD_2 = np.fft.ifft2(SIM_FIELD_2_FFT).real

        SIM_FIELD_3 = np.zeros(SIM_FIELD_2.shape)
        if qu_or_eb == 'qu':
            SIM_FIELD_2, SIM_FIELD_3 = convert_eb_qu(
                SIM_FIELD_2, SIM_FIELD_3, mapparams, eb_to_qu=1)

        SIM = np.asarray([SIM_FIELD_1, SIM_FIELD_2, SIM_FIELD_3])

    if bl is not None:
        if np.ndim(bl) != 2:
            bl = cl_to_cl2d(el, bl, mapparams)
        SIM = np.fft.ifft2(np.fft.fft2(SIM) * bl).real

    SIM -= np.mean(SIM)
    return SIM


# ---------------------------------------------------------------------------
# Polarisation rotation helper
# ---------------------------------------------------------------------------

def convert_eb_qu(map1, map2, flatskymapparams, eb_to_qu=1):
    """Convert between E/B and Q/U polarisation representations.

    Parameters
    ----------
    map1, map2 : ndarray
        Input polarisation maps.
    flatskymapparams : list
        [nx, ny, dx, dy] — see :func:`get_lxly`.
    eb_to_qu : int
        If 1 convert E/B → Q/U; if 0 convert Q/U → E/B.

    Returns
    -------
    map1_mod, map2_mod : ndarray
        Rotated polarisation maps.
    """
    lx, ly = get_lxly(flatskymapparams)
    angle = get_lxly_az_angle(lx, ly)
    map1_fft, map2_fft = np.fft.fft2(map1), np.fft.fft2(map2)
    if eb_to_qu:
        map1_mod = np.fft.ifft2(np.cos(angle) * map1_fft - np.sin(angle) * map2_fft).real
        map2_mod = np.fft.ifft2(np.sin(angle) * map1_fft + np.cos(angle) * map2_fft).real
    else:
        map1_mod = np.fft.ifft2(np.cos(angle) * map1_fft + np.sin(angle) * map2_fft).real
        map2_mod = np.fft.ifft2(-np.sin(angle) * map1_fft + np.cos(angle) * map2_fft).real
    return map1_mod, map2_mod


# ---------------------------------------------------------------------------
# Profile estimation
# ---------------------------------------------------------------------------

def radial_profile(z, xy=None, bin_size=1., minbin=0., maxbin=10.,
                   to_arcmins=1):
    """Compute the radial profile of a real- or Fourier-space image.

    Parameters
    ----------
    z : ndarray
        2D image.
    xy : tuple of ndarray, optional
        Pre-computed (x, y) coordinate arrays.  Computed from *z* when *None*.
    bin_size : float
        Radial bin width.
    minbin, maxbin : float
        Radial range.
    to_arcmins : int
        If 1, multiply radius by 60 (convert degrees to arcminutes).

    Returns
    -------
    ndarray, shape (nbins, 3)
        Columns: bin centre, mean value, error on mean.
    """
    z = np.asarray(z)
    if xy is None:
        x, y = np.indices(z.shape)
    else:
        x, y = xy

    radius = (x ** 2. + y ** 2.) ** 0.5
    if to_arcmins:
        radius *= 60.

    binarr = np.arange(minbin, maxbin, bin_size)
    radprf = np.zeros((len(binarr), 3))
    hit_count = []

    for b, bin_lo in enumerate(binarr):
        ind = np.where((radius >= bin_lo) & (radius < bin_lo + bin_size))
        radprf[b, 0] = bin_lo + bin_size / 2.
        hits = len(np.where(abs(z[ind]) > 0.)[0])
        if hits > 0:
            radprf[b, 1] = np.sum(z[ind]) / hits
            radprf[b, 2] = np.std(z[ind])
        hit_count.append(hits)

    hit_count = np.asarray(hit_count)
    std_mean = np.sum(radprf[:, 2] * hit_count) / np.sum(hit_count)
    errval = std_mean / hit_count ** 0.5
    radprf[:, 2] = errval
    return radprf