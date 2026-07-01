import numpy as np

from foregrounds_diffusion.flatmaps import _build_ell_bin_cache, bandpass_filter, map2cl


# ---------------------------------------------------------------------------
# Power-spectrum summary statistics
# ---------------------------------------------------------------------------

def mean_cls(maps_nhw, mapparams, lmin, lmax, binsize):
    """Compute mean auto-power spectrum over a stack of maps.

    Parameters
    ----------
    maps_nhw : ndarray, shape (N, H, W)
        Stack of flat-sky maps.
    mapparams : list
        [nx, ny, dx, dy] — see :func:`~foregrounds_diffusion.flatmaps.get_lxly`.
    lmin, lmax : float
        Multipole range.
    binsize : float
        Bin width in ℓ.

    Returns
    -------
    el : ndarray
        Bin centres.
    mean_cl : ndarray
        Mean power spectrum across maps.
    std_cl : ndarray
        Standard deviation across maps.
    """
    cache = _build_ell_bin_cache(mapparams, binsize=binsize, minbin=lmin, maxbin=lmax)
    cls = []
    for m in maps_nhw:
        el, cl = map2cl(mapparams, m, binsize=binsize, minbin=lmin, maxbin=lmax,
                        _ell_bin_cache=cache)
        cls.append(cl)
    cls = np.array(cls)
    return el, cls.mean(axis=0), cls.std(axis=0)


def mean_cross_cls(maps1, maps2, mapparams, lmin, lmax, binsize):
    """Compute mean cross-power spectrum between two stacks of maps.

    Parameters
    ----------
    maps1, maps2 : ndarray, shape (N, H, W)
        Two stacks of flat-sky maps.
    mapparams : list
        [nx, ny, dx, dy].
    lmin, lmax : float
        Multipole range.
    binsize : float
        Bin width in ℓ.

    Returns
    -------
    el : ndarray
    mean_cl : ndarray
    std_cl : ndarray
    """
    cache = _build_ell_bin_cache(mapparams, binsize=binsize, minbin=lmin, maxbin=lmax)
    cls = []
    for m1, m2 in zip(maps1, maps2):
        el, cl = map2cl(mapparams, m1, m2, binsize=binsize, minbin=lmin, maxbin=lmax,
                        _ell_bin_cache=cache)
        cls.append(cl)
    cls = np.array(cls)
    return el, cls.mean(axis=0), cls.std(axis=0)


# ---------------------------------------------------------------------------
# Higher-order statistics (bispectrum / trispectrum proxies)
# ---------------------------------------------------------------------------

def compute_summed_moments(cib_arr, tsz_arr, bp_filters):
    """Compute S2, S3, S4 of the summed CIB+tSZ field per ℓ-band.

    Parameters
    ----------
    cib_arr : ndarray, shape (N, H, W)
    tsz_arr : ndarray, shape (N, H, W)
    bp_filters : list of ndarray
        2D bandpass filters from :func:`~foregrounds_diffusion.flatmaps.get_lpf_hpf`.

    Returns
    -------
    ndarray, shape (N, len(bp_filters), 3)
        Columns: variance (S2), skewness (S3), excess kurtosis (S4).
    """
    N = len(cib_arr)
    moments = np.zeros((N, len(bp_filters), 3))
    for b, bp in enumerate(bp_filters):
        for i in range(N):
            filtered = bandpass_filter(cib_arr[i] + tsz_arr[i], bp)
            var = np.var(filtered)
            s2 = var
            s3 = np.mean(filtered ** 3) / var ** 1.5 if var > 0 else 0.
            s4 = (np.mean(filtered ** 4) / var ** 2 - 3.) if var > 0 else 0.
            moments[i, b] = [s2, s3, s4]
    return moments


def _cross_moments_one_map(cib, tsz, bp_filters):
    """Compute 12 cross-moments for a single map pair across all ℓ-bands."""
    L = len(bp_filters)
    out = np.zeros((L, 12))
    for b, bp in enumerate(bp_filters):
        a      = bandpass_filter(cib, bp)
        bfield = bandpass_filter(tsz, bp)
        out[b, 0]  = np.mean(a ** 2)
        out[b, 1]  = np.mean(bfield ** 2)
        out[b, 2]  = np.mean(a * bfield)
        out[b, 3]  = np.mean(a ** 3)
        out[b, 4]  = np.mean(bfield ** 3)
        out[b, 5]  = np.mean(a ** 2 * bfield)
        out[b, 6]  = np.mean(a * bfield ** 2)
        out[b, 7]  = np.mean(a ** 4)
        out[b, 8]  = np.mean(bfield ** 4)
        out[b, 9]  = np.mean(a ** 3 * bfield)
        out[b, 10] = np.mean(a ** 2 * bfield ** 2)
        out[b, 11] = np.mean(a * bfield ** 3)
    return out


def compute_cross_moments(cib_arr, tsz_arr, bp_filters, n_jobs=1):
    """Compute all 12 cross-moments per ℓ-band (a=CIB, b=tSZ).

    Moments: S2^{aa}, S2^{bb}, S2^{ab},
             S3^{aaa}, S3^{bbb}, S3^{aab}, S3^{abb},
             S4^{aaaa}, S4^{bbbb}, S4^{aaab}, S4^{aabb}, S4^{abbb}.

    Parameters
    ----------
    cib_arr : ndarray, shape (N, H, W)
    tsz_arr : ndarray, shape (N, H, W)
    bp_filters : list of ndarray
    n_jobs : int
        Number of parallel workers (joblib).  1 = serial (default).
        −1 = use all cores.

    Returns
    -------
    moments : ndarray, shape (N, len(bp_filters), 12)
    labels : list of str
    """
    labels = ['S2aa', 'S2bb', 'S2ab',
              'S3aaa', 'S3bbb', 'S3aab', 'S3abb',
              'S4aaaa', 'S4bbbb', 'S4aaab', 'S4aabb', 'S4abbb']
    N = len(cib_arr)

    if n_jobs != 1:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=n_jobs)(
            delayed(_cross_moments_one_map)(cib_arr[i], tsz_arr[i], bp_filters)
            for i in range(N)
        )
        return np.stack(rows, axis=0), labels

    moments_out = np.zeros((N, len(bp_filters), 12))
    for i in range(N):
        moments_out[i] = _cross_moments_one_map(cib_arr[i], tsz_arr[i], bp_filters)
    return moments_out, labels
