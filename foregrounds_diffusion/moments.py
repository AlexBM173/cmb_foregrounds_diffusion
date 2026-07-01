"""Power-spectrum and higher-order moment statistics.

This module computes the summary statistics used to validate DDPM samples
against the AGORA training maps (paper §4.1 and §4.2).  All functions
accept stacks of flat-sky maps and return statistics averaged over the
ensemble, making them suitable for comparing large sets of DDPM samples.

Power spectra
-------------
:func:`mean_cls` — mean auto-power spectrum C_ℓ over N maps.
:func:`mean_cross_cls` — mean cross-power spectrum between two map stacks.

Both functions accept ``n_jobs=-1`` to parallelise over maps using joblib,
and share a pre-computed ℓ-bin cache to amortise the FFT grid construction.

Higher-order moments
--------------------
:func:`compute_summed_moments` — variance (S2), skewness (S3), and excess
kurtosis (S4) of the bandpass-filtered sum field CIB + tSZ.  Equivalent to
the collapsed bispectrum and trispectrum proxies used in the paper.

:func:`compute_cross_moments` — the full set of 12 cross-moments between
bandpass-filtered CIB (a) and tSZ (b) fields::

    S2^{aa}, S2^{bb}, S2^{ab},
    S3^{aaa}, S3^{bbb}, S3^{aab}, S3^{abb},
    S4^{aaaa}, S4^{bbbb}, S4^{aaab}, S4^{aabb}, S4^{abbb}

These are computed per ℓ-band by applying 2D bandpass filters from
:func:`~foregrounds_diffusion.flatmaps.get_lpf_hpf` before taking moments.
The output shape is ``(N, n_bands, 12)`` where N is the number of maps.

See tutorial ``docs/tutorials/07_higher_order_stats.ipynb`` for the full
comparison between AGORA, DDPM samples, and a Gaussian baseline.
"""

import numpy as np

from foregrounds_diffusion.flatmaps import _build_ell_bin_cache, bandpass_filter, map2cl

# ---------------------------------------------------------------------------
# Power-spectrum summary statistics
# ---------------------------------------------------------------------------


def _mean_cls_one(m, mapparams, lmin, lmax, binsize, cache):
    return map2cl(mapparams, m, binsize=binsize, minbin=lmin, maxbin=lmax, _ell_bin_cache=cache)


def _mean_cross_cls_one(m1, m2, mapparams, lmin, lmax, binsize, cache):
    return map2cl(
        mapparams, m1, m2, binsize=binsize, minbin=lmin, maxbin=lmax, _ell_bin_cache=cache
    )


def mean_cls(maps_nhw, mapparams, lmin, lmax, binsize, n_jobs=1):
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
    n_jobs : int
        Number of parallel workers (joblib).  1 = serial (default).  -1 = all cores.

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

    if n_jobs != 1:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs)(
            delayed(_mean_cls_one)(m, mapparams, lmin, lmax, binsize, cache) for m in maps_nhw
        )
        el = results[0][0]
        cls = np.array([r[1] for r in results])
        return el, cls.mean(axis=0), cls.std(axis=0)

    cls = []
    for m in maps_nhw:
        el, cl = map2cl(
            mapparams, m, binsize=binsize, minbin=lmin, maxbin=lmax, _ell_bin_cache=cache
        )
        cls.append(cl)
    cls = np.array(cls)
    return el, cls.mean(axis=0), cls.std(axis=0)


def mean_cross_cls(maps1, maps2, mapparams, lmin, lmax, binsize, n_jobs=1):
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
    n_jobs : int
        Number of parallel workers (joblib).  1 = serial (default).  -1 = all cores.

    Returns
    -------
    el : ndarray
    mean_cl : ndarray
    std_cl : ndarray
    """
    cache = _build_ell_bin_cache(mapparams, binsize=binsize, minbin=lmin, maxbin=lmax)

    if n_jobs != 1:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs)(
            delayed(_mean_cross_cls_one)(m1, m2, mapparams, lmin, lmax, binsize, cache)
            for m1, m2 in zip(maps1, maps2)
        )
        el = results[0][0]
        cls = np.array([r[1] for r in results])
        return el, cls.mean(axis=0), cls.std(axis=0)

    cls = []
    for m1, m2 in zip(maps1, maps2):
        el, cl = map2cl(
            mapparams, m1, m2, binsize=binsize, minbin=lmin, maxbin=lmax, _ell_bin_cache=cache
        )
        cls.append(cl)
    cls = np.array(cls)
    return el, cls.mean(axis=0), cls.std(axis=0)


# ---------------------------------------------------------------------------
# Higher-order statistics (bispectrum / trispectrum proxies)
# ---------------------------------------------------------------------------


def _summed_moments_one_map(cib, tsz, bp_filters):
    """Compute S2, S3, S4 for a single CIB+tSZ map pair across all ℓ-bands."""
    B = len(bp_filters)
    out = np.zeros((B, 3))
    for b, bp in enumerate(bp_filters):
        filtered = bandpass_filter(cib + tsz, bp)
        var = np.var(filtered)
        out[b, 0] = var
        out[b, 1] = np.mean(filtered**3) / var**1.5 if var > 0 else 0.0
        out[b, 2] = (np.mean(filtered**4) / var**2 - 3.0) if var > 0 else 0.0
    return out


def compute_summed_moments(cib_arr, tsz_arr, bp_filters, n_jobs=1):
    """Compute S2, S3, S4 of the summed CIB+tSZ field per ℓ-band.

    Parameters
    ----------
    cib_arr : ndarray, shape (N, H, W)
    tsz_arr : ndarray, shape (N, H, W)
    bp_filters : list of ndarray
        2D bandpass filters from :func:`~foregrounds_diffusion.flatmaps.get_lpf_hpf`.
    n_jobs : int
        Number of parallel workers (joblib).  1 = serial (default).  -1 = all cores.

    Returns
    -------
    ndarray, shape (N, len(bp_filters), 3)
        Columns: variance (S2), skewness (S3), excess kurtosis (S4).
    """
    if n_jobs != 1:
        from joblib import Parallel, delayed

        rows = Parallel(n_jobs=n_jobs)(
            delayed(_summed_moments_one_map)(cib_arr[i], tsz_arr[i], bp_filters)
            for i in range(len(cib_arr))
        )
        return np.stack(rows, axis=0)

    N = len(cib_arr)
    moments = np.zeros((N, len(bp_filters), 3))
    for b, bp in enumerate(bp_filters):
        for i in range(N):
            filtered = bandpass_filter(cib_arr[i] + tsz_arr[i], bp)
            var = np.var(filtered)
            moments[i, b, 0] = var
            moments[i, b, 1] = np.mean(filtered**3) / var**1.5 if var > 0 else 0.0
            moments[i, b, 2] = (np.mean(filtered**4) / var**2 - 3.0) if var > 0 else 0.0
    return moments


def _cross_moments_one_map(cib, tsz, bp_filters):
    """Compute 12 cross-moments for a single map pair across all ℓ-bands."""
    L = len(bp_filters)
    out = np.zeros((L, 12))
    for b, bp in enumerate(bp_filters):
        a = bandpass_filter(cib, bp)
        bfield = bandpass_filter(tsz, bp)
        out[b, 0] = np.mean(a**2)
        out[b, 1] = np.mean(bfield**2)
        out[b, 2] = np.mean(a * bfield)
        out[b, 3] = np.mean(a**3)
        out[b, 4] = np.mean(bfield**3)
        out[b, 5] = np.mean(a**2 * bfield)
        out[b, 6] = np.mean(a * bfield**2)
        out[b, 7] = np.mean(a**4)
        out[b, 8] = np.mean(bfield**4)
        out[b, 9] = np.mean(a**3 * bfield)
        out[b, 10] = np.mean(a**2 * bfield**2)
        out[b, 11] = np.mean(a * bfield**3)
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
    labels = [
        "S2aa",
        "S2bb",
        "S2ab",
        "S3aaa",
        "S3bbb",
        "S3aab",
        "S3abb",
        "S4aaaa",
        "S4bbbb",
        "S4aaab",
        "S4aabb",
        "S4abbb",
    ]
    N = len(cib_arr)

    if n_jobs != 1:
        from joblib import Parallel, delayed

        rows = Parallel(n_jobs=n_jobs)(
            delayed(_cross_moments_one_map)(cib_arr[i], tsz_arr[i], bp_filters) for i in range(N)
        )
        return np.stack(rows, axis=0), labels

    moments_out = np.zeros((N, len(bp_filters), 12))
    for i in range(N):
        moments_out[i] = _cross_moments_one_map(cib_arr[i], tsz_arr[i], bp_filters)
    return moments_out, labels
