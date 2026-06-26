"""Peak and minima counts for flat-sky CMB foreground maps.

Implements the peak and minima counting statistics from
Sabyr, Hill & Haiman (2024), arXiv:2410.21247, adapted for
flat-sky patches rather than full-sky maps.

The pipeline:
1. Smooth each patch with a Gaussian kernel at one or more angular scales.
2. Identify local maxima (peaks) and local minima using scipy's
   ``maximum_filter`` / ``minimum_filter``.
3. Bin the peak/minima pixel values by their amplitude in units of the
   map standard deviation ν = T / σ.
4. Return counts as a function of ν for comparison across map ensembles.

No external packages beyond numpy and scipy are required — the LensTools
dependency used by Sabyr et al. is not needed for flat-sky patches.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def smooth_map(patch, fwhm_arcmin, pixel_res_arcmin=1.40625):
    """Smooth a 2D patch with a Gaussian kernel.

    Parameters
    ----------
    patch : ndarray, shape (H, W)
        Input flat-sky map in any units.
    fwhm_arcmin : float
        Full-width at half-maximum of the Gaussian smoothing kernel in arcmin.
    pixel_res_arcmin : float
        Pixel resolution in arcmin/pixel (default: 6°/256px ≈ 1.406 arcmin).

    Returns
    -------
    ndarray, shape (H, W)
        Smoothed map.
    """
    sigma_pixels = (fwhm_arcmin / pixel_res_arcmin) / (2 * np.sqrt(2 * np.log(2)))
    return gaussian_filter(patch.astype(np.float64), sigma=sigma_pixels)


# ---------------------------------------------------------------------------
# Peak and minima identification
# ---------------------------------------------------------------------------

def find_peaks(patch, filter_size=3):
    """Find local maxima (peaks) in a 2D map.

    A pixel is a peak if it equals the maximum of its neighbourhood defined
    by ``filter_size × filter_size``.

    Parameters
    ----------
    patch : ndarray, shape (H, W)
        Smoothed 2D map.
    filter_size : int
        Side length of the sliding window for local maximum detection.

    Returns
    -------
    ndarray of float
        Pixel values at all peak locations.
    """
    local_max = maximum_filter(patch, size=filter_size) == patch
    # Exclude boundary pixels to avoid edge artefacts
    border = filter_size // 2
    local_max[:border, :] = False
    local_max[-border:, :] = False
    local_max[:, :border] = False
    local_max[:, -border:] = False
    return patch[local_max]


def find_minima(patch, filter_size=3):
    """Find local minima in a 2D map.

    Parameters
    ----------
    patch : ndarray, shape (H, W)
        Smoothed 2D map.
    filter_size : int
        Side length of the sliding window for local minimum detection.

    Returns
    -------
    ndarray of float
        Pixel values at all minima locations.
    """
    local_min = minimum_filter(patch, size=filter_size) == patch
    border = filter_size // 2
    local_min[:border, :] = False
    local_min[-border:, :] = False
    local_min[:, :border] = False
    local_min[:, -border:] = False
    return patch[local_min]


# ---------------------------------------------------------------------------
# Binned counts
# ---------------------------------------------------------------------------

def count_peaks_binned(patches_nhw, thresholds, fwhm_arcmin,
                       pixel_res_arcmin=1.40625, filter_size=3):
    """Compute mean peak counts per map as a function of threshold ν.

    Following Sabyr et al. (2024), thresholds are defined in units of the
    per-map standard deviation: ν = T / σ.  Peaks with ν > threshold are
    counted, giving a cumulative count curve.

    Parameters
    ----------
    patches_nhw : ndarray, shape (N, H, W)
        Stack of flat-sky patches.
    thresholds : array_like
        Threshold values in units of σ (e.g. ``np.linspace(-4, 4, 40)``).
    fwhm_arcmin : float
        Gaussian smoothing scale in arcmin applied before peak finding.
    pixel_res_arcmin : float
        Pixel resolution in arcmin.
    filter_size : int
        Neighbourhood size for local maximum detection.

    Returns
    -------
    counts : ndarray, shape (N, len(thresholds))
        Peak counts per map per threshold bin.
    """
    N = len(patches_nhw)
    counts = np.zeros((N, len(thresholds)))

    for i, patch in enumerate(patches_nhw):
        smoothed = smooth_map(patch, fwhm_arcmin, pixel_res_arcmin)
        sigma = smoothed.std()
        if sigma == 0:
            continue
        nu_map = smoothed / sigma
        peak_vals = find_peaks(nu_map, filter_size=filter_size)
        for t, thresh in enumerate(thresholds):
            counts[i, t] = (peak_vals > thresh).sum()

    return counts


def count_minima_binned(patches_nhw, thresholds, fwhm_arcmin,
                        pixel_res_arcmin=1.40625, filter_size=3):
    """Compute mean minima counts per map as a function of threshold ν.

    Minima with ν < threshold are counted (threshold should be negative).

    Parameters
    ----------
    patches_nhw : ndarray, shape (N, H, W)
        Stack of flat-sky patches.
    thresholds : array_like
        Threshold values in units of σ (e.g. ``np.linspace(-4, 0, 20)``).
    fwhm_arcmin : float
        Gaussian smoothing scale in arcmin.
    pixel_res_arcmin : float
        Pixel resolution in arcmin.
    filter_size : int
        Neighbourhood size for local minimum detection.

    Returns
    -------
    counts : ndarray, shape (N, len(thresholds))
        Minima counts per map per threshold bin.
    """
    N = len(patches_nhw)
    counts = np.zeros((N, len(thresholds)))

    for i, patch in enumerate(patches_nhw):
        smoothed = smooth_map(patch, fwhm_arcmin, pixel_res_arcmin)
        sigma = smoothed.std()
        if sigma == 0:
            continue
        nu_map = smoothed / sigma
        minima_vals = find_minima(nu_map, filter_size=filter_size)
        for t, thresh in enumerate(thresholds):
            counts[i, t] = (minima_vals < thresh).sum()

    return counts


# ---------------------------------------------------------------------------
# Multi-scale convenience wrapper
# ---------------------------------------------------------------------------

def compute_peak_minima_counts(patches_nhw, thresholds_peaks, thresholds_minima,
                               smoothing_scales_arcmin=(1.0, 2.5, 5.0),
                               pixel_res_arcmin=1.40625, filter_size=3):
    """Compute peak and minima counts at multiple smoothing scales.

    Parameters
    ----------
    patches_nhw : ndarray, shape (N, H, W)
        Stack of flat-sky patches in physical units (e.g. µK).
    thresholds_peaks : array_like
        Threshold values ν for peak counting (e.g. ``np.linspace(-1, 5, 30)``).
    thresholds_minima : array_like
        Threshold values ν for minima counting (e.g. ``np.linspace(-5, 1, 30)``).
    smoothing_scales_arcmin : tuple of float
        Gaussian FWHM smoothing scales in arcmin.
    pixel_res_arcmin : float
        Pixel resolution in arcmin.
    filter_size : int
        Neighbourhood size for local extremum detection.

    Returns
    -------
    results : dict
        Nested dictionary keyed by smoothing scale (arcmin), then
        ``'peaks'`` and ``'minima'``, each containing an ndarray of
        shape (N, len(thresholds)).

    Examples
    --------
    >>> results = compute_peak_minima_counts(
    ...     tsz_patches, thresholds_peaks, thresholds_minima)
    >>> mean_peaks = results[1.0]['peaks'].mean(axis=0)
    """
    results = {}
    for fwhm in smoothing_scales_arcmin:
        print(f"Computing counts at FWHM = {fwhm} arcmin …")
        peaks = count_peaks_binned(
            patches_nhw, thresholds_peaks, fwhm,
            pixel_res_arcmin=pixel_res_arcmin, filter_size=filter_size)
        minima = count_minima_binned(
            patches_nhw, thresholds_minima, fwhm,
            pixel_res_arcmin=pixel_res_arcmin, filter_size=filter_size)
        results[fwhm] = {'peaks': peaks, 'minima': minima}
    return results
