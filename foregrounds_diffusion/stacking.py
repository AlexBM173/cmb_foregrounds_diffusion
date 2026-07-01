"""tSZ cluster stacking utilities.

Stacking is a signal-recovery technique that averages cutouts centred on
known or candidate cluster positions, suppressing uncorrelated noise and
revealing the mean tSZ profile of the underlying population.

This module implements the stacking analysis from paper §4.2, which compares
the stacked tSZ cluster profiles in AGORA maps against DDPM samples to test
whether the model reproduces the spatial structure of galaxy clusters.

Workflow
--------
1. :func:`select_snr_pixels` identifies local SNR peaks in the tSZ maps
   within a chosen SNR bin (e.g. 3–5 or 5–∞), returning ``(patch_idx, row,
   col)`` coordinates.
2. :func:`extract_cutouts` extracts square thumbnails of a given size centred
   on each peak coordinate.
3. The returned cutout stack is averaged by the caller to produce a stacked
   profile image.

See tutorial ``docs/tutorials/09_tsz_stacking.ipynb`` for the full analysis,
including stacked-profile comparisons across SNR bins.
"""

import numpy as np

# ---------------------------------------------------------------------------
# tSZ cluster stacking utilities
# ---------------------------------------------------------------------------


def select_snr_pixels(tsz_maps_nhw, snr_min, snr_max, min_separation=5):
    """Find local SNR-peak coordinates within a given SNR bin.

    Parameters
    ----------
    tsz_maps_nhw : ndarray, shape (N, H, W)
    snr_min : float
        Lower SNR bound (inclusive).
    snr_max : float or None
        Upper SNR bound (exclusive).  *None* means no upper bound.
    min_separation : int
        Minimum pixel separation between selected peaks.

    Returns
    -------
    list of tuple
        Each element is ``(patch_idx, row, col)``.
    """
    from scipy.ndimage import maximum_filter

    coords = []
    for i, m in enumerate(tsz_maps_nhw):
        noise = m.std()
        if noise == 0:
            continue
        snr_map = m / noise
        local_max = maximum_filter(snr_map, size=min_separation) == snr_map
        in_bin = (snr_map >= snr_min) & local_max
        if snr_max is not None:
            in_bin &= snr_map < snr_max
        for ri, rj in np.argwhere(in_bin):
            coords.append((i, int(ri), int(rj)))
    print(f"SNR [{snr_min}, {snr_max}): {len(coords)} peaks found")
    return coords


def extract_cutouts(maps_nhw, coords, cutout_size, max_cutouts=500):
    """Extract square cutouts centred on ``(patch_idx, row, col)`` coordinates.

    Parameters
    ----------
    maps_nhw : ndarray, shape (N, H, W)
    coords : list of tuple
        Coordinates from :func:`select_snr_pixels`.
    cutout_size : int
        Side length of the square cutout in pixels.
    max_cutouts : int
        Maximum number of cutouts to extract (caps memory use).

    Returns
    -------
    ndarray, shape (M, cutout_size, cutout_size) or None
        Stacked cutouts, or *None* if no valid cutouts were found.
    """
    half = cutout_size // 2
    cutouts = []
    for patch_idx, ri, rj in coords[:max_cutouts]:
        m = maps_nhw[patch_idx]
        ri0, ri1 = ri - half, ri + half
        rj0, rj1 = rj - half, rj + half
        if ri0 < 0 or ri1 > m.shape[0] or rj0 < 0 or rj1 > m.shape[1]:
            continue
        cutouts.append(m[ri0:ri1, rj0:rj1])
    return np.array(cutouts, dtype=np.float32) if cutouts else None
