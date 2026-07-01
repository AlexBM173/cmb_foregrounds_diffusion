"""2D Gaussian fitting and summary statistics for flat-sky maps.

This module provides two distinct capabilities:

2D Gaussian fitting
-------------------
Used to characterise the angular resolution of beam-convolved or filtered
maps by fitting a 2D Gaussian to the stacked profile of point sources or
the central peak of a two-point correlation function.

:func:`gaussian` — parameterised 2D Gaussian (height, centre, widths, angle).
:func:`moments` — estimate Gaussian parameters from image moments.
:func:`fitgaussian` — fit a 2D Gaussian to a 2D array via ``scipy.optimize``.
:func:`fitting_func` — convenience wrapper that fits and optionally returns the residual map; also enforces parameter bounds.

Summary statistics
------------------
:func:`stats` — compute mean, standard deviation, minimum, and maximum of a
    flat-sky map, optionally masking zero pixels.  Used for quick sanity
    checks on patch statistics before and after normalisation.
"""

import numpy as np
from scipy import optimize

# ---------------------------------------------------------------------------
# Gaussian fitting
# ---------------------------------------------------------------------------


def gaussian(height, center_x, center_y, width_x, width_y):
    """Return a 2D Gaussian function with the given parameters.

    Parameters
    ----------
    height : float
        Peak amplitude.
    center_x, center_y : float
        Centre coordinates.
    width_x, width_y : float
        Standard deviations along x and y.

    Returns
    -------
    callable
        Function ``f(x, y)`` evaluating the Gaussian at *(x, y)*.
    """
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x, y: (
        height * np.exp(-(((center_x - x) / width_x) ** 2 + ((center_y - y) / width_y) ** 2) / 2)
    )


def moments(data):
    """Estimate 2D Gaussian parameters from image moments.

    Parameters
    ----------
    data : ndarray, shape (ny, nx)
        2D image.

    Returns
    -------
    tuple
        ``(height, x, y, width_x, width_y)``
    """
    total = data.sum()
    xgrid, ygrid = np.indices(data.shape)
    x = (xgrid * data).sum() / total
    y = (ygrid * data).sum() / total
    col = data[:, int(y)]
    width_x = np.sqrt(abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum())
    height = data.max()
    return height, x, y, width_x, width_y


def fitgaussian(data):
    """Fit a 2D Gaussian to an image using least squares.

    Parameters
    ----------
    data : ndarray, shape (ny, nx)
        2D image.

    Returns
    -------
    tuple
        ``(height, x, y, width_x, width_y)`` — best-fit parameters.
    """
    params = moments(data)

    def errorfunction(p):
        """Residual vector for least-squares Gaussian fitting."""
        return np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)

    p, _ = optimize.leastsq(errorfunction, params)
    return p


def fitting_func(p, p0, xgrid, ygrid, tmap, lbounds=None, ubounds=None, fixed=None, return_fit=0):
    """Evaluate or fit a 2D Gaussian model on a pixel grid.

    Used internally by :func:`~foregrounds_diffusion.preprocessing.get_mask_using_gaussian_fitting`.

    Parameters
    ----------
    p : array_like
        Current parameter vector ``[baseline, amp, x_cen, y_cen, width, ...]``.
    p0 : array_like
        Reference parameter vector (used to restore fixed parameters).
    xgrid, ygrid : ndarray
        Pixel coordinate grids.
    tmap : ndarray
        Target map (used as fall-back return on bound violations).
    lbounds, ubounds : array_like, optional
        Lower and upper parameter bounds.
    fixed : array_like of int, optional
        Indices of parameters to hold fixed at *p0*.
    return_fit : int
        If 1, return the model image; if 0, return the residual vector.

    Returns
    -------
    ndarray
        Residual vector (when ``return_fit=0``) or model image (when ``return_fit=1``).
    """
    if hasattr(fixed, "__len__"):
        p[fixed] = p0[fixed]

    if hasattr(lbounds, "__len__"):
        linds = abs(p) < abs(lbounds)
        if len(linds) > 0:
            return tmap
        p[linds] = lbounds[linds]

    if hasattr(ubounds, "__len__"):
        uinds = abs(p) > abs(ubounds)
        if len(uinds) > 0:
            return tmap
        p[uinds] = ubounds[uinds]

    def _gaussian(p, xp, yp):
        if len(p) > 6:
            wx, wy = p[4], p[5]
            rota = np.radians(p[6])
            xp_rot = np.radians(xp / 60.0) * np.cos(rota) - np.radians(yp / 60.0) * np.sin(rota)
            yp_rot = np.radians(xp / 60.0) * np.sin(rota) + np.radians(yp / 60.0) * np.cos(rota)
            xp = np.degrees(xp_rot) * 60.0
            yp = np.degrees(yp_rot) * 60.0
        else:
            wx = wy = p[4]
        return p[0] + p[1] * np.exp(-(((p[2] - xp) / wx) ** 2 + ((p[3] - yp) / wy) ** 2) / 2.0)

    if not return_fit:
        return np.ravel(_gaussian(p, xgrid, ygrid) - tmap)
    else:
        return _gaussian(p, xgrid, ygrid)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def stats(maps):
    """Return (min, max, mean, std) of an array.

    Parameters
    ----------
    maps : ndarray
        Input array.

    Returns
    -------
    tuple of float
        ``(min, max, mean, std)``
    """
    return np.min(maps), np.max(maps), np.mean(maps), np.std(maps)
