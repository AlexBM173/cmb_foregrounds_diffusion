import numpy as np
from scipy import optimize
import scipy.ndimage as ndimage


def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters."""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x, y: height * np.exp(
        -(((center_x - x) / width_x) ** 2 + ((center_y - y) / width_y) ** 2) / 2
    )


def moments(data):
    """Returns (height, x, y, width_x, width_y) from distribution moments."""
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
    """Returns (height, x, y, width_x, width_y) found by a least-squares fit."""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)
    p, _success = optimize.leastsq(errorfunction, params)
    return p


def fitting_func(
    p,
    p0,
    xgrid,
    ygrid,
    tmap,
    lbounds=None,
    ubounds=None,
    fixed=None,
    return_fit=0,
):
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

    def gaussian_local(p, xp, yp):
        if len(p) > 6:
            wx, wy = p[4], p[5]
            rota = np.radians(p[6])
            xp = np.radians(xp / 60.0) * np.cos(rota) - np.radians(yp / 60.0) * np.sin(
                rota
            )
            yp = np.radians(xp / 60.0) * np.sin(rota) + np.radians(yp / 60.0) * np.cos(
                rota
            )

            xp = np.degrees(xp) * 60.0
            yp = np.degrees(yp) * 60.0
        else:
            wx = wy = p[4]
        g = p[0] + p[1] * np.exp(-(((p[2] - xp) / wx) ** 2 + ((p[3] - yp) / wy) ** 2) / 2)
        return g

    if not return_fit:
        return np.ravel(gaussian_local(p, xgrid, ygrid) - tmap)
    return gaussian_local(p, xgrid, ygrid)


def get_mask_using_gaussian_fitting(
    nonpeak_mask,
    mul_width_by_factor=2,
    ini_height=0.0,
    ini_amp=1.0,
    ini_rot=0.0,
    ini_blob_size_in_pixels=10.0,
    use_elliptical_gaussian=False,
    perform_apod=True,
):
    # Fit Gaussian to each blob to get centroids and build a mask around them.
    ny, nx = nonpeak_mask.shape
    x = y = np.arange(nx)
    xgrid, ygrid = np.meshgrid(x, y)

    height = ini_height
    amp = ini_amp
    rot = ini_rot
    wx = wy = ini_blob_size_in_pixels

    non_zero_yinds, non_zero_xinds = np.where(nonpeak_mask == 1)
    howmany = len(non_zero_yinds)

    total_mask = np.zeros(nonpeak_mask.shape)
    for cntr, (x, y) in enumerate(zip(non_zero_xinds, non_zero_yinds)):
        if not (cntr < 10 or cntr > howmany - 10):
            continue

        x_cen_ini, y_cen_ini = x, y
        if use_elliptical_gaussian:
            p0 = [height, amp, x_cen_ini, y_cen_ini, wx, wy, rot]
        else:
            p0 = [height, amp, x_cen_ini, y_cen_ini, wx]

        p1, _success = optimize.leastsq(
            fitting_func, p0[:], args=(p0, xgrid, ygrid, nonpeak_mask)
        )
        _nonpeak_mask_fit = fitting_func(
            p1, p1, xgrid, ygrid, nonpeak_mask, return_fit=1
        )

        x_fit, y_fit, x_width = p1[2:]
        width_for_mask = x_width * mul_width_by_factor
        rad_grid = np.hypot(xgrid - x_fit, ygrid - y_fit)
        inds_to_mask = np.where(rad_grid <= width_for_mask)
        curr_mask = np.zeros(nonpeak_mask.shape)
        curr_mask[inds_to_mask] = 1.0
        total_mask = total_mask + curr_mask

    final_mask = np.ones(nonpeak_mask.shape)
    final_mask[total_mask != 0] = 0.0

    if perform_apod:
        pix = 1
        radius = (nx * pix) / 10.0
        npix_cos = int(radius / pix)
        ker = np.hanning(npix_cos)
        ker2d = np.asarray(np.sqrt(np.outer(ker, ker)))

        final_mask = ndimage.convolve(final_mask, ker2d)
        final_mask /= final_mask.max()
    return final_mask


def get_peak_masks(
    tmap,
    mask_threshold_sigma_units=10,
    mask_radius_pixel_units=0,
    perform_apod=1,
    mask_shape="circle",
    taper_radius_fac=2.0,
):
    """
    Get masking using sigma clipping.

    Returns
    -------
    peak_mask: array, shape is tmap.shape.
        Binary peak mask.
    mask: array shape is tmap.shape.
        Binary or apodised mask with some finite radius around the peaks.
    """
    assert mask_radius_pixel_units >= 0

    peak_mask = np.ones(tmap.shape)
    inds = np.where(
        abs(tmap) > abs(np.mean(tmap)) + mask_threshold_sigma_units * np.std(tmap)
    )
    peak_mask[inds] = 0.0

    if mask_radius_pixel_units > 0:
        x_grid, y_grid = np.indices(tmap.shape)
        assert mask_shape in ["circle", "square"]

        mask = np.ones(y_grid.shape)
        for (i, j) in zip(inds[0], inds[1]):
            y, x = x_grid[j, i], y_grid[j, i]
            if mask_shape == "circle":
                radius = np.sqrt(((x - x_grid) ** 2.0 + (y - y_grid) ** 2.0))
                inds_to_mask = np.where((radius <= mask_radius_pixel_units))
            else:
                inds_to_mask = np.where(
                    (abs(x - x_grid) < mask_radius_pixel_units)
                    & (abs(y - y_grid) < mask_radius_pixel_units)
                )

            mask[inds_to_mask[0], inds_to_mask[1]] = 0.0

        taper_radius = mask_radius_pixel_units * taper_radius_fac
        if perform_apod:
            ker = np.hanning(taper_radius)
            ker2d = np.asarray(np.sqrt(np.outer(ker, ker)))
            mask = ndimage.convolve(mask, ker2d)
            mask /= mask.max()
    else:
        mask = peak_mask

    return peak_mask, mask


def boundary_apod_mask(
    x_grid,
    y_grid,
    mask_radius,
    perform_apod=True,
    mask_shape="circle",
    taper_radius_fac=6.0,
):
    """Create a binary or apodised boundary mask."""
    assert mask_shape in ["circle", "square"]

    mask = np.ones(y_grid.shape)

    if mask_shape == "circle":
        radius = np.sqrt((x_grid**2.0 + y_grid**2.0))
        inds_to_mask = np.where((radius <= mask_radius))
    else:
        inds_to_mask = np.where(
            (abs(x_grid) < mask_radius) & (abs(y_grid) < mask_radius)
        )

    mask[inds_to_mask[0], inds_to_mask[1]] = 0.0

    taper_radius = mask_radius * taper_radius_fac
    if perform_apod:
        ker = np.hanning(taper_radius)
        ker2d = np.asarray(np.sqrt(np.outer(ker, ker)))
        mask = ndimage.convolve(mask, ker2d)
        mask /= mask.max()

    return mask
