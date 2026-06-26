import numpy as np
from scipy import optimize
from scipy.ndimage import sobel, binary_erosion

from foregrounds_diffusion.flatmaps import map2cl, bandpass_filter


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
    return lambda x, y: height * np.exp(
        -(((center_x - x) / width_x) ** 2
          + ((center_y - y) / width_y) ** 2) / 2)


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
    errorfunction = lambda p: np.ravel(
        gaussian(*p)(*np.indices(data.shape)) - data)
    p, _ = optimize.leastsq(errorfunction, params)
    return p


def fitting_func(p, p0, xgrid, ygrid, tmap,
                 lbounds=None, ubounds=None, fixed=None, return_fit=0):
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
    if hasattr(fixed, '__len__'):
        p[fixed] = p0[fixed]

    if hasattr(lbounds, '__len__'):
        linds = abs(p) < abs(lbounds)
        if len(linds) > 0:
            return tmap
        p[linds] = lbounds[linds]

    if hasattr(ubounds, '__len__'):
        uinds = abs(p) > abs(ubounds)
        if len(uinds) > 0:
            return tmap
        p[uinds] = ubounds[uinds]

    def _gaussian(p, xp, yp):
        if len(p) > 6:
            wx, wy = p[4], p[5]
            rota = np.radians(p[6])
            xp_rot = np.radians(xp / 60.) * np.cos(rota) - np.radians(yp / 60.) * np.sin(rota)
            yp_rot = np.radians(xp / 60.) * np.sin(rota) + np.radians(yp / 60.) * np.cos(rota)
            xp = np.degrees(xp_rot) * 60.
            yp = np.degrees(yp_rot) * 60.
        else:
            wx = wy = p[4]
        return p[0] + p[1] * np.exp(
            -(((p[2] - xp) / wx) ** 2 + ((p[3] - yp) / wy) ** 2) / 2.)

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
    cls = []
    for m in maps_nhw:
        el, cl = map2cl(mapparams, m, binsize=binsize, minbin=lmin, maxbin=lmax)
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
    cls = []
    for m1, m2 in zip(maps1, maps2):
        el, cl = map2cl(mapparams, m1, m2, binsize=binsize, minbin=lmin, maxbin=lmax)
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


def compute_cross_moments(cib_arr, tsz_arr, bp_filters):
    """Compute all 12 cross-moments per ℓ-band (a=CIB, b=tSZ).

    Moments: S2^{aa}, S2^{bb}, S2^{ab},
             S3^{aaa}, S3^{bbb}, S3^{aab}, S3^{abb},
             S4^{aaaa}, S4^{bbbb}, S4^{aaab}, S4^{aabb}, S4^{abbb}.

    Parameters
    ----------
    cib_arr : ndarray, shape (N, H, W)
    tsz_arr : ndarray, shape (N, H, W)
    bp_filters : list of ndarray

    Returns
    -------
    moments : ndarray, shape (N, len(bp_filters), 12)
    labels : list of str
    """
    N, L = len(cib_arr), len(bp_filters)
    moments_out = np.zeros((N, L, 12))
    labels = ['S2aa', 'S2bb', 'S2ab',
              'S3aaa', 'S3bbb', 'S3aab', 'S3abb',
              'S4aaaa', 'S4bbbb', 'S4aaab', 'S4aabb', 'S4abbb']
    for b, bp in enumerate(bp_filters):
        for i in range(N):
            a = bandpass_filter(cib_arr[i], bp)
            bfield = bandpass_filter(tsz_arr[i], bp)
            moments_out[i, b, 0]  = np.mean(a ** 2)
            moments_out[i, b, 1]  = np.mean(bfield ** 2)
            moments_out[i, b, 2]  = np.mean(a * bfield)
            moments_out[i, b, 3]  = np.mean(a ** 3)
            moments_out[i, b, 4]  = np.mean(bfield ** 3)
            moments_out[i, b, 5]  = np.mean(a ** 2 * bfield)
            moments_out[i, b, 6]  = np.mean(a * bfield ** 2)
            moments_out[i, b, 7]  = np.mean(a ** 4)
            moments_out[i, b, 8]  = np.mean(bfield ** 4)
            moments_out[i, b, 9]  = np.mean(a ** 3 * bfield)
            moments_out[i, b, 10] = np.mean(a ** 2 * bfield ** 2)
            moments_out[i, b, 11] = np.mean(a * bfield ** 3)
    return moments_out, labels


# ---------------------------------------------------------------------------
# Minkowski functionals
# ---------------------------------------------------------------------------

def compute_mfs(maps_nhw, norm_fn, thresholds):
    """Compute Minkowski functionals M0, M1, M2 across a stack of maps.

    Requires the ``quantimpy`` package (Boelens & Tchelepi 2021).

    Parameters
    ----------
    maps_nhw : ndarray, shape (N, H, W)
    norm_fn : callable
        Normalisation function mapping a 2D array to [0, 1]
        (e.g. :func:`~foregrounds_diffusion.preprocessing.apply_maxmin_normalization`).
    thresholds : array_like
        Intensity thresholds ν.

    Returns
    -------
    M0, M1, M2 : ndarray, each shape (N, len(thresholds))
        M0 = area fraction, M1 = perimeter, M2 = Euler characteristic.
    """
    from quantimpy import minkowski as mk
    M0 = np.zeros((len(maps_nhw), len(thresholds)))
    M1 = np.zeros_like(M0)
    M2 = np.zeros_like(M0)
    for i, m in enumerate(maps_nhw):
        m_norm = np.ascontiguousarray(norm_fn(m), dtype=np.float32)
        for t, thresh in enumerate(thresholds):
            binary = np.ascontiguousarray(m_norm > thresh, dtype=bool)
            area, length, euler = mk.functionals(binary)
            M0[i, t] = area
            M1[i, t] = length
            M2[i, t] = euler
    return M0, M1, M2


# ---------------------------------------------------------------------------
# Minkowski tensors
# ---------------------------------------------------------------------------

def _tensor_W012(binary_map):
    """W^{0,2}_1 — interface normal tensor.

    Sums the outer product n⊗n of the outward unit normal n over all
    boundary pixels of the excursion set.  Sensitive to the isotropy of
    cluster boundary shapes.
    """
    interior = binary_erosion(binary_map, border_value=0)
    boundary = binary_map & ~interior
    if not boundary.any():
        return np.eye(2) * 0.5

    gx = sobel(binary_map.astype(np.float64), axis=1)
    gy = sobel(binary_map.astype(np.float64), axis=0)
    bx, by = gx[boundary], gy[boundary]
    norm = np.sqrt(bx ** 2 + by ** 2)
    valid = norm > 0
    if not valid.any():
        return np.eye(2) * 0.5
    nx, ny = bx[valid] / norm[valid], by[valid] / norm[valid]
    return np.array([[np.sum(nx * nx), np.sum(nx * ny)],
                     [np.sum(nx * ny), np.sum(ny * ny)]])


def _tensor_W200(binary_map, centred=True):
    """W^{2,0}_0 — area inertia tensor.

    Sums the outer product r⊗r of pixel position vectors r over all
    interior pixels of the excursion set.  Measures elongation of the
    filled region.
    """
    coords = np.argwhere(binary_map).astype(np.float64)
    if len(coords) < 2:
        return np.eye(2) * 0.5
    if centred:
        coords -= coords.mean(axis=0)
    return coords.T @ coords / len(coords)


def _tensor_W201(binary_map, centred=True):
    """W^{2,0}_1 — boundary position tensor.

    Sums the outer product r⊗r of pixel position vectors r over boundary
    pixels only.  Hybrid between W012 and W200: position-weighted boundary
    measure.
    """
    interior = binary_erosion(binary_map, border_value=0)
    boundary = binary_map & ~interior
    coords = np.argwhere(boundary).astype(np.float64)
    if len(coords) < 2:
        return np.eye(2) * 0.5
    if centred:
        coords -= coords.mean(axis=0)
    return coords.T @ coords / len(coords)


_TENSOR_FUNCTIONS = {
    'W012': _tensor_W012,
    'W200': _tensor_W200,
    'W201': _tensor_W201,
}

MINKOWSKI_TENSOR_DESCRIPTIONS = {
    'W012': (
        "W^{0,2}_1 — interface normal tensor: sums n⊗n over boundary pixels "
        "(Sobel-estimated outward normals). Probes isotropy of cluster boundary "
        "shapes; recommended default."
    ),
    'W200': (
        "W^{2,0}_0 — area inertia tensor: sums r⊗r over interior pixels. "
        "Measures elongation of the filled excursion region."
    ),
    'W201': (
        "W^{2,0}_1 — boundary position tensor: sums r⊗r over boundary pixels. "
        "Hybrid: position-weighted boundary measure, intermediate sensitivity "
        "between W012 and W200."
    ),
}


def _eigendecompose_2x2(W):
    """Return anisotropy β and major-axis orientation θ for a 2×2 tensor.

    Parameters
    ----------
    W : ndarray, shape (2, 2)
        Symmetric positive-semidefinite tensor.

    Returns
    -------
    beta : float
        λ_min / λ_max ∈ [0, 1].  1 = perfectly isotropic.
    theta : float
        Angle of the major eigenvector in radians, wrapped to (-π/2, π/2].
    """
    eigenvalues, eigenvectors = np.linalg.eigh(W)   # ascending order
    lam_min, lam_max = eigenvalues[0], eigenvalues[1]
    if lam_max <= 0:
        return 1.0, 0.0
    beta = float(lam_min / lam_max)
    vx, vy = eigenvectors[:, 1]                     # major eigenvector
    theta = float(np.arctan2(vy, vx))
    if theta > np.pi / 2:
        theta -= np.pi
    elif theta <= -np.pi / 2:
        theta += np.pi
    return beta, theta


def compute_minkowski_tensors(maps_nhw, norm_fn, thresholds,
                               tensor_types=('W012',), centred=True):
    """Compute Minkowski tensor anisotropy indices across intensity thresholds.

    For each map and threshold ν, the map is binarised to the excursion set
    K = {x : T(x) > ν}.  The requested rank-2 Minkowski tensor(s) are
    computed from K, then eigendecomposed to give the anisotropy index
    β = λ_min/λ_max and the major-axis orientation θ.

    Parameters
    ----------
    maps_nhw : ndarray, shape (N, H, W)
        Stack of flat-sky patches.
    norm_fn : callable
        Normalisation function applied to each 2D patch before thresholding
        (e.g. ``apply_maxmin_normalization``).  Pass ``lambda x: x`` to use
        raw pixel values.
    thresholds : array_like
        Intensity thresholds ν in the normalised range.
    tensor_types : tuple of str
        Which tensor(s) to compute.  Any subset of:

        ``'W012'``
            W^{0,2}_1 — interface normal tensor (default).  Boundary normals
            n⊗n via Sobel gradients.  Most sensitive to boundary shape.
        ``'W200'``
            W^{2,0}_0 — area inertia tensor.  Interior pixel positions r⊗r.
            Measures elongation of filled regions.
        ``'W201'``
            W^{2,0}_1 — boundary position tensor.  Boundary pixel positions
            r⊗r.  Hybrid between W012 and W200.
    centred : bool
        If True (default), subtract the excursion-set centroid from position
        vectors before forming the tensor.  Affects W200 and W201 only.

    Returns
    -------
    results : dict
        Keyed by tensor type string.  Each value is a dict:

        ``'beta'`` : ndarray, shape (N, len(thresholds))
            Anisotropy index β = λ_min/λ_max ∈ [0, 1].  1 = isotropic.
        ``'theta'`` : ndarray, shape (N, len(thresholds))
            Major-axis orientation in radians ∈ (-π/2, π/2].

    Notes
    -----
    β = 1 and θ = 0 are returned for empty or full excursion sets (fewer
    than 2 interior pixels or fewer than 2 exterior pixels).  As a sanity
    check, a Gaussian random field should give β ≈ 1 at all thresholds.

    Examples
    --------
    >>> thresholds = np.linspace(0.05, 0.95, 50)
    >>> results = compute_minkowski_tensors(
    ...     maps, apply_maxmin_normalization, thresholds,
    ...     tensor_types=('W012', 'W200'))
    >>> beta_W012 = results['W012']['beta']   # shape (N, 50)
    >>> theta_W012 = results['W012']['theta'] # shape (N, 50)
    """
    thresholds = np.asarray(thresholds)
    N, T = len(maps_nhw), len(thresholds)

    results = {
        tt: {'beta': np.ones((N, T)), 'theta': np.zeros((N, T))}
        for tt in tensor_types
    }

    for i, m in enumerate(maps_nhw):
        m_norm = np.ascontiguousarray(norm_fn(m), dtype=np.float64)
        for t, nu in enumerate(thresholds):
            binary = m_norm > nu
            n_interior = int(binary.sum())
            if n_interior < 2 or n_interior > binary.size - 2:
                continue                              # trivially isotropic
            for tt in tensor_types:
                fn = _TENSOR_FUNCTIONS[tt]
                kwargs = {'centred': centred} if tt in ('W200', 'W201') else {}
                W = fn(binary, **kwargs)
                beta, theta = _eigendecompose_2x2(W)
                results[tt]['beta'][i, t] = beta
                results[tt]['theta'][i, t] = theta

    return results


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
            in_bin &= (snr_map < snr_max)
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