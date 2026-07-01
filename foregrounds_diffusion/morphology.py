import numpy as np
from scipy.ndimage import sobel, binary_erosion


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
                               tensor_types=('W012',), centred=True, n_jobs=1):
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
    n_jobs : int
        Number of parallel workers (joblib).  1 = serial (default).
        −1 = use all cores.

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

    if n_jobs != 1:
        from joblib import Parallel, delayed
        chunks = np.array_split(np.arange(N), abs(n_jobs) if n_jobs != -1 else N)
        parts = Parallel(n_jobs=n_jobs)(
            delayed(compute_minkowski_tensors)(
                maps_nhw[idx], norm_fn, thresholds, tensor_types, centred, 1
            )
            for idx in chunks if len(idx) > 0
        )
        return {tt: {stat: np.concatenate([p[tt][stat] for p in parts], axis=0)
                     for stat in ('beta', 'theta')}
                for tt in tensor_types}

    results = {
        tt: {'beta': np.ones((N, T)), 'theta': np.zeros((N, T))}
        for tt in tensor_types
    }

    for i, m in enumerate(maps_nhw):
        m_norm = np.ascontiguousarray(norm_fn(m), dtype=np.float64)
        # Vectorise threshold binarisation: compute all T masks at once
        # to reduce Python-loop overhead and improve cache locality.
        all_binary = m_norm[np.newaxis] > thresholds[:, np.newaxis, np.newaxis]
        for t, binary in enumerate(all_binary):
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
