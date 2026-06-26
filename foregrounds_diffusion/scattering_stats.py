"""Scattering transform statistics for flat-sky CMB foreground patches.

Wraps the ``scattering`` package from Cheng et al.
(https://github.com/SihaoCheng/scattering_transform) to compute
S1 (first-order) and S2 (second-order) scattering coefficients,
and the scattering covariance, for ensembles of flat-sky patches.

These statistics capture non-Gaussian structure at multiple angular
scales and orientations, providing a richer description of the CIB
and tSZ fields than the power spectrum alone.

Installation
------------
The ``scattering`` package is not on PyPI. Clone the repository and
add it to your Python path:

    git clone https://github.com/SihaoCheng/scattering_transform.git
    # Then either:
    #   (a) copy the `scattering/` folder into your project, or
    #   (b) add to sys.path in your notebook:
    #       import sys; sys.path.insert(0, '/path/to/scattering_transform')

Or install ``kymatio`` as a pip-installable alternative (see notes below):

    pip install kymatio

Notes
-----
This module tries to import ``scattering`` (Cheng et al.) first.
If not found it falls back to ``kymatio`` with a warning.
The Cheng et al. implementation is preferred because it is faster
for large batches and exposes the scattering covariance C11 directly.
"""

import numpy as np
import warnings

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _get_backend():
    """Return ('cheng', module) or ('kymatio', module) or raise."""
    try:
        import scattering
        return 'cheng', scattering
    except ImportError:
        pass
    try:
        import kymatio
        return 'kymatio', kymatio
    except ImportError:
        raise ImportError(
            "No scattering transform backend found.\n"
            "Install one of:\n"
            "  (a) git clone https://github.com/SihaoCheng/scattering_transform.git\n"
            "      and add the repo root to sys.path, or\n"
            "  (b) pip install kymatio"
        )


# ---------------------------------------------------------------------------
# Coefficient computation
# ---------------------------------------------------------------------------

def compute_scattering_coefficients(patches_nhw, J=5, L=4, device=None):
    """Compute first- and second-order scattering coefficients.

    Uses the Cheng et al. ``scattering`` package if available,
    otherwise falls back to ``kymatio``.

    Parameters
    ----------
    patches_nhw : ndarray, shape (N, H, W)
        Stack of flat-sky patches.  All patches must have the same
        spatial dimensions H × W = 256 × 256.
    J : int
        Number of dyadic scales (default 5, giving scales 2¹…2⁵ pixels).
    L : int
        Number of orientations per scale (default 4).
    device : str or None
        ``'cuda'``, ``'cpu'``, or ``None`` (auto-detect).

    Returns
    -------
    dict with keys:
        ``'S0'`` : ndarray, shape (N, 1)
            Zeroth-order coefficient (mean pixel value).
        ``'S1'`` : ndarray, shape (N, J)
            First-order scattering coefficients, orientation-averaged
            (``S1_iso`` in Cheng et al. notation).
        ``'S2'`` : ndarray, shape (N, J, J, L)
            Second-order scattering coefficients — cross-scale coupling
            at pairs (j1, j2) as a function of orientation difference l
            (``S2_iso`` in Cheng et al. notation).
        ``'S1_mean'`` : ndarray, shape (J,)
            Mean S1 across all N patches.
        ``'S2_mean'`` : ndarray, shape (J, J, L)
            Mean S2 across all N patches.
        ``'J'``, ``'L'`` : int
            Parameters used.
    """
    backend, mod = _get_backend()
    N, H, W = patches_nhw.shape
    patches_f32 = patches_nhw.astype(np.float32)

    if backend == 'cheng':
        import torch
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Cheng et al. Scattering2d uses 'gpu'/'cpu', not 'cuda'
        cheng_device = 'gpu' if device == 'cuda' else 'cpu'

        st_calc = mod.Scattering2d(M=H, N=W, J=J, L=L, device=cheng_device)
        # Pass tensor on CPU — the class handles GPU transfer internally
        s_mean = st_calc.scattering_coef_simple(torch.tensor(patches_f32))

        S0 = s_mean['S0'].cpu().numpy()      # (N, 1)
        S1 = s_mean['S1_iso'].cpu().numpy()  # (N, J)
        S2 = s_mean['S2_iso'].cpu().numpy()  # (N, J, J, L)

    elif backend == 'kymatio':
        import torch
        from kymatio.numpy import Scattering2D

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        warnings.warn(
            "Using kymatio backend. The Cheng et al. scattering package "
            "is preferred for speed and scattering covariance support.",
            UserWarning)

        scattering2d = Scattering2D(J=J, shape=(H, W), L=L)
        Sx = scattering2d(patches_f32)  # (N, 1+J*L+..., H/2^J, W/2^J)

        # Average over spatial dimensions then orientations for S1
        Sx_mean = Sx.mean(axis=(-2, -1))  # (N, n_coeffs)
        n0 = 1
        S0 = Sx_mean[:, :n0]
        # kymatio S1: J*L coefficients → average over L to match Cheng S1_iso
        S1_full = Sx_mean[:, n0:n0 + J * L].reshape(N, J, L)
        S1 = S1_full.mean(axis=-1)  # (N, J)
        # S2: fill upper triangle (j1, j2, l=0) as approximation
        S2_flat = Sx_mean[:, n0 + J * L:]
        S2 = np.zeros((N, J, J, L))
        idx = 0
        for j1 in range(J):
            for j2 in range(j1 + 1, J):
                for l in range(L):
                    if idx < S2_flat.shape[1]:
                        S2[:, j1, j2, l] = S2_flat[:, idx]
                        idx += 1

    return {
        'S0': S0,
        'S1': S1,
        'S2': S2,
        'S1_mean': S1.mean(axis=0),
        'S2_mean': S2.mean(axis=0),
        'J': J,
        'L': L,
    }


def compute_scattering_covariance(patches_nhw, J=5, L=4, device=None):
    """Compute the scattering covariance C11 (modulus × modulus correlations).

    Only available with the Cheng et al. backend.  The scattering covariance
    captures cross-scale correlations that the scattering mean misses,
    making it sensitive to non-Gaussian structure at multiple scales.

    Parameters
    ----------
    patches_nhw : ndarray, shape (N, H, W)
        Stack of flat-sky patches.
    J : int
        Number of dyadic scales.
    L : int
        Number of orientations.
    device : str or None
        ``'cuda'``, ``'cpu'``, or ``None`` (auto-detect).

    Returns
    -------
    dict
        Full scattering covariance dictionary from Cheng et al.,
        with keys ``'C11_iso'``, ``'C01_iso'``, ``'S1'``, etc.
        Returns ``None`` if the Cheng et al. backend is not available.
    """
    try:
        import scattering
        import torch
    except ImportError:
        warnings.warn(
            "Scattering covariance requires the Cheng et al. scattering "
            "package. Install from: "
            "https://github.com/SihaoCheng/scattering_transform",
            UserWarning)
        return None

    N, H, W = patches_nhw.shape
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cheng_device = 'gpu' if device == 'cuda' else 'cpu'

    patches_f32 = patches_nhw.astype(np.float32)
    st_calc = scattering.Scattering2d(M=H, N=W, J=J, L=L, device=cheng_device)
    s_cov = st_calc.scattering_cov(torch.tensor(patches_f32))

    # Move all tensors to CPU numpy
    return {k: v.cpu().numpy() if hasattr(v, 'cpu') else v
            for k, v in s_cov.items()}


# ---------------------------------------------------------------------------
# Summary statistics for comparison
# ---------------------------------------------------------------------------

def scattering_summary(coeffs, scale_idx=None):
    """Extract a flat summary vector from scattering coefficients.

    Useful for computing residuals or distances between ensembles.

    Parameters
    ----------
    coeffs : dict
        Output of :func:`compute_scattering_coefficients`.
    scale_idx : list of int, optional
        Subset of j indices (scales) to include.  All scales used if None.

    Returns
    -------
    ndarray, shape (N, n_features)
        Flattened scattering features per map.
    """
    J = coeffs['J']
    scales = scale_idx if scale_idx is not None else list(range(J))

    S1 = coeffs['S1'][:, scales]             # (N, n_scales)
    # S2 upper triangle j2 > j1, all orientation differences
    S2_list = []
    for j1 in scales:
        for j2 in range(j1 + 1, J):
            S2_list.append(coeffs['S2'][:, j1, j2, :])  # (N, L)

    parts = [S1]
    if S2_list:
        parts.append(np.concatenate(S2_list, axis=1))

    return np.concatenate(parts, axis=1)
