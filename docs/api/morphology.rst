morphology — Minkowski functionals and tensors
==============================================

``morphology`` computes topological and geometrical statistics on excursion
sets — the subsets of pixels that exceed a given intensity threshold ν.
These statistics complement the power spectrum by probing non-Gaussian
morphological features such as the shapes and connectivity of tSZ clusters.

.. rubric:: Minkowski functionals

The three scalar functionals characterise excursion-set topology as a
function of threshold ν (paper §4.1, Figure 6):

- **M0**: area fraction — fraction of pixels above ν
- **M1**: perimeter — total boundary length normalised by map area
- **M2**: Euler characteristic — connected components minus holes

Requires ``pip install quantimpy``:

.. code-block:: python

    from foregrounds_diffusion.morphology import compute_mfs
    from foregrounds_diffusion.preprocessing import apply_maxmin_normalization

    thresholds = np.linspace(0.05, 0.95, 19)
    M0, M1, M2 = compute_mfs(maps_nhw, apply_maxmin_normalization, thresholds, n_jobs=-1)
    # Each: (N, len(thresholds))

.. rubric:: Minkowski tensors

:func:`compute_minkowski_tensors` is a tensorial extension that additionally
captures anisotropy and orientation per excursion set.  It returns the
anisotropy index β = λ_min / λ_max ∈ [0, 1] (β = 1 is perfectly isotropic)
and major-axis orientation θ for three tensor types (W021, W200, W201).
W021 (boundary normal tensor) is recommended as the default:

.. code-block:: python

    from foregrounds_diffusion.morphology import compute_minkowski_tensors

    results = compute_minkowski_tensors(
        maps_nhw, apply_maxmin_normalization, thresholds,
        tensor_types=["W021"], centred=True,
    )
    beta = results["W021"]["beta"]   # (N, len(thresholds))
    theta = results["W021"]["theta"]

See :doc:`../tutorials/12_minkowski_tensors` for β(ν) curves, polar
orientation histograms, and DDPM/AGORA residuals.

.. rubric:: API

.. automodule:: foregrounds_diffusion.morphology
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource
