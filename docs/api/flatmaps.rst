flatmaps — Flat-sky Fourier utilities
=====================================

``flatmaps`` is the mathematical foundation of the analysis pipeline.
Every function that works in harmonic space — power spectra, Gaussian
realisations, filtering, polarisation rotation — lives here.

All functions operate on **flat-sky maps**: 2D NumPy arrays parameterised by::

    flatskymapparams = [nx, ny, dx, dy]   # dx, dy in arcminutes/pixel

For the 6° × 6° training patches used in this project the standard value is
``[256, 256, 1.40625, 1.40625]``.  Pass this list to every function that
accepts a ``flatskymapparams`` argument.

.. rubric:: Common patterns

Compute a mean power spectrum over N maps:

.. code-block:: python

    from foregrounds_diffusion.flatmaps import map2cl
    from foregrounds_diffusion.moments import mean_cls

    mapparams = [256, 256, 1.40625, 1.40625]
    el, mean_cl, std_cl = mean_cls(maps_nhw, mapparams, lmin=300, lmax=4000, binsize=60)

Apply a low-pass filter at ℓ = 7000:

.. code-block:: python

    from foregrounds_diffusion.flatmaps import get_lpf_hpf, bandpass_filter

    lpf = get_lpf_hpf(mapparams, 7000, filter_type=0)
    filtered = bandpass_filter(raw_map, lpf)

Draw a correlated Gaussian realisation:

.. code-block:: python

    from foregrounds_diffusion.flatmaps import make_gaussian_realisation

    sim = make_gaussian_realisation(mapparams, el, cl_cib, cl2=cl_tsz, cl12=cl_cross)
    # sim[0] = CIB-like field, sim[1] = correlated tSZ-like field

.. rubric:: GPU acceleration

:func:`map2cl_torch` computes all N power spectra in a single batched
``torch.fft.fft2`` call, giving a substantial speedup over calling
:func:`map2cl` in a Python loop.  See :doc:`../tutorials/13_benchmarks`
for timing comparisons.

.. rubric:: API

.. automodule:: foregrounds_diffusion.flatmaps
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource
