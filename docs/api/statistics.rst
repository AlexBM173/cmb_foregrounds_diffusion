statistics — 2D Gaussian fitting and summary stats
==================================================

``statistics`` provides two utilities used for quick map characterisation.

.. rubric:: 2D Gaussian fitting

Used to measure the effective beam or PSF of a filtered or beam-convolved
map by fitting a 2D Gaussian to a stacked source profile or a correlation
function peak.  The typical workflow is:

.. code-block:: python

    from foregrounds_diffusion.statistics import fitgaussian, fitting_func

    # Fit a Gaussian to a 2D thumbnail (e.g. a stacked tSZ cluster profile)
    params = fitgaussian(thumbnail)
    # params: (height, center_x, center_y, width_x, width_y)

    # Or use fitting_func for bounds enforcement and optional residual return
    tmap_fit = fitting_func(thumbnail, return_fit=1)   # returns fitted image
    residual  = fitting_func(thumbnail, return_fit=0)  # returns data - fit

The initial parameter guess is estimated from image moments via
:func:`moments`, so no manual initialisation is required.

.. rubric:: Summary statistics

:func:`stats` reports the mean, std, min, and max of a map, optionally
excluding zero-valued (masked) pixels:

.. code-block:: python

    from foregrounds_diffusion.statistics import stats

    mean, std, vmin, vmax = stats(cib_patch, log=True, mask_zero=True)

.. rubric:: API

.. automodule:: foregrounds_diffusion.statistics
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource
