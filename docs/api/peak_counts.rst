peak_counts — Peak and minima counting
======================================

``peak_counts`` implements the peak and minima counting statistics from
Sabyr, Hill & Haiman (2024, arXiv:2410.21247), adapted for flat-sky patches.

Peak counts measure how many pixels exceed a given signal-to-noise threshold
ν = T/σ, and minima counts measure how many pixels fall below -ν.  Together
they characterise the one-point distribution and local topology of the field
in a way that is sensitive to non-Gaussianity — a Gaussian field has a
symmetric peak/minima distribution, while tSZ maps are skewed toward positive
peaks due to massive clusters.

Requires only ``numpy`` and ``scipy`` — no LensTools dependency.

.. rubric:: Usage

.. code-block:: python

    from foregrounds_diffusion.peak_counts import compute_peak_minima_counts

    nu_bins = np.linspace(-4, 4, 17)   # ν = T/σ thresholds
    smoothing_arcmin = 1.0             # Gaussian smoothing scale

    result = compute_peak_minima_counts(
        maps_nhw, nu_bins, smoothing_arcmin, dx_arcmin=1.40625
    )
    # result["peak_counts"]: (N, len(nu_bins)-1)
    # result["minima_counts"]: (N, len(nu_bins)-1)
    # result["nu_centres"]: bin centres

To smooth at multiple angular scales before counting:

.. code-block:: python

    from foregrounds_diffusion.peak_counts import smooth_map, find_peaks, count_peaks_binned

    smoothed = smooth_map(maps_nhw, sigma_arcmin=2.0, dx_arcmin=1.40625)
    peaks = find_peaks(smoothed)                           # local maxima coordinates
    counts = count_peaks_binned(smoothed, peaks, nu_bins) # (N, n_bins)

See :doc:`../tutorials/10_peak_minima_counts` for comparisons between AGORA,
DDPM, and Gaussian maps across multiple smoothing scales.

.. rubric:: API

.. automodule:: foregrounds_diffusion.peak_counts
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource
