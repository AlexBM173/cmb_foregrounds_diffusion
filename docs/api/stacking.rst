stacking — tSZ cluster stacking
================================

``stacking`` implements the SNR-selected cluster stacking analysis from
paper §4.2.  Stacking averages cutouts centred on tSZ signal peaks,
suppressing uncorrelated pixel noise and revealing the mean radial profile
of the underlying cluster population.

The analysis is carried out in SNR bins rather than mass bins because the
patch-based pipeline has no direct access to the halo catalogue at inference
time.  Comparing AGORA and DDPM stacked profiles tests whether the model
reproduces not just the pixel statistics but the spatial structure of
individual clusters.

.. rubric:: Workflow

.. code-block:: python

    from foregrounds_diffusion.stacking import select_snr_pixels, extract_cutouts

    # 1. Find peaks in the 3–5σ SNR bin across all tSZ maps
    coords = select_snr_pixels(tsz_maps_nhw, snr_min=3.0, snr_max=5.0,
                               min_separation=5)

    # 2. Extract 32×32 pixel cutouts centred on each peak
    cutouts = extract_cutouts(tsz_maps_nhw, coords, cutout_size=32)
    # cutouts: (M, 32, 32)

    # 3. Stack
    stacked_profile = cutouts.mean(axis=0)   # (32, 32)

Run the same workflow on DDPM samples to compare against AGORA.
See :doc:`../tutorials/09_tsz_stacking` for the full analysis, including
radial profile plots and SNR-bin comparisons.

.. rubric:: API

.. automodule:: foregrounds_diffusion.stacking
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource
