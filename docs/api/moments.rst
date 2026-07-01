moments — Power spectra and higher-order statistics
===================================================

``moments`` is the primary validation module.  It computes the summary
statistics used throughout the paper to compare DDPM-generated maps against
AGORA training maps and a Gaussian baseline.

.. rubric:: Power spectra

:func:`mean_cls` and :func:`mean_cross_cls` average power spectra over an
ensemble of maps.  Both functions cache the ℓ-bin grid and support joblib
parallelism via ``n_jobs=-1``:

.. code-block:: python

    from foregrounds_diffusion.moments import mean_cls, mean_cross_cls

    mapparams = [256, 256, 1.40625, 1.40625]

    # Auto-spectrum of CIB maps
    el, cl_cib, err_cib = mean_cls(cib_maps, mapparams, lmin=300, lmax=4000,
                                    binsize=60, n_jobs=-1)

    # Cross-spectrum between CIB and tSZ
    el, cl_cross, err_cross = mean_cross_cls(cib_maps, tsz_maps, mapparams,
                                              lmin=300, lmax=4000, binsize=60)

.. rubric:: Higher-order moments

:func:`compute_cross_moments` computes 12 cross-moments of bandpass-filtered
CIB (a) and tSZ (b) fields per ℓ-band — variance, skewness, and kurtosis
terms up to 4th order:

.. code-block:: python

    from foregrounds_diffusion.flatmaps import get_lpf_hpf
    from foregrounds_diffusion.moments import compute_cross_moments

    # Build one bandpass filter per ℓ-band
    bp_edges = [(300 + i*720, 300 + (i+1)*720) for i in range(8)]
    bp_filters = [get_lpf_hpf(mapparams, edges, filter_type=2) for edges in bp_edges]

    moments, labels = compute_cross_moments(cib_maps, tsz_maps, bp_filters, n_jobs=-1)
    # moments: (N, 8, 12),  labels: ['S2aa', 'S2bb', 'S2ab', ...]

See :doc:`../tutorials/07_higher_order_stats` for the full comparison between
AGORA, DDPM, and Gaussian maps.

.. rubric:: API

.. automodule:: foregrounds_diffusion.moments
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource
