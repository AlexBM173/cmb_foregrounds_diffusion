Data conventions
================

Array layout
------------

Raw ``.npy`` files saved by the preprocessing pipeline are **channels-last**:
shape ``(N, H, W, C)``.  PyTorch requires channels-first ``(N, C, H, W)``, so
``train.py`` transposes on load.

- ``C = 1`` for single-channel files (separate CIB and tSZ files).
- ``C = 2`` for joint files (CIB channel 0, tSZ channel 1).
- DDPM samples output by ``sample.py`` are channels-first ``(N, 2, H, W)``.

Patch geometry
--------------

All patches are **6° × 6°** flat-sky projections of HEALPix maps, pixelised
onto a **256 × 256** grid.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Value
   * - ``nx``, ``ny``
     - 256
   * - ``dx``, ``dy`` (arcmin/pixel)
     - 1.40625  (= 6° × 60 / 256)
   * - ``flatskymapparams``
     - ``[256, 256, 1.40625, 1.40625]``

Pass ``flatskymapparams`` to all ``flatmaps`` functions.

Normalisation
-------------

.. warning::

   The normalisation scheme is currently **contested** between notebooks — see
   inconsistency #7 in ``docs/paper_code_inconsistencies.md``.  Notebook 03 (the
   data *producer*) z-scores **both** channels and saves ``_zscore_`` files with
   ``norm_params = [cib_mean, cib_std, tsz_mean, tsz_std]``, which matches the
   ``denormalize_dm_maps`` (``x·std+mean``) paragraph below and the checkpoint
   name ``v3_zscore_...``.  The table immediately below documents the *legacy*
   min–max scheme still referenced by notebooks 06–14's load filenames.  Confirm
   which ``.npy`` files exist on disk and how the checkpoint was trained before
   relying on either.

Two normalisation schemes have been used, one per channel (legacy scheme shown;
see the warning above):

.. list-table::
   :header-rows: 1

   * - Channel
     - Scheme
     - File suffix
     - Notes
   * - CIB 150 GHz
     - Min–max → [0, 1]
     - ``_zero``
     - CIB pixels are non-negative by construction
   * - tSZ Compton-y
     - Z-score (μ=0, σ=1)
     - ``_norm``
     - tSZ spans negative and positive values

Normalisation parameters (``cib_mean, cib_std, tsz_mean, tsz_std``) are saved
alongside the patches as ``norm_params_{ptsrc}mJy.npy``.  Use
:func:`~foregrounds_diffusion.preprocessing.denormalize_dm_maps` to invert.

Masking and filtering
---------------------

- **Point-source mask**: sources brighter than 2 mJy at 150 GHz are masked;
  masked pixels are set to zero (not NaN).
- **Low-pass filter**: modes above ℓ = 7000 are removed; negative pixels
  arising from filtering artefacts are zeroed.
- **Patch rejection**: patches where all pixels are zero (footprint entirely
  within the Galactic-plane mask) are discarded before training.

Multipole conventions
----------------------

:func:`~foregrounds_diffusion.flatmaps.map2cl` returns the **unbinned power
spectrum** :math:`C_\ell` in map-unit² sr.  To convert to
:math:`D_\ell = \ell(\ell+1)C_\ell / 2\pi`:

.. code-block:: python

   el, cl, _ = mean_cls(maps, flatskymapparams, lmin=300, lmax=4000, binsize=60)
   dl = el * (el + 1) / (2 * np.pi) * cl
