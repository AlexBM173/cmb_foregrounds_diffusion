preprocessing — Normalisation and patch extraction
==================================================

``preprocessing`` handles everything between raw HEALPix FITS files and
the training-ready ``.npy`` arrays consumed by the model.  It also provides
the inverse normalisation functions needed to convert DDPM samples back to
physical units after generation.

.. rubric:: Pipeline context

The full preprocessing workflow runs across three tutorial notebooks:

1. :doc:`../tutorials/01_halo_catalogue` — build the filtered MDPL2 halo catalogue
2. :doc:`../tutorials/02_masking` — apply point-source and cluster masks
3. :doc:`../tutorials/03_patch_extraction` — extract patches, filter, normalise, save

.. rubric:: Normalisation

Two schemes are used:

.. list-table::
   :header-rows: 1

   * - Channel
     - Function
     - Inverse
     - Range
   * - CIB
     - :func:`apply_maxmin_normalization`
     - :func:`renormalize_dm_maps`
     - [0, 1]
   * - tSZ
     - :func:`apply_stdnorm`
     - :func:`denormalize_dm_maps`
     - z-score (μ=0, σ=1)

After sampling, convert DDPM output back to physical units:

.. code-block:: python

    from foregrounds_diffusion.preprocessing import denormalize_dm_maps

    samples = np.load("samples.npy")           # (N, 2, 256, 256), channels-first
    norm = np.load("norm_params_2mJy.npy")     # [cib_mean, cib_std, tsz_mean, tsz_std]
    samples_phys = denormalize_dm_maps(samples, *norm)

.. rubric:: Dataset splitting

:func:`split_data_to_tensors` performs an 80/10/10 train/val/test split,
converts from channels-last ``(N, H, W, C)`` to channels-first
``(N, C, H, W)``, and returns ``torch.Tensor`` objects:

.. code-block:: python

    from foregrounds_diffusion.preprocessing import split_data_to_tensors

    data = np.load("CIB_map_150GHz_256_st6_zscore_2mJy_lp.npy")  # (N, H, W, 1)
    train, val, test = split_data_to_tensors(data)

.. rubric:: API

.. automodule:: foregrounds_diffusion.preprocessing
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource
