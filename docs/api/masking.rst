masking — Peak masks and AGORA cluster masks
============================================

``masking`` provides two distinct families of functions used at different
stages of the pipeline.

.. rubric:: Flat-sky peak masks (post-sampling)

After drawing samples from the DDPM, bright point-source peaks in the
generated CIB maps are masked before computing statistics.  This mirrors
the 2 mJy masking applied during training data preparation:

.. code-block:: python

    from foregrounds_diffusion.masking import get_peak_masks, inpaint_masked_regions

    mask = get_peak_masks(cib_map, mask_threshold_sigma_units=3.0,
                          mask_radius_pixel_units=3.0)
    inpainted = inpaint_masked_regions(cib_map, mask)

:func:`boundary_apod_mask` and :func:`get_mask_using_gaussian_fitting` are
variants used for apodisation and adaptive threshold fitting respectively.

.. rubric:: Full-sky HEALPix masks (preprocessing)

These functions run once during data preparation on the cluster, operating
on full-sky HEALPix maps before patch extraction.  They require
``healpy``, ``astropy``, and the MDPL2 halo catalogue produced by
:doc:`../tutorials/01_halo_catalogue`.

:func:`get_point_source_mask_in_healpix` masks CIB point sources above a
flux threshold (default 2 mJy at 150 GHz) using sigma-clipping:

.. code-block:: python

    from foregrounds_diffusion.masking import get_point_source_mask_in_healpix

    mask = get_point_source_mask_in_healpix(cib_map_jy_sr, freq_ghz=150,
                                             threshold_mjy=2.0, nside=8192)

:func:`get_apodised_mdpl2_cluster_mask` builds apodised circular masks
around MDPL2 clusters with M_500c ≥ 3×10¹⁴ M☉:

.. code-block:: python

    from foregrounds_diffusion.masking import get_apodised_mdpl2_cluster_mask

    cluster_mask = get_apodised_mdpl2_cluster_mask(
        nside=2048,
        halo_cat_fname="data/halo_catalogue/halo_catalogue_m500gt3e14.npy.npz",
        m500c_threshold=3e14,
    )

See :doc:`../tutorials/02_masking` for the full masking workflow.

.. rubric:: API

.. automodule:: foregrounds_diffusion.masking
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource
