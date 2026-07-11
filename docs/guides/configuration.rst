Configuration reference
=======================

The whole pipeline is driven by a single YAML file. Every stage
(:doc:`quickstart`) takes ``--config <file>``, and every run copies its config,
the config's SHA256 hash, and the git commit into ``runs/<run_name>/`` so an
artefact always traces back to an exact configuration and code state.

.. contents:: Sections
   :local:
   :depth: 1

Getting started with a config
------------------------------

Three configs ship with the repository:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Purpose
   * - ``config/default.yaml``
     - Fully annotated template — **the canonical, field-by-field reference**.
       Every field on this page is commented there. Copy it to start a new run.
   * - ``config/v4_eval.yaml``
     - The two-field (CIB + tSZ) run written up in the report.
   * - ``config/v5_4ch.yaml``
     - The four-field (CIB + tSZ + kSZ + κ) run.

Copy the template, edit it, and validate before running anything:

.. code-block:: bash

   cp config/default.yaml config/my_run.yaml
   # edit run_name, data.patches_dir, and any settings you want to change
   python config/validate.py config/my_run.yaml

Validation rejects unknown keys, out-of-range values, and incompatible
settings with a path-qualified message (e.g.
``training.ema_decay: must be in (0, 1)``). The schema and every check live in
``config/validate.py``.

.. note::

   **Not every field is wired to executable code yet.** Two groups are
   validated (so a config is self-documenting and future-proof) but not
   consumed by the current stages:

   - The whole ``data`` and ``preprocessing`` sections *describe* the manual
     preprocessing performed by ``scripts/vm_preprocessing/`` — they are not
     read by ``run.py`` (see :ref:`preprocessing-note`). The exceptions are
     ``data.res``, ``preprocessing.point_source_mjy`` and ``model.channels``,
     which ``run.py`` uses to locate the training-ready ``.npy`` patch files.
   - Several ``model``/``training`` fields are fixed in the code
     (:ref:`fixed-settings`); a config that asks for a different value fails
     loudly rather than silently training a different model.

``run_name`` and ``output``
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Field
     - Default
     - Meaning
   * - ``run_name``
     - ``example_run``
     - Label for the run; artefacts go to ``<output.base_dir>/<run_name>/``.
       Must match ``[A-Za-z0-9._-]+``.
   * - ``output.base_dir``
     - ``runs``
     - Parent directory for all run outputs.

``data`` — raw maps and patch geometry
--------------------------------------

Describes the raw AGORA inputs and how patches are cut. Consumed by the manual
preprocessing scripts; ``run.py`` only reads ``res`` (and ``patches_dir``).

.. list-table::
   :header-rows: 1
   :widths: 25 22 53

   * - Field
     - Default
     - Meaning
   * - ``cib_map`` / ``tsz_map``
     - AGORA FITS
     - Full-sky HEALPix input maps (Globus paths in the README's Data section).
   * - ``halo_catalogue``
     - ``…m500gt3e14.npz``
     - Halo catalogue used to build the cluster mask.
   * - ``frequency_ghz``
     - ``150``
     - Observing frequency of the maps.
   * - ``nside_in`` / ``nside_out``
     - ``8192`` / ``2048``
     - HEALPix resolution of the raw maps / the downgraded maps used for patch
       extraction. Powers of two; ``nside_out ≤ nside_in``.
   * - ``patch_deg`` / ``step_deg``
     - ``6.0`` / ``6.0``
     - Patch side length and centre spacing in degrees
       (``step_deg < patch_deg`` ⇒ overlapping patches).
   * - ``res``
     - ``256``
     - Pixels per patch side (6° / 256 px = 1.40625′/px). **Used by run.py** to
       build patch filenames.
   * - ``gal_cut_deg`` / ``pole_cut_deg``
     - ``20.0`` / ``6.0``
     - Exclude patch centres within this angle of the Galactic plane / poles.
   * - ``train_size`` / ``val_size`` / ``test_size``
     - ``0.8`` / ``0.1`` / ``0.1``
     - Train/val/test split fractions; must sum to 1.
   * - ``seed``
     - ``42``
     - ``np.random.default_rng`` seed for the split. **Must match between
       training and evaluation** or the test set differs.
   * - ``patches_dir``
     - ``null``
     - Directory holding the training-ready ``.npy`` patches. ``null`` ⇒
       ``<run_dir>/data/patches``. Point it at the output of the preprocessing
       scripts, e.g. ``data/low_pass/2mJy``.

``preprocessing`` — masking, filtering, normalisation
-----------------------------------------------------

Records the choices baked into the patches. Consumed by the manual
preprocessing scripts; ``run.py`` only reads ``point_source_mjy`` (to build
patch filenames).

.. list-table::
   :header-rows: 1
   :widths: 32 20 48

   * - Field
     - Default
     - Meaning
   * - ``lowpass.type``
     - ``sharp``
     - Low-pass profile at ``ell_max``: ``sharp`` | ``cosine`` | ``wiener``.
   * - ``lowpass.ell_max``
     - ``7000``
     - Low-pass cut-off multipole.
   * - ``normalisation``
     - ``zscore``
     - ``zscore`` (both channels; canonical) | ``minmax``. Fixed at ``zscore``
       in the current pipeline (:ref:`fixed-settings`).
   * - ``point_source_mjy``
     - ``2``
     - Flux-density point-source masking threshold (mJy). **Used by run.py** to
       build patch filenames.
   * - ``cluster_mask.enabled``
     - ``true``
     - Whether to apply the apodised cluster mask.
   * - ``cluster_mask.m500c_min``
     - ``3.0e+14``
     - Mask haloes above this mass (M\ :sub:`⊙`). Write exponents as ``3.0e+14``
       (a bare ``3.0e14`` parses as a string under YAML 1.1).
   * - ``cluster_mask.theta500_multiplier``
     - ``3.0``
     - Apodised mask radius = multiplier × θ\ :sub:`500c`.
   * - ``inpainting``
     - ``gaussian_noise``
     - Fill for masked pixels before filtering.
   * - ``augmentation``
     - ``true``
     - 8× augmentation (4 rotations × mirror) applied at training time.

``model`` — U-Net / diffusion architecture
------------------------------------------

Must match a checkpoint to load it. Only ``dim`` and ``channels`` are
configurable today; the rest are :ref:`fixed <fixed-settings>`.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Field
     - Default
     - Meaning
   * - ``dim``
     - ``64``
     - Base U-Net channel width. **Configurable** (v4 uses 64, v5 uses 96).
   * - ``channels``
     - ``2``
     - Number of map channels. **Configurable** — 2 (CIB, tSZ) or 4 (+ kSZ, κ).
   * - ``dim_mults``
     - ``[1, 2, 4, 8]``
     - Channel-width multipliers per resolution level. *Fixed.*
   * - ``flash_attn``
     - ``true``
     - Memory-efficient attention (CUDA compute ≥ 8.0). *Fixed.*
   * - ``timesteps``
     - ``1000``
     - Forward-process length. *Fixed.*
   * - ``noise_schedule``
     - ``sigmoid``
     - ``linear`` | ``cosine`` | ``sigmoid`` (package default). *Fixed.*
   * - ``objective``
     - ``pred_v``
     - ``pred_v`` | ``pred_noise`` | ``pred_x0``. *Fixed.*
   * - ``auto_normalize``
     - ``false``
     - **Must be false** for z-score data; ``true`` rescales samples
       ``[-1,1]→[0,1]`` on output and corrupts amplitudes. *Fixed.*

``training``
------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Field
     - Default
     - Meaning
   * - ``batch_size``
     - ``16``
     - Per-GPU batch size.
   * - ``lr``
     - ``1.0e-4``
     - Adam learning rate.
   * - ``lr_scheduler`` / ``warmup_steps``
     - ``none`` / ``0``
     - LR schedule. *Fixed* at ``none`` / ``0``.
   * - ``train_num_steps``
     - ``100000``
     - Total optimisation steps.
   * - ``gradient_accumulate_every``
     - ``2``
     - Steps per optimiser update (effective batch = ``batch_size × this``).
       *Fixed.*
   * - ``ema_decay``
     - ``0.995``
     - EMA decay; EMA weights are used for sampling. *Fixed.*
   * - ``mixed_precision``
     - ``bf16``
     - ``no`` | ``fp16`` | ``bf16``. *Fixed* at ``bf16``.
   * - ``save_and_sample_every``
     - ``5000``
     - Checkpoint interval (steps).
   * - ``milestone_num_samples``
     - ``25``
     - Samples generated at each checkpoint; must be a perfect square, or 0 to
       skip milestone sampling.
   * - ``resume_from_checkpoint``
     - ``false``
     - Continue from the latest ``model-*.pt`` in ``checkpoints/``.
   * - ``num_gpus``
     - ``1``
     - Informational; ``accelerate`` handles device placement.

``sampling``
------------

.. list-table::
   :header-rows: 1
   :widths: 28 20 52

   * - Field
     - Default
     - Meaning
   * - ``num_samples``
     - ``640``
     - Total maps to generate (``ceil(num_samples / batch_size)`` batches).
   * - ``batch_size``
     - ``16``
     - Per-batch sample count.
   * - ``ddim_steps``
     - ``250``
     - ``null`` = full DDPM (``model.timesteps`` steps); an integer in
       ``[1, timesteps]`` enables DDIM (≈ ``timesteps/ddim_steps``× faster).
       The reported runs use ``null`` (full 1000-step ancestral sampling).
   * - ``output_format``
     - ``npy``
     - Only ``npy`` is supported (fits/h5 planned).
   * - ``compile``
     - ``false``
     - ``torch.compile`` the U-Net (one-off warm-up cost).
   * - ``rescale_cib`` / ``rescale_tsz``
     - ``null``
     - Opt-in post-sampling scalar rescale (paper §3.2); positive when set.

``evaluation``
--------------

.. list-table::
   :header-rows: 1
   :widths: 25 28 47

   * - Field
     - Default
     - Meaning
   * - ``n_jobs``
     - ``8``
     - CPU workers for parallelised statistics (``-1`` = all cores).
   * - ``ilc_noise_file``
     - ``data/ilc/…npy``
     - ILC residual-noise spectra used by the ``noise_tiers`` of the moments.
   * - ``noise_seed``
     - ``42``
     - Seeds the per-(tier, source) noise realisations.
   * - ``statistics``
     - 7 core stats
     - The list of statistics to compute. Remove a name to skip it; add an
       extension name to enable it.

Each statistic has its own parameter block (``lmin``, ``lmax``, ``n_maps``,
``noise_tiers`` …) keyed by name. The statistics understood by the evaluator,
and the keys each accepts, are defined in ``KNOWN_STATISTICS`` in
``config/validate.py``:

- **Two-point:** ``power_spectrum``, ``cross_spectrum``
- **Higher-order:** ``moments``, ``cross_moments``
- **Morphology:** ``minkowski_functionals``, ``minkowski_tensors``,
  ``pixel_histograms``
- **Peaks:** ``peak_counts``, ``minima_counts``
- **Multi-scale:** ``scattering_transforms``
- **Cluster stacking:** ``tsz_stacking`` (2- and 4-field),
  ``kappa_on_tsz_stacking`` and ``ksz_stacking`` (4-field)

See ``config/default.yaml`` for every statistic's parameters with inline notes,
and the tutorial notebooks 06–12 for what each statistic measures.

``wandb`` (optional)
--------------------

``enabled`` (default ``false``), ``project``, ``entity``, ``tags``. The
pipeline runs without ``wandb`` installed; set ``enabled: true`` (and
``WANDB_API_KEY`` in the environment) to log losses, sample grids, and the
sample ``.npy`` artefact.

``slurm`` (deferred)
--------------------

``partition``, ``qos``, ``account``, ``num_gpus``, ``mem``, ``time``,
``mail_user``. Job-script *generation* is deferred (the project migrated off
HPC); these fields are validated so configs written for a cluster still pass.
See :doc:`hpc_slurm` for the reference SLURM scripts.

.. _fixed-settings:

Settings currently fixed in code
--------------------------------

To guarantee a config never claims a model it did not train, ``run.py`` rejects
any config that changes these fields from their fixed value (see
``_FIXED_SETTINGS`` in ``run.py``):

``model.dim_mults``, ``model.flash_attn``, ``model.timesteps``,
``model.noise_schedule``, ``model.objective``, ``model.auto_normalize``,
``training.gradient_accumulate_every``, ``training.ema_decay``,
``training.mixed_precision``, ``training.lr_scheduler``,
``training.warmup_steps``, and ``preprocessing.normalisation``.

``model.dim`` and ``model.channels`` **are** honoured. Making the rest
configurable is a planned follow-up.

.. _preprocessing-note:

A note on preprocessing
-----------------------

``run.py preprocess`` **does not run preprocessing** — it checks whether the
training-ready patch files already exist and, if not, points you at the scripts
that produce them. The actual preprocessing lives in
``scripts/vm_preprocessing/`` (and the tutorial notebooks 01–03) and currently
uses hardcoded paths rather than reading this config. The ``data`` and
``preprocessing`` sections above therefore *document* the choices those scripts
make; they are the reference for how the shipped patches were built. See
:doc:`quickstart` for the full step sequence.
