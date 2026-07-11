Quickstart
==========

The whole pipeline runs from **one YAML file and a few CLI entries**. Each of
the four stages is ``python run.py <stage> --config <file>``; every field the
config accepts is documented in :doc:`configuration`.

.. contents:: Contents
   :local:
   :depth: 1

Prerequisites
-------------

1. **Install the package and dependencies** — see :doc:`installation`.
2. **Activate the environment** before running anything.
3. **Get the training-ready patches.** Training reads per-channel ``.npy``
   patch files (see :ref:`preprocess-stage`). If you already have them (e.g.
   from an artifact store), point ``data.patches_dir`` at their directory and
   skip preprocessing.

The pipeline in four stages
---------------------------

Copy the annotated template, edit it, and validate before running:

.. code-block:: bash

   cp config/default.yaml config/my_run.yaml   # edit run_name, data.patches_dir, …
   python config/validate.py config/my_run.yaml

Then run each stage. Every stage writes into ``runs/<run_name>/`` alongside a
copy of the config, its SHA256 hash, and the git commit:

.. code-block:: bash

   python run.py preprocess --config config/my_run.yaml   # checks patches exist
   python run.py train      --config config/my_run.yaml   # → runs/<run>/checkpoints/
   python run.py sample     --config config/my_run.yaml   # → runs/<run>/samples/
   python run.py evaluate   --config config/my_run.yaml   # → runs/<run>/{stats,plots}/

Add ``--dry-run`` to any stage to print what it would do without executing.
``python run.py all --config …`` chains the stages. On multiple GPUs, prefix
any stage with ``accelerate launch`` (e.g.
``accelerate launch run.py train --config …``).

.. _preprocess-stage:

Preprocess
~~~~~~~~~~

``run.py preprocess`` **verifies** that the training-ready patches named by the
config exist; it does not itself produce them. Producing them from the raw
full-sky AGORA maps is done by standalone scripts (the path used for the actual
runs), which currently use hardcoded paths rather than the config:

.. code-block:: bash

   # Two-channel (CIB + tSZ):
   python scripts/vm_preprocessing/nb01_run.py            # filter halo lightcone
   python scripts/vm_preprocessing/nb02_run.py            # masking → masked FITS
   python scripts/vm_preprocessing/nb03_run.py            # patch extraction + z-score

   # Four-channel adds kSZ + κ:
   python scripts/vm_preprocessing/nb02b_mask_ksz_kappa.py
   python scripts/vm_preprocessing/nb03b_extract_4ch.py

Then set ``data.patches_dir`` to the output directory (e.g.
``data/low_pass/2mJy``). The tutorial notebooks
``01_halo_catalogue`` → ``03_patch_extraction`` mirror these steps with full
explanation and are the best starting point for understanding the pipeline.
The ``data`` and ``preprocessing`` config sections *document* the choices these
scripts bake into the patches — see :ref:`preprocessing-note`.

Train
~~~~~

.. code-block:: bash

   python run.py train --config config/my_run.yaml

Checkpoints and milestone sample previews land in ``runs/<run>/checkpoints/``.
The train/val/test split is config-controlled (``data.seed``,
``data.train_size``) and **must match** between training and evaluation. Set
``training.resume_from_checkpoint: true`` to continue from the latest
checkpoint.

Sample
~~~~~~

.. code-block:: bash

   python run.py sample --config config/my_run.yaml

Uses the latest checkpoint in ``runs/<run>/checkpoints/`` (or pass
``--checkpoint PATH``). For faster generation set ``sampling.ddim_steps`` (e.g.
250); the reported runs use full 1000-step ancestral sampling.

Evaluate
~~~~~~~~

.. code-block:: bash

   python run.py evaluate --config config/my_run.yaml

Computes the configured statistics for the Agora test split, a Gaussian
baseline, and the DDPM samples, caching each under ``runs/<run>/stats/*.npz``
and writing figures to ``runs/<run>/plots/`` plus a one-line-per-statistic
``summary.md``. Re-running with the caches present recomputes nothing and only
rewrites the figures, so plot styling can be iterated for free.

Provided configs and reproducing results
-----------------------------------------

Two run configs ship with their cached statistics, so the figures regenerate
without re-sampling:

.. code-block:: bash

   python run.py evaluate --config config/v4_eval.yaml   # two-field (CIB + tSZ)
   python run.py evaluate --config config/v5_4ch.yaml    # four-field (+ kSZ + κ)

See :doc:`configuration` for the full settings reference and ``CHANGELOG.md``
for per-generation provenance.

Using the package as a library
------------------------------

The stage runners are thin wrappers over the importable
``foregrounds_diffusion`` package, which you can also call directly.

Load patches and measure power spectra:

.. code-block:: python

   import numpy as np
   from pathlib import Path
   from foregrounds_diffusion.moments import mean_cls

   PATCHES_DIR = Path("data/low_pass/2mJy")
   flatskymapparams = [256, 256, 1.40625, 1.40625]  # [nx, ny, dx, dy] arcmin

   cib_maps = np.load(PATCHES_DIR / "CIB_map_150GHz_256_st6_zscore_2mJy_lp.npy")
   agora_cib = cib_maps[:, :, :, 0]   # (N, 256, 256), channels-last on disk

   el, mean_cl, std_cl = mean_cls(
       agora_cib, flatskymapparams, lmin=300, lmax=4000, binsize=60, n_jobs=-1
   )

Compute higher-order cross moments:

.. code-block:: python

   from foregrounds_diffusion.flatmaps import get_lpf_hpf
   from foregrounds_diffusion.moments import compute_cross_moments

   tsz_maps = np.load(PATCHES_DIR / "tSZ3_map_150GHz_256_st6_zscore_2mJy_lp.npy")
   agora_tsz = tsz_maps[:, :, :, 0]

   bp_edges = [(300 + i * 720, 300 + (i + 1) * 720) for i in range(8)]
   bp_filters = [get_lpf_hpf(flatskymapparams, e, filter_type=2) for e in bp_edges]
   moments, labels = compute_cross_moments(agora_cib, agora_tsz, bp_filters, n_jobs=-1)

Denormalise DDPM samples to physical units:

.. code-block:: python

   from foregrounds_diffusion.preprocessing import denormalize_dm_maps

   samples_raw = np.load("runs/v4_zscore_2mJy_a100/samples/samples_v4_zscore_2mJy_a100.npy")
   cib_mean, cib_std, tsz_mean, tsz_std = np.load(PATCHES_DIR / "norm_params_2mJy.npy")
   samples = denormalize_dm_maps(samples_raw, cib_mean, cib_std, tsz_mean, tsz_std)
   ddpm_cib, ddpm_tsz = samples[:, 0], samples[:, 1]   # (N, 256, 256) each
