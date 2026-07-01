Quickstart
==========

This guide shows the most common operations: loading preprocessed patches,
computing power spectra, and generating samples from a trained checkpoint.

Load patches and measure power spectra
---------------------------------------

.. code-block:: python

   import numpy as np
   from pathlib import Path
   from foregrounds_diffusion.flatmaps import map2cl
   from foregrounds_diffusion.moments import mean_cls

   PATCHES_DIR = Path("data/low_pass/2mJy")
   flatskymapparams = [256, 256, 1.40625, 1.40625]  # [nx, ny, dx, dy] arcmin

   cib_maps = np.load(PATCHES_DIR / "CIB_map_150GHz_256_st6_minmax_2mJy_zero_lp.npy")
   # shape: (N, 256, 256, 1) — channels-last

   agora_cib = cib_maps[:, :, :, 0]   # (N, 256, 256)

   el, mean_cl, std_cl = mean_cls(agora_cib, flatskymapparams, lmin=300, lmax=4000, binsize=60)

   # Optional: parallel over all CPU cores
   el, mean_cl, std_cl = mean_cls(
       agora_cib, flatskymapparams, lmin=300, lmax=4000, binsize=60, n_jobs=-1
   )

Compute higher-order moments
-----------------------------

.. code-block:: python

   from foregrounds_diffusion.flatmaps import get_lpf_hpf
   from foregrounds_diffusion.moments import compute_cross_moments

   tsz_maps = np.load(PATCHES_DIR / "tSZ3_map_150GHz_256_st6_minmax_2mJy_norm_lp.npy")
   agora_tsz = tsz_maps[:, :, :, 0]

   bp_edges = [(300 + i * 720, 300 + (i + 1) * 720) for i in range(8)]
   bp_filters = [get_lpf_hpf(flatskymapparams, e, filter_type=2) for e in bp_edges]

   moments, labels = compute_cross_moments(agora_cib, agora_tsz, bp_filters, n_jobs=-1)
   # moments: (N, 8, 12)  labels: ['S2aa', 'S2bb', ...]

Sample from a trained checkpoint
----------------------------------

.. code-block:: bash

   accelerate launch foregrounds_diffusion/sample.py \
     --checkpoint results/my_run_v1/model-20.pt \
     --batches 10 --batch-size 16 \
     --output data/low_pass/2mJy/samples.npy

Then load and evaluate:

.. code-block:: python

   from foregrounds_diffusion.preprocessing import denormalize_dm_maps

   samples_raw = np.load("data/low_pass/2mJy/samples.npy")  # (N, 2, 256, 256)
   norm_params = np.load(PATCHES_DIR / "norm_params_2mJy.npy")
   cib_mean, cib_std, tsz_mean, tsz_std = norm_params

   samples = denormalize_dm_maps(samples_raw, cib_mean, cib_std, tsz_mean, tsz_std)
   ddpm_cib = samples[:, 0]   # (N, 256, 256)
   ddpm_tsz = samples[:, 1]
