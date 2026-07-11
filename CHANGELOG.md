# Changelog

Model/run provenance for the CMB-foregrounds diffusion project. This records
what changed between the three training generations (v3 → v4 → v5), which the
git history alone does not make obvious. Dates are UTC.

> **Provenance caveat.** Each run directory under `runs/<name>/` stamps a
> `git_commit.txt`, but `run.py` rewrites it on *every* invocation (including
> `evaluate`), so it reflects the last `run.py` call for that run, not the commit
> the model was trained at. Both current runs read `292ce91…-dirty` after the
> July 2026 cleanup re-ran evaluate. Treat the training commit as approximate;
> the run `config.yaml` (also stamped in the run dir) is the reliable record of
> the parameters actually used.

## Runs

### v5 — four-channel (CIB + tSZ + kSZ + CMB-lensing κ)
- **Config:** `config/v5_4ch.yaml` · run dir `runs/v5_4ch_zscore_2mJy_a100/`
- **Model:** `channels=4`, `dim=96` (raised from v4's 64 for the joint 4-field),
  `dim_mults=(1,2,4,8)`, 1000 diffusion timesteps, z-score normalisation.
- **Training:** 100k steps on an A100, in two resumable Colab sessions
  (~47 h GPU time total); final loss ≈ 0.17.
- **Sampling:** 640 maps, 1000-step ancestral (no DDIM), unclamped
  (`UnclampedGaussianDiffusion`); ~2 h 40 m.
- **Data:** patches from `scripts/vm_preprocessing/nb02b_mask_ksz_kappa.py` +
  `nb03b_extract_4ch.py` (not the notebook-03 path), synced from GCS. κ is the
  raytraced lensing convergence; kSZ in µK.
- **Split:** seed 42, `train_size=0.8` → 560 train / 141 test of 701 patches.
- **Headline result:** DDPM matches the Gaussian core and truncates rare peaks;
  cluster-stacking deficit in every SNR bin.

### v4 — two-channel (CIB + tSZ)
- **Config:** `config/v4_eval.yaml` · run dir `runs/v4_zscore_2mJy_a100/`
- **Model:** `channels=2`, `dim=64`, `dim_mults=(1,2,4,8)`, 1000 timesteps,
  z-score normalisation.
- **Training:** 100k steps on an A100 (~29.6 h per session notes — not verified
  against a surviving training log; no `train_*.log` exists for v4).
- **Sampling:** 640 maps, 1000-step ancestral, unclamped; ~1 h 49 m.
- **Split:** seed 42, `train_size=0.8` → 560 train / 141 test of 701 patches.
- This is the generation written up for the MPhil report.

### v3 — earliest z-score checkpoint (superseded)
- Referenced only by the checkpoint name `v3_zscore_...`. No `runs/v3*`
  directory survives, so
  its provenance is undocumented. Established the z-score normalisation scheme
  that v4/v5 inherit, deviating from the paper's min-max (CIB) + std-norm (tSZ).

## Notable code changes (July 2026 cleanup)

- Config-driven train/test split: `pipeline/train.py` now honours
  `--seed`/`--train-size` from the config instead of hardcoding 42 / 0.8
  (verified bit-identical for the shipped configs, so v4/v5 results are intact).
- `.gitignore` excludes large binaries by extension; notebook outputs stripped
  and enforced by an `nbstripout` pre-commit hook; SLURM scripts relocated to
  `scripts/slurm/`.
- The `x₀`-clamp fix (`UnclampedGaussianDiffusion`, `foregrounds_diffusion/`
  `sample.py`) is required for z-score models; the stock sampler crushes maps
  into ±1.

> A history rewrite to purge large AGORA FITS blobs (planned) will change all
> commit SHAs. When it runs, the old→new mapping should be appended here.
