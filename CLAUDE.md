# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Activate the virtual environment before running any Python code:
```bash
# From the home directory (~):
source activate_diffusion_project_env.sh
```

## Common Commands

The current default workflow is config-driven via `run.py` (one YAML per run):

```bash
python run.py train    --config config/v5_4ch.yaml   # → runs/<run_name>/checkpoints/
python run.py sample   --config config/v5_4ch.yaml   # → runs/<run_name>/samples/
python run.py evaluate --config config/v5_4ch.yaml   # → runs/<run_name>/{stats,plots}/
python config/validate.py config/v5_4ch.yaml         # validate a config
```

The legacy flag-based entry points below still work (the root `train.py` is a
shim over `pipeline/train.py`); they write to `results/<run>/` rather than
`runs/<run>/` and do not read the YAML config.

```bash
# Train the model (from repo root)
accelerate launch train.py --run-name my_run_v1

# Train with Weights & Biases logging
accelerate launch train.py --run-name my_run_v1 --wandb

# Resume an interrupted run from its latest checkpoint in results/my_run_v1/
accelerate launch train.py --run-name my_run_v1 --resume

# Sample from a trained checkpoint
accelerate launch foregrounds_diffusion/sample.py \
  --checkpoint results/my_run_v1/model-20.pt \
  --batches 10 --batch-size 16 \
  --output data/low_pass/2mJy/samples.npy

# Sample with Weights & Biases logging
accelerate launch foregrounds_diffusion/sample.py \
  --checkpoint results/my_run_v1/model-20.pt \
  --batches 10 --batch-size 16 \
  --output data/low_pass/2mJy/samples.npy --wandb

# Fast DDIM sampling (250 steps instead of 1000, ~4× faster, no retraining needed)
accelerate launch foregrounds_diffusion/sample.py \
  --checkpoint results/my_run_v1/model-20.pt \
  --batches 10 --batch-size 16 \
  --output data/low_pass/2mJy/samples_ddim250.npy \
  --sampling-timesteps 250
```

## SLURM (cluster)

Edit the variables at the top of each script, then submit:

The SLURM scripts live in `scripts/slurm/`. Override `REPO_DIR`/`VENV_DIR`
(env vars or edit in-script) for the cluster paths, then submit:

```bash
sbatch scripts/slurm/train.sh    # single GPU, 1-12h wall time
sbatch scripts/slurm/sample.sh   # 4 GPUs, 2h wall time
```

Key variables in each script:

| Script | Variable | Purpose |
|---|---|---|
| `scripts/slurm/train.sh` | `RUN_NAME` | Run label; checkpoints go to `results/<RUN_NAME>/` |
| `scripts/slurm/train.sh` | `USE_WANDB` | `true` / `false` — enables `--wandb` flag |
| `scripts/slurm/sample.sh` | `CHECKPOINT` | Path to `.pt` checkpoint to sample from |
| `scripts/slurm/sample.sh` | `OUTPUT` | Output `.npy` path |
| `scripts/slurm/sample.sh` | `BATCHES` / `BATCH_SIZE` | Total samples = `BATCHES × BATCH_SIZE × 4` GPUs |
| `scripts/slurm/sample.sh` | `USE_WANDB` | `true` / `false` — enables `--wandb` flag |

## Weights & Biases

WandB is **opt-in**: pass `--wandb` or set `WANDB=1` in the environment. The API key must be set before running:

```bash
export WANDB_API_KEY=<your_key>   # add to ~/.bashrc for persistence
```

When enabled:
- **Training**: logs `train/loss` per step and CIB/tSZ sample image grids at each checkpoint milestone. WandB project: `cmb_foregrounds_diffusion`.
- **Sampling**: logs sample images and saves the output `.npy` as a WandB artifact.

## Architecture

The pipeline is:
1. **HEALPix maps** (on remote cluster) → **flat-sky patches** → **normalised `.npy` arrays** in `data/low_pass/{ptsrc}mJy/`
2. **Training** (`train.py`): loads CIB + tSZ `.npy` files, stacks into `(N, 2, 256, 256)` tensors, applies 8× augmentation (4 rotations × flip), and trains a U-Net DDPM via `denoising-diffusion-pytorch`
3. **Sampling** (`sample.py`): loads a checkpoint, generates batches of correlated CIB–tSZ pairs, saves as `.npy`

### Package modules (`foregrounds_diffusion/`)

| Module | Responsibility |
|---|---|
| `flatmaps.py` | Flat-sky Fourier utilities: `get_lxly`, `map2cl`, `cl2map`, `make_gaussian_realisation`, `radial_profile`, polarisation E/B↔Q/U conversion |
| `preprocessing.py` | Normalisation (`apply_maxmin_normalization`, `apply_stdnorm`), HEALPix patch extraction (`FlatCutter`, `get_patch_centers`), Fourier filtering (`get_lpf_hpf`, `wiener_filter`), dataset splitting |
| `statistics.py` | 2D Gaussian fitting (`gaussian`, `moments`, `fitgaussian`, `fitting_func`) and summary stats (`stats`) |
| `moments.py` | Power-spectrum summaries (`mean_cls`, `mean_cross_cls`) and higher-order moments (`compute_summed_moments`, `compute_cross_moments`) |
| `morphology.py` | Minkowski functionals (`compute_mfs`) and Minkowski tensors (`compute_minkowski_tensors`, `MINKOWSKI_TENSOR_DESCRIPTIONS`, `_tensor_W021/W200/W201`, `_eigendecompose_2x2`) |
| `stacking.py` | tSZ cluster stacking utilities (`select_snr_pixels`, `extract_cutouts`) |
| `masking.py` | Flat-sky peak masks (`get_peak_masks`, `inpaint_masked_regions`, `boundary_apod_mask`, `get_mask_using_gaussian_fitting`) and AGORA MDPL2 cluster/point-source masks (`get_point_source_mask_in_healpix`, `get_apodised_mdpl2_cluster_mask`, etc.) |
| `peak_counts.py` | Peak and minima counting statistics (Sabyr et al. 2024): `smooth_map`, `find_peaks`, `find_minima`, `count_peaks_binned`, `count_minima_binned`, `compute_peak_minima_counts`. numpy/scipy only. |
| `scattering_stats.py` | Scattering transform statistics: `compute_scattering_coefficients` (S1, S2), `compute_scattering_covariance` (C11, Cheng et al. backend only), `scattering_summary`. Requires Cheng et al. repo or `kymatio`. |
| `train.py` | Training entry point (not a library module — run via `accelerate launch`). CLI: `--run-name`, `--steps`, `--batch-size`, `--lr`, `--save-every`, `--num-samples` (0 skips milestone sampling), `--resume` (continue from latest `model-*.pt`), `--wandb` |
| `sample.py` | Sampling entry point. CLI: `--checkpoint`, `--batches`, `--batch-size`, `--output`, `--channels`, `--sampling-timesteps` (DDIM), `--rescale-cib`/`--rescale-tsz` (opt-in post-sampling scalar rescaling, paper §3.2), `--compile` (torch.compile the U-Net), `--wandb` |

### Key data conventions

- **Channels-last on disk**: raw `.npy` arrays are `(N, H, W, C)` — transposed to channels-first `(N, C, H, W)` before entering PyTorch
- **Preprocessing choices**: low-pass filter cuts `ℓ > 7000`; negative pixels from filtering artifacts are zeroed; point sources masked at 2 mJy threshold (masked pixels set to zero, not NaN)
- **Normalisation**: ⚠ **contested between notebooks.** Notebook 03 (the data producer) currently **z-scores both channels** and saves `_zscore_` files with `norm_params = [cib_mean, cib_std, tsz_mean, tsz_std]`; denormalise DDPM output with `denormalize_dm_maps` (`x·std+mean`, both channels). Older docs/notebooks (and `renormalize_dm_maps`) assume CIB min-max to `[0, 1]` (`_zero` suffix) + tSZ std-norm (`_norm` suffix). The checkpoint name `v3_zscore_...` favours z-score. Confirm against the on-disk files and the checkpoint's training normalisation before relying on either.
- **Train/val/test split**: 80/10/10 by default, seeded with `np.random.default_rng(seed=42)`
- **Model architecture**: U-Net with `dim=64`, `dim_mults=(1,2,4,8)`, `flash_attn=True`, 2 channels, 1000 diffusion timesteps

### Map parameters

`flatskymapparams = [nx, ny, dx, dy]` where `dx`, `dy` are pixel resolution in **arcminutes**. Patch size is 6°×6° projected to 256×256 pixels.

### Remote data

Raw files are on two Globus collections:

| Collection | Path | Files |
|---|---|---|
| **Agora** | `/components/cib/len/act/nocc/` | `agora_len_mag_cibmap_act_150ghz.fits` (Jy/sr) |
| **Agora** | `/components/tsz/len/` | `agora_ltszNG_bahamas80_bnd_unb_1.0e+12_1.0e+18_lensed.fits` (Compton-y) |
| **agora** | `halolc/` | `haloslc_rot_*.npz` (halo catalogue slices) |

See `docs/tutorials/01_halo_catalogue.ipynb` through `03_patch_extraction.ipynb` for the full preprocessing pipeline.

## Reference docs

- `docs/notebook_summaries.md` — description of every notebook in the repo, what each does, and how it maps to paper sections and `foregrounds_diffusion/` module functions.
- The **Future extensions** section of `README.md` — proposed extensions with scientific motivation and starting points (larger sky patches, conditional generation, additional foreground components, Bayesian integration, faster sampling, and more).
- `docs/tutorials/` — fourteen notebooks (01–14) covering the full pipeline from raw data to results. Notebooks 01–09 are the core tutorial sequence; 10–12 are post-paper extension evaluations (peak/minima counts, scattering transforms, Minkowski tensors); 13 is profiling/benchmarks and 14 generates the paper figures. Each has a summary cell describing inputs, outputs, key module functions, and the corresponding paper section.
