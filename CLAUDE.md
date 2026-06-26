# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Activate the virtual environment before running any Python code:
```bash
# From the home directory (~):
source activate_diffusion_project_env.sh
```

## Common Commands

```bash
# Train the model (from repo root)
accelerate launch train.py --run-name my_run_v1

# Train with Weights & Biases logging
accelerate launch train.py --run-name my_run_v1 --wandb

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
```

## SLURM (cluster)

Edit the variables at the top of each script, then submit:

```bash
sbatch train_slurm.sh    # single GPU, 1-12h wall time
sbatch sample_slurm.sh   # 4 GPUs, 2h wall time
```

Key variables in each script:

| Script | Variable | Purpose |
|---|---|---|
| `train_slurm.sh` | `RUN_NAME` | Run label; checkpoints go to `results/<RUN_NAME>/` |
| `train_slurm.sh` | `USE_WANDB` | `true` / `false` — enables `--wandb` flag |
| `sample_slurm.sh` | `CHECKPOINT` | Path to `.pt` checkpoint to sample from |
| `sample_slurm.sh` | `OUTPUT` | Output `.npy` path |
| `sample_slurm.sh` | `BATCHES` / `BATCH_SIZE` | Total samples = `BATCHES × BATCH_SIZE × 4` GPUs |
| `sample_slurm.sh` | `USE_WANDB` | `true` / `false` — enables `--wandb` flag |

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
| `morphology.py` | Minkowski functionals (`compute_mfs`) and Minkowski tensors (`compute_minkowski_tensors`, `MINKOWSKI_TENSOR_DESCRIPTIONS`, `_tensor_W012/W200/W201`, `_eigendecompose_2x2`) |
| `stacking.py` | tSZ cluster stacking utilities (`select_snr_pixels`, `extract_cutouts`) |
| `masking.py` | Flat-sky peak masks (`get_peak_masks`, `inpaint_masked_regions`, `boundary_apod_mask`, `get_mask_using_gaussian_fitting`) and AGORA MDPL2 cluster/point-source masks (`get_point_source_mask_in_healpix`, `get_apodised_mdpl2_cluster_mask`, etc.) |
| `peak_counts.py` | Peak and minima counting statistics (Sabyr et al. 2024): `smooth_map`, `find_peaks`, `find_minima`, `count_peaks_binned`, `count_minima_binned`, `compute_peak_minima_counts`. numpy/scipy only. |
| `scattering_stats.py` | Scattering transform statistics: `compute_scattering_coefficients` (S1, S2), `compute_scattering_covariance` (C11, Cheng et al. backend only), `scattering_summary`. Requires Cheng et al. repo or `kymatio`. |
| `train.py` | Training entry point (not a library module — run via `accelerate launch`). CLI: `--run-name`, `--steps`, `--batch-size`, `--lr`, `--wandb` |
| `sample.py` | Sampling entry point. CLI: `--checkpoint`, `--batches`, `--batch-size`, `--output`, `--channels`, `--wandb` |
| `redundant/` | Old scripts kept for reference; not part of the active codebase |

### Key data conventions

- **Channels-last on disk**: raw `.npy` arrays are `(N, H, W, C)` — transposed to channels-first `(N, C, H, W)` before entering PyTorch
- **Preprocessing choices**: low-pass filter cuts `ℓ > 7000`; negative pixels from filtering artifacts are zeroed; point sources masked at 2 mJy threshold (masked pixels set to zero, not NaN)
- **Normalisation**: CIB uses min-max to `[0, 1]` (`_zero` suffix files); tSZ uses std-normalisation (`_norm` suffix files)
- **Train/val/test split**: 80/10/10 by default, seeded with `np.random.default_rng(seed=42)`
- **Model architecture**: U-Net with `dim=64`, `dim_mults=(1,2,4,8)`, `flash_attn=True`, 2 channels, 1000 diffusion timesteps

### Map parameters

`flatskymapparams = [nx, ny, dx, dy]` where `dx`, `dy` are pixel resolution in **arcminutes**. Patch size is 6°×6° projected to 256×256 pixels.

### Remote data

Raw AGORA maps live on the cluster at:
```
/sptlocal/analysis/ymap/sims/mdpl2/data/v0.7/bahamas80_scal1.000/mask_radio_cib_2.0mjy/
```
The `preprocessing.ipynb` notebook documents the full preprocessing pipeline from raw HEALPix maps to the training `.npy` arrays.

## Reference docs

- `docs/notebook_summaries.md` — description of every notebook in the repo, what each does, and how it maps to paper sections and `foregrounds_diffusion/` module functions.
- `docs/paper_code_inconsistencies.md` — documented inconsistencies between the paper (Prabhu et al.) and the current codebase, covering masking, normalisation, augmentation, post-sampling rescaling, and noise schedule parameters.
- `docs/potential_extensions.md` — ten proposed extensions with scientific motivation, implementation starting points, and known obstacles. Covers larger sky patches, conditional generation, additional foreground components, Bayesian integration, faster sampling, and more.
- `docs/tutorials/` — twelve notebooks (01–12) covering the full pipeline from raw data to results. Notebooks 01–09 are the core tutorial sequence; 10–12 are post-paper extension evaluations (peak/minima counts, scattering transforms, Minkowski tensors). Each has a summary cell describing inputs, outputs, key module functions, and the corresponding paper section.
