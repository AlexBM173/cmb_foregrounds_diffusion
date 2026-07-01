# Learning Correlated Astrophysical Foregrounds with Denoising Diffusion Probabilistic Models

[![PyPI version](https://img.shields.io/pypi/v/foregrounds_diffusion?style=flat-square)](https://pypi.org/project/foregrounds_diffusion/)
[![ReadTheDocs](https://img.shields.io/readthedocs/cmb-foregrounds-diffusion?style=flat-square)](https://cmb-foregrounds-diffusion.readthedocs.io/)
[![CI Tests](https://img.shields.io/github/actions/workflow/status/AlexBM173/cmb_foregrounds_diffusion/tests.yml?branch=main&style=flat-square&label=tests)](https://github.com/AlexBM173/cmb_foregrounds_diffusion/actions?query=workflow%3Atests)

## Overview

This repository implements a denoising diffusion probabilistic model (DDPM) pipeline for generating realistic synthetic maps of extragalactic cosmic microwave background (CMB) foregrounds. The model learns to generate correlated pairs of Cosmic Infrared Background (CIB) and thermal Sunyaev–Zeldovich (tSZ) maps from AGORA cosmological simulations, reproducing the statistical properties—power spectra, higher-order moments, and morphology—of the training data while preserving physically important cross-channel correlations.

The DDPM can be deployed as a differentiable prior in Bayesian inference pipelines (e.g., CMB lensing or kSZ analyses), as a tool for forecasting survey noise properties and component separation fidelity, or as a data augmentation pipeline for testing downstream analysis codes. The model is trained on 6°×6° flat-sky patches at 256×256 pixel resolution and includes options for fast sampling via DDIM acceleration.

This work is part of the MPhil in Data Intensive Science programme at the University of Cambridge.

## Architecture

The pipeline consists of three stages:

1. **Data Preparation**: Raw HEALPix maps from the AGORA BAHAMAS simulation (hosted on Globus) are patched into 6°×6° flat-sky cutouts, masked at point-source and cluster thresholds, low-pass filtered at ℓ > 7000, and normalised to training-ready `.npy` arrays.

2. **Training**: Paired CIB and tSZ patches are stacked into 2-channel tensors of shape (N, 2, 256, 256), augmented with 4 rotations × horizontal flip (8× total), and used to train a U-Net-based DDPM via the denoising-diffusion-pytorch library. The U-Net architecture has `dim=64`, `dim_mults=(1,2,4,8)`, and flash attention is enabled for efficiency. The diffusion schedule uses 1000 timesteps with a sigmoid noise schedule.

3. **Sampling**: A trained checkpoint generates batches of correlated CIB–tSZ map pairs. Standard sampling uses full DDPM (1000 reverse steps); DDIM sampling with fewer timesteps (e.g., 250 steps) is ~4× faster with minimal quality loss.

### Package Modules

The `foregrounds_diffusion/` package provides the following modules:

| Module | Responsibility |
|---|---|
| `flatmaps.py` | Flat-sky Fourier utilities: power-spectrum conversion (`map2cl`, `cl2map`), map generation (`make_gaussian_realisation`), radial profiling, polarisation E/B↔Q/U conversion. |
| `preprocessing.py` | Data normalisation (`apply_maxmin_normalization`, `apply_stdnorm`), HEALPix patch extraction (`FlatCutter`, `get_patch_centers`), Fourier filtering (`get_lpf_hpf`, `bandpass_filter`, `wiener_filter`), and dataset splitting. |
| `statistics.py` | 2D Gaussian fitting (`gaussian`, `moments`, `fitgaussian`) and summary statistics (`stats`). |
| `moments.py` | Power-spectrum summaries (`mean_cls`, `mean_cross_cls`) and higher-order moments (`compute_summed_moments`, `compute_cross_moments`). |
| `morphology.py` | Minkowski functionals (`compute_mfs`) and Minkowski tensors (`compute_minkowski_tensors`). |
| `stacking.py` | tSZ cluster stacking utilities (`select_snr_pixels`, `extract_cutouts`). |
| `masking.py` | Flat-sky peak masks (`get_peak_masks`, `inpaint_masked_regions`) and AGORA MDPL2 cluster/point-source masks (`get_point_source_mask_in_healpix`, `get_apodised_mdpl2_cluster_mask`, etc.). |
| `peak_counts.py` | Peak and minima counting statistics following Sabyr et al. (2024): `smooth_map`, `find_peaks`, `count_peaks_binned`, `compute_peak_minima_counts`. Requires only numpy/scipy. |
| `scattering_stats.py` | Scattering transform statistics: `compute_scattering_coefficients` (S1, S2), `compute_scattering_covariance` (C11), `scattering_summary`. Supports Cheng et al. or kymatio backends. |
| `train.py` | Training entry point (run via `accelerate launch train.py`). CLI: `--run-name`, `--steps`, `--batch-size`, `--lr`, `--wandb`. |
| `sample.py` | Sampling entry point (run via `accelerate launch sample.py`). CLI: `--checkpoint`, `--batches`, `--batch-size`, `--output`, `--sampling-timesteps` (DDIM), `--wandb`. |

## Installation

### From PyPI

```bash
pip install foregrounds_diffusion
```

### Optional Extras

The package includes optional dependencies for additional functionality:

```bash
# Development and testing
pip install foregrounds_diffusion[dev]

# Acceleration via Numba and quantimpy (Minkowski functionals)
pip install foregrounds_diffusion[fast]

# Building Sphinx documentation locally
pip install foregrounds_diffusion[docs]

# All of the above
pip install foregrounds_diffusion[dev,fast,docs]
```

### From Source

Clone the repository and install in editable mode:

```bash
git clone https://github.com/AlexBM173/cmb_foregrounds_diffusion.git
cd cmb_foregrounds_diffusion
pip install -e ".[dev]"
```

## Data

### Globus Collections

The raw simulation files are distributed across two Globus collections. You will need a Globus account and the Globus Connect Personal client to transfer them.

**Collection: Agora** — full-sky HEALPix simulation maps (NSIDE=8192):

| File | Globus path | Units |
|---|---|---|
| `agora_len_mag_cibmap_act_150ghz.fits` | `/components/cib/len/act/nocc/` | Jy/sr |
| `agora_len_mag_cibmap_act_150ghz.fits` | `/components/cib/len/act/uK/` | µK |
| `agora_ltszNG_bahamas80_bnd_unb_1.0e+12_1.0e+18_lensed.fits` | `/components/tsz/len/` | Compton-y |

The preprocessing pipeline uses the Jy/sr CIB map and the Compton-y tSZ map. The µK CIB variant is provided for reference.

**Collection: agora** — halo catalogue slices:

| Files | Globus path |
|---|---|
| `haloslc_rot_*.npz` | `halolc/` |

The catalogue slices are concatenated and filtered by `docs/tutorials/01_halo_catalogue.ipynb` to produce `data/halo_catalogue/halo_catalogue_m500gt3e14.npz`, which is used by the cluster masking step.

### Preprocessing

The full preprocessing pipeline runs across the first three tutorial notebooks:

1. **`01_halo_catalogue.ipynb`** — concatenates halo catalogue slices, filters to M_500c ≥ 3×10¹⁴ M☉, saves `data/halo_catalogue/halo_catalogue_m500gt3e14.npz`
2. **`02_masking.ipynb`** — loads raw FITS maps, applies 2 mJy point-source masking and apodised cluster masks, saves `data/cib_150_masked.fits` and `data/tsz_150_masked.fits`
3. **`03_patch_extraction.ipynb`** — extracts 6°×6° flat-sky patches at 256×256 resolution, low-pass filters at ℓ = 7000, normalises (CIB: z-score; tSZ: z-score), saves training-ready `.npy` arrays

**Expected local data layout after preprocessing:**

```
data/
├── agora_len_mag_cibmap_act_150ghz.fits         # raw CIB map (from Globus)
├── agora_ltszNG_bahamas80_...lensed.fits         # raw tSZ map (from Globus)
├── cib_150_masked.fits                           # after 02_masking
├── tsz_150_masked.fits                           # after 02_masking
├── halo_catalogue/
│   └── halo_catalogue_m500gt3e14.npz             # after 01_halo_catalogue
└── low_pass/
    └── 2mJy/
        ├── CIB_map_150GHz_256_st6_zscore_2mJy_lp.npy   # training-ready CIB
        ├── tSZ3_map_150GHz_256_st6_zscore_2mJy_lp.npy  # training-ready tSZ
        ├── gaussian_cib_tsz_2mJy_lp.npy                # Gaussian baseline
        └── norm_params_2mJy.npy                         # normalisation stats
```

## Quick Start

### Training

Train a new model with the default configuration:

```bash
accelerate launch foregrounds_diffusion/train.py --run-name my_run_v1
```

To enable Weights & Biases logging (see the [Weights & Biases](#weights--biases) section for setup):

```bash
accelerate launch foregrounds_diffusion/train.py --run-name my_run_v1 --wandb
```

Checkpoints are saved to `results/my_run_v1/model-{step}.pt` every 5 steps (configurable via `--checkpoint-freq`).

### Sampling with Full DDPM (1000 steps)

Generate samples from a trained checkpoint:

```bash
accelerate launch foregrounds_diffusion/sample.py \
  --checkpoint results/my_run_v1/model-20.pt \
  --batches 10 \
  --batch-size 16 \
  --output data/low_pass/2mJy/samples.npy
```

This generates 10 × 16 = 160 correlated CIB–tSZ patch pairs and saves them as a single `.npy` file with shape (160, 2, 256, 256).

### Sampling with DDIM (250 steps, ~4× faster)

Use DDIM for faster sampling with minimal quality loss:

```bash
accelerate launch foregrounds_diffusion/sample.py \
  --checkpoint results/my_run_v1/model-20.pt \
  --batches 10 \
  --batch-size 16 \
  --output data/low_pass/2mJy/samples_ddim250.npy \
  --sampling-timesteps 250
```

The `--sampling-timesteps` argument accepts any integer < 1000. Typical choices are 50 (very fast, ~1s/patch), 100 (fast, ~2s/patch), or 250 (good quality/speed trade-off, ~0.5s/patch).

## Weights & Biases

Weights & Biases (WandB) integration is **optional** and off by default. Both training and sampling can log to WandB with the `--wandb` flag.

### Setup

Set your WandB API key before running:

```bash
export WANDB_API_KEY=<your_key>
```

To persist the key across sessions, add the line to your `~/.bashrc` or `~/.zshrc`:

```bash
echo 'export WANDB_API_KEY=<your_key>' >> ~/.bashrc
```

### Logging

When enabled with the `--wandb` flag:

**Training:**
- Logs `train/loss` per step
- Logs CIB and tSZ sample image grids at each checkpoint milestone
- Project name: `cmb_foregrounds_diffusion`

**Sampling:**
- Logs sample image grids (visualisation of generated CIB and tSZ patches)
- Saves the output `.npy` file as a WandB artifact for lineage tracking

### Example with WandB

```bash
export WANDB_API_KEY=<your_key>
accelerate launch foregrounds_diffusion/train.py --run-name my_run_v1 --wandb
```

## SLURM and HPC Clusters

For users with access to HPC clusters running SLURM, two shell scripts are provided to streamline job submission.

### Training on a Single GPU

Edit `train_slurm.sh` to configure your run, then submit:

```bash
# Edit the variables at the top of the file
vim train_slurm.sh

# Submit the job
sbatch train_slurm.sh
```

**Configuration Variables in `train_slurm.sh`:**

| Variable | Default | Purpose |
|---|---|---|
| `RUN_NAME` | `run_v1` | Run label; checkpoints saved to `results/<RUN_NAME>/` |
| `USE_WANDB` | `false` | Set to `true` to enable Weights & Biases logging |

The script allocates:
- 1 GPU (Ampere, A100)
- 8 CPU cores
- 128 GB RAM
- 1–12 hour wall time

### Sampling on Four GPUs

Edit `sample_slurm.sh` to configure your sampling run, then submit:

```bash
# Edit the variables at the top of the file
vim sample_slurm.sh

# Submit the job
sbatch sample_slurm.sh
```

**Configuration Variables in `sample_slurm.sh`:**

| Variable | Default | Purpose |
|---|---|---|
| `CHECKPOINT` | `results/run_v1/model-20.pt` | Path to trained checkpoint |
| `OUTPUT` | `data/low_pass/2mJy/samples.npy` | Output `.npy` file path |
| `BATCHES` | `10` | Number of sampling batches |
| `BATCH_SIZE` | `16` | Samples per batch per GPU; total samples = `BATCHES × BATCH_SIZE × 4` |
| `SAMPLING_TIMESTEPS` | (empty) | Leave empty for full DDPM (1000 steps); set to an integer (e.g., `250`) for DDIM |
| `USE_WANDB` | `false` | Set to `true` to enable Weights & Biases logging |

The script allocates:
- 4 GPUs (Ampere, A100)
- 8 CPU cores per GPU
- 128 GB RAM
- 2 hour wall time

### Multi-GPU DDIM Sampling Example

To sample 640 CIB–tSZ patches with DDIM in 250 steps on the cluster:

```bash
# Edit sample_slurm.sh:
# BATCHES=10
# BATCH_SIZE=16
# SAMPLING_TIMESTEPS=250

sbatch sample_slurm.sh
# Total samples generated: 10 × 16 × 4 GPUs = 640 patches
# Expected wall time: ~30 minutes for 250-step DDIM sampling
```

## Development

### Running Tests

Install development dependencies and run the test suite:

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

### Pre-commit Hooks

Install pre-commit hooks to lint and format code before each commit:

```bash
pre-commit install
```

The hooks run ruff for linting and formatting, plus checks for trailing whitespace, YAML/TOML validity, and merge conflicts.

### Building Documentation Locally

Install documentation dependencies and build the Sphinx docs:

```bash
pip install -e ".[docs]"
sphinx-build docs/ docs/_build/html
```

The built HTML documentation will be in `docs/_build/html/index.html`. Alternatively, use:

```bash
make -C docs html
```

Documentation is automatically deployed to https://cmb-foregrounds-diffusion.readthedocs.io/ on each push to the `main` branch.

## Citation

If you use this code in your research, please cite:

```bibtex
@thesis{BlakeMartin2026,
  author    = {Alex Blake Martin},
  title     = {Learning Correlated Astrophysical Foregrounds with Denoising Diffusion Probabilistic Models},
  year      = {2026},
  school    = {University of Cambridge},
  type      = {MPhil thesis},
}
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
