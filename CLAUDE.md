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
accelerate launch foregrounds_diffusion/train.py

# Sample from a trained checkpoint
accelerate launch foregrounds_diffusion/sample.py \
  --checkpoint results/model-14.pt \
  --batches 5 --batch-size 16 \
  --output data/low_pass/2mJy/samples.npy
```

## Architecture

The pipeline is:
1. **HEALPix maps** (on remote cluster) → **flat-sky patches** → **normalised `.npy` arrays** in `data/low_pass/{ptsrc}mJy/`
2. **Training** (`train.py`): loads CIB + tSZ `.npy` files, stacks into `(N, 2, 256, 256)` tensors, applies 8× augmentation (4 rotations × flip), and trains a U-Net DDPM via `denoising-diffusion-pytorch`
3. **Sampling** (`sample.py`): loads a checkpoint, generates batches of correlated CIB–tSZ pairs, saves as `.npy`

### Package modules (`foregrounds_diffusion/`)

| Module | Responsibility |
|---|---|
| `flatmaps.py` | Flat-sky Fourier utilities: `get_lxly`, `map2cl`, `cl2map`, `make_gaussian_realisation`, `radial_profile`, polarisation E/B↔Q/U conversion |
| `preprocessing.py` | Normalisation (`apply_maxmin_normalization`, `apply_stdnorm`), HEALPix patch extraction (`FlatCutter`, `get_patch_centers`), Fourier filtering (`get_lpf_hpf`, `wiener_filter`), masking, train/val/test splitting |
| `statistics.py` | 2D Gaussian fitting (`fitgaussian`, `fitting_func`) and summary stats |
| `get_cluster_source_mask_for_agora.py` | Apodised cluster/point-source mask generation for AGORA MDPL2 maps |
| `train.py` | Training entry point (not a library module — run via `accelerate launch`) |
| `sample.py` | Sampling entry point with CLI (`--checkpoint`, `--batches`, `--batch-size`, `--output`, `--channels`) |
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
- `docs/tutorials/` — nine stub notebooks (01–09) laying out a self-contained tutorial sequence from raw data to results, one per pipeline stage. Each has a summary cell describing inputs, outputs, key module functions, and the corresponding paper section.
