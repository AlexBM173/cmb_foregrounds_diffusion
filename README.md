# Learning Correlated Astrophysical Foregrounds with Denoising Diffusion Probabilistic Models

[![PyPI version](https://img.shields.io/pypi/v/foregrounds_diffusion?style=flat-square)](https://pypi.org/project/foregrounds_diffusion/)
[![ReadTheDocs](https://img.shields.io/readthedocs/cmb-foregrounds-diffusion?style=flat-square)](https://cmb-foregrounds-diffusion.readthedocs.io/)
[![CI Tests](https://img.shields.io/github/actions/workflow/status/AlexBM173/cmb_foregrounds_diffusion/tests.yml?branch=main&style=flat-square&label=tests)](https://github.com/AlexBM173/cmb_foregrounds_diffusion/actions?query=workflow%3Atests)

## Overview

This repository trains a **denoising diffusion probabilistic model (DDPM)** — a
generative model that learns to synthesise images by reversing a gradual
noising process — to produce realistic, statistically correlated maps of
extragalactic microwave-sky foregrounds. Rather than modelling each component
in isolation, the model generates all channels *jointly*, preserving the
physical cross-correlations between them.

Two generations of the model live in this repository:

- **Two-field (v4):** Cosmic Infrared Background (**CIB**) and thermal
  Sunyaev–Zeldovich (**tSZ**). This is the generation written up in the report.
- **Four-field (v5):** adds kinetic Sunyaev–Zeldovich (**kSZ**) and CMB-lensing
  convergence (**κ**).

The four channels are:

| Channel | What it is |
|---|---|
| **CIB** | Cosmic Infrared Background — redshifted thermal emission from dust in star-forming galaxies, integrated along the line of sight. Positively skewed, with bright point sources. |
| **tSZ** | thermal Sunyaev–Zeldovich effect — a CMB spectral distortion from inverse-Compton scattering of CMB photons off hot electrons in galaxy-cluster gas. At 150 GHz it appears as a temperature *decrement* (negative). |
| **kSZ** | kinetic Sunyaev–Zeldovich effect — a Doppler shift of the CMB imprinted by the bulk line-of-sight velocity of free electrons. Sign-symmetric, so its *mean* vanishes at clusters while its variance does not. |
| **κ** | CMB-lensing convergence — the projected matter over-density that gravitationally lenses the CMB. Dimensionless; traces the same structures that host clusters, so it correlates with tSZ. |

The trained model can serve as a differentiable prior in Bayesian inference
(e.g. CMB-lensing or kSZ pipelines), as a forecasting tool for survey noise and
component-separation fidelity, or as an augmentation source for testing
downstream analysis codes. Maps are 6°×6° flat-sky patches at 256×256 pixels.

This work is part of the MPhil in Data Intensive Science programme at the
University of Cambridge.

## Pipeline

The pipeline has four stages, each driven by a single YAML config (see
[Quickstart](#quickstart)):

1. **Preprocess** — full-sky **HEALPix** maps (the equal-area sphere
   pixelisation the raw inputs use) from the **AGORA** simulated-sky package
   (with cluster gas from the **BAHAMAS** hydrodynamic simulation) are masked at
   point-source and cluster thresholds, low-pass filtered at ℓ > 7000, cut into
   6°×6° flat-sky patches, and z-score normalised to training-ready `.npy`
   arrays.
2. **Train** — per-channel patches are stacked into `(N, C, 256, 256)` tensors
   (C = 2 or 4), augmented 8× (4 rotations × horizontal flip), and used to train
   a **U-Net** DDPM (a convolutional encoder–decoder with skip connections,
   using memory-efficient *flash attention*) via the
   `denoising-diffusion-pytorch` library. 1000 diffusion timesteps; `dim=64`
   for v4, `dim=96` for v5.
3. **Sample** — a trained checkpoint generates batches of jointly-correlated
   map stacks. The reported results use full 1000-step ancestral sampling;
   **DDIM** (a deterministic sampler that skips steps) is available for ~`1000/N`×
   faster generation at some quality cost.
4. **Evaluate** — a suite of summary statistics (power/cross spectra,
   higher-order moments, Minkowski functionals/tensors, cluster stacking,
   peak/minima counts, scattering transforms) is computed for the Agora test
   split, a Gaussian baseline, and the DDPM samples, with figures written per
   run.

> **Note on z-score normalisation and the sampler.** Because the maps are
> z-scored (not scaled to [-1, 1]), the stock diffusion sampler's hard clamp of
> the predicted clean image to [-1, 1] destroys the output. This repository
> ships `UnclampedGaussianDiffusion` (`foregrounds_diffusion/sample.py`) to
> remove that clamp.

## Repository layout

The importable package is `foregrounds_diffusion/`; the config-driven stage
runners live in `pipeline/`.

| Module | Responsibility |
|---|---|
| `foregrounds_diffusion/flatmaps.py` | Flat-sky Fourier utilities: `map2cl`/`cl2map`, `make_gaussian_realisation`, `make_correlated_gaussian_fields`, `radial_profile`, E/B↔Q/U conversion. |
| `foregrounds_diffusion/preprocessing.py` | Normalisation (`apply_maxmin_normalization`, `apply_stdnorm`), HEALPix patch extraction (`FlatCutter`, `get_patch_centers`), Fourier filtering (`get_lpf_hpf`, `wiener_filter`), dataset splitting. |
| `foregrounds_diffusion/statistics.py` | 2D Gaussian fitting (`fitgaussian`) and summary statistics. |
| `foregrounds_diffusion/moments.py` | Power-spectrum summaries (`mean_cls`, `mean_cross_cls`) and higher-order moments (`compute_summed_moments`, `compute_cross_moments`). |
| `foregrounds_diffusion/morphology.py` | Minkowski functionals (`compute_mfs`) and Minkowski tensors (`compute_minkowski_tensors`). |
| `foregrounds_diffusion/peak_counts.py` | Peak/minima counts (Sabyr et al. 2024); numpy/scipy only. |
| `foregrounds_diffusion/scattering_stats.py` | Scattering-transform statistics (S1/S2, covariance C01/C11). Cheng et al. or `kymatio` backend. |
| `foregrounds_diffusion/stacking.py` | tSZ cluster stacking (`select_snr_pixels`, `extract_cutouts`). |
| `foregrounds_diffusion/masking.py` | Flat-sky peak masks and AGORA/MDPL2 cluster & point-source masks. |
| `foregrounds_diffusion/plot_style.py` | Shared publication plot style (`apply`). |
| `foregrounds_diffusion/sample.py` | Sampling entry point + `UnclampedGaussianDiffusion`. |
| `pipeline/train.py` | Training implementation (invoked by `run.py train`). |
| `pipeline/evaluate.py` | Cached evaluation-statistics engine (invoked by `run.py evaluate`). |
| `pipeline/rundir.py` | `runs/<name>/` layout, config stamping, git provenance. |

## Installation

### From PyPI

```bash
pip install foregrounds_diffusion
```

### From source

```bash
git clone https://github.com/AlexBM173/cmb_foregrounds_diffusion.git
cd cmb_foregrounds_diffusion
pip install -e ".[dev]"
```

### Optional extras

```bash
pip install foregrounds_diffusion[dev]    # pytest + coverage
pip install foregrounds_diffusion[fast]   # numba + quantimpy (Minkowski functionals)
pip install foregrounds_diffusion[docs]   # Sphinx toolchain
pip install foregrounds_diffusion[wandb]  # Weights & Biases logging
```

### Scattering-transform backend (optional)

The scattering-transform statistics prefer the Cheng et al. backend. It is not
on PyPI, so clone it into the repository root (the loader looks for
`scattering_transform/scattering/`); if it is absent, the code falls back to
`kymatio`.

```bash
git clone https://github.com/SihaoCheng/scattering_transform.git
git -C scattering_transform checkout 04f36a6   # pinned commit used for the results
```

## Quickstart

One YAML file drives every stage. `config/default.yaml` is the fully documented
template; `config/v4_eval.yaml` (two-field) and `config/v5_4ch.yaml` (four-field)
are the run configs behind the results. Each run writes all artefacts to
`runs/<run_name>/` alongside a copy of the config, its SHA256 hash, and the git
commit, so every result traces back to an exact configuration and code state.

```bash
cp config/default.yaml config/my_run.yaml     # edit run_name, paths, settings
python config/validate.py config/my_run.yaml  # validate before running

python run.py preprocess --config config/my_run.yaml   # checks patches exist (see Data)
python run.py train      --config config/my_run.yaml   # → runs/<run>/checkpoints/
python run.py sample     --config config/my_run.yaml   # → runs/<run>/samples/
python run.py evaluate   --config config/my_run.yaml   # → runs/<run>/{stats,plots}/
```

Every field the config accepts is documented — inline in `config/default.yaml`
(the annotated template) and as a reference table in
`docs/guides/configuration.rst`. Add `--dry-run` to any stage to print what
would run without executing it. The
`evaluate` stage caches every statistic under `runs/<run>/stats/*.npz`, so
re-running it only regenerates figures unless a parameter changes.

The legacy flag-based entry points still work — the root `train.py` is a shim
over `pipeline/train.py`, and `foregrounds_diffusion/sample.py` runs standalone
— but they write to `results/<run>/` and do not read the YAML config. Prefer the
config-driven workflow above.

## Data

### Globus collections

The raw simulation files are distributed across two Globus collections; you need
a Globus account and Globus Connect Personal to transfer them.

**Collection `Agora`** — full-sky HEALPix maps (NSIDE=8192):

| File | Globus path | Units |
|---|---|---|
| `agora_len_mag_cibmap_act_150ghz.fits` | `/components/cib/len/act/nocc/` | Jy/sr |
| `agora_ltszNG_bahamas80_bnd_unb_1.0e+12_1.0e+18_lensed.fits` | `/components/tsz/len/` | Compton-y |

kSZ and κ full-sky maps come from the corresponding AGORA components for the
four-field run.

**Collection `agora`** — halo catalogue slices `haloslc_rot_*.npz` under
`halolc/`, concatenated and filtered to M₅₀₀c ≥ 3×10¹⁴ M☉ for cluster masking.

### Preprocessing

`python run.py preprocess` only **verifies** that the training-ready patches
named by the config exist — it does not itself produce them. Patch production
from the raw full-sky maps runs as standalone scripts (the path used for the
actual runs), which currently use hardcoded paths rather than the YAML; the
`data:`/`preprocessing:` config sections *document* the choices they bake in:

```
scripts/vm_preprocessing/nb01_run.py           # filter halo lightcone
scripts/vm_preprocessing/nb02_run.py           # 2-channel masking (CIB + tSZ)
scripts/vm_preprocessing/nb02b_mask_ksz_kappa.py  # kSZ + κ masking (4-channel)
scripts/vm_preprocessing/nb03_run.py           # 2-channel patch extraction + norm
scripts/vm_preprocessing/nb03b_extract_4ch.py  # 4-channel patch extraction + norm
```

The tutorial notebooks `docs/tutorials/01_halo_catalogue.ipynb` →
`03_patch_extraction.ipynb` mirror the same steps with explanation and are the
best starting point for understanding the pipeline. Training-ready arrays land
in `data/low_pass/<ptsrc>mJy/` (e.g. `CIB_map_150GHz_256_st6_zscore_2mJy_lp.npy`,
`norm_params_2mJy.npy`). These files are large and git-ignored; regenerate them
from the raw maps or pull them from your own artifact store.

## Running each stage

### Training

```bash
python run.py train --config config/v5_4ch.yaml
```

Checkpoints and sample previews land in `runs/<run>/checkpoints/`. The split is
config-controlled (`data.seed`, `data.train_size`) and must match between
training and evaluation — both default to seed 42 and an 0.8 train fraction
(560 train / 141 test of 701 patches).

### Sampling

```bash
python run.py sample --config config/v5_4ch.yaml
```

For faster DDIM sampling, set `sampling.ddim_steps` in the config (e.g. 250).
The reported v4/v5 results use full 1000-step ancestral sampling, which runs at
roughly 10–15 s/patch on an A100.

### Evaluation

```bash
python run.py evaluate --config config/v5_4ch.yaml
```

Figures are written to `runs/<run>/plots/` with a `2f_`/`4f_` prefix indicating
the field count; a one-line-per-statistic `summary.md` lands in
`runs/<run>/stats/`.

### Weights & Biases (optional)

WandB is opt-in via `--wandb` (or `wandb.enabled` in the config). Set
`WANDB_API_KEY` in your environment first. Training logs per-step loss and
milestone sample grids; sampling logs image grids and the output `.npy` as an
artifact.

## Reproducing the results

The two run directories ship their cached statistics, so the figures can be
regenerated without re-sampling:

```bash
python run.py evaluate --config config/v4_eval.yaml   # two-field
python run.py evaluate --config config/v5_4ch.yaml    # four-field
```

With the caches present this recomputes nothing (`N/N cached, 0 recomputed`) and
only rewrites figures — so plot styling can be iterated for free. See
`CHANGELOG.md` for per-generation provenance.

## Future extensions

Directions that build on the current two-/four-field model, roughly ordered by
tractability given the existing codebase. Items already realised are noted.

- **Larger sky patches.** Training uses 6°×6° patches and degrades at 10°×10°
  because the number of independent patches falls as patch-area⁻¹. Patch-diffusion
  or hierarchical coarse→fine generation could yield coherent realisations over
  survey-scale areas.
- **Conditional generation on parameters.** Condition the U-Net on σ₈, Ωm, or
  feedback amplitude (classifier-free guidance) so the prior can marginalise over
  simulation/real-sky mismatch. Requires a multi-cosmology training set (WebSky,
  Agora/BAHAMAS variants).
- **More foreground components.** kSZ and CMB-lensing κ are already added in the
  four-field (v5) model; radio galaxies and Galactic dust remain. Sparse,
  point-source-like fields (radio) are likely better handled by compositing a
  parametric source model onto the DDPM background than by learning them jointly.
- **Multi-frequency CIB SED.** Jointly generate CIB at 95/150/857 GHz with an
  inter-frequency cross-power loss, or condition on a continuous frequency
  embedding to interpolate to unseen bands — needed for ILC component separation.
- **DDPM prior in Bayesian inference.** The reverse process exposes the score
  ∇log p(x); surfacing it as a standalone gradient and combining it with a
  Gaussian likelihood would let the model act as a foreground prior in
  lensing/kSZ pipelines (e.g. MUSE).
- **Faster sampling.** DDIM is available now (`sampling.ddim_steps`); distilling
  to a consistency model or retraining with flow matching would cut 1000-step
  generation to a handful of steps for covariance/inference loops.
- **Better extreme-value pixels.** The dominant failure mode is under-production
  of rare high-amplitude pixels (massive clusters, bright sources), which drives
  the tSZ tail and cluster-stacking deficit. Importance-sampling high-SNR patches,
  `min_snr_loss_weight=True`, or conditional cluster inpainting are candidate fixes.
- **Real observational data.** Apply `FlatCutter` to SPT-3G/ACT maps and compare
  spectra, histograms, and Minkowski functionals against DDPM samples to test the
  Agora simulation and the model together (requires noise/beam modelling).
- **Paper-faithful preprocessing.** Bring the masking pipeline fully in line with
  the published method: flux-based (mJy) point-source masking, Gaussian inpainting
  of masked regions (vs zero-fill), and θ₅₀₀c-scaled cluster mask radii.
- **Extra validation statistics (implemented).** Wavelet scattering transforms
  (`scattering_stats.py`) and Minkowski tensors (`morphology.py`) were added as
  non-Gaussian diagnostics beyond the paper's power-spectrum/MF/moment suite.

## Compute environments

- **Local / single machine** — the Quickstart commands above.
- **Google Cloud + Colab Pro+** — preprocessing on a VM, training/sampling on a
  Colab A100 with artifacts in a GCS bucket; the resumable launchers are in
  `scripts/colab/`. See `docs/guides/gcp_colab.rst`.
- **HPC / SLURM** — batch scripts in `scripts/slurm/` wrap the same entry
  points; override `REPO_DIR`/`VENV_DIR` for your cluster. See
  `docs/guides/hpc_slurm.rst`.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v            # test suite
pre-commit install          # ruff + nbstripout + hygiene hooks
sphinx-build docs/ docs/_build/html   # build docs locally ([docs] extra)
```

Documentation deploys to https://cmb-foregrounds-diffusion.readthedocs.io/ on
each push to `main`.

## Contributing

Contributions are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the
developer quick start (setup, tests, style, PR flow) and
`docs/guides/contributing.rst` for the full guide on adding modules, tutorials,
and config settings.

## Citation

If you use this software, please cite both the codebase (see
[`CITATION.cff`](CITATION.cff)) and the accompanying thesis:

```bibtex
@software{foregrounds_diffusion,
  author = {Prabhu, Karthik and Raghunathan, Srinivasan and
            Blake Martín, Alex and Bolliet, Boris},
  title  = {foregrounds_diffusion: DDPM pipeline for correlated CIB and tSZ CMB foregrounds},
  year   = {2026},
  url    = {https://github.com/AlexBM173/cmb_foregrounds_diffusion},
}

@thesis{BlakeMartin2026,
  author = {Alex Blake Martín},
  title  = {Learning Correlated Astrophysical Foregrounds with Denoising Diffusion Probabilistic Models},
  year   = {2026},
  school = {University of Cambridge},
  type   = {MPhil thesis},
}
```

## License

MIT License — see the LICENSE file.

## Acknowledgements

This project builds on the original
[`diffusion_model`](https://github.com/Karthikprabhu22/diffusion_model)
codebase by **Karthik Prabhu** (University of California, Davis) and
**Srinivasan Raghunathan**, which established the DDPM pipeline for correlated
CMB foregrounds that this repository extends. It also relies on the
[`denoising-diffusion-pytorch`](https://github.com/lucidrains/denoising-diffusion-pytorch)
library by Phil Wang (lucidrains).

The MPhil project was supervised by **Boris Bolliet** (University of Cambridge).
