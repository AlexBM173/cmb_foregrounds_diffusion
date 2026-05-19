# Notebook Summaries

Descriptions of each notebook in the codebase, their relation to the paper ("Learning Correlated Astrophysical Foregrounds with Denoising Diffusion Probabilistic Models", Prabhu et al.), and their relation to the `foregrounds_diffusion/` module.

---

## `preprocessing.ipynb` (repo root)

The entry-point for all data preparation. Builds the halo catalogue from 197 lightcone slice files, applies point-source and cluster masking to the full-sky HEALPix maps, low-pass filters at ℓ = 7000, extracts 6°×6° flat-sky patches at 256×256 resolution, min-max normalises them to [0, 1], and saves them as `.npy` files — producing the training data consumed by `train.py`.

**Paper relation:** Implements the pipeline described in §2 and Figure 1 (the four-step flowchart). It is where the discrepancies noted in `paper_code_inconsistencies.md` live: point sources are masked by sigma-clipping in K_CMB rather than via a calibrated mJy threshold, and masked pixels are zeroed rather than Gaussian-inpainted.

**Module relation:** Calls `get_patch_centers` and `FlatCutter.rotate_to_pole_and_interpolate` from `preprocessing.py`, and `apply_maxmin_normalization` (defined inline here, later extracted into `preprocessing.py`). The cluster masking uses `get_apodised_mdpl2_cluster_mask` from `get_cluster_source_mask_for_agora.py`.

---

## `docs/00_model.ipynb`

Defines and inspects the model. Instantiates the `Unet` + `GaussianDiffusion` + `Trainer1D` stack, loads the stacked CIB+tSZ training patches, applies the 8× augmentation, and sets up the trainer. Generates a model graph via `torchview` and produces a parameter breakdown (35.7 M parameters across encoder/decoder/attention stages). The learning rate here is `5e-5`, differing from `train.py`'s `1e-4`.

**Paper relation:** Corresponds to Appendix A (Table 1 — U-Net architecture, 35.7 M parameters, sigmoid schedule, v-prediction objective). It is the interactive equivalent of `train.py` and is how the training configuration was explored before being finalised.

**Module relation:** Does not import from `foregrounds_diffusion/` — it directly uses `denoising_diffusion_pytorch`. The data loading and augmentation logic (`augment_images_unique`) is duplicated from `train.py`.

---

## `docs/01_map_cuts.ipynb`

A cleaner, standalone reproduction of the flat-sky patch extraction pipeline. Loads full-sky CIB and tSZ FITS files, applies point-source masking (zeroing) and low-pass filtering (ℓ > 7000), zeros negative pixels introduced by the filter, extracts patches with `FlatCutter`, and saves them. Also generates Gaussian realisations from the measured power spectra to serve as the Gaussian baseline for comparisons.

**Paper relation:** The data preparation half of §2. The Gaussian realisations it produces are the "Gaussian simulations" baseline used throughout the results (§4 figures).

**Module relation:** Uses `get_patch_centers` and `FlatCutter` from `preprocessing.py`, and `cl2map` and `make_gaussian_realisation` from `flatmaps.py`.

---

## `docs/02_visualization-joint.ipynb`

Loads training maps, test maps, and DDPM-generated samples (960 patches), denormalises the DDPM output via `renormalize_dm_maps`, and performs quantitative comparison. Computes auto- and cross-power spectra using `map2cl`, plots pixel-intensity histograms, and calculates correlation coefficients between frequency channels (95, 150, 857 GHz).

**Paper relation:** Produces the figures and numbers behind §4.3 (power spectra comparison, Figure 4) and §4.4 (pixel histograms, Figure 5), and Appendix B (multi-frequency correlation coefficients, Figure 9).

**Module relation:** Heavily uses `map2cl` from `flatmaps.py` and `renormalize_dm_maps` from `preprocessing.py`. The post-sampling rescaling done here via `renormalize_dm_maps` is the two-step affine transform flagged in `paper_code_inconsistencies.md` as diverging from the paper's scalar-multiply description.

---

## `docs/03_compute_moments-joint.ipynb`

Computes the full set of 12 cross-moments (2nd, 3rd, 4th order) between the CIB and tSZ channels for training maps, DDPM samples, and Gaussian realisations. Works by bandpass-filtering each map into 8 ℓ-bands of width 720, then computing all combinations of auto- and cross-channel moments (S2ᵃᵃ, S2ᵇᵇ, S2ᵃᵇ, S3ᵃᵃᵃ, … S4ᵃᵇᵇᵇ). Adds three tiers of ILC residual noise (SPT-3G, S4-Wide, S4-Ultra Deep). Saves results as `(801, 8, 12)` arrays.

**Paper relation:** Generates the data behind Appendix C (Figures 10 and 11 — the full breakdown of individual and cross-channel bispectra and trispectra). The 12 moment labels match exactly the paper's notation.

**Module relation:** Uses `get_lpf_hpf` from `preprocessing.py` for bandpass filter construction, and `cl2map` from `flatmaps.py`. Reloads saved moment arrays via `load_all_moments` from `preprocessing.py`.

---

## `docs/03_compute_moments-sum.ipynb`

A simpler companion to the joint moments notebook. Sums CIB and tSZ into a single channel, then computes only variance (S2), skewness (S3), and excess kurtosis (S4) per ℓ-band. The 3-moment arrays are what appear in the main body of the paper rather than the full 12-moment cross-breakdown.

**Paper relation:** Produces the data for Figure 7 (§4.6 — collapsed equilateral bispectrum S3 and trispectrum S4 of the summed CIB+tSZ+noise signal). This is the primary non-Gaussianity result in the paper.

**Module relation:** Uses the same bandpass infrastructure as the joint moments notebook. Reloads previously saved moment arrays via `load_all_moments` from `preprocessing.py`.

---

## `docs/05_plots.ipynb`

The paper-figure production notebook. Pulls together all previously computed outputs (power spectra, moments, histograms, stacks) and formats them for publication. Additionally computes Minkowski functionals (M0 area, M1 perimeter, M2 Euler characteristic) via the external Boelens & Tchelepi package across 50 intensity thresholds, and runs the multi-frequency (95/150/857 GHz) analysis. Uses `apply_stdnorm` from `preprocessing.py` for display normalisation of sample tiles.

**Paper relation:** Is the direct source of essentially every figure in the paper: Figure 1 (pipeline schematic), Figure 2 (visual comparison), Figure 4 (power spectra), Figure 5 (histograms), Figure 6 (Minkowski functionals), Figure 7 (bispectra/trispectra), Figure 8 (multi-frequency), Figure 9 (correlation coefficients).

**Module relation:** Uses `map2cl` from `flatmaps.py`, and `apply_stdnorm` and `renormalize_dm_maps` from `preprocessing.py`. The stacking plot is imported from results computed in `stack_tsz_based_on_snr.ipynb`.

---

## `docs/scratch.ipynb`

Exploratory notebook for investigating the full-sky maps before committing to a pipeline. Reads raw tSZ and CIB maps, examines the effect of cluster and point-source masks on pixel statistics (e.g. masking reduces tSZ std from 4.11 K to 3.25 K), and experiments with generating non-Gaussian realisations by sampling from the empirical pixel PDF via inverse-CDF. Also reconstructs correlated CIB+tSZ maps using measured cross-power spectra.

**Paper relation:** No direct correspondence to a paper section — this is pre-paper exploration. The non-Gaussian realisation method (sampling from PDF) was not adopted; the paper uses Gaussian realisations from power spectra as the baseline instead.

**Module relation:** Uses `make_gaussian_realisation` from `flatmaps.py`. Defines local variants of `cl2map` and `sample_random_dist` that are prototypes for functions later absorbed into `flatmaps.py` and `preprocessing.py`.

---

## `docs/stack_tsz_based_on_snr.ipynb`

Implements the tSZ stacking analysis. Selects pixels exceeding SNR thresholds (5–10σ, 10–20σ, ≥20σ) in both Agora and DDPM tSZ maps, extracts 22-pixel (≈31') cutouts around each, stacks them, and computes 1D radial profiles. Saves the stacked images and profiles to `tsz_extracts/` and final figures to `plots/tsz_stacks.pdf` and `plots/tsz_stacks_radial_profile.pdf`.

**Paper relation:** Directly implements §4.2 and produces Figure 3. The SNR bins, number of stacked clusters (263k/60k/3.9k for Agora), the 8% agreement finding, and the 2-halo term observation in the radial profiles all come from this notebook.

**Module relation:** Uses `radial_profile` from `flatmaps.py` to convert the 2D stacked image into the 1D curves shown in Figure 3. The SNR selection logic is self-contained in the notebook.
