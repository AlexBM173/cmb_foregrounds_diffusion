# Notebook Summaries

Descriptions of each notebook in the codebase, their relation to the paper ("Learning Correlated Astrophysical Foregrounds with Denoising Diffusion Probabilistic Models", Prabhu et al.), and their relation to the `foregrounds_diffusion/` module. Also covers new extension modules added after the paper.

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

---

## `docs/tutorials/10_peak_minima_counts.ipynb`

Evaluates non-Gaussian structure in CIB and tSZ patches using peak and minima counts following Sabyr, Hill & Haiman (2024, arXiv:2410.21247). For each of three smoothing scales (θ_s = 1, 2.5, 5 arcmin FWHM), identifies local maxima and minima via `scipy`'s maximum/minimum filter, then bins counts as a function of amplitude threshold ν = T/σ (30 bins over ν ∈ [−1, 5] for peaks, ν ∈ [−5, 1] for minima). Plots mean ± std for Agora, DDPM, and Gaussian baseline, plus normalised residuals (Agora − DDPM)/σ_Agora. Uses 120 test maps per source. Outputs `plots/peak_minima_counts.pdf` and `plots/peak_minima_residuals.pdf`.

**Paper relation:** No direct paper section — this is a proposed extension statistic. Peak counts are a well-established non-Gaussianity probe sensitive to cluster abundance at different signal levels, complementing the bandpass moment analysis in §4.6.

**Module relation:** Uses `compute_peak_minima_counts` from `peak_counts.py`. Map loading and denormalisation follow the same pattern as the other evaluation notebooks: loads norm params from `norm_params_2mJy.npy`, applies the 80/20 train/test split with `seed=42`, and denormalises DDPM samples from the `new_samples_*.npy` file.

---

## `docs/tutorials/11_scattering_transforms.ipynb`

Computes scattering transform coefficients for CIB and tSZ patches and compares Agora, DDPM, and Gaussian baseline distributions. Supports two backends: the Cheng et al. repo (recommended; must be cloned into the project root) or `kymatio` (pip-installable, slower, no C11). Uses J = 5 wavelet scales and L = 4 orientations. Computes:

- **S1** (first-order): mean wavelet modulus at each scale j, averaged over orientations — plotted on a log scale, related to the power spectrum.
- **S2** (second-order): cross-scale coupling matrix for pairs (j1, j2) with j2 > j1 — visualised as a DDPM/Agora ratio heatmap per channel.
- **Flattened residuals**: full scattering feature vector (S1 + S2 concatenated via `scattering_summary`) normalised by Agora σ, shown as a bar chart.
- **C11** (optional, Cheng et al. only): scattering covariance matrix, mean |C11_iso| visualised as a j2-j3 heatmap.

Outputs `plots/scattering_S1.pdf`, `plots/scattering_S2_ratio.pdf`, `plots/scattering_residuals.pdf`, and optionally `plots/scattering_C11.pdf`.

**Paper relation:** No direct paper section — scattering transforms are a proposed extension statistic. They capture non-Gaussian multi-scale correlations complementary to the bandpass moments in §4.6 and Appendix C, and have been used in CMB/LSS analysis (e.g. Mallat 2012, Cheng et al. 2020).

**Module relation:** Uses `compute_scattering_coefficients`, `compute_scattering_covariance`, and `scattering_summary` from `scattering_stats.py`. Map loading and denormalisation are identical in structure to notebook 10.

---

## Package Modules (post-paper extensions)

### `foregrounds_diffusion/peak_counts.py`

Implements peak and minima counting statistics for flat-sky patches, following Sabyr, Hill & Haiman (2024, arXiv:2410.21247). Requires only numpy and scipy — no LensTools dependency needed for flat-sky data.

**Public API:**

| Function | Description |
|---|---|
| `smooth_map(patch, fwhm_arcmin, pixel_res_arcmin)` | Applies a Gaussian kernel (FWHM in arcmin) to a single 2D patch using `scipy.ndimage.gaussian_filter`. Converts FWHM to σ in pixels using the 6°/256px resolution. |
| `find_peaks(patch, filter_size=3)` | Returns pixel values at local maxima via `maximum_filter`. Boundary pixels (within `filter_size//2` of the edge) are excluded to avoid edge artefacts. |
| `find_minima(patch, filter_size=3)` | Returns pixel values at local minima via `minimum_filter`, with the same boundary exclusion. |
| `count_peaks_binned(patches_nhw, thresholds, fwhm_arcmin, ...)` | Smooths each patch, normalises by per-map σ to get ν = T/σ, then counts peaks above each threshold. Returns shape `(N, len(thresholds))`. |
| `count_minima_binned(patches_nhw, thresholds, fwhm_arcmin, ...)` | Same as above for minima below each threshold. |
| `compute_peak_minima_counts(patches_nhw, thresholds_peaks, thresholds_minima, smoothing_scales_arcmin, ...)` | Top-level convenience wrapper. Loops over smoothing scales and returns a nested dict keyed by FWHM, then `'peaks'`/`'minima'`, each an `(N, len(thresholds))` array. |

**Used by:** `docs/tutorials/10_peak_minima_counts.ipynb`

---

### `foregrounds_diffusion/scattering_stats.py`

Wraps scattering transform backends to compute S1/S2 scattering coefficients and the scattering covariance C11 for ensembles of flat-sky patches. Tries to import the Cheng et al. `scattering` package first (faster, exposes C11); falls back to `kymatio` with a warning. Both require PyTorch.

**Backend setup:**
- Cheng et al. (preferred): `git clone https://github.com/SihaoCheng/scattering_transform.git` into the project root, then add to `sys.path`. Also requires `pip install appdirs` (missing from the repo's declared dependencies). The module translates `device='cuda'` → `'gpu'` internally since Cheng et al. uses `'gpu'`/`'cpu'` strings, not `'cuda'`.
- kymatio (fallback): `pip install kymatio`. C11 is unavailable; S2 is reconstructed from the flattened kymatio output by iterating over j2 > j1 pairs; S1 is averaged over orientations to match the `S1_iso` convention.

**Public API:**

| Function | Description |
|---|---|
| `compute_scattering_coefficients(patches_nhw, J=5, L=4, device=None)` | Computes S0 `(N, 1)`, S1 `(N, J)` (orientation-averaged, `S1_iso` in Cheng et al. notation), and S2 `(N, J, J, L)` (cross-scale coupling as a function of orientation difference, `S2_iso`). Auto-detects GPU. Returns a dict also containing `S1_mean`, `S2_mean`, `J`, `L`. |
| `compute_scattering_covariance(patches_nhw, J=5, L=4, device=None)` | Computes the full scattering covariance (C11_iso, C01_iso, etc.) via the Cheng et al. backend. Returns `None` with a warning if only kymatio is available. |
| `scattering_summary(coeffs, scale_idx=None)` | Flattens S1 and the upper-triangle S2 entries (j2 > j1) into a single feature vector of shape `(N, n_features)`, suitable for computing per-feature residuals between ensembles. |

**Used by:** `docs/tutorials/11_scattering_transforms.ipynb`

---

## `docs/tutorials/12_minkowski_tensors.ipynb`

Computes rank-2 Minkowski tensors for CIB and tSZ patches and compares anisotropy between Agora, DDPM, and Gaussian baseline. For each intensity threshold ν the map is binarised to the excursion set K = {x : T(x) > ν} and three tensor types are computed, each eigendecomposed to give β = λ_min/λ_max ∈ [0, 1] (anisotropy index; 1 = isotropic) and θ (major-axis orientation). All three tensor types are run so their different sensitivities can be compared before committing to a specific choice:

- **W012** (W^{0,2}_1): boundary normal tensor via Sobel-estimated outward normals n⊗n. Probes isotropy of cluster boundary shapes. Recommended default.
- **W200** (W^{2,0}_0): area inertia tensor r⊗r over interior pixels. Measures elongation of filled excursion regions.
- **W201** (W^{2,0}_1): boundary position tensor r⊗r over boundary pixels. Hybrid sensitivity.

Plots: (1) β(ν) 2×3 grid (rows = CIB/tSZ, cols = tensor type) with mean ± std bands; (2) polar θ histograms at ν = 0.2, 0.5, 0.8 using W012 — a uniform distribution is the Gaussian sanity check; (3) W012 residuals (β_Agora − β_DDPM)/σ_Agora with 1σ reference lines. Outputs `plots/minkowski_tensors_beta.pdf`, `plots/minkowski_tensors_theta.pdf`, `plots/minkowski_tensors_residuals.pdf`.

**Paper relation:** No direct paper section — Minkowski tensors are a post-paper extension. They generalise the scalar Minkowski functionals in Figure 6 (§4.5) by adding directional information, exposing morphological anisotropy that MFs miss.

**Module relation:** Uses `compute_minkowski_tensors` from `morphology.py`. Map loading and denormalisation follow the same pattern as notebooks 10–11.

**Reference:** Schroder-Turk et al. (2013), *New J. Phys.* 15 083028.

---

## Package module structure (post-refactor)

`statistics.py` was split into focused modules. The table below shows where each function now lives.

### `foregrounds_diffusion/statistics.py`

Gaussian fitting only.

| Function | Description |
|---|---|
| `gaussian(height, cx, cy, wx, wy)` | Returns a 2D Gaussian callable `f(x, y)`. |
| `moments(data)` | Estimates Gaussian parameters from image moments. |
| `fitgaussian(data)` | Fits a 2D Gaussian by least squares via `scipy.optimize`. |
| `fitting_func(p, p0, xgrid, ygrid, tmap, ...)` | Evaluates/fits a Gaussian model on a pixel grid; used by `masking.get_mask_using_gaussian_fitting`. |
| `stats(maps)` | Returns `(min, max, mean, std)` of an array. |

### `foregrounds_diffusion/moments.py`

Power spectra and higher-order moments.

| Function | Description |
|---|---|
| `mean_cls(maps_nhw, mapparams, lmin, lmax, binsize)` | Mean auto-power spectrum over a stack of maps. |
| `mean_cross_cls(maps1, maps2, mapparams, lmin, lmax, binsize)` | Mean cross-power spectrum between two stacks. |
| `compute_summed_moments(cib_arr, tsz_arr, bp_filters)` | S2/S3/S4 of the summed CIB+tSZ field per ℓ-band. Returns `(N, B, 3)`. |
| `compute_cross_moments(cib_arr, tsz_arr, bp_filters)` | All 12 cross-moments (S2ᵃᵃ … S4ᵃᵇᵇᵇ) per ℓ-band. Returns `(N, B, 12)`. |

### `foregrounds_diffusion/morphology.py`

Minkowski functionals and tensors.

| Function / object | Description |
|---|---|
| `compute_mfs(maps_nhw, norm_fn, thresholds)` | Minkowski functionals M0/M1/M2 via `quantimpy`. Returns three `(N, T)` arrays. |
| `compute_minkowski_tensors(maps_nhw, norm_fn, thresholds, tensor_types=('W012',), centred=True)` | Eigendecomposes each Minkowski tensor at each threshold. Returns dict keyed by tensor type → `{'beta': (N,T), 'theta': (N,T)}`. |
| `MINKOWSKI_TENSOR_DESCRIPTIONS` | Dict of human-readable descriptions keyed by tensor type string. |
| `_tensor_W012(binary_map)` | W^{0,2}_1: Σ n⊗n over boundary pixels (Sobel normals). |
| `_tensor_W200(binary_map, centred=True)` | W^{2,0}_0: Σ r⊗r over interior pixels. |
| `_tensor_W201(binary_map, centred=True)` | W^{2,0}_1: Σ r⊗r over boundary pixels. |
| `_eigendecompose_2x2(W)` | Returns β = λ_min/λ_max and θ (major axis, wrapped to (-π/2, π/2]). |

### `foregrounds_diffusion/stacking.py`

tSZ cluster stacking.

| Function | Description |
|---|---|
| `select_snr_pixels(tsz_maps_nhw, snr_min, snr_max, min_separation=5)` | Finds local SNR-peak coordinates within a given SNR bin. Returns list of `(patch_idx, row, col)`. |
| `extract_cutouts(maps_nhw, coords, cutout_size, max_cutouts=500)` | Extracts square cutouts centred on the given coordinates. Returns `(M, size, size)` or `None`. |

### `foregrounds_diffusion/masking.py`

Flat-sky and HEALPix masking (merged from `preprocessing.py` and `get_cluster_source_mask_for_agora.py`).

| Function | Description |
|---|---|
| `get_peak_masks(tmap, ...)` | Sigma-clipping peak mask with optional apodisation. |
| `inpaint_masked_regions(hmap, mask, rng=None)` | Replaces masked pixels with Gaussian noise matching unmasked statistics. |
| `boundary_apod_mask(x_grid, y_grid, mask_radius, ...)` | Apodised boundary mask on a 2D grid. |
| `get_mask_using_gaussian_fitting(nonpeak_mask, ...)` | Fits Gaussians to mask blobs and builds a smooth mask. |
| `get_mdpl2_halo_cat(halo_cat_fname, ...)` | Loads the MDPL2 halo catalogue (.npy or .npz). |
| `get_cluster_mask_radius(m500c)` | Returns mask radius in arcmin for a given M_500c. |
| `get_point_source_mask_in_healpix(freq, hmap_Mjy_per_sr, ...)` | Identifies point-source pixels above a flux threshold. |
| `get_mdpl2_conversion_factors_K_to_MjyperSr(expname, band)` | Looks up K → MJy/sr conversion factors by experiment and band. |
| `apodize_binary_mask_prof(binary_mask, ...)` | Apodises a HEALPix binary mask using a cosine profile. |
| `get_apodised_mdpl2_cluster_mask(nside, halo_cat_fname, ...)` | Builds an apodised full-sky cluster mask from the MDPL2 halo catalogue. |
