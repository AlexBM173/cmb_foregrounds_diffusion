# Paper–Code Inconsistencies

Comparison between "Learning Correlated Astrophysical Foregrounds with Denoising Diffusion Probabilistic Models" (Prabhu et al.) and the current codebase.

---

## 1. Cluster mask inpainting — zero-fill vs. Gaussian noise

**Paper (§2):** "Masked regions are inpainted with Gaussian random values with a mean and standard deviation corresponding to the entire map."

**Code:** The preprocessing notebook (`preprocessing.ipynb`, cell 8) sets cluster-masked pixels to zero with `np.where(mask, 0, map)`. No Gaussian inpainting function exists anywhere in the module. The only related function, `replace_zeros_with_neighbor_avg` in `preprocessing.py:252`, fills with neighbour averages, not Gaussian noise.

---

## 2. Point-source masking method — sigma-clip vs. flux threshold

**Paper (§2):** "We mask sources brighter than 2 mJy using a single-pixel mask." The module includes `get_point_source_mask_in_healpix` (`get_cluster_source_mask_for_agora.py:75`) which implements the proper flux-based (mJy) identification in HEALPix units.

**Code:** The preprocessing notebook (cell 8) uses `astropy.stats.sigma_clip(cib_map, sigma=10)` directly on the map in K_CMB units. Sigma-clipping on map values is statistically different from a calibrated mJy flux threshold, and the proper masking function in the module is not called.

---

## 3. Cluster masking thresholds and radii

**Paper (§2):** Masks clusters with M500c ≥ 3×10¹⁴ M☉, with circular radii of **3θ₅₀₀c to 5θ₅₀₀c** depending on mass, and a **minimum radius of 10'**.

**Code (two problems):**

- `get_apodised_mdpl2_cluster_mask` (`get_cluster_source_mask_for_agora.py:228`) defaults to `m500c_threshold=5e13` — 6× lower than the paper's 3×10¹⁴.
- `get_cluster_mask_radius` (`get_cluster_source_mask_for_agora.py:45`) returns **fixed arcminute values** (3', 5', 8', 10') regardless of the cluster's θ₅₀₀c. Its minimum is **3'** (for m < 10¹⁴), not 10'. A θ₅₀₀c-based mode exists via `howmanythetaforclusters > 0` but is not the default.

---

## 4. Post-sampling variance rescaling — scalar factor vs. affine transform

**Paper (§3.2):** "We multiply each DDPM sample by a single global factor: the ratio of the standard deviation of all the Agora samples to that of all the generated samples" — citing specific factors 1.0328 (CIB) and 1.1425 (tSZ).

**Code:** `renormalize_dm_maps` (`preprocessing.py:103`) applies a two-step affine transform: first a range rescaling to match `[tr_min, tr_max]`, then optionally a mean-and-variance match `(x − μ_dm) × (σ_tr/σ_dm) + μ_tr`. This is not a simple scalar multiply. Furthermore, `sample.py` calls no rescaling at all — it saves raw model output without any post-processing, and neither do the statistics notebooks (06–12). **Consequence for the thesis:** because `compute_cross_moments` (`moments.py:226`) returns *raw, un-normalised* moments (e.g. `np.mean(a**3)`), any amplitude offset propagates nonlinearly into every S3/S4 cross-moment (S4 ∝ amplitude⁴) and quadratically into power spectra. The rescaling decision — reproduce the paper's scalar, or deliberately omit and state so — must be made explicitly before generating thesis figures.

---

## 5. Data augmentation — "random" vs. exhaustive systematic

**Paper (§2):** "We also apply a **random** rotation and flip to each patch to increase the number of training patches."

**Code:** `augment_images_unique` applies **all 8 combinations** (4 rotations × 2 flip states) deterministically to every patch, giving an exact 8× expansion. This is deterministic and exhaustive, not random selection. Note there are **two divergent implementations**: the tested package version (`preprocessing.py:461`, `torch.stack` of per-image rot90×flip) and an inline copy in the root training script (`train.py:164`, `torch.cat([images, r1, r2, r3] + flips])`) with a different output ordering. The model was trained with the untested inline copy.

---

## 6. β₁₀₀₀ inconsistency within the paper itself

**Paper §3.1:** β₁₀₀₀ = **0.02**
**Paper Appendix A:** β₁₀₀₀ = **10⁻² = 0.01**

These differ by a factor of 2. The code constructs `GaussianDiffusion` without specifying `beta_schedule` kwargs, so it uses the library defaults in `denoising-diffusion-pytorch` v2.2.5 (sigmoid schedule with its own hardcoded endpoints). Whether the library's default endpoints match either paper value is not verified.

---

## 7. Normalisation scheme — internal contradiction between notebooks 03 and 06 (**cluster-day blocker**)

The single most error-prone interface in the pipeline is under-specified and the tutorial notebooks disagree with each other and with `CLAUDE.md`.

**Notebook 03 (patch extraction — the data *producer*):** z-scores **both** channels inline
(`cib_norm = (cib_maps - cib_mean) / cib_std`, comment "CIB: global Z-score"; same for tSZ),
saves `CIB_map_150GHz_256_st6_zscore_2mJy_lp.npy` and `tSZ3_..._zscore_..._lp.npy`, and writes
`norm_params_2mJy.npy = np.array([cib_mean, cib_std, tsz_mean, tsz_std])`. Its imports of
`apply_maxmin_normalization` / `apply_stdnorm` are dead — neither is called.

**Notebook 06 (power spectra — a data *consumer*):** loads
`CIB_map_150GHz_256_st6_minmax_2mJy_zero_lp.npy` and `tSZ3_..._minmax_..._norm_lp.npy` —
**filenames notebook 03 does not produce.** A fresh 03 → 06 run raises `FileNotFoundError`.
Its inline comment mislabels the array as `[cib_min, cib_max, tsz_mean, tsz_std]`, but the *code*
correctly unpacks `cib_mean, cib_std, tsz_mean, tsz_std = norm_params` and denormalises via
`denormalize_dm_maps` (z-score inverse `x·std + mean` for **both** channels) — self-consistent
with notebook 03's z-score scheme. So the denormalisation *maths* is correct; only the *filenames*
and the *comment* are stale.

**`CLAUDE.md` / `README.md`:** describe CIB as min-max `[0, 1]` (`_zero` suffix) and tSZ as
std-norm (`_norm` suffix). This matches the legacy `renormalize_dm_maps` (min-max affine) and the
old sample filename `new_samples_cib_tsz_2mJy_zero_norm_...` but **contradicts** the current
z-score scheme in notebook 03. The checkpoint name `v3_zscore_no_cib_cluster_mask` favours the
z-score scheme being current truth.

**Before running any statistics on the cluster:**
1. Confirm which `.npy` files physically exist in `data/low_pass/2mJy/` (`_zscore_` vs `_minmax_..._zero`/`_norm`).
2. Confirm the normalisation the checkpoint was **trained** with (a z-score-trained model must be denormalised with `denormalize_dm_maps`, *not* the min-max `renormalize_dm_maps`).
3. Align the load filenames in notebooks 06–14 to match whatever 03 actually produced, and update `CLAUDE.md`/`README.md` to the true scheme.

Until (1)–(2) are confirmed, do not trust any figure's absolute amplitude.
