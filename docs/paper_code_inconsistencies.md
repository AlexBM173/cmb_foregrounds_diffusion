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

**Code:** `renormalize_dm_maps` (`preprocessing.py:52`) applies a two-step affine transform: first a range rescaling to match `[tr_min, tr_max]`, then optionally a mean-and-variance match `(x − μ_dm) × (σ_tr/σ_dm) + μ_tr`. This is not a simple scalar multiply. Furthermore, `sample.py` calls no rescaling at all — it saves raw model output without any post-processing.

---

## 5. Data augmentation — "random" vs. exhaustive systematic

**Paper (§2):** "We also apply a **random** rotation and flip to each patch to increase the number of training patches."

**Code:** `augment_images_unique` (`train.py:71`) applies **all 8 combinations** (4 rotations × 2 flip states) deterministically to every patch, giving an exact 8× expansion. This is deterministic and exhaustive, not random selection.

---

## 6. β₁₀₀₀ inconsistency within the paper itself

**Paper §3.1:** β₁₀₀₀ = **0.02**  
**Paper Appendix A:** β₁₀₀₀ = **10⁻² = 0.01**

These differ by a factor of 2. The code constructs `GaussianDiffusion` without specifying `beta_schedule` kwargs, so it uses the library defaults in `denoising-diffusion-pytorch` v2.2.5 (sigmoid schedule with its own hardcoded endpoints). Whether the library's default endpoints match either paper value is not verified.
