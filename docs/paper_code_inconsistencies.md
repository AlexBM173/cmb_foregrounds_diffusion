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

**Code:** `renormalize_dm_maps` (`preprocessing.py:103`) applies a two-step affine transform: first a range rescaling to match `[tr_min, tr_max]`, then optionally a mean-and-variance match `(x − μ_dm) × (σ_tr/σ_dm) + μ_tr`. This is not a simple scalar multiply. Furthermore, `sample.py` calls no rescaling at all — it saves raw model output without any post-processing, and neither do the statistics notebooks (06–12). **Consequence for the thesis:** because `compute_cross_moments` (`moments.py:226`) returns *raw, un-normalised* moments (e.g. `np.mean(a**3)`), any amplitude offset propagates nonlinearly into every S3/S4 cross-moment (S4 ∝ amplitude⁴) and quadratically into power spectra. The rescaling decision — reproduce the paper's scalar, or deliberately omit and state so — must be made explicitly before generating thesis figures. `sample.py` now provides opt-in `--rescale-cib` / `--rescale-tsz` flags (off by default; backed by `rescale_samples` in `preprocessing.py`) so the paper's per-channel scalars (1.0328 / 1.1425) can be applied at sampling time without changing the default raw-output behaviour.

---

## 5. Data augmentation — "random" vs. exhaustive systematic

**Paper (§2):** "We also apply a **random** rotation and flip to each patch to increase the number of training patches."

**Code:** `augment_images_unique` applies **all 8 combinations** (4 rotations × 2 flip states) deterministically to every patch, giving an exact 8× expansion. This is deterministic and exhaustive, not random selection. Note there are **two divergent implementations**: the tested package version (`preprocessing.py:461`, `torch.stack` of per-image rot90×flip) and an inline copy in the root training script (`train.py:164`, `torch.cat([images, r1, r2, r3] + flips])`) with a different output ordering. The model was trained with the untested inline copy.

---

## 6. Noise schedule — paper's β values are unreliable reporting; code uses library-default sigmoid

**Paper §3.1:** "sigmoid schedule βt ranging from β₁ = 10⁻⁴ to β₁₀₀₀ = **0.02**"
**Paper Appendix A:** "sigmoid noise schedule, with β₁ = 10⁻⁴ and β₁₀₀₀ = **10⁻²**"

**Code:** `GaussianDiffusion` is constructed without `beta_schedule` kwargs, so training uses the `denoising-diffusion-pytorch` v2.2.5 default: the **Jabri et al. (arXiv:2212.11972) sigmoid schedule** (`sigmoid_beta_schedule`, `start=-3, end=3, tau=1`, β clipped to ≤ 0.999), which parameterises ᾱₜ with a sigmoid and derives βₜ from its ratios. Computed values (T = 1000, float64, exactly as the library does): β₁ = 3.0028×10⁻⁴, β₅₀₀ = 3.31×10⁻³, β₁₀₀₀ = 0.999 (the only clipped step; just 10 of 1000 betas exceed 0.1), ᾱ₁₀₀₀ = 3.0×10⁻⁷. The near-1 terminal β is by design in ᾱ-parameterised schedules and is not a bug — the "all βₜ ≪ 1" intuition belongs to the linear schedule.

**Neither paper value matches the code, and the paper's numbers are internally incoherent:**

- No sigmoid parameterisation produces §3.1's (10⁻⁴, 0.02) — those are verbatim the Ho et al. **linear**-schedule textbook endpoints attached to the word "sigmoid".
- Appendix A's (10⁻⁴, 10⁻²) matches the endpoints of the *old DDIM-repo* "sigmoid-on-β" schedule (`β = σ(linspace(−6,6))·(β_end−β_start)+β_start` → β₁ = 1.24×10⁻⁴, β₁₀₀₀ = 9.98×10⁻³), but that schedule **has never existed in the lucidrains library** (verified against the library's git history: the only sigmoid ever added is the Jabri one, Dec 2022). Using it would have required custom schedule code, and the paper describes its implementation as building on the lucidrains repo (its ref. [47] + §3.1 footnote) with no mention of modification. It would also leave ᾱ₁₀₀₀ = 6.3×10⁻³ (~8% residual signal amplitude at t = T) — a known terminal-SNR defect that ᾱ-parameterised schedules exist to fix.

**Resolution (confirmed from git history):** Prabhu et al. ran the library defaults. This repo's initial commit `a3804d5` (author karthikprabhu22, the paper's first author, May 2025) contains his original `train.py` and `sample.py`, both constructing `GaussianDiffusion(model, image_size=256, timesteps=1000)` with **no other kwargs** — so the sigmoid (Jabri) schedule and `pred_v` objective are certain, not inferred. No upstream commit ever changed this construction (the only upstream `train.py` change, `b638b66`, switched channels 3→2 for the CIB+tSZ paper model and lr 5e-5→1e-4). The paper correctly names the defaults it looked up ("sigmoid", v-prediction, T = 1000, U-Net depths 64/128/256/512) but quoted textbook β endpoints instead of the actual values, inconsistently between sections. Treat the reproduction as *matching* the paper's training setup despite the quoted numbers. Accurate wording for the report: *sigmoid (Jabri et al.) schedule, β from 3.0×10⁻⁴ to 0.999 (clipped, final step only), ᾱ_T ≈ 3×10⁻⁷*.

Note the objective corollary: the paper's "velocity-prediction objective (Salimans & Ho)" is the library default `objective='pred_v'`, which the code — passing no `objective` kwarg — also inherits. Paper and code agree here; documented because the code never states it explicitly.

Minor related find: the paper's stated learning rate 1×10⁻⁴ is correct for the final 2-channel model (`b638b66` set it), but the earlier 3-channel CIB frequency experiment (paper Appendix B) was configured at 5e-5 in the initial commit — the paper reports a single lr for both.

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

**Git-history evidence (resolves the provenance):** the legacy min-max/std-norm convention is the
**paper's** convention, not a stale doc. Prabhu's own `train.py` (initial commit `a3804d5`, final
upstream form `b638b66`) loads `CIB_..._minmax_2mJy_zero_lp.npy` + `tSZ_..._minmax_2mJy_norm_lp.npy`
and passes no `auto_normalize` kwarg — so the published model trained with the library default
`auto_normalize=True` (data mapped [0,1]↔[−1,1] internally), coherent for min-max CIB but odd for
std-normed tSZ. The z-score-both scheme in notebook 03, and `auto_normalize=False` in the current
train/sample code, are this project's later **deliberate deviations** — document them as such in the
report rather than as ambiguity. (v4 run confirms: z-score both channels,
`norm_params = [20.371, 4.744, −4.979, 3.521]`, `auto_normalize=False` at train and sample time.)

**Before running any statistics on the cluster:**
1. Confirm which `.npy` files physically exist in `data/low_pass/2mJy/` (`_zscore_` vs `_minmax_..._zero`/`_norm`).
2. Confirm the normalisation the checkpoint was **trained** with (a z-score-trained model must be denormalised with `denormalize_dm_maps`, *not* the min-max `renormalize_dm_maps`).
3. Align the load filenames in notebooks 06–14 to match whatever 03 actually produced, and update `CLAUDE.md`/`README.md` to the true scheme.

Until (1)–(2) are confirmed, do not trust any figure's absolute amplitude.

---

## 8. Sampler — paper silent; authors used 1000-step ancestral DDPM; current pipeline uses DDIM-250

**Paper:** never states which sampler generated its results — no mention of DDIM, implicit models,
reduced-step sampling, or η anywhere. §3.1 gives only the ancestral reverse kernel
pθ(x_{t−1}|x_t) = N(µθ, βt). Appendix A claims "sampling takes roughly 1–2 seconds per patch" on an
A100, which looks optimistic for 1000 sequential U-Net passes. **Measured (v4 run, 5 Jul 2026,
Colab A100-80GB, memory-efficient attention, fp16, batch 16): ~6.1 it/s → ~164 s per 1000-step
batch → ~10 s per patch ancestral** — the paper's figure is ~5–10× optimistic for the sampler its
authors verifiably used, and would only fit DDIM at ~100–250 steps or much larger batches. Another
loosely-reported number, consistent with #6.

**Verified from git history:** Prabhu's original `sample.py` (initial commit `a3804d5`) calls
`diffusion.sample(batch_size=16)` on a `GaussianDiffusion` built with no `sampling_timesteps` kwarg
→ `sampling_timesteps = timesteps = 1000` → `is_ddim_sampling = False` → the **stochastic
ancestral `p_sample` loop, 1000 steps**, drawing noise with the clipped posterior variance
σt² = β̃t = βt(1−ᾱ_{t−1})/(1−ᾱt) — *not* the σt² = βt the paper's §3.1 states (the library has no
σt² = βt code path at all). Sampling ran under fp16 `accelerate`. No upstream commit ever modified
`sample.py`.

**Current pipeline:** samples with **DDIM at 250 steps, η = 0 (deterministic)** via
`--sampling-timesteps` / config `sampling.ddim_steps`. This is a deliberate deviation (4× fewer
steps, zero injected noise); state it in the report with the standard justification (Song et al.
2021: DDIM sampling from DDPM-trained weights preserves quality at large step counts) rather than
claiming consistency with the paper's sampling procedure.

---

## 9. Library samplers clamp predicted x₀ to [−1, 1] — fatal for z-score models (**fixed 5 Jul 2026**)

`denoising-diffusion-pytorch` hard-codes an x₀ clamp in both samplers, with no kwarg to disable:
ancestral `p_sample` → `p_mean_variance(clip_denoised=True)` and `ddim_sample` →
`model_predictions(clip_x_start=True)`. The clamp originates in Ho et al.'s official DDPM
implementation (images normalised to [−1, 1], so clipping to the valid range is correct there);
lucidrains inherited it in his first working sampler (commit `79e0765`, Sep 2020) and added the
DDIM-path clamp with DDIM itself (`931a5af`, Jul 2022). It has therefore been in every sampler this
repo ever invoked — since the initial commit (May 2025).

**Impact:** for models trained on z-scored data with `auto_normalize=False` (v4: tSZ decrements
≈ −23σ, CIB tails +12σ), unpatched sampling crushes every map into ≈[−1, 1] — verified end-to-end
with a stub denoiser: the stock class returns samples pinned at exactly ±1 while the true x₀ sits
at ≈ −65. Training weights are unaffected (the loss path never clamps).

**Paper-era note:** the paper's min-max CIB ([0, 1] + `auto_normalize=True` → internal [−1, 1])
made the clamp harmless for CIB, but its std-normed tSZ (`apply_stdnorm` is a plain z-score) had
internal values far outside [−1, 1], so the clamp plausibly suppressed the paper's own tSZ
extremes — a candidate mechanism for its documented tSZ extreme-pixel deficit and the larger tSZ
variance-rescaling factor (1.1425 vs CIB's 1.0328). Hypothesis only; not verifiable without their
checkpoint.

**Fix:** `UnclampedGaussianDiffusion` in `foregrounds_diffusion/sample.py` overrides both paths to
force the clamp off; `build_model` always returns it, covering the CLI, `sample_slurm.sh`, and
`run.py sample`. Regression tests in `tests/test_sample.py`. Caveat: `pipeline/train.py` milestone
sampling still uses the stock class — its W&B preview images are amplitude-compressed (cosmetic
only; do not judge model quality from them).
