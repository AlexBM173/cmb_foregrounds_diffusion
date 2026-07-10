# Potential Project Extensions

Extensions are grouped by theme, roughly ordered from most to least directly tractable given the current codebase. Each entry notes the scientific motivation, the implementation starting point, and the main obstacle.

---

## 1. Scaling to Larger Sky Patches

**Motivation:** The paper trains on 6°×6° patches and notes significant degradation at 10°×10°, limiting applicability to surveys that span thousands of square degrees. Most downstream analyses (lensing, kSZ) need coherent foreground realisations over much larger areas.

**Approach:** The Patch Diffusion framework (cited in §5.2) trains a diffusion model that can generate overlapping patches and stitch them consistently. An alternative is a hierarchical model: train a coarse 10°×10° model to capture large-scale structure, then a fine 6°×6° model conditioned on the coarse output to recover small-scale features. Tiling with overlap and blending (e.g. feathering at boundaries) is a simpler baseline worth establishing first.

**Starting point:** `train.py` — vary `STEP_SIZE` and `RES`; monitor the power spectrum deficit that appears at large scales as a diagnostic.

**Obstacle:** The number of available non-overlapping patches falls sharply with patch size (∝ patch area⁻¹), which is the root cause of the degradation. Any solution requires either more training data (other simulations) or a model that can exploit the overlap between patches.

---

## 2. Conditional Generation on Cosmological or Astrophysical Parameters

**Motivation:** Currently the DDPM learns the foreground distribution for a single set of cosmological and astrophysical parameters (MDPL2/BAHAMAS). Any mismatches between the training simulation and the real sky introduce uncontrolled biases. A model conditioned on parameters (σ₈, Ωm, feedback amplitude) could marginalise over these uncertainties in inference.

**Approach:** Replace the unconditional `Unet` with a class-conditioned or continuous-parameter-conditioned variant. The `denoising_diffusion_pytorch` library supports classifier-free guidance: concatenate a parameter embedding to the time embedding at each U-Net block. Training would require maps from multiple simulation runs at varied parameters — WebSky, Agora variants, or the BAHAMAS suite would all be candidates.

**Starting point:** `foregrounds_diffusion/diffusion.py` (currently a stub) and `train.py` — add a `condition` argument to the `Unet` constructor.

**Obstacle:** No multi-cosmology simulation dataset currently exists at the required resolution and sky coverage. This makes the extension primarily a waiting game for data, though the code architecture can be prepared in advance.

---

## 3. Extending to Additional Foreground Components

**Motivation:** Real CMB analyses must contend with radio galaxies, the cosmic optical background, the kinetic SZ (kSZ) effect, and Galactic dust leakage in addition to CIB and tSZ. The paper demonstrates joint CIB–tSZ generation and multi-frequency CIB; the natural next step is a larger joint model.

**Approach — kSZ:** Add a third channel to the `Unet` (`channels=3`) trained on the Agora kSZ map. kSZ is spectrally flat (unlike tSZ) and probes line-of-sight velocities, so its cross-correlation with CIB and tSZ encodes distinct astrophysics. The preprocessing pipeline in `preprocessing.py` already supports multi-channel input via `split_data_to_tensors`.

**Approach — radio galaxies:** Radio galaxies are sparser and more point-source-like than CIB, making them harder to learn without very large patch counts. A hybrid approach — model the diffuse CIB/tSZ field with the DDPM and sprinkle radio sources drawn from a parametric number count model — may be more practical than learning them jointly.

**Starting point:** `train.py` — change `channels=2` to `channels=3` and update the data loading to concatenate a third map.

**Obstacle:** kSZ is an order of magnitude fainter than tSZ at the noise levels of current experiments, making the model's effective dynamic range challenging. Normalisation strategy needs rethinking.

---

## 4. Multi-Frequency Spectral Energy Distribution Modelling

**Motivation:** Appendix B shows the DDPM can jointly generate CIB at 95, 150, and 857 GHz and largely reproduces inter-frequency correlations, but with mild over-correlation at high multipoles. Physically, the CIB SED varies with redshift because different populations contribute differently at each frequency; correctly learning this decorrelation is essential for ILC-based foreground cleaning.

**Approach:** Train the 3-channel (95/150/857 GHz) CIB model with explicit inter-frequency cross-power as an auxiliary loss term. Alternatively, use a frequency-conditioned model that generates a map at any requested frequency by conditioning on a continuous frequency embedding — this allows interpolation to bands not in the training set.

**Starting point:** `docs/tutorials/04_model_and_training.ipynb` — the commented-out multi-frequency data loading block in `train.py` is the natural starting point.

**Obstacle:** The 857 GHz map decorrelates from 95/150 GHz because of lower-redshift galaxy contributions. Capturing this requires either many training patches (to sample the rare large-scale structure driving decorrelation) or physically motivated inductive biases not present in a standard U-Net.

---

## 5. Integration into a Bayesian Inference Pipeline

**Motivation:** The paper identifies this as a key application (§5.1): replacing Gaussian foreground priors with the trained DDPM prior inside inference frameworks like MUSE. This would allow the DDPM to directly inform CMB lensing or kSZ analyses rather than being used only as a simulation generator.

**Approach:** The score function ∇ log p(x) is directly available from the trained DDPM — at each diffusion timestep t, the model predicts the score. This can be used as an unnormalised prior gradient in Langevin MCMC or variational inference. Concretely: given a noisy CMB observation d = s_CMB + s_fg + n, sample s_fg using the DDPM score as the foreground prior and a Gaussian likelihood for n.

**Starting point:** `foregrounds_diffusion/sample.py` — the reverse diffusion loop already computes the score implicitly; it needs to be exposed as a standalone function and combined with a data likelihood term.

**Obstacle:** Diffusion-model posteriors are expensive to evaluate (1000 reverse steps per sample) and the score is only well-defined in the diffusion model's normalised pixel space, complicating coupling with a physical likelihood in physical units.

---

## 6. Faster Sampling via Consistency or Flow-Matching Models

**Motivation:** The current model requires 1000 reverse diffusion steps per sample. While the paper reports 1–2 seconds per patch on an A100, generating thousands of patches for covariance matrix estimation or likelihood calls in an inference loop would benefit from faster samplers.

**Approach:** Distil the trained DDPM into a consistency model (Song et al. 2023), which can generate samples in as few as 1–4 steps with minimal quality loss. Alternatively, retrain from scratch using flow matching (Lipman et al. 2022), which learns a simpler straight-path ODE between noise and data and typically converges faster. The `denoising_diffusion_pytorch` library supports DDIM sampling (a deterministic ODE solver) as an immediate lower-cost option without retraining.

**Starting point:** `foregrounds_diffusion/sample.py` — add a `--sampling-timesteps` argument passed to `GaussianDiffusion`; the library already supports this via the `sampling_timesteps` parameter, enabling DDIM acceleration without retraining.

**Obstacle:** Consistency distillation requires the base DDPM to be well-trained first and introduces its own instabilities. DDIM is free but the quality degradation at very low step counts (< 50) needs to be benchmarked against the power spectra and higher-order statistics.

---

## 7. Improved Handling of Extreme-Value Pixels

**Motivation:** The paper identifies under-reproduction of rare, high-amplitude pixels (massive clusters, bright point sources) as the primary failure mode, leading to a deficit in the Poisson tail of the power spectrum that requires post-hoc variance rescaling to correct. This is the root cause of the ~19% discrepancy in the high-SNR stacking result (§4.2).

**Approach — data augmentation:** Oversample training patches that contain high-SNR pixels (importance-weighted sampling) so the model sees more examples of extreme structures during training. Weights can be set proportional to the maximum pixel absolute value in each patch.

**Approach — conditional inpainting:** Train a separate small model to generate realistic cluster profiles conditioned on mass and redshift from the halo catalogue, then composite these onto the DDPM background. This decouples the rare-event problem from the bulk foreground statistics.

**Approach — min-SNR loss weighting:** The `GaussianDiffusion` constructor already supports `min_snr_loss_weight=True` (enabled via a flag). This down-weights loss at early (high-noise) timesteps where the model struggles to learn fine structure, potentially improving high-frequency fidelity.

**Starting point:** `train.py` — set `min_snr_loss_weight=True` in the `GaussianDiffusion` constructor as a zero-cost first experiment.

**Obstacle:** Oversampling rare patches may hurt the model's overall calibration for the bulk of the distribution. The composite approach requires accurate cluster profile models, reintroducing simulation dependence.

---

## 8. Applying to Real Observational Data

**Motivation:** The model is trained entirely on simulations. Testing whether DDPM-generated maps agree with real SPT-3G or ACT observations would validate the Agora simulation itself and determine whether the generative model can be used in real analyses.

**Approach:** Extract 6°×6° patches from SPT-3G or ACT temperature maps in the same way patches are extracted from HEALPix simulations (the `FlatCutter` class works on real data too). Compare the power spectra, histograms, and Minkowski functionals of real patches against DDPM samples. Discrepancies would flag either simulation inaccuracies or model failures.

**Starting point:** `foregrounds_diffusion/preprocessing.py` — `FlatCutter` and `get_patch_centers` work on any HEALPix map; the only change is the input data path.

**Obstacle:** Real maps contain instrumental noise, beam effects, and scan-strategy artefacts not present in the simulations. A direct pixel-level comparison requires careful noise modelling and beam deconvolution. The ILC noise spectra already in `data/ilc/` are a starting point for noise characterisation.

---

## 9. Fixing the Preprocessing Inconsistencies

**Motivation:** Several discrepancies between the paper and the code (documented in `paper_code_inconsistencies.md`) affect the fidelity of the training data: point sources are masked by sigma-clipping rather than flux-based thresholding, cluster masks are zero-filled rather than Gaussian-inpainted, and the cluster mask radius defaults are wrong. Correcting these would bring the code into alignment with the published method and potentially improve sample quality.

**Approach:**
- Replace `sigma_clip` in `preprocessing.ipynb` with `get_point_source_mask_in_healpix` (already implemented in `get_cluster_source_mask_for_agora.py`).
- Add a Gaussian inpainting step after cluster masking: draw values from N(μ_map, σ_map) at masked pixel locations.
- Set `m500c_threshold=3e14` and `howmanythetaforclusters=4` (midpoint of the paper's 3–5× range) in `get_apodised_mdpl2_cluster_mask`.
- Retrain on the corrected data and compare power spectra and moment statistics against the current model to quantify the impact.

**Starting point:** `docs/tutorials/02_masking.ipynb` — this tutorial notebook is the natural place to implement the corrected pipeline before it feeds into patch extraction.

**Obstacle:** Regenerating training data requires access to the full-sky FITS maps on the remote cluster. The impact of each correction is unknown without retraining, which takes ~30 hours per run.

---

## 10. Wavelet Scattering Transform as an Additional Validation Statistic ✓ *Implemented*

**Motivation:** The paper validates against power spectra, histograms, Minkowski functionals, and collapsed bispectra/trispectra. Wavelet scattering transforms (WST) are sensitive to multi-scale non-Gaussian structure in a way that is complementary to the harmonic-space moment approach and have been applied to CMB and large-scale structure analyses.

**Implementation:** `foregrounds_diffusion/scattering_stats.py` — `compute_scattering_coefficients` (S1: N×J orientation-averaged, S2: N×J×J×L cross-scale coupling), `compute_scattering_covariance` (C11, Cheng et al. backend only), `scattering_summary` (flattened feature vector). Uses the Cheng et al. `scattering_transform` repo (cloned to project root) with `kymatio` as a fallback. See `docs/tutorials/11_scattering_transforms.ipynb` for the full comparison against Agora, DDPM, and Gaussian baseline.

**Remaining work:** Run on the full test set (100+ maps) and interpret the S2 cross-scale ratio matrix — deviations from 1 in the DDPM/Agora ratio identify which scale pairs the model fails to reproduce. C11 scattering covariance is available but not yet analysed.

**Obstacle:** WST coefficient interpretation is less established in the CMB foreground literature than power spectra or bispectra, making it harder to relate findings back to physical quantities. It is primarily useful as an additional empirical check rather than a physically motivated diagnostic.

---

## 11. Minkowski Tensors as a Morphological Anisotropy Statistic ✓ *Implemented*

**Motivation:** The scalar Minkowski functionals in Figure 6 measure area (M0), perimeter (M1), and Euler characteristic (M2) of excursion sets but discard all directional information. Minkowski tensors are rank-2 tensorial generalisations that additionally encode the *anisotropy* and *orientation* of morphological features — cluster boundary shapes, filament directions — providing a sensitive test of whether the DDPM reproduces the full geometry of the foreground fields, not just their scalar topology.

**Implementation:** `foregrounds_diffusion/morphology.py` — `compute_minkowski_tensors(maps_nhw, norm_fn, thresholds, tensor_types, centred)` returns β = λ_min/λ_max ∈ [0, 1] (anisotropy index) and θ (major-axis orientation) per map per threshold, for three tensor types:
- **W021** (W^{0,2}_1): boundary normal tensor via Sobel-estimated outward normals n⊗n. Probes isotropy of cluster boundary shapes. Recommended default.
- **W200** (W^{2,0}_0): area inertia tensor r⊗r over interior pixels. Measures elongation of filled excursion regions.
- **W201** (W^{2,0}_1): boundary position tensor r⊗r over boundary pixels. Hybrid sensitivity.

See `docs/tutorials/12_minkowski_tensors.ipynb` for β(ν) curves (2×3 grid across channels and tensor types), polar θ histograms, and DDPM/Agora residuals.

**Remaining work:** Run the tutorial on the full test set, choose the most informative tensor type(s) for the specific CIB/tSZ morphology, and determine whether the DDPM reproduces the anisotropy of tSZ clusters (which are expected to be mildly anisotropic due to filamentary infall) versus CIB (which should be closer to isotropic on average). A joint CIB×tSZ tensor (cross-channel anisotropy alignment) could be a further extension.

**Starting point:** `docs/tutorials/12_minkowski_tensors.ipynb` — fully runnable; choose between W021/W200/W201 based on the β(ν) plot output.

**Obstacle:** Sobel-based normal estimation has O(1 pixel) error on sharp boundaries; this is negligible for the 256×256 patches in use but could matter for very small excursion sets (ν near 0.95). Smoothing the binary mask before gradient estimation is a simple fix if needed. Interpretation requires care at extreme thresholds where excursion sets are nearly empty or nearly full — those entries are already masked to β = 1 in the implementation.

**Reference:** Schroder-Turk et al. (2013), *New J. Phys.* 15 083028.

---

## 12. Multi-Channel (3–5 Field) Generation — Deadline Feasibility Assessment (added 6 Jul 2026)

Extends #3 with a concrete go/no-go against the 12 July submission and a plan
for generalising the evaluation suite past two fields. Timeline anchors:
results writing Wed 8 Jul, pipeline polish Thu 10 Jul, submission Sun 12 Jul
23:59 BST (GitLab locks); one full training run costs ~30 h wall on an A100
(100k steps, observed for v4) plus ~2 h sampling and ~1 h evaluation.

### Verdict per candidate channel

| Channel | Data availability | Preprocessing cost | Physics interest | Pre-deadline feasible? |
|---|---|---|---|---|
| **kSZ (3rd)** | Agora lensed kSZ map exists on the Globus collection | ~½ day (same mask + patch pipeline, new FITS) | High — spectrally flat, sign-symmetric, correlated with tSZ via the same haloes; tests whether the model learns a *weak* correlated field | **Marginal** — only if training launches Tue 7 Jul and works first try; collides with the compute cutoff and leaves no retry budget |
| 2nd CIB frequency (95 or 857 GHz) | Agora CIB maps at multiple frequencies | Lowest (~½ day, identical pipeline) | Moderate — near-unity cross-correlation tests correlated-channel fidelity, less new physics | Same marginal timeline; scientifically the cheapest win but also the least novel |
| Radio galaxies | Agora radio catalogue | High — point-source-dominated field needs its own masking/normalisation decisions | High but risky — tails far heavier than tSZ, exactly the failure mode v4 already shows | **No** (post-submission) |
| 4–5 channels combined | — | Sum of the above | — | **No** (post-submission) |

**Recommendation:** do not gate the report on any new training run. Land the
*channel-generic evaluation suite* (below) as code — it is cheap, testable
without a new model, and strengthens the assessed repo — and present this
section's analysis as the feasibility study in the report's extensions
discussion. If Tuesday frees up unexpectedly, kSZ-3-channel is the only
candidate worth launching, with the explicit fallback that the report ships
on v4 results alone.

### Generalising the evaluation suite to C > 2 fields

Current state: `pipeline/evaluate.py` passes `(cib, tsz)` pairs; caches are
keyed `{statistic}__{source}.npz`. Change the source container to
`(N, C, H, W)` arrays plus a `channel_names` list from the config
(`channels: [cib_150, tsz_150, ksz_150]`, per-channel `norm_params` of length
2C), and key caches `{statistic}__{source}__{channel|pair}` so existing
two-channel caches stay valid under the names `cib`/`tsz`.

Per-statistic work:

- **Trivial per-channel loops** (power spectrum, pixel histograms — needs
  per-channel bin ranges in config, MFs, MTs, peaks/minima, WST S1/S2 and
  single-field covariance): iterate `channel_names` instead of the literal
  `[("cib", cib), ("tsz", tsz)]`.
- **Pairwise statistics** (cross-spectrum, two-field WST covariance): loop
  `itertools.combinations(channel_names, 2)` — C(3,2)=3, C(5,2)=10 pairs;
  runtime scales accordingly (two-field WST is the expensive one: ~35 min per
  pair per 100 maps on CPU, so cap `n_maps` or pairs in config).
- **Summed moments**: already field-count-agnostic (sum all channels + one
  noise realisation).
- **Cross moments**: the 12-label set is hard-coded for (a, b). Generalise to
  all multiset moments ⟨∏ ch_i^{m_i}⟩ of order 2–4: 6/10/15 combinations for
  C=3 (31 panels) — generate labels programmatically and split plots by
  moment order. This is the largest single code change.
- **tSZ stacking**: keep keyed to the tSZ channel by name (it is a
  tSZ-specific statistic). Optional new statistic: cross-stacking — cutouts
  of *other* channels at tSZ-selected peak locations, which directly probes
  the learned inter-channel correlation at cluster sites.
- **Model/training plumbing**: `Unet(channels=C)` and the sampler are already
  parameterised; `run.py`'s `_FIXED_SETTINGS` guard currently rejects
  `channels != 2` and would be lifted as part of Tier 3.

Estimated effort: ~1 day for the evaluate.py generalisation + tests (mock
3-channel data, no trained model needed), ~½ day preprocessing per new
channel, 30 h training + ~3 h sampling/evaluation per model. The code
generalisation fits Thu 10 Jul if the report is on track; everything
requiring GPU time is post-submission (GitHub v0.2).

**Obstacle (inherited from #3):** kSZ is ~an order of magnitude fainter than
tSZ; per-channel z-scoring equalises training amplitudes, but the v4 tSZ
tail-deficit finding suggests weak heavy-tailed channels are exactly where
the model underperforms — build the amplitude diagnostics (fixed-reference
histograms, stacking counts) into the multi-channel suite from day one.
