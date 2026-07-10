# Results / Discussion / Conclusion / Abstract — writing plan

Target session: **Wednesday 8 July, afternoon** (after coursework viva).
Everything referenced here already exists — no new computation is required.
Sources of truth: `runs/v4_zscore_2mJy_a100/stats/summary.md` (headline numbers),
`runs/v4_zscore_2mJy_a100/plots/*.png` (report-convention figures),
`docs/paper_code_inconsistencies.md` (reproducibility findings).

## 0. Pre-writing figure sweep (~30 min)

1. Copy the run-dir plots into `report/figures/` and convert the keepers to PDF
   (matplotlib re-export at publication style if time allows; PNG at 150 dpi is
   acceptable fallback).
2. Suggested cut (8 figures): map panels (from samples, NB14-style), power +
   cross spectra, pixel histograms, tSZ stacking, Minkowski functionals,
   summed moments, MT β/θ/r(ν), WST (S1 ⊕ S2 residuals + covariance rows).
   Peaks/minima and cross-moments can go to an appendix if page count bites.
3. Captions must state the error-bar convention once each: means from
   noiseless maps, error bars from the S4-Ultra Deep ILC tier, no debiasing,
   141 Agora / 141 Gaussian / up-to-640 DDPM maps (n_maps per statistic).

## 1. Results section — subsection order and headline numbers

Mirror the methodology order. For each subsection: figure, the quoted numbers,
one-sentence takeaway.

1. **Visual comparison** — CIB/tSZ sample panels vs Agora patches. Takeaway:
   qualitatively indistinguishable structure, correct CIB skew and tSZ
   decrement morphology, unclamped range (CIB [−4.8, +9.3]σ, tSZ [−28.1, +2.7]σ).
2. **Power spectra + rescaling decision (headline result)** —
   max |Agora−DDPM| residual 0.82σ (CIB), 0.58σ (tSZ) in per-patch scatter
   units; every band sub-σ. Measured would-be rescaling factors:
   CIB α = 0.984 ± 0.001 (band-power fit; ratio crosses unity 0.94→1.01, so
   **no rescaling applied — new result vs Prabhu's 1.0328**);
   tSZ band-power ratio 1.9 (ℓ≈330) → 1.14 (ℓ≈1400) → 1.34 (ℓ≈3900),
   would-be α 1.09 (power-weighted) / 1.12 (ℓ<2000) / 1.33 (per-map std) —
   scale-dependent, so none applied. Global-std (Prabhu-literal) estimators
   1.07 / 1.53 are inflated by the missing patch-mean scatter (0.019 vs 0.272
   CIB, 0.024 vs 0.102 tSZ in z-units) — the ℓ≲300 super-patch modes a
   patch-independent sampler cannot produce.
3. **Pixel histograms** (physical μK) — CIB PDF agreement through the tail;
   tSZ deep-decrement deficit: tail mass 0.4–0.6× Agora over −10…−50 μK.
4. **tSZ stacking** — at fixed Agora-referenced σ_T: DDPM/Agora stacked-peak
   counts 4418/4362 (5–10σ), 179/253 (10–20σ), 6/14 (≥20σ) — deficit grows
   with depth; profile shapes agree in the 5–10σ bin. Emphasise this uses the
   fixed-absolute-threshold convention (the earlier "+30–49% excess" was a
   per-map-normalisation artifact — worth one sentence in Discussion).
5. **Minkowski functionals** — quantimpy conventions; describe agreement per
   M0/M1/M2 from the plot.
6. **Bispectrum/trispectrum** — summed standardised S3/S4 vs the Gaussian null;
   raw cross moments (Appendix?) amplify the tSZ amplitude deficit as
   amplitude^(m+n) — state the convention explicitly (report §methodology now
   does).
7. **Peaks/minima** — per-map σ convention (shape-only statistic, blind to
   amplitude by construction — say so); describe residuals from the plot.
8. **Minkowski tensors** — max β residual 0.64σ (CIB, W021, ν=0.33); all
   channels/tensors ≤ 0.64σ; r(ν) within ±0.7σ everywhere while the Gaussian
   baseline reaches ≈4σ (tSZ W200/W201 low ν); DDPM reproduces the θ≈0
   orientation peak the Gaussian lacks.
9. **Wavelet scattering transforms** (4 tests) — S1/S2 residuals; covariance:
   max |Agora−DDPM| 2.13σ / 1.28σ / 2.13σ (CIB / tSZ / two-field cross),
   means 0.22σ / 0.32σ / 0.26σ over 631 / 631 / 2262 iso coefficients.
   Takeaway: the model captures multi-scale non-Gaussian structure the
   Gaussian baseline cannot, with residual tSZ offsets consistent with the
   amplitude deficit.

## 2. Discussion — argument skeleton

1. **The tSZ deficit is one coherent story, not many.** Per-map std 0.648 vs
   0.861 (z-units), low-ℓ band-power ratio 1.9, histogram tail 0.4–0.6×,
   stacked-peak counts 0.32× at ≥20σ, raw S3ᵇᵇᵇ/S4ᵇᵇᵇᵇ undershoot, broad
   +0.3–0.9σ offsets across WST coefficients. Interpretation: the model
   under-produces the rare deep-decrement tail (massive clusters); plausible
   mechanisms — MSE/EMA variance compression plus undertraining on
   heavy-tailed, z-scored data. Same sign as Prabhu's tSZ factor (1.1425);
   our map-level deficit is larger.
2. **Missing super-patch variance** — inherent to independent 6° patch
   generation; irrelevant for ℓ>300 statistics but disqualifying for any use
   needing ℓ≲300 correlations; connects to extension #1 (larger patches).
3. **Normalisation conventions determine what a statistic can see** — per-map
   σ statistics (peaks, minima) probe shape only; fixed-reference statistics
   (stacking, histograms, spectra) carry the amplitude information. The
   stacking artifact is the cautionary example.
4. **Reproducibility findings vs Prabhu et al.** — CIB needs no rescaling
   (contrast 1.0328); sampler and step count now documented (ancestral 1000;
   the paper's 1–2 s/patch is ~5–10× optimistic for that sampler); the
   library x₀ clamp had to be disabled for z-score models — cite
   `docs/paper_code_inconsistencies.md` as released artefact.
5. **Applications and limits** — pipeline testing / ILC validation with
   documented amplitude caveats; two channels at one frequency, fixed
   cosmology; point to extensions (conditional generation, more channels —
   see feasibility note in `potential_extensions.md` §12, faster samplers).

## 3. Conclusion (one to two paragraphs)

Reproduced the Prabhu et al. DDPM framework end-to-end on Agora CIB+tSZ;
sub-σ agreement on two-point statistics and most higher-order statistics with
a fully cached, config-driven evaluation pipeline (11 statistics incl. four
WST tests); two quantified departures — no CIB amplitude correction needed,
and a scale-dependent tSZ fluctuation deficit concentrated in the deep tail;
released code + documented inconsistencies as a reproducibility contribution.

## 4. Abstract (write last, ~200 words)

Sentence skeleton: context (simulation-based CMB foreground modelling) →
method (DDPM, Agora 150 GHz CIB+tSZ, 701 6°×6° patches, 100k steps, A100) →
evaluation (test set of 141 unseen patches, Gaussian baseline, 11 summary
statistics, S4-Ultra Deep ILC error convention) → numbers (power spectra
sub-σ; WST covariance mean residuals ≲ 0.3σ; MT r(ν) within ±0.7σ where the
Gaussian baseline fails at 4σ) → findings (CIB requires no post-sampling
rescaling, in contrast to the original work; tSZ shows a scale-dependent
deficit traced to the deep-decrement tail) → conclusion (DDPMs are viable
foreground emulators for pipeline testing, with amplitude caveats quantified).

## 5. Wednesday session running order

| Slot | Task |
|---|---|
| 0:00–0:30 | Figure sweep + captions (§0) |
| 0:30–3:00 | Results subsections 1–9 (§1) |
| 3:00–4:30 | Discussion (§2) |
| 4:30–5:00 | Conclusion + Abstract (§3–4) |
| 5:00–5:30 | `/verify-citations`, full compile, read-through |

Fallbacks if time is short: fold subsections 5–7 into one "morphology and
higher-order moments" subsection; move cross-moments and peaks/minima figures
to an appendix; the abstract skeleton can be filled in Thursday morning.
