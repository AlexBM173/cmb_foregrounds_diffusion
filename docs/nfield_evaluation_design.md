# N-field evaluation design note

**Status:** decisions resolved (7 Jul 2026) — ready to implement. Scopes the
generalisation of the evaluation suite (`pipeline/evaluate.py`) from the 2-field
CIB+tSZ model to the 4-field v5 run (CIB, tSZ, kSZ, CMB-lensing κ) and, more
generally, to any `C`-channel model. See §8 for the resolved decisions.

Related: [potential_extensions.md](potential_extensions.md) §12 (C>2 evaluation),
`config/v5_4ch.yaml`, `scripts/vm_preprocessing/nb03b_extract_4ch.py`.

---

## 1. What exists today (2-field)

`pipeline/evaluate.py` is a registry of 11 statistic classes
(`STATISTIC_REGISTRY`): `PowerSpectrum`, `CrossSpectrum`, `Moments`,
`CrossMoments`, `PixelHistograms`, `MinkowskiFunctionals`, `MinkowskiTensors`,
`TszStacking`, `PeakCounts`, `MinimaCounts`, `ScatteringTransforms`.

Key 2-field assumptions baked in:

- **`load_sources`** returns `name -> (cib, tsz)`, each `(N, H, W)`, for three
  sources: `agora` (test split), `ddpm` (samples), `gaussian` (baseline).
- **Denormalisation** uses a **2-entry** `norm_params = [cib_mean, cib_std,
  tsz_mean, tsz_std]` via `denormalize_dm_maps(...)`.
- **`cross_spectrum` / `cross_moments`** assume the single CIB×tSZ pair.
- **`moments`** works on the **summed** CIB+tSZ map (total 150 GHz intensity).
- **ILC noise** (`total_ilc_residuals[tier]['mv']`) is added as one `cl2map`
  realisation per patch, the **same realisation in both channels**.
- **`tsz_stacking`** stacks on tSZ SNR peaks.
- Caching: one `stats/<statistic>__<source>.npz` + JSON metadata per
  (statistic, source); re-runs reuse caches, config changes invalidate them.

The v5 patch extraction (`nb03b`) already writes an **8-entry** `norm_params`
(`[cib_mean, cib_std, tsz_mean, tsz_std, ksz_mean, ksz_std, kappa_mean,
kappa_std]`) and per-channel `.npy` files.

---

## 2. Design principles

1. **Channel-generic, not 4-hardcoded.** Drive everything off an ordered
   channel list with per-channel metadata (label, unit, plot colour, and a
   `noise_model` tag). `C=2` must keep reproducing v4 exactly.
2. **Per-channel denorm** from the `2C`-entry `norm_params`.
3. **Respect units.** CIB/tSZ/kSZ are µK; κ is dimensionless. Never combine
   across unit classes (see §4).
4. **Preserve the caching contract.** Cached arrays simply gain a channel (or
   channel-pair) axis; the `<statistic>__<source>.npz` layout is unchanged.

Proposed channel metadata (single source of truth, e.g. in the config or a
small module table):

| label | unit | noise_model | summable |
|---|---|---|---|
| cib   | µK | ilc_residual | yes |
| tsz   | µK | ilc_residual | yes |
| ksz   | µK | ilc_residual | yes |
| kappa | — (dimensionless) | lensing_n0 | no |

---

## 3. Per-statistic generalisation

**Category A — per-field (mechanical: channel loop + more panels).**
`power_spectrum` (→ C auto-spectra), `pixel_histograms` (→ C PDFs),
`minkowski_functionals`, `peak_counts`, `minima_counts`,
`scattering_transforms`, `minkowski_tensors`. No conceptual change.

**Category B — cross-field (combinatorial; the payload of a joint model).**
- `cross_spectrum`: 1 pair → **C(C−1)/2 = 6** pairs for C=4 (CIB×tSZ, CIB×kSZ,
  CIB×κ, tSZ×kSZ, tSZ×κ, kSZ×κ). These are the headline new diagnostics:
  whether the DDPM learned the joint structure, not just the marginals.
- `cross_moments`: currently 12 mixed CIB×tSZ combinations. **Decision:
  compute the full generalisation — all C(C−1)/2 = 6 pairs × 12 = 72
  combinations** — and cache them all. Meaningful subsets are chosen at
  **plot time**, not computation time, so nothing is prematurely discarded.
  Cache as a `(pair, combination, …)`-indexed array with a pair-label axis.

**Category C — the summed-channel decision (`moments`).** The summed map is
physical only within a unit class. Sum the **temperature** channels
(CIB+tSZ+kSZ = total 150 GHz foreground temperature) for the summed-moment
statistic; compute **κ moments per-field** (Category A). Gate the sum on the
`summable` flag so a mixed-unit sum can never be formed.

**Category D — field-specific and new cross-field stacks (highest-value new
science).** `tsz_stacking` stays tSZ-specific. Add, as new statistics enabled
by the extra fields:
- **`kappa_on_tsz_stacking`** — stack κ on tSZ cluster peaks → mean cluster
  convergence profile; direct test of κ–tSZ cross-morphology.
- **`ksz_stacking`** — kSZ on cluster locations (velocity-weighted / pairwise).

---

## 4. Key design decisions (call these before coding)

1. **κ carries no ILC noise.** ILC residual is a CMB-temperature
   component-separation residual, valid for CIB/tSZ/kSZ only. κ's observational
   noise is lensing **reconstruction** noise N₀^κκ (a CMB-S4 quadratic-estimator
   curve) — a different object. **Decision:** run all κ statistics *noiseless*
   for the deadline; add a `lensing_n0` noise model as future work. The
   `noise_model` metadata tag routes each channel to the right (or no) noise.
2. **Summed moments = temperature channels only** (§3 Category C).
3. **Units per output.** Auto-spectra: µK² (temperature) or dimensionless
   (κ). Cross-spectra: mixed (e.g. CIB×κ is µK·dimensionless) — label axes per
   pair; no cross-unit rescaling.
4. **Gaussian baseline — build the N-field generator.** `nb03b` did not produce
   a 4-field Gaussian baseline (`make_gaussian_realisation` is 2-field).
   **Decision: implement an N-field correlated-Gaussian generator** — a genuine
   framework extension, not deadline scaffolding. Approach:
   - Measure the full **C×C cross-power-spectrum matrix** `S(ℓ)` from the AGORA
     test patches (auto + all cross `C_ℓ`).
   - At each Fourier mode, impose the covariance by **Cholesky factor**
     `S(ℓ) = L(ℓ) L(ℓ)ᵀ`: draw C independent complex white-noise fields and
     multiply by `L(ℓ)`, then inverse-FFT to C correlated Gaussian maps.
   - **Gotcha:** measured cross-spectra can make `S(ℓ)` non-positive-definite
     (estimation noise). Regularise per-ℓ (clip negative eigenvalues / nearest-
     PSD) before the Cholesky, and log how often it triggers.
   - Lives in `flatmaps.py` alongside `make_gaussian_realisation` (e.g.
     `make_correlated_gaussian_fields(mapparams, ell, cl_matrix)`); the 2-field
     function stays as the `C=2` special case. Saves
     `gaussian_<C>field_<ptsrc>mJy_lp.npy`, which `load_sources` picks up as the
     `gaussian` source. This is the baseline that isolates what structure is
     genuinely non-Gaussian (the whole point of the project) for all C fields.

---

## 5. Why we do not directly compare v4 and v5 (report-ready rationale)

A natural question is whether the 4-field model reproduces CIB and tSZ *better
or worse* than the dedicated 2-field model (v4). We deliberately **do not** frame
a head-to-head v4-vs-v5 comparison as a result, because it would not be a
controlled experiment: two independent variables changed alongside the field
count, so any measured difference is uninterpretable.

1. **Model capacity differs.** v5 uses a base width `dim=96`; v4 used `dim=64`
   (≈2.2× the parameters). A difference in CIB/tSZ fidelity could be caused by
   the added capacity rather than by joint 4-field training. Isolating the
   field-count effect would require a *capacity-matched* 2-field control (a
   dim=96 CIB+tSZ model), which was outside the compute budget for this work.

2. **The training data itself differs.** v4's CIB/tSZ patches were extracted
   through a patch-cutter code path that applied a spin-2 (polarisation-style)
   rotation to the last two fields — a small cross-contamination of the CIB and
   tSZ maps. v5 extracts every channel independently and is free of this effect.
   Consequently the two models were trained on subtly different CIB/tSZ maps,
   **and their AGORA reference distributions differ too** (the v4 reference was
   extracted the same, affected, way). There is no common ground truth against
   which both models can be scored.

Because capacity and training data both changed, a raw v4-vs-v5 delta cannot be
attributed to "adding kSZ and κ." Rather than report a confounded and
potentially misleading comparison, **we evaluate v5 on its own terms** — its
fidelity to its own (clean) AGORA test split across all auto- and
cross-statistics — and treat the *new* cross-correlations (κ×tSZ, κ×CIB, kSZ×…)
as the scientific payload of the 4-field extension. A properly controlled
"does adding fields help the marginals?" ablation is noted as future work
requiring a capacity-matched baseline trained on the corrected pipeline.

---

## 6. Scope tiers (deadline-driven; 12 Jul)

| Tier | Work | Rationale |
|---|---|---|
| **1 — must** | `load_sources` → C-channel + `2C` denorm + channel metadata; `power_spectrum` (C autos); `cross_spectrum` (6 pairs); `pixel_histograms` (C) | Core marginals + all cross-correlations = headline |
| **2 — high** | N-field correlated-Gaussian generator (§4.4) → `gaussian` source restored; `kappa_on_tsz_stacking` + `ksz_stacking` (both) | Non-Gaussian baseline for all C fields + the unique 4-field cross-morphology science |
| **3 — completeness** | `moments` (3-temp sum + κ separate); `cross_moments` (all 72, subset chosen at plot time); MFs, peak/minima, scattering, MT (C per-field) | Full statistic coverage |
| **defer** | κ lensing-N₀ noise (κ stats run noiseless) | Realism; needs an S4 N₀ curve |

Note: the v4-vs-v5 comparison is intentionally **not** a work item (§5).

---

## 7. Implementation checklist (Tier 1)

- [ ] Channel metadata table (label, unit, colour, noise_model, summable).
- [ ] `load_sources`: return `name -> (C, N, H, W)` (or dict by label);
      per-channel denorm from `2C` `norm_params`; DDPM from `(N, C, H, W)`
      samples; `gaussian` source optional/absent (§4.4).
- [ ] `PowerSpectrum`: loop channels → C auto-spectra; per-channel units.
- [ ] `CrossSpectrum`: enumerate C(C−1)/2 pairs; per-pair axis labels/units.
- [ ] `PixelHistograms`: C PDFs; per-channel ranges from config.
- [ ] Config schema (`config/validate.py`): channel list / cross-pair
      selection / per-channel hist ranges; keep 2-field configs valid.
- [ ] Backward-compat check: `C=2` reproduces v4 caches bit-for-bit.
- [ ] Cache-key/metadata: include channel labels so a channel-set change
      invalidates stale caches.

---

## 8. Resolved decisions (7 Jul 2026)

1. **Cross-moments** — compute **all** 6 pairs × 12 = 72 combinations; select
   meaningful ones at plot time (§3 Category B).
2. **Gaussian baseline** — **build** the N-field correlated-Gaussian generator
   (§4.4); a lasting framework extension, not a stopgap.
3. **v4-vs-v5 comparison** — **not performed**; confounded by capacity and
   training-data changes. Rationale written up in §5 for the report.
4. **New stacks** — implement **both** `kappa_on_tsz_stacking` and
   `ksz_stacking`.
