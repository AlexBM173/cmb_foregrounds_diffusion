# Development Plan

Six-phase plan covering a full test suite, profiling and optimisation, parallelisation,
public documentation, package distribution, and CI/CD. Phases are largely independent,
with the following ordering constraints:
- Phase 1 (tests) should precede Phase 2 (profiling) so regressions are caught during optimisation
- Phase 2 baseline measurements should be complete before Phase 3 (parallelisation) benchmarks are run
- Phase 4 (docs) should be reasonably complete before Phase 5 (PyPI)

---

## Scope

**Thesis deadline: 2026-07-08.**
All phases are pre-submission. Phase 0 (model extensions requiring HPC) is excluded
because the cluster is down. Phases 1–6 are all in scope.

**Top open risk:** the normalisation scheme is contested between notebooks 03 and 06
(inconsistency #7 in `docs/paper_code_inconsistencies.md`) and must be resolved on the
cluster before any statistic's absolute amplitude is trusted — see the ⚠ BLOCKER in the
Cluster action plan below.

### Status

| Phase / section | Status |
|---|---|
| §1 Full unit + integration test suite | ✅ Complete — 185 tests (180 pass, 5 skip); §2.6 equivalence gate now runs in CI |
| §2.1–2.3 Profiling baseline sweeps | ✅ Complete |
| §2.4 Benchmark notebook | ✅ Complete (top-4 bottlenecks; stacking/scattering 2e–2f deprioritised) |
| §2.6a Numba JIT | ✅ Skipped — scipy owns 67% of cost, accumulation < 3% |
| §2.6b–c NumPy vectorisation + ℓ-bin precompute | ✅ Complete |
| §2.6f `torch.compile` sampling | ✅ Complete (opt-in `--compile` flag) |
| Post-sampling rescaling (inconsistency #4) | ✅ Opt-in `--rescale-cib`/`--rescale-tsz` flags added (off by default); apply/omit decision deferred to cluster day |
| Test hardening | ✅ Distinct-channel cross-moment tests + non-Gaussian (lognormal) fixture added |
| §3.2 `n_jobs` parallelisation (all functions) | ✅ Complete |
| §3.3 GPU port `map2cl_torch` | ✅ Complete |
| §3.9 Parallel benchmarks | ◑ Partial — Fig 11 (strong scaling) + summary table done on local CPU; Figs 12–15 cluster-dependent |
| §6.1–6.4 CI foundation (tests.yml + lint.yml) | ✅ Complete |
| §6.5 Additional CI/CD | ✅ Complete — pip-audit, dependency-review, pre-commit, docs build CI, equivalence gate in CI, `twine check`, concurrency groups |
| §7 Codebase cleanup | ✅ Complete (+ removed stale root `sample.py` and empty `diffusion.py`) |
| §8 Publication-quality plots | ✅ Complete |
| §9 Notebook variable naming consistency | ✅ Complete |
| §4 Documentation + ReadTheDocs | ✅ Complete — guides (incl. background, contributing), API prose, module docstrings, notebook summaries 01–14, RTD theme, docs CI with `-W`/`fail_on_warning` |
| §5 PyPI distribution | ✅ Complete (publish.yml + Trusted Publisher; `v0.1.0` live on PyPI, `v0.1.1` tagged) |
| ⚠ Normalisation scheme (inconsistency #7) | **BLOCKER** — notebooks 03/06 disagree; resolve on the cluster before trusting statistics (see Cluster action plan) |
| §3.4–3.5 Multi-GPU + MPI evaluation | To do (cluster dependent) |
| §3.7 SLURM array eval jobs | To do (needs `eval.py`; optional — notebooks 06–09 are the critical path) |

---

## Cluster action plan — when the cluster comes back online

**Deadline: 2026-07-08. Time budget: run sampling first, then statistics in
parallel, then paper figures. Everything below is copy-paste ready.**

The cluster action plan is the only remaining critical-path work. §3.4, §3.5,
and §3.7 below contain the supporting code; this section is the operational
checklist.

> **⚠ BLOCKER — resolve before Step 2 (statistics).** The tutorial notebooks
> disagree on normalisation (see inconsistency #7 in
> `docs/paper_code_inconsistencies.md`). Notebook 03 z-scores **both** channels
> and saves `..._zscore_...` files + `norm_params = [cib_mean, cib_std,
> tsz_mean, tsz_std]`; notebook 06 loads legacy `..._minmax_..._zero`/`_norm`
> filenames that 03 does not produce. On login, run
> `ls data/low_pass/2mJy/*.npy` and confirm **which files actually exist** and
> **which normalisation the checkpoint was trained with**. A z-score-trained
> model must be denormalised with `denormalize_dm_maps` (`x·std+mean`, both
> channels), never `renormalize_dm_maps`. Align the load filenames in notebooks
> 06–14 to the real ones before trusting any figure's absolute amplitude.
>
> **Critical path = notebooks 06–09 run interactively (Step 2).** `eval.py`
> (§3.10) and the SLURM array (§3.7) are *optional* convenience for multi-checkpoint
> convergence plots and are **not** required for the core thesis figures.

### Step 0 — First things on login

```bash
# Pull latest code and activate the project env
cd ~/projects/cmb_foregrounds_diffusion
git pull
source ~/activate_diffusion_project_env.sh

# Verify GPU availability and check what checkpoint exists
squeue -u $USER
ls results/*/model-*.pt | sort -V | tail -5
nvidia-smi   # if on a login node with GPUs
```

Identify the best checkpoint. The default is `results/v3_zscore_no_cib_cluster_mask/model-20.pt`.
Confirm it exists and note its path — you will use it in every command below.

---

### Step 1 — Sample (submit immediately)

Edit `sample_slurm.sh` and set:

```bash
CHECKPOINT="results/v3_zscore_no_cib_cluster_mask/model-20.pt"
OUTPUT="data/low_pass/2mJy/samples_v3_ddpm1000.npy"
BATCHES=40        # 40 × 16 × 4 GPUs = 2560 samples
BATCH_SIZE=16
SAMPLING_TIMESTEPS=""   # full DDPM (1000 steps); or set to 250 for DDIM
USE_WANDB="false"
```

Then submit:
```bash
sbatch sample_slurm.sh
```

While this runs (expected: ~2 h on 4 A100s for 2560 samples at 1000 steps,
~30 min for 250-step DDIM), proceed to Step 2 to set up the statistics run.

Also generate a Gaussian baseline if one does not already exist at
`data/low_pass/2mJy/gaussian_cib_tsz_2mJy_lp.npy` — see tutorial 05 section
on Gaussian baseline generation.

---

### Step 2 — Run statistics notebooks (as soon as samples land)

Notebooks 06–12 are independent of each other and can be submitted as
separate SLURM jobs or run interactively in a Jupyter session. Minimum
required for the thesis figures: **06, 07, 08, 09**. Notebooks 10–12 are
extended statistics.

Notebook dependencies:
- 06, 07, 08, 09, 10, 11, 12 all load `samples_v3_ddpm1000.npy` — run
  after Step 1 completes.
- 14 (paper figures) depends on outputs from 06–09.

**Interactive Jupyter session (if cluster allows):**
```bash
# Request an interactive GPU node
srun --account=mphil-dis-sl2-gpu --partition=ampere \
     --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=04:00:00 \
     --pty bash

source ~/activate_diffusion_project_env.sh
cd ~/projects/cmb_foregrounds_diffusion
jupyter lab --no-browser --port=8888 &
# Then SSH tunnel: ssh -NL 8888:localhost:8888 <cluster>
```

Run notebooks 06–09 first (core paper statistics). Then 10–12 if time
permits. Then 14 to generate the final paper figures.

**Alternatively, convert and run as SLURM batch jobs:**
```bash
# One SLURM job per notebook — submit all at once after samples land
for NB in 06 07 08 09; do
    sbatch --wrap="source ~/activate_diffusion_project_env.sh && \
        jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=7200 \
        --ExecutePreprocessor.kernel_name=python3 \
        --inplace docs/tutorials/${NB}_*.ipynb" \
        --account=mphil-dis-sl2-gpu --partition=ampere \
        --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=04:00:00 \
        --output=logs/nb_${NB}_%j.out
done
```

Each notebook saves its figures to `plots/` and its computed statistics to
intermediate `.npy` / `.npz` files — check the output path in the first
config cell of each notebook.

---

### Step 3 — Multi-GPU statistics (§3.4)

For the computationally heavy statistics (Minkowski tensors, cross-moments)
over the full 2560-sample set, use the joblib parallel wrapper that's already
in the codebase. The `n_jobs` parameter is available on:
- `compute_minkowski_tensors` — pass `n_jobs=-1`
- `compute_cross_moments` — use `parallel_cross_moments` wrapper (§3.2)
- `compute_peak_minima_counts` — pass `n_jobs=-1`

In each statistics notebook, set at the top:
```python
N_JOBS = 16   # or however many CPUs the SLURM job was allocated
```
and pass it to the relevant calls. The notebooks already import `joblib` via
the module functions — no extra setup needed.

---

### Step 4 — SLURM array over checkpoints (§3.7, optional)

If time permits and you want statistics at multiple training milestones
(to show convergence in the thesis), see the `eval_slurm_array.sh` template
in §3.7 below. This requires `eval.py` (§3.10 step 5), which has not yet
been written — skip this step unless convergence plots are needed for the thesis.

---

### Step 5 — Paper figures (notebook 14)

Run tutorial 14 last, after 06–09 have finished. It loads the precomputed
statistics from those notebooks and assembles the final paper figures.

Check `14_paper_figures.ipynb` cell 1 for the paths it expects. Update
`CHECKPOINT` and the sample file path to match what was generated in Step 1.

All figures are saved to `plots/paper/` as both PDF (for LaTeX) and PNG
(300 dpi backup). Commit the figures directory to a separate branch or
attach to the thesis submission — do NOT commit large PDFs to `main`.

---

### What is already done (no cluster needed)

| Item | Status |
|---|---|
| Trained checkpoint `v3_zscore_no_cib_cluster_mask/model-20.pt` | ✅ On cluster |
| `sample_slurm.sh` with DDIM flag | ✅ Ready to submit |
| DDIM sampling (`--sampling-timesteps 250`) | ✅ Implemented and tested |
| Statistics modules (06–12) | ✅ All notebooks complete |
| Paper figures notebook (14) | ✅ Written, needs sample data |
| `n_jobs` on `compute_minkowski_tensors`, `compute_cross_moments`, `compute_peak_minima_counts`, `select_snr_pixels` | ✅ In codebase |
| `map2cl_torch` GPU port | ✅ In codebase, equivalence-tested |
| All statistics unit-tested | ✅ 125 tests pass |

---

## Phase 1 — Testing Suite

### 1.1 Infrastructure

- **Framework:** `pytest` with `pytest-cov` for coverage reporting
- **Structure:**
  ```
  tests/
    conftest.py            # shared fixtures (synthetic maps, flatskymapparams)
    test_flatmaps.py
    test_preprocessing.py
    test_statistics.py
    test_moments.py
    test_morphology.py
    test_stacking.py
    test_masking.py
    test_peak_counts.py
    test_scattering_stats.py
    integration/
      test_power_spectrum_roundtrip.py
      test_preprocessing_pipeline.py
  ```
- **Fixtures (`conftest.py`):**
  - `rng` — seeded `np.random.default_rng(42)`
  - `flatskymapparams` — `[64, 64, 1.40625, 1.40625]` (small maps for speed)
  - `flatskymapparams_256` — `[256, 256, 1.40625, 1.40625]` (production size, used by benchmark tests)
  - `gaussian_patch` — single 64×64 Gaussian realisation
  - `patch_stack` — `(16, 64, 64)` stack of Gaussian patches
  - `patch_stack_256` — `(16, 256, 256)` stack of Gaussian patches at production resolution (required by §2.7 benchmark tests)
  - `binary_map` — 64×64 binary excursion set at a fixed threshold

**Pixel-scale convention note:** the canonical pixel scale throughout this project
is **1.40625 arcmin/pixel** (exact: 6° × 60 arcmin/° ÷ 256 pixels). This value is
used by `peak_counts.smooth_map` (its default), by the `flatskymapparams` fixtures
above, by `map2cl_torch` (§3.3), and by tutorials 10, 11, and 12. Tutorials 06, 07,
and 09 still contain `flatskymapparams = [256, 256, 1.41, 1.41]` — update these to
1.40625 before running tests or comparisons that cross module boundaries. Do not
pass `pixel_res_arcmin=1.41` explicitly to `smooth_map`; its default of 1.40625
is now authoritative.

### 1.2 Unit tests per module

**`test_flatmaps.py`**
- `get_lxly`: shape, dtype, zero at DC
- `map2cl`: output shape, positivity, symmetry under map flip
- `cl2map`: round-trip `cl2map → map2cl` recovers input spectrum within sample variance
- `make_gaussian_realisation` (single-field): pixel variance matches input `Cl` amplitude
- `make_gaussian_realisation` (correlated two-field path): generate a correlated pair by passing `cl2=` and `cl12=` kwargs; measure auto-spectra of each field and the cross-spectrum with `map2cl`; assert recovered auto-spectra match the input `Cl` arrays within sample variance and that the sign of the cross-spectrum matches `cl12`. This is the actual scientific control used in the thesis comparison — the correlated path exercises the code branch that matters for CIB×tSZ validation.
- `radial_profile`: monotonically spaced bins, correct output shape
- `bandpass_filter`: energy outside band is suppressed

**`test_preprocessing.py`**
- `apply_maxmin_normalization`: output in `[0, 1]`, min=0, max=1
- `apply_maxmin_normalization` round-trip: normalise then invert (scale by `(max - min)` and shift by `min`) recovers the original array within `atol=1e-6`; this tests that `renormalize_dm_maps` (which applies the inverse) preserves physical amplitudes — a known post-sampling inconsistency flagged in `docs/paper_code_inconsistencies.md`. The round-trip test passes only when `train_maps` (the reference used for inversion) is the same realisation that was originally min-max normalised. Shape contract: `dm_maps` must be channels-first `(N, C, H, W)` and `train_maps` channels-last `(N, H, W, C)` — `renormalize_dm_maps` transposes internally
- `apply_stdnorm`: output mean≈0, std≈1 per channel; input must be channels-last `(..., C)` — the on-disk convention (do not transpose before calling)
- `apply_stdnorm` round-trip: normalise then invert (multiply by stored std, add stored mean) recovers original within `atol=1e-6`
- Power-spectrum amplitude preservation: min-max normalise a Gaussian realisation, invert with `renormalize_dm_maps`, measure power spectrum with `map2cl`; assert ratio to original power spectrum is within 1% at each ℓ-bin
- `get_lpf_hpf`: low-pass kills high-ℓ; high-pass kills low-ℓ
- `augment_images_unique`: output has 8× the input count; no duplicate tensors
- `load_all_moments`: returns correct shape given mock `.npy` files (monkeypatch)

**`test_statistics.py`**
- `gaussian`: callable; correct value at centre
- `moments`: returns 6-tuple; centre estimates correct on a synthetic Gaussian image
- `fitgaussian`: fitted centre within 1 pixel of true centre on a noiseless image
- `stats`: correct min/max/mean/std on known array

**`test_moments.py`**
- `mean_cls`: returns tuple `(el, mean_cl, std_cl)` each shape `(n_bins,)`; `mean_cl` values positive for auto-spectra
- `mean_cross_cls`: cross-spectrum of independent maps is near zero (within noise)
- `compute_summed_moments`: shape `(N, n_bands, 3)`; Gaussian input gives S3≈0, S4≈0
- `compute_cross_moments`: returns tuple `(moments_out, labels)` where `moments_out` shape is `(N, n_bands, 12)` and `labels` is a list of 12 strings; assert `labels == ['S2aa','S2bb','S2ab','S3aaa','S3bbb','S3aab','S3abb','S4aaaa','S4bbbb','S4aaab','S4aabb','S4abbb']` (exact order, confirmed from `moments.py`) and `moments_out.shape == (N, n_bands, 12)`

**`test_morphology.py`**
- `_eigendecompose_2x2`: identity matrix gives β=1, θ=0; known anisotropic tensor gives correct β
- `_tensor_W012`: all-ones binary map gives isotropic tensor (β≈1)
- `_tensor_W200`: circular excursion set gives β≈1
- `compute_minkowski_tensors`: shape `{'W012': {'beta': (N,T), 'theta': (N,T)}}`; β ∈ [0,1]
- `compute_mfs` (requires `quantimpy`): marked `pytest.mark.optional`; returns tuple `(M0, M1, M2)` each shape `(N, T)` (M0 ≡ V0 area fraction, M1 ≡ V1 perimeter, M2 ≡ V2 Euler characteristic — same quantities, different naming convention); assert M0 decreasing with threshold

**`test_stacking.py`**
- `select_snr_pixels`: returns list of tuples; all coordinates within map bounds; min_separation enforced
- `extract_cutouts`: output shape `(M, size, size)`; returns `None` for empty coords; boundary exclusion works

**`test_masking.py`** (flat-sky only; HEALPix functions require `healpy` and are cluster-only)
- `inpaint_masked_regions`: masked pixels replaced; unmasked pixels unchanged
- `get_peak_masks`: mask where map > threshold; output shape matches input
- `boundary_apod_mask`: values in `[0, 1]`; zero at mask centre; one far from mask

**`test_peak_counts.py`**
- `smooth_map`: output shape unchanged; constant map unchanged by smoothing
- `find_peaks`: local maximum detected at correct location in synthetic map
- `find_minima`: local minimum detected correctly
- `count_peaks_binned`: shape `(N, len(thresholds))`; counts non-negative
- `compute_peak_minima_counts`: nested dict structure; shapes consistent across smoothing scales

**`test_scattering_stats.py`**
- Import handled gracefully when neither backend is available (mock both)
- `scattering_summary`: `scattering_summary` consumes a `coeffs` dict returned by `compute_scattering_coefficients` (keys `'J'`, `'S1'`, `'S2'`), not raw patches directly. Test sequence: call `compute_scattering_coefficients(patches)` to obtain `coeffs`, then call `scattering_summary(coeffs, scale_idx=None)`; assert output shape `(N, n_features)`. Do not hard-code the feature count formula `J + L*J*(J-1)/2` because it depends on the backend and J/L parameters; instead compute the reference feature count from a single-map batch (`N=1`) and assert the multi-map output's feature dimension matches that reference.

### 1.3 Integration tests

**`test_power_spectrum_roundtrip.py`**
- Generate a Gaussian realisation from a known power-law `Cl`
- Measure `Cl` back with `map2cl`
- Assert recovered spectrum within 20% of input at each ℓ-bin (loose tolerance for small maps)

**`test_preprocessing_pipeline.py`**
- Synthetic `(8, 64, 64, 2)` array (channels-last, as on disk) through normalisation → `split_data_to_tensors` → `augment_images_unique` → DataLoader
- Pass explicit `train_size=0.8, val_size=0.1, test_size=0.1` to match project usage (the function default is 70/15/15, not 80/10/10)
- Assert augmented training count = 64 (8 patches × 8× augmentation), dtype float32, channels-first shape `(64, 2, 64, 64)`, values in expected range

---

## Phase 2 — Profiling, Benchmarking, and Optimisation

The workflow is: **measure → understand → optimise → re-measure → document**.
All profiling is done twice — before and after each optimisation — so the
improvement is quantified and plotted. Results live in a dedicated notebook.

---

### 2.1 Profiling infrastructure

**Tools:**

| Tool | Purpose |
|---|---|
| `cProfile` + `snakeviz` | Call-graph profiling; interactive flame chart in browser |
| `line_profiler` | Line-by-line timing inside a single function |
| `memory_profiler` | Line-by-line memory usage |
| `tracemalloc` | Peak memory and allocation tracebacks (stdlib, no install) |
| `pytest-benchmark` | Automated, statistically robust timing with CI integration |
| `timeit` | Microbenchmark of isolated expressions |

Install:
```bash
pip install snakeviz line-profiler memory-profiler pytest-benchmark
```

**Standard harness** used for every function below:

```python
import cProfile, pstats, tracemalloc, timeit

def profile_fn(fn, *args, n_repeat=5, **kwargs):
    # Wall-clock time (median of n_repeat)
    times = timeit.repeat(lambda: fn(*args, **kwargs), number=1, repeat=n_repeat)

    # Peak memory
    tracemalloc.start()
    fn(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Call graph
    pr = cProfile.Profile()
    pr.enable(); fn(*args, **kwargs); pr.disable()
    stats = pstats.Stats(pr).sort_stats('cumtime')

    return {
        'time_median_s': sorted(times)[n_repeat // 2],
        'time_min_s':    min(times),
        'peak_mem_mb':   peak / 1024**2,
        'stats':         stats,
    }
```

---

### 2.2 Functions to profile

Priority is proportional to call frequency in the evaluation pipeline.

**`flatmaps.py`**
- `map2cl(flatskymapparams, flatskymap1, flatskymap2=None, binsize=None, minbin=100, maxbin=10000)` — called N times per evaluation; FFT-based; note `minbin`/`maxbin` not `lmin`/`lmax`
- `cl2map(mapparams, cl, el=None)` — used in Gaussian baseline generation; `el` is optional (defaults to the map's own ℓ grid when `None`)
- `make_gaussian_realisation(mapparams, el, cl)` — called ~1000× to build baseline
- `bandpass_filter(fmap, bp)` — applies a pre-computed 2D filter (`bp` from `get_lpf_hpf`); called once per ℓ-band per map per evaluation run
- `radial_profile(z, xy, bin_size, ...)` — called per SNR bin in stacking

**`moments.py`**
- `mean_cls(maps_nhw, mapparams, lmin, lmax, binsize)` — wraps `map2cl`; scales with N
- `mean_cross_cls(maps1, maps2, ...)` — same
- `compute_summed_moments(cib, tsz, bp_filters)` — inner loop over N × B bands; dominant cost in tutorial 07
- `compute_cross_moments(cib, tsz, bp_filters)` — 12 combinations; heaviest function in the package

**`morphology.py`**
- `compute_mfs(maps_nhw, norm_fn, thresholds)` — loops N × T; calls `quantimpy`
- `compute_minkowski_tensors(maps_nhw, norm_fn, thresholds, tensor_types)` — loops N × T × 3 types; expected bottleneck
- `_tensor_W012(binary_map)` — inner kernel; called N × T × 1 times
- `_tensor_W200(binary_map)` — same
- `_tensor_W201(binary_map)` — same

**`peak_counts.py`**
- `smooth_map(patch, fwhm_arcmin, pixel_res_arcmin)` — scipy gaussian_filter; called N × S times
- `count_peaks_binned(patches_nhw, thresholds, fwhm_arcmin)` — outer loop
- `compute_peak_minima_counts(patches_nhw, ...)` — full pipeline; expected ~linear in N

**`stacking.py`**
- `select_snr_pixels(tsz_maps_nhw, snr_min, snr_max, min_separation)` — applies `scipy.ndimage.maximum_filter` per map for vectorised local-maximum detection; scales as O(N·HW)
- `extract_cutouts(maps_nhw, coords, cutout_size)` — numpy slicing; likely fast

**`scattering_stats.py`**
- `compute_scattering_coefficients(patches_nhw, J, L, device)` — GPU/CPU torch; measure on both
- `compute_scattering_covariance(patches_nhw, J, L, device)` — most expensive scattering call

---

### 2.3 Scaling analysis

For each function in §2.2, sweep the relevant input dimensions and record
time and peak memory. Use log-spaced values to reveal power-law scaling.

**Dimensions to sweep:**

| Dimension | Values | Relevant functions |
|---|---|---|
| N (number of maps) | 1, 5, 10, 50, 100, 500 | all |
| H = W (map side length, pixels) | 32, 64, 128, 256 | `map2cl`, `cl2map`, `bandpass_filter`, `compute_minkowski_tensors`, `smooth_map` |
| T (number of thresholds) | 5, 10, 25, 50, 100 | `compute_mfs`, `compute_minkowski_tensors` |
| B (number of ℓ-bands) | 2, 4, 8, 16 | `compute_summed_moments`, `compute_cross_moments` |
| S (number of smoothing scales) | 1, 2, 3, 5 | `compute_peak_minima_counts` |

**Expected scaling laws** (to be verified empirically):

| Function | Expected time scaling | Expected memory scaling |
|---|---|---|
| `map2cl` | O(N · HW log HW) | O(HW) |
| `compute_summed_moments` | O(N · B · HW log HW) | O(B · HW) |
| `compute_cross_moments` | O(N · B · HW log HW) | O(B · HW) |
| `compute_minkowski_tensors` | O(N · T · HW) | O(N · T · HW) if vectorised |
| `_tensor_W012` | O(HW) | O(HW) |
| `select_snr_pixels` | O(N · HW) | O(HW) |
| `smooth_map` | O(N · HW) | O(HW) |

Fit the empirical slope in log-log space:
```python
from scipy.stats import linregress
slope, intercept, *_ = linregress(np.log(Ns), np.log(times))
# slope ≈ 1.0 means linear in N; slope ≈ 2.0 means quadratic
```

---

### 2.4 Benchmark notebook

Create `docs/tutorials/13_benchmarks.ipynb`. Structure:

**Section 1 — Setup**
- Import profiling tools and build synthetic fixtures at each size
- Define `sweep(fn, dim_name, dim_values, fixed_kwargs)` helper that calls `profile_fn`
  for each value and returns a DataFrame of `{dim, time_s, mem_mb}`

**Section 2 — Baseline measurements (pre-optimisation)**
One subsection per function group:
- 2a. Fourier utilities (`map2cl`, `cl2map`, `bandpass_filter`)
- 2b. Moment statistics (`compute_summed_moments`, `compute_cross_moments`)
- 2c. Minkowski tensors (`compute_minkowski_tensors` + inner kernels)
- 2d. Peak counts (`compute_peak_minima_counts`)
- 2e. Stacking (`select_snr_pixels`, `extract_cutouts`)
- 2f. Scattering transforms (CPU vs GPU)

**Section 3 — Figures (pre-optimisation)**
See §2.5 below.

**Section 4 — Optimisations applied**
Brief description of each change made (links to the relevant commit), with
the specific code snippet before and after.

**Section 5 — Post-optimisation measurements**
Repeat the same sweeps from Section 2 using the optimised implementations.

**Section 6 — Before/after comparison figures**
See §2.5 below.

**Section 7 — Scaling law summary table**
| Function | Pre slope | Post slope | Pre time (N=100, 256²) | Post time | Speedup |
|---|---|---|---|---|---|
| `compute_minkowski_tensors` | | | | | |
| `compute_cross_moments` | | | | | |
| `select_snr_pixels` | | | | | |
| ... | | | | | |

**Section 8 — Parallel scaling** (populated in §3.9)
- 8a. Strong scaling (Figure 11)
- 8b. Weak scaling (Figure 12)
- 8c. MPI communication overhead (Figure 13)
- 8d. GPU vs CPU throughput (Figure 14)
- 8e. Multi-GPU throughput (Figure 15)
- 8f. Parallel scaling summary table

---

### 2.5 Figures

All figures saved to `plots/benchmarks/` and embedded in the benchmark notebook.

**Figure 1 — Wall-clock time vs N (log-log), one panel per function group**
```
x-axis: N (number of maps), log scale
y-axis: median wall-clock time (seconds), log scale
series: one line per function; fitted power-law slope annotated
```

**Figure 2 — Wall-clock time vs map size H×W (log-log)**
```
x-axis: map side length (pixels), log scale [32, 64, 128, 256]
y-axis: time (seconds), log scale
series: map2cl, compute_minkowski_tensors, smooth_map
annotation: O(HW log HW) reference line for FFT functions
```

**Figure 3 — Peak memory vs N, per function**
```
x-axis: N (number of maps)
y-axis: peak memory (MB)
series: one line per function
dashed line: available RAM for reference
```

**Figure 4 — Peak memory vs map size H×W**
```
x-axis: map side length (pixels)
y-axis: peak memory (MB)
annotation: highlight the 256² production size
```

**Figure 5 — Before/after speedup bar chart (N=100, H=W=256)**
```
x-axis: function name
y-axis: speedup factor (pre_time / post_time, log scale)  # >1 means faster
colour: green if ≥2×, yellow if 1.2–2×, red if <1.2×
```

**Figure 6 — Before/after wall-clock time comparison (grouped bars)**
```
For the top-5 slowest functions:
grouped bars: [pre_time, post_time] per function
error bars: min/max over n_repeat=10 runs
```

**Figure 7 — Before/after memory comparison (grouped bars)**
Same layout as Figure 6 but for peak memory.

**Figure 8 — cProfile flame chart (snakeviz HTML)**
Embed a static screenshot of the snakeviz flame chart for
`compute_minkowski_tensors` before and after optimisation.
Export with:
```python
import cProfile
cProfile.run('compute_minkowski_tensors(...)', 'profile_pre.prof')
# then: snakeviz profile_pre.prof   (opens browser)
```

**Figure 9 — Line-profiler output table**
For the single most expensive function, embed the `line_profiler` table
(% time per line) as a styled DataFrame in the notebook.
```python
from line_profiler import LineProfiler
lp = LineProfiler()
lp.add_function(compute_minkowski_tensors)
lp.add_function(_tensor_W012)
lp.enable_by_count()
compute_minkowski_tensors(...)
lp.disable_by_count()
lp.print_stats()
```

**Figure 10 — Scaling exponent summary (heatmap)**
```
rows: function name
cols: input dimension (N, H, T, B)
cell value: fitted power-law exponent (0=constant, 1=linear, 2=quadratic)
colourmap: green (linear or better) → red (superlinear)
```

---

### 2.6 Optimisations

Applied after baseline measurements are recorded. Each optimisation is benchmarked
immediately after implementation, before moving to the next.

**Correctness gate (mandatory before any benchmark):**
Every optimised implementation must pass an equivalence test against the reference
before its benchmark numbers are recorded. Run this check in CI:
```python
import numpy as np
ref = original_fn(test_input, **kwargs)
opt = optimised_fn(test_input, **kwargs)
# For ndarray outputs:
assert np.allclose(ref, opt, rtol=1e-5, atol=1e-8), "optimised output diverges from reference"
# For tuple outputs (e.g. compute_mfs, compute_cross_moments):
for r, o in zip(ref, opt):
    assert np.allclose(r, o, rtol=1e-5, atol=1e-8)
# For torch tensors (§3.3):
assert torch.allclose(ref_tensor, opt_tensor, rtol=1e-5, atol=1e-8)
```
Add these as `tests/benchmarks/test_equivalence.py` so CI catches regressions
at every commit, not just during initial optimisation.

**a) Numba JIT — candidate pending profiling**

**Important constraint:** the true bottleneck in `_tensor_W012` and `_tensor_W201`
is `scipy.ndimage.binary_erosion` and `scipy.ndimage.sobel` — both are pre-compiled
C/Fortran and cannot run inside a Numba `nopython=True` kernel. Only the
normal-vector accumulation loop (after the scipy calls return) is JIT-eligible.
Profile first (§2.4 Section 2c) to confirm the accumulation is a material fraction
of total time before investing in Numba.

If profiling shows the accumulation loop is ≥ 30% of `compute_minkowski_tensors`
time, apply JIT to that loop only:

```python
import numba

@numba.jit(nopython=True, cache=True)
def _accumulate_normals(bx, by):
    """Accumulate W012 tensor from boundary normal components."""
    W = np.zeros((2, 2))
    for i in range(len(bx)):
        nx, ny = bx[i], by[i]
        norm = np.sqrt(nx*nx + ny*ny)
        if norm > 0:
            nx /= norm; ny /= norm
            W[0,0] += nx*nx; W[0,1] += nx*ny
            W[1,0] += nx*ny; W[1,1] += ny*ny
    return W
```

Use `cache=True` so compilation is skipped on subsequent calls (important in CI).
Warm up the JIT cache with a small dummy call before the benchmark.
If the accumulation fraction is < 30%, skip Numba and rely on NumPy vectorisation
(§2.6b) and GPU binarisation (§3.3) instead.

**b) NumPy threshold vectorisation**

`compute_minkowski_tensors` and `compute_mfs` loop over T thresholds in Python.
Binarise the entire stack at once:
```python
binary_stack = maps_nhw[:, None, :, :] > thresholds[None, :, None, None]  # (N, T, H, W)
```
Then process each `(n, t)` slice with the JIT kernel. Removes the Python threshold
loop and enables better cache locality.

**Memory warning:** at production scale (N=100, T=100, H=W=256) the `(N, T, H, W)` bool
array occupies ~655 MB. If this exceeds available RAM, chunk over N rather than materialising
the full array: process `binary_stack = chunk[:, None] > thresholds[None, :, None, None]`
inside the existing N loop and keep T vectorised.

**c) `mean_cls` / `mean_cross_cls` — pre-compute ℓ-bin mask**

Currently recomputes the ℓ-bin assignment array inside each `map2cl` call.
Compute once outside the loop:
```python
lbin_idx = np.digitize(ell_2d.ravel(), bins)   # computed once
# then reuse across all N maps
```

**d) `select_snr_pixels` — batch `maximum_filter` across N maps**

`select_snr_pixels` calls `scipy.ndimage.maximum_filter` once per map in a Python
loop over N. The filter itself is O(HW) and dominates; the Python loop adds overhead.
Two vectorisation strategies (choose after profiling §2.4 2e):

1. **Restrict to the SNR-mask bounding box.** Compute the bounding box of pixels
   in the SNR range before calling `maximum_filter` and pass only that sub-array.
   For typical tSZ maps where clusters occupy a small sky fraction this reduces the
   effective HW per call substantially.

2. **Single batched `maximum_filter` call over the N-stack.** `scipy.ndimage.maximum_filter`
   accepts N-dimensional arrays. Use `size=(1, min_separation, min_separation)` to filter
   spatially but not across the N axis — this replicates `size=min_separation` per-map
   semantics exactly, including scipy's even/odd origin handling:
   ```python
   from scipy.ndimage import maximum_filter

   # snr_stack_nhw must already be per-map SNR-normalised (each map divided by its
   # own std) before calling this, mirroring what select_snr_pixels does per map.
   # size=(1, min_separation, min_separation) is the direct 3-D equivalent of the
   # per-map size=min_separation call; do NOT substitute 2*min_separation+1 here.
   local_max_stack = maximum_filter(snr_stack_nhw, size=(1, min_separation, min_separation))
   ```
   One C-level call replaces the Python loop and improves cache utilisation.
   Correctness gate: output must match the per-map loop on the same input before
   any benchmark is recorded.

**e) Memory layout — C-contiguous enforcement**

Add `maps = np.ascontiguousarray(maps)` at the entry point of `map2cl` and
`fmap = np.ascontiguousarray(fmap)` inside `bandpass_filter`. `get_lpf_hpf` is
defined in both `flatmaps.py` (line 51, canonical — this is the version called by
`bandpass_filter` in the same module) and `preprocessing.py` (line 285, duplicate
retained for historical compatibility). The canonical `flatmaps.get_lpf_hpf` has signature
`get_lpf_hpf(flatskymapparams, lmin_lmax, filter_type=0)` where `lmin_lmax` is a
scalar (for low- or high-pass) or a `(lmin, lmax)` pair (for band-pass,
`filter_type=2`). It does not receive a map array, so contiguity enforcement
does not apply to it. Prevents silent internal copies in
numpy's FFT when arrays arrive in non-standard memory order.

**f) `torch.compile` for sampling — ✅ implemented**

```python
diffusion.model = torch.compile(diffusion.model)   # PyTorch 2.0+
```
Compiles the U-Net denoiser (called once per reverse step) rather than the
whole `GaussianDiffusion`, whose `.sample()` has Python control flow.
Expected 20–40% speedup on repeated forward passes after a one-off warm-up.
Exposed as an **opt-in** `--compile` flag on `sample.py` (default off) and a
`USE_COMPILE` toggle in `sample_slurm.sh`.

**g) Cython — fallback if Numba insufficient**

For the Minkowski tensor boundary accumulation if Numba JIT does not reach
the target speedup:
```
foregrounds_diffusion/
  _morphology_cy.pyx
  _morphology_cy.pxd
```
Build via `pyproject.toml`; keep a pure-Python fallback for environments without
a C compiler.

---

### 2.7 pytest-benchmark integration

Add `tests/benchmarks/` with one file per module, using `pytest-benchmark`
for statistically robust, reproducible timings. **Each benchmark file must also
contain the equivalence test from §2.6 so correctness and speed are validated
in the same CI run.**

```python
# tests/benchmarks/test_bench_morphology.py
import numpy as np
import pytest
from foregrounds_diffusion.morphology import compute_minkowski_tensors

thresholds = np.linspace(-3, 3, 25)

def test_minkowski_tensors_equivalence(patch_stack_256):
    """Optimised output must match reference before benchmarking."""
    ref = compute_minkowski_tensors(patch_stack_256, lambda x: x, thresholds)
    opt = compute_minkowski_tensors_v2(patch_stack_256, lambda x: x, thresholds)
    # ref and opt are dicts of {tensor_key: {stat: (N,T) array}}
    for key in ref:
        for stat in ref[key]:
            assert np.allclose(ref[key][stat], opt[key][stat], rtol=1e-5, atol=1e-8)

def test_minkowski_tensors_baseline(benchmark, patch_stack_256):
    # patch_stack_256 is the (16, 256, 256) fixture from conftest.py
    benchmark(compute_minkowski_tensors, patch_stack_256, lambda x: x, thresholds)

def test_minkowski_tensors_optimised(benchmark, patch_stack_256):
    # compute_minkowski_tensors_v2 is the Numba/vectorised version from §2.6a-b
    benchmark(compute_minkowski_tensors_v2, patch_stack_256, lambda x: x, thresholds)
```

Run and save a JSON baseline:
```bash
pytest tests/benchmarks/ --benchmark-save=baseline
pytest tests/benchmarks/ --benchmark-compare=baseline --benchmark-compare-fail=mean:20%
```

The `--benchmark-compare-fail` flag makes CI fail if any benchmark regresses
by more than 20% against the saved baseline.

---

## Phase 3 — Parallelisation

The evaluation pipeline is embarrassingly parallel over N maps on the CPU side,
and the training/sampling pipeline already uses `accelerate` for multi-GPU. This
phase documents where parallelism applies, how to implement it at each scope level
(process, node, cluster), and how to benchmark the gains alongside the single-core
results from Phase 2.

---

### 3.1 Parallelism landscape

| Scope | Tool | Best for |
|---|---|---|
| Single node, multi-core (CPU) | `joblib.Parallel` | Any loop over N maps |
| Single node, multi-GPU | `torch.multiprocessing` / `accelerate` | GPU statistics, sampling |
| Multi-node, no shared memory | `mpi4py` | Large-scale evaluation across nodes |
| Multi-node, deep learning | `accelerate` + DDP (MULTI_GPU) | Multi-node training at current model size; ZeRO only if model > VRAM |
| Coarse-grained cluster tasks | SLURM array jobs | Evaluation over many checkpoints/seeds |
| Async I/O overlap | `DataLoader(num_workers=N)` | Training data pipeline |

---

### 3.2 Embarrassingly parallel CPU functions

The following functions are independent per map and have no inter-map communication.
They can all be parallelised with the same pattern: chunk the N axis, process each
chunk in a separate worker, concatenate results.

| Function | Actual return type | Merge strategy |
|---|---|---|
| `compute_minkowski_tensors` | `dict` mapping each tensor key to `{'beta': (N,T), 'theta': (N,T)}` | merge each leaf array with `np.concatenate(..., axis=0)`; see `parallel_minkowski_tensors` below |
| `compute_mfs` | tuple `(M0, M1, M2)` each `(N, T)` | `tuple(np.concatenate([r[i] for r in results], axis=0) for i in range(3))` |
| `compute_cross_moments` | tuple `(moments_out, labels)` where `moments_out` is `(N, B, 12)` and `labels` is a fixed list of str | concatenate `moments_out` along axis 0; `labels` is identical for every chunk — take from `results[0][1]` |
| `compute_summed_moments` | `ndarray (N, B, 3)` | `np.concatenate` along axis 0 |
| `mean_cls` | tuple `(el, mean_cl, std_cl)` already averaged over N — no per-map array | not directly parallelisable via chunk-and-concat; parallelise the internal per-map loop with `joblib` instead (wrap the `for m in maps_nhw` loop) |
| `compute_peak_minima_counts` | nested `dict` of arrays with N along axis 0 | recurse over leaves with `np.concatenate(..., axis=0)` |
| `smooth_map` | `(H, W)` | applied per map inside the caller loop; no merge needed |
| `extract_cutouts` | `(M, size, size)` | `np.concatenate` along axis 0 — **note:** if `extract_cutouts` accepts a `max_cutouts` cap (default 500), that cap applies *per chunk* in the parallel version, so the total number of returned cutouts can reach `n_jobs × 500`; this differs from the serial result where the cap is global. Either disable the cap or enforce a post-concatenation trim when results must match serial output exactly. |

**Canonical joblib pattern:**

```python
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import numpy as np

def _chunk(arr, n_jobs):
    k, rem = divmod(len(arr), n_jobs)
    return [arr[i*k + min(i,rem):(i+1)*k + min(i+1,rem)] for i in range(n_jobs)]

def parallel_minkowski_tensors(maps_nhw, norm_fn, thresholds, n_jobs=-1):
    n = n_jobs if n_jobs > 0 else cpu_count()
    chunks = _chunk(maps_nhw, n)
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_minkowski_tensors)(chunk, norm_fn, thresholds)
        for chunk in chunks
    )
    # results is a list of dicts; merge tensor-by-tensor
    merged = {}
    for tensor_key in results[0]:
        merged[tensor_key] = {
            stat: np.concatenate([r[tensor_key][stat] for r in results], axis=0)
            for stat in results[0][tensor_key]
        }
    return merged
```

Set `n_jobs=-1` to use all physical cores. Use `backend="loky"` (default) for
CPU-bound tasks; use `backend="threading"` only when the function releases the GIL
(e.g. pure NumPy/SciPy code).

**Dual-array functions (`compute_summed_moments`, `compute_cross_moments`):**

Both functions accept two aligned arrays (`cib` and `tsz`) that must be chunked
together — the single-array `_chunk` pattern above would misalign the inputs.
Use `zip` to keep the two stacks in lockstep:

```python
def parallel_cross_moments(cib, tsz, bp_filters, n_jobs=-1):
    n = n_jobs if n_jobs > 0 else cpu_count()
    cib_chunks = _chunk(cib, n)
    tsz_chunks  = _chunk(tsz, n)
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_cross_moments)(c, t, bp_filters)
        for c, t in zip(cib_chunks, tsz_chunks)
    )
    moments_out = np.concatenate([r[0] for r in results], axis=0)
    labels = results[0][1]   # identical for every chunk — take from first result
    return moments_out, labels
```

Apply the same `zip(cib_chunks, tsz_chunks)` pattern for
`parallel_summed_moments` wrapping `compute_summed_moments(cib, tsz, bp_filters)`.

**Add `n_jobs` parameter to each function** in the public API so users can opt in
without importing `joblib` directly:

```python
def compute_minkowski_tensors(maps_nhw, norm_fn, thresholds, n_jobs=1):
    if n_jobs != 1:
        return parallel_minkowski_tensors(maps_nhw, norm_fn, thresholds, n_jobs)
    # existing single-threaded implementation ...
```

The default `n_jobs=1` preserves current behaviour; no code that uses the function
needs to change.

---

### 3.3 GPU acceleration for statistics

Several CPU-bound functions can be ported to PyTorch to exploit GPU parallelism.
The key criterion is whether the cost of transferring data to/from the GPU is
amortised across the batch — worth it for N ≥ 50 on 256² maps.

**`map2cl` → `torch.fft.rfft2`**

```python
import torch, math

def map2cl_torch(maps_nhw: torch.Tensor, lbin_idx_rfft, bin_counts, n_bins,
                 dx_arcmin: float):
    """
    maps_nhw      : (N, H, W) float tensor on GPU
    lbin_idx_rfft : (H*(W//2+1),) long tensor — ℓ-bin index for each rfft2 pixel,
                    derived from get_lxly applied to the rfft2 frequency grid
                    (NOT from ell_2d.ravel() which covers the full fft2 grid)
    bin_counts    : (n_bins,) float tensor — number of rfft2 pixels per bin
    dx_arcmin     : pixel size in arcminutes (same for x and y)
    """
    N, H, W = maps_nhw.shape
    dx_rad = math.radians(dx_arcmin / 60.)
    # Physical normalisation: matches CPU map2cl which computes
    # |fft2(map) * dx_rad|^2 / (nx * ny)
    norm = dx_rad ** 2 / (H * W)
    fft   = torch.fft.rfft2(maps_nhw)              # (N, H, W//2+1) complex
    power = (fft.real**2 + fft.imag**2) * norm     # (N, H, W//2+1)
    flat  = power.reshape(N, -1)                   # (N, H*(W//2+1))
    # Allocate n_bins+1 so the sentinel index (= n_bins) set by build_lbin_idx_rfft
    # for out-of-range pixels has a valid write target; scatter_add_ with index n_bins
    # into an n_bins-wide tensor causes a CUDA index-out-of-bounds crash.
    cl    = torch.zeros(N, n_bins + 1, device=maps_nhw.device)
    cl.scatter_add_(1, lbin_idx_rfft.expand(N, -1), flat)
    return cl[:, :n_bins] / bin_counts              # discard sentinel column; normalise by hits
```

**Note on `lbin_idx_rfft`:** `torch.fft.rfft2` returns only the non-redundant half
of the spectrum (`W//2+1` columns), so the ℓ-bin assignment must be built from the
rfft2 frequency grid — not from `ell_2d.ravel()` which covers the full `(H, W)`
fft2 grid. Pre-compute once before any batched call:

```python
from foregrounds_diffusion.flatmaps import get_lxly
import numpy as np, torch

def build_lbin_idx_rfft(mapparams, binsize=None, minbin=100, maxbin=10000):
    """Build ℓ-bin index tensor for map2cl_torch, matching CPU map2cl bin edges.

    get_lxly returns (lx, ly) on the full (H, W) fft2 grid.  rfft2 keeps
    columns 0 … W//2 (inclusive), so slice the first W//2+1 columns only.
    The binsize default mirrors the smallest lx spacing that map2cl uses.
    """
    nx, ny, dx, dy = mapparams
    lx, ly = get_lxly(mapparams)                        # (H, W) each
    ell_2d = np.sqrt(lx**2 + ly**2)
    ell_rfft = ell_2d[:, :nx//2 + 1]                   # (H, W//2+1)
    if binsize is None:
        binsize = lx[0, 1] - lx[0, 0]                  # smallest ℓ step
    bins = np.arange(minbin, maxbin, binsize)            # matches radial_profile bin edges
    n_bins = len(bins)
    lbin_idx = np.digitize(ell_rfft.ravel(), bins) - 1  # 0-indexed; range -1 … n_bins-1
    valid = (lbin_idx >= 0) & (lbin_idx < n_bins)
    lbin_idx[~valid] = n_bins                            # sentinel for out-of-range
    bin_counts = np.bincount(lbin_idx[valid], minlength=n_bins).astype(np.float32)
    # Note: radial_profile divides each bin sum by the count of NONZERO pixels
    # ('hits'), not the total number of pixels in the bin.  For Gaussian PSDs
    # every pixel is non-zero and the two counts agree; the rtol=1e-4 tolerance
    # in the equivalence test below absorbs any residual float32/float64
    # difference, so no correction is needed here.
    return (torch.from_numpy(lbin_idx).long(),          # (H*(W//2+1),)
            torch.from_numpy(bin_counts),               # (n_bins,)
            n_bins)
```

Call `build_lbin_idx_rfft(mapparams)` once, move the tensors to GPU, and pass them
to every `map2cl_torch` call.

**Equivalence test (mandatory before benchmarking):**
```python
import torch, numpy as np
from foregrounds_diffusion.flatmaps import map2cl

rng = np.random.default_rng(42)
maps_np = rng.standard_normal((8, 256, 256)).astype(np.float32)
mapparams = [256, 256, 1.40625, 1.40625]

# CPU reference (per-map, then stack)
el_ref, cl_ref = zip(*[map2cl(mapparams, m) for m in maps_np])
cl_ref = np.stack(cl_ref)   # (8, n_bins)

# GPU port
maps_t = torch.from_numpy(maps_np).cuda()
cl_gpu = map2cl_torch(maps_t, lbin_idx_rfft, bin_counts, n_bins, dx_arcmin=1.40625)
assert torch.allclose(torch.from_numpy(cl_ref).cuda(), cl_gpu, rtol=1e-4, atol=1e-8), \
    "map2cl_torch output does not match CPU map2cl"
```
(Loose `rtol=1e-4` tolerates float32 vs float64 accumulation differences.)

This computes all N power spectra in a single batched FFT call, avoiding the Python
loop over N maps that the CPU version requires.

**Minkowski tensor binarisation on GPU**

The threshold broadcast `maps_nhw[:, None] > thresholds[None, :, None, None]` is
already vectorisable; running it on a GPU tensor gives a (N, T, H, W) bool array
in microseconds.

**Scattering transforms** (`scattering_stats.py`) are already torch-based and
benefit from GPU automatically; no changes needed.

---

### 3.4 Multi-GPU on a single node (evaluation)

For evaluation runs on a single 4-GPU node (as available on CSD3 Ampere nodes),
distribute N maps across GPUs with `torch.multiprocessing`:

```python
import torch.multiprocessing as mp

def _worker(rank, maps_chunk, result_queue, fn, kwargs):
    """fn must be a torch-native function that accepts a GPU tensor and returns
    a GPU tensor (e.g. map2cl_torch).  Do NOT pass numpy statistics functions
    here — they do not accept CUDA tensors."""
    device = torch.device(f"cuda:{rank}")
    inp = torch.from_numpy(maps_chunk).to(device)   # numpy → GPU tensor
    out = fn(inp, **kwargs)                          # fn returns GPU tensor
    result_queue.put((rank, out.cpu().numpy()))      # GPU → CPU → numpy

def multi_gpu_eval(maps_nhw, fn, n_gpus=4, **kwargs):
    """Distribute maps across GPUs. fn must be torch-native (accept/return GPU tensors).
    For numpy statistics functions use joblib (§3.2) instead."""
    chunks = np.array_split(maps_nhw, n_gpus)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(i, chunks[i], q, fn, kwargs))
             for i in range(n_gpus)]
    for p in procs: p.start()
    results = [q.get() for _ in procs]
    for p in procs: p.join()
    results.sort(key=lambda x: x[0])
    return np.concatenate([r for _, r in results], axis=0)
```

**Alternatively, use `accelerate` for evaluation** — it handles device placement,
gather/scatter, and mixed precision automatically:

```python
from accelerate import Accelerator

accelerator = Accelerator()
dataset = MapDataset(maps_nhw)
loader  = DataLoader(dataset, batch_size=32)
loader  = accelerator.prepare(loader)

all_results = []
for batch in loader:
    out = compute_statistic(batch)           # runs on accelerator.device
    all_results.append(accelerator.gather(out))
results = torch.cat(all_results).cpu().numpy()
```

This approach works identically on 1, 4, or 32 GPUs without code changes —
only the `accelerate config` needs updating.

---

### 3.5 Multi-node parallelism with `mpi4py`

For analysis across O(1000) maps distributed over multiple CSD3 nodes, use MPI
via `mpi4py`. The pattern is: rank 0 holds all maps and scatters chunks; each rank
computes its local statistics; rank 0 gathers and merges.

**Install:**
```bash
pip install mpi4py   # uses the system MPI; on CSD3 load the appropriate OpenMPI module
                     # first (exact string varies by partition — check `module avail openmpi`)
```

**Generic scatter–compute–gather wrapper:**

```python
from mpi4py import MPI
import numpy as np

def mpi_parallel_eval(maps_nhw, fn, **kwargs):
    """Scatter maps over MPI ranks, compute fn on each chunk, gather results.

    comm.scatter with a list of objects (not a flat numpy buffer) does not
    require equal chunk sizes — no padding needed.  Error handling uses
    comm.Abort() to prevent ranks from deadlocking if one raises an exception.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Rank 0 splits into variable-length chunks; list scatter handles unequal sizes
    if rank == 0:
        chunks = np.array_split(maps_nhw, size)   # list of arrays, possibly unequal
    else:
        chunks = None

    local_chunk = comm.scatter(chunks, root=0)    # each rank gets its slice

    try:
        local_result = fn(local_chunk, **kwargs)
    except Exception as exc:
        print(f"[rank {rank}] error in fn: {exc}", flush=True)
        comm.Abort(1)                             # prevents other ranks from hanging
        raise

    all_results = comm.gather(local_result, root=0)

    if rank == 0:
        # For ndarray results: np.concatenate(all_results, axis=0)
        # For tuple results (e.g. compute_mfs returns (M0,M1,M2)):
        #   tuple(np.concatenate([r[i] for r in all_results], axis=0) for i in range(3))
        # For dict-of-arrays (e.g. compute_minkowski_tensors):
        #   recurse over leaves with np.concatenate(..., axis=0)
        return np.concatenate(all_results, axis=0)
```

**Run with `mpirun` on a single node:**
```bash
mpirun -n 4 python eval_mpi.py
```

**Run across multiple CSD3 nodes via SLURM:**
```bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1      # 1 MPI rank per node (use CPU cores within each via joblib)
srun python eval_mpi.py
```

Combine MPI across nodes with `joblib` within each node (§3.2) for a hybrid
parallelism strategy: `n_nodes × n_cores_per_node` total workers.

---

### 3.6 Multi-node training (DEFERRED — post-thesis)

**Status: explicitly deferred per scope triage (§"Deadline and scope triage").
The dim=64 U-Net fits comfortably in a single Ampere GPU and trains in hours, so
multi-node training provides no convergence benefit for the current thesis work.
This section documents the DDP approach for future reference.**

Current training uses single-node, single-GPU (`accelerate launch --num_processes 1`).
The appropriate multi-GPU / multi-node strategy for this model size is
**DDP (DistributedDataParallel)**, not DeepSpeed ZeRO, because:
- The dim=64 U-Net has ~50 M parameters; all fit in VRAM without sharding
- ZeRO-2 optimizer state partitioning adds communication overhead with no memory benefit
- DDP scales linearly for gradient synchronisation without the DeepSpeed dependency

**`accelerate config` for multi-node DDP:**
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_machines: 4
num_processes: 16   # 4 nodes × 4 GPUs each
machine_rank: 0     # override per node via SLURM env var
main_process_ip: <head_node_ip>
main_process_port: 29500
mixed_precision: fp16
```

**`train_slurm_multinode.sh`:**
```bash
#!/bin/bash
#SBATCH --job-name=cmb_multinode
#SBATCH --account=mphil-dis-sl2-gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1   # one launcher per node; accelerate manages all 4 local GPUs
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --partition=ampere

RUN_NAME="multinode_run_v1"
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

srun accelerate launch \
    --num_processes 16 \
    --num_machines 4 \
    --machine_rank $SLURM_NODEID \
    --main_process_ip $HEAD_NODE \
    --main_process_port 29500 \
    train.py --run-name "$RUN_NAME"
```

`--ntasks-per-node=1` ensures `srun` launches exactly one task (one `accelerate launch`
process) per node. `accelerate` then spawns 4 sub-processes internally (16 total / 4 nodes).
Setting `--ntasks-per-node=4` would cause `srun` to start 4 competing launchers per node,
each trying to own all 4 GPUs and all sharing the same `$SLURM_NODEID`, resulting in 64
conflicting processes. SLURM populates `$SLURM_NODEID` (0-indexed node rank) automatically.

**Note on Ampere GPU variants:** CSD3 Ampere nodes are predominantly 80 GB A100s; some
partitions have 40 GB variants. The dim=64 U-Net is comfortable in either. If experimenting
with `compute_scattering_covariance` (which builds a large intermediate tensor), confirm
available VRAM with `nvidia-smi` before running and reduce batch size if needed.

---

### 3.7 SLURM array jobs for coarse-grained evaluation

**Purpose:** generate statistics at multiple training milestones in parallel,
to show convergence curves in the thesis (e.g. power spectrum error vs
training step). Each array task samples from one checkpoint and runs all
statistics, saving results to a per-milestone NPZ.

**Prerequisite:** `foregrounds_diffusion/eval.py` must exist (§3.10 step 5).

**`eval_slurm_array.sh`:**

```bash
#!/bin/bash
#SBATCH --job-name=cmb_eval
#SBATCH --account=mphil-dis-sl2-gpu
#SBATCH --array=0-9              # tasks 0–9 → checkpoints model-5 through model-50
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=ampere
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err

TASK_ID=$SLURM_ARRAY_TASK_ID
RUN="v3_zscore_no_cib_cluster_mask"
CHECKPOINT="results/${RUN}/model-$((TASK_ID * 5 + 5)).pt"
OUTPUT="results/eval/${RUN}/stats_milestone_${TASK_ID}.npz"
AGORA_CIB="data/low_pass/2mJy/CIB_map_150GHz_256_st6_zscore_2mJy_lp.npy"
AGORA_TSZ="data/low_pass/2mJy/tSZ3_map_150GHz_256_st6_zscore_2mJy_lp.npy"
NORM_PARAMS="data/low_pass/2mJy/norm_params_2mJy.npy"

mkdir -p results/eval/${RUN} logs

source "$HOME/activate_diffusion_project_env.sh"

accelerate launch foregrounds_diffusion/eval.py \
    --checkpoint  "$CHECKPOINT" \
    --output      "$OUTPUT" \
    --agora-cib   "$AGORA_CIB" \
    --agora-tsz   "$AGORA_TSZ" \
    --norm-params "$NORM_PARAMS" \
    --n-samples   512 \
    --batch-size  16 \
    --n-jobs      16
```

Submit:
```bash
mkdir -p logs
sbatch eval_slurm_array.sh
# monitor
squeue -u $USER
```

Collect and compare after all tasks finish:
```python
import numpy as np, glob

files = sorted(glob.glob("results/eval/v3_zscore_no_cib_cluster_mask/stats_milestone_*.npz"))
milestones = []
for f in files:
    d = np.load(f, allow_pickle=True)
    milestones.append({
        'step':    int(d['step']),
        'cl_cib':  d['cl_cib'],    # (n_bins,) mean power spectrum error
        'cl_tsz':  d['cl_tsz'],
        'moments': d['moments'],    # (n_bands, 12) cross-moment residuals
    })
milestones.sort(key=lambda x: x['step'])

# Plot convergence: power spectrum χ² vs training step
import matplotlib.pyplot as plt
steps = [m['step'] for m in milestones]
cl_err = [np.mean((m['cl_cib'] - 1)**2) for m in milestones]  # fractional error
plt.plot(steps, cl_err, marker='o')
plt.xlabel('Training step'); plt.ylabel('Mean CIB Cℓ fractional error²')
plt.yscale('log')
```

---

### 3.8 Training data pipeline

Within training, I/O is rarely the bottleneck at 256² patches but can become one
at larger sizes or on slow shared filesystems.

**Overlapping I/O with GPU compute:**
```python
DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=8,          # 8 background processes preload batches
    pin_memory=True,        # allocate in pinned (page-locked) host memory for fast H2D
    prefetch_factor=2,      # keep 2 batches queued per worker
    persistent_workers=True # keep worker processes alive between epochs
)
```

**Lustre striping on CSD3** (relevant if data lives on `/rds/` or `/sptlocal/`):
```bash
lfs setstripe -c 4 data/low_pass/   # set stripe count for the directory
```
**Important:** `lfs setstripe` only applies to **new files** created in the directory
after the command runs. Existing `.npy` files retain their original stripe layout.
To stripe existing files, copy them through the newly-striped directory:
```bash
lfs setstripe -c 4 /rds/project/<project>/data_striped/
cp data/low_pass/*.npy /rds/project/<project>/data_striped/
```
This pre-fragments the `.npy` files across storage servers so multiple DataLoader
workers can read simultaneously without contention.

---

### 3.9 Parallelisation benchmarks

Extend `docs/tutorials/13_benchmarks.ipynb` with **Section 8 — Parallel scaling**
(add this section to the notebook structure defined in §2.4) covering parallel
scaling. New figures to add:

**Figure 11 — Strong scaling: time vs n_workers (fixed N=500, 256²)**
```
x-axis: number of workers (1, 2, 4, 8, 16, 32)
y-axis: wall-clock time (seconds)
series: compute_minkowski_tensors, compute_cross_moments, compute_peak_minima_counts
reference line: ideal linear speedup (t₁ / n_workers)
```
Strong scaling efficiency = (t₁ / (n × tₙ)) × 100%. Efficiency >80% at 8 workers
is a reasonable target for these functions; expect degradation above 16 due to
process spawn overhead and memory bandwidth saturation.

**Figure 12 — Weak scaling: time vs n_workers (fixed N=50 maps per worker)**
```
x-axis: number of workers (1, 2, 4, 8, 16)
y-axis: wall-clock time (seconds), should be flat for ideal scaling
series: same functions as Figure 11
annotation: +10% and +20% tolerance bands
```

**Figure 13 — Communication overhead fraction**
```
For MPI runs (multi-node):
x-axis: number of nodes (1, 2, 4, 8)
y-axis: fraction of total time spent in scatter/gather (not compute)
annotation: target <10% for this workload
```

**Figure 14 — GPU vs CPU speedup for torch-ported functions**
```
x-axis: N (number of maps)
y-axis: CPU time / GPU time
series: map2cl_torch, compute_minkowski_tensors (after GPU port)
annotation: PCIe transfer breakeven point
dashed: speedup = 1 (breakeven)
```

**Figure 15 — Multi-GPU evaluation throughput (maps per second)**
```
x-axis: number of GPUs (1, 2, 4)
y-axis: maps processed per second
series: per-function throughput
```

Also add a **parallel scaling summary table** to the benchmark notebook:

| Function | Serial (N=500) | 8 cores | 4 GPUs | Strong eff. @8 | Notes |
|---|---|---|---|---|---|
| `compute_minkowski_tensors` | | | | | |
| `compute_cross_moments` | | | | | |
| `map2cl` | | | | | |
| `compute_peak_minima_counts` | | | | | |

---

### 3.10 Implementation order

1. ✅ Add `n_jobs` parameter to all functions in §3.2 (one PR per module)
2. ✅ Benchmark `joblib` parallel on local machine (Figure 11, 12)
3. ✅ Port `map2cl` to torch; benchmark GPU speedup (Figure 14)
4. Write `mpi4py` wrapper and test on 2 CSD3 nodes (Figure 13) — **cluster dependent**
5. Create `foregrounds_diffusion/eval.py` — **cluster dependent; full spec below**
6. Add `train_slurm_multinode.sh` (DDP, no DeepSpeed) — **deferred post-thesis**
7. Benchmark multi-GPU evaluation (Figure 15) — **cluster dependent**

---

#### `eval.py` — full specification

**Purpose:** single-command pipeline that samples from a checkpoint, loads
AGORA truth patches, builds a Gaussian baseline, runs all evaluation
statistics on all three sets, and saves results to NPZ.  Used by
`eval_slurm_array.sh` (§3.7) to evaluate multiple checkpoints in parallel.

**CLI:**
```
accelerate launch foregrounds_diffusion/eval.py \
    --checkpoint  results/run/model-20.pt \
    --output      results/eval/stats.npz \
    --agora-cib   data/low_pass/2mJy/CIB_...npy \
    --agora-tsz   data/low_pass/2mJy/tSZ3_...npy \
    --norm-params data/low_pass/2mJy/norm_params_2mJy.npy \
    --n-samples   512 \
    --batch-size  16 \
    --sampling-timesteps 250 \   # optional DDIM; default = full 1000-step DDPM
    --n-jobs 16
```

**Implementation skeleton:**

```python
"""Unified evaluation script: sample → statistics → NPZ output."""
import argparse, numpy as np, torch
from pathlib import Path
from accelerate import Accelerator
# NOTE: sample.py exposes build_model, load_checkpoint, and sample(diffusion,
# accelerator, num_batches, batch_size) — there is no `sample_batches`.
from foregrounds_diffusion.sample import build_model, load_checkpoint, sample
from foregrounds_diffusion.preprocessing import denormalize_dm_maps
from foregrounds_diffusion.flatmaps import make_gaussian_realisation
from foregrounds_diffusion.moments import mean_cls, mean_cross_cls, compute_cross_moments
from foregrounds_diffusion.morphology import compute_minkowski_tensors
from foregrounds_diffusion.peak_counts import compute_peak_minima_counts

FLATSKYMAPPARAMS = [256, 256, 1.40625, 1.40625]
BP_EDGES = [(100, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 7000)]
THRESHOLDS = np.linspace(-3, 3, 30)
SMOOTHING_SCALES = [1.0, 2.5, 5.0]   # arcmin FWHM, matching notebook 10
PEAK_THRESHOLDS = np.linspace(-3, 3, 25)
MINIMA_THRESHOLDS = np.linspace(-3, 0, 15)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',  required=True)
    p.add_argument('--output',      required=True)
    p.add_argument('--agora-cib',   required=True)
    p.add_argument('--agora-tsz',   required=True)
    p.add_argument('--norm-params', required=True)
    p.add_argument('--n-samples',   type=int, default=512)
    p.add_argument('--batch-size',  type=int, default=16)
    p.add_argument('--sampling-timesteps', type=int, default=None)
    p.add_argument('--n-jobs',      type=int, default=1)
    return p.parse_args()


def run_statistics(cib, tsz, mapparams, bp_edges, n_jobs):
    """Run all evaluation statistics on a (N,H,W) CIB and tSZ stack."""
    from foregrounds_diffusion.flatmaps import get_lpf_hpf, bandpass_filter

    # Build bandpass filters once
    bp_filters = [
        get_lpf_hpf(mapparams, (lmin, lmax), filter_type=2)
        for lmin, lmax in bp_edges
    ]

    # Power spectra
    el, cl_cib, cl_cib_std = mean_cls(
        cib, mapparams, lmin=100, lmax=7000, binsize=200
    )
    _,  cl_tsz, cl_tsz_std = mean_cls(
        tsz, mapparams, lmin=100, lmax=7000, binsize=200
    )
    # Cross-spectrum: use mean_cross_cls (mean_cls has NO flatskymap2 arg)
    _, cl_cross, cl_cross_std = mean_cross_cls(
        cib, tsz, mapparams, lmin=100, lmax=7000, binsize=200
    )

    # Cross-moments (12 combinations)
    moments_out, labels = compute_cross_moments(
        cib, tsz, bp_filters, n_jobs=n_jobs
    )

    # Minkowski tensors
    mt = compute_minkowski_tensors(
        tsz, lambda x: x, THRESHOLDS,
        tensor_types=['W012'], n_jobs=n_jobs
    )

    # Peak / minima counts
    pk = compute_peak_minima_counts(
        tsz, PEAK_THRESHOLDS, MINIMA_THRESHOLDS, SMOOTHING_SCALES
    )

    return dict(
        el=el,
        cl_cib=cl_cib,     cl_cib_std=cl_cib_std,
        cl_tsz=cl_tsz,     cl_tsz_std=cl_tsz_std,
        cl_cross=cl_cross, cl_cross_std=cl_cross_std,
        moments=moments_out, moment_labels=np.array(labels),
        mt_beta=mt['W012']['beta'],
        pk=pk,
    )


def main():
    args = parse_args()

    # ---- 1. Load AGORA patches and denormalise ----
    # norm_params_2mJy.npy is a plain ndarray [cib_mean, cib_std, tsz_mean, tsz_std]
    # written by notebook 03 (BOTH channels z-scored) — NOT a dict, no .item().
    # See inconsistency #7 in docs/paper_code_inconsistencies.md: confirm on the
    # cluster that the checkpoint was z-score-trained before trusting this.
    cib_mean, cib_std, tsz_mean, tsz_std = np.load(args.norm_params)
    agora_cib_raw = np.load(args.agora_cib)  # (N, H, W) or (N, H, W, 1)
    agora_tsz_raw = np.load(args.agora_tsz)
    if agora_cib_raw.ndim == 4:
        agora_cib_raw = agora_cib_raw[..., 0]
        agora_tsz_raw = agora_tsz_raw[..., 0]
    n = min(args.n_samples, len(agora_cib_raw))
    # z-score inverse (x * std + mean) for both channels
    agora_cib = agora_cib_raw[:n] * cib_std + cib_mean
    agora_tsz = agora_tsz_raw[:n] * tsz_std + tsz_mean

    # ---- 2. Generate DDPM samples ----
    # sample() requires an Accelerator and gathers across GPUs; total returned
    # per call = num_batches * batch_size * num_processes.
    accelerator = Accelerator(split_batches=True, mixed_precision="fp16")
    diffusion = build_model(channels=2, sampling_timesteps=args.sampling_timesteps)
    diffusion = diffusion.to(accelerator.device)
    diffusion = load_checkpoint(diffusion, args.checkpoint, accelerator)
    n_batches = (args.n_samples + args.batch_size - 1) // args.batch_size
    samples = sample(diffusion, accelerator, num_batches=n_batches,
                     batch_size=args.batch_size)          # (N, 2, H, W), z-score space
    # Denormalise both channels with the z-score inverse (see #7)
    samples = denormalize_dm_maps(samples[:n], cib_mean, cib_std, tsz_mean, tsz_std)
    ddpm_cib = samples[:, 0]
    ddpm_tsz = samples[:, 1]

    # ---- 3. Gaussian baseline ----
    el_agora, cl_agora_cib, _ = mean_cls(agora_cib, FLATSKYMAPPARAMS, 100, 7000, 200)
    _, cl_agora_tsz, _ = mean_cls(agora_tsz, FLATSKYMAPPARAMS, 100, 7000, 200)
    gauss_cib = np.stack([
        make_gaussian_realisation(FLATSKYMAPPARAMS, el_agora, cl_agora_cib)
        for _ in range(n)
    ])
    gauss_tsz = np.stack([
        make_gaussian_realisation(FLATSKYMAPPARAMS, el_agora, cl_agora_tsz)
        for _ in range(n)
    ])

    # ---- 4. Statistics on all three sets ----
    print('Running statistics on AGORA ...')
    stats_agora = run_statistics(agora_cib, agora_tsz, FLATSKYMAPPARAMS,
                                 BP_EDGES, args.n_jobs)
    print('Running statistics on DDPM ...')
    stats_ddpm  = run_statistics(ddpm_cib,  ddpm_tsz,  FLATSKYMAPPARAMS,
                                 BP_EDGES, args.n_jobs)
    print('Running statistics on Gaussian ...')
    stats_gauss = run_statistics(gauss_cib, gauss_tsz, FLATSKYMAPPARAMS,
                                 BP_EDGES, args.n_jobs)

    # ---- 5. Save ----
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        step=int(Path(args.checkpoint).stem.split('-')[-1]),
        checkpoint=args.checkpoint,
        **{f'agora_{k}': v for k, v in stats_agora.items()},
        **{f'ddpm_{k}':  v for k, v in stats_ddpm.items()},
        **{f'gauss_{k}': v for k, v in stats_gauss.items()},
    )
    print(f'Saved → {args.output}')


if __name__ == '__main__':
    main()
```

**Key notes for implementation:**
- `sample.py` already exposes reusable `build_model`, `load_checkpoint`, and
  `sample(diffusion, accelerator, num_batches, batch_size)` — `eval.py` imports
  these directly. Because `sample()` gathers across GPUs, the number of samples
  returned per call is `num_batches × batch_size × num_processes`; size
  `--n-samples` and `--batches` accordingly, and run under `accelerate launch`.
- `norm_params_2mJy.npy` is a plain `np.ndarray` of shape `(4,)` ordered
  `[cib_mean, cib_std, tsz_mean, tsz_std]` (notebook 03). Load with positional
  unpacking — **no `.item()`, no dict keys**. **Before trusting amplitudes,
  resolve inconsistency #7**: confirm the checkpoint's training normalisation
  and that the `_zscore_` files (not the legacy `_minmax_` names in notebook 06)
  are the ones on disk. A z-score-trained model must be denormalised with
  `denormalize_dm_maps`, never the min-max `renormalize_dm_maps`.
- Minkowski functionals (`compute_mfs`) require `quantimpy` which may not
  be installed on the cluster — wrap in a `try/except ImportError` and skip
  if not available; log a warning.
- `compute_peak_minima_counts` returns a nested dict; use
  `np.savez(..., pk=np.array(pk, dtype=object))` to preserve the structure,
  or flatten to arrays before saving.

---

## Phase 4 — Documentation and ReadTheDocs

### 4.1 Docstring audit

All public functions should have NumPy-style docstrings covering:
- One-line summary
- `Parameters` section with types and shapes
- `Returns` section with types and shapes
- `Notes` for any non-obvious behaviour (e.g. normalisation conventions, edge cases)

Priority order: `flatmaps` → `preprocessing` → `moments` → `morphology` → `masking`.
`statistics`, `stacking`, `peak_counts` are already reasonably documented.

### 4.2 Sphinx setup

```
docs/
  conf.py
  index.rst
  api/
    index.rst           # auto-generated from docstrings via autodoc
  guides/
    installation.rst
    quickstart.rst
    data_conventions.rst
  notebooks/            # rendered via nbsphinx
    (symlinks to docs/tutorials/*.ipynb)
  _static/
  requirements.txt      # sphinx deps for RTD build
```

**`docs/conf.py` key settings:**
```python
import sys, os
# Allow autodoc to find the package source without installing it as a package.
# Required when RTD installs only docs/requirements.txt (no `pip install .`).
sys.path.insert(0, os.path.abspath('..'))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",      # NumPy docstring support
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",       # for ℓ, θ, β notation
    "nbsphinx",                 # render tutorial notebooks
    "sphinx_copybutton",
]
html_theme = "furo"             # clean, mobile-friendly

# Mock heavy packages that are not installed on RTD.
# autodoc_mock_imports prevents import errors at Sphinx build time but does NOT
# prevent pip from installing those packages if they appear in install dependencies —
# that is why the .readthedocs.yaml must NOT install the package itself (see §4.3).
autodoc_mock_imports = [
    "torch", "torch.fft", "torch.multiprocessing",
    "healpy", "accelerate",
    "denoising_diffusion_pytorch",
    "numba", "quantimpy", "kymatio",
    "astropy",
]

# Do not re-execute notebooks on RTD — they require FITS data not available there.
# Notebooks should be committed with pre-executed outputs.
nbsphinx_execute = "never"
```

**`docs/requirements.txt`:**
```
sphinx>=7
furo
nbsphinx
sphinx-copybutton
sphinx-autodoc-typehints
# lightweight scientific stack — available on RTD ubuntu-22.04 without GPU deps
numpy>=1.26
scipy>=1.10
```

### 4.3 ReadTheDocs configuration

**`.readthedocs.yaml`** (repo root):
```yaml
version: 2
build:
  os: ubuntu-22.04
  tools:
    python: "3.11"
python:
  install:
    - requirements: docs/requirements.txt
    # Do NOT use `method: pip / path: .` here — that would install the package
    # including all its core dependencies (torch, healpy, etc.) from
    # [project.dependencies], even when extra_requirements: [docs] is set.
    # The sys.path.insert in conf.py makes the source importable without install.
sphinx:
  configuration: docs/conf.py
```

RTD rebuilds automatically on every push to `main` via a GitHub webhook that RTD
installs when you connect the repo. No extra CI step is needed — RTD polls GitHub
or receives the webhook and triggers its own build pipeline.

Add `[docs]` optional dependency group to `pyproject.toml` (used for local doc
builds and CI; NOT used by RTD — RTD uses `docs/requirements.txt` directly):
```toml
[project.optional-dependencies]
docs = ["sphinx>=7", "furo", "nbsphinx", "sphinx-copybutton",
        "sphinx-autodoc-typehints", "ipykernel"]
```

### 4.4 ReadTheDocs setup steps

1. Push `.readthedocs.yaml` and `docs/conf.py` to GitHub (ensure `autodoc_mock_imports` and `nbsphinx_execute = "never"` are set as in §4.2)
2. Go to readthedocs.org → Import project → connect `AlexBM173/cmb_foregrounds_diffusion`
3. Set default branch to `main`; enable "build on every push"
4. Trigger first build; if autodoc still raises `ImportError`, add the failing package to `autodoc_mock_imports` in `conf.py`
5. To build versioned docs for a tagged release (e.g. `v0.1.0`), go to RTD → Versions → activate the tag — RTD does not auto-activate new tags
6. Add RTD badge to `README.md`

### 4.5 Content plan

| Page | Source |
|---|---|
| Installation | New `.rst` — venv setup, optional deps |
| Quickstart | New `.rst` — load data, run `mean_cls`, plot |
| Data conventions | Extract from `CLAUDE.md` |
| API reference | Auto-generated from docstrings |
| Tutorials 01–12 | Rendered notebooks via `nbsphinx` |
| Contributing | New `.rst` — how to add modules, run tests |

---

## Phase 5 — Distribution and PyPI

### 5.1 Source distribution and wheels

**Source distribution (sdist):** a `.tar.gz` of the source tree — what pip uses when
no pre-built wheel is available for the target platform.

**Wheel (bdist_wheel):** a pre-built `.whl` archive. For pure-Python packages (no
Cython) this is a single `py3-none-any` wheel. If Cython extensions are added
(§2.6g), platform-specific wheels (`linux_x86_64`, `macosx_arm64`, etc.) must
be built separately — use `cibuildwheel` for this (see §5.3).

Build both with:
```bash
pip install build
python -m build          # produces dist/foregrounds_diffusion-*.tar.gz and *.whl
```

### 5.2 `pyproject.toml` audit

Before publishing, ensure `pyproject.toml` is complete:

```toml
[project]
name = "foregrounds-diffusion"
version = "0.1.0"                        # or use dynamic versioning (see below)
description = "Denoising diffusion models for correlated CMB foreground simulation"
readme = "README.md"
license = { text = "MIT" }
authors = [{ name = "Alexander Blake Martin", email = "alexbm173@gmail.com" }]
requires-python = ">=3.11"
keywords = ["CMB", "diffusion models", "astrophysics", "foregrounds"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Astronomy",
    "Programming Language :: Python :: 3.11",
]
dependencies = [
    "numpy>=1.26",
    "scipy>=1.10",
    "torch>=2.0",
    "healpy",
    "denoising-diffusion-pytorch",
    "accelerate",
]

[project.optional-dependencies]
dev  = ["pytest", "pytest-cov"]
docs = ["sphinx>=7", "furo", "nbsphinx", "sphinx-copybutton",
        "sphinx-autodoc-typehints", "ipykernel"]
fast = ["numba", "quantimpy"]            # optional performance/feature extras

[project.urls]
Homepage      = "https://github.com/AlexBM173/cmb_foregrounds_diffusion"
Documentation = "https://cmb-foregrounds-diffusion.readthedocs.io"
Repository    = "https://github.com/AlexBM173/cmb_foregrounds_diffusion"
```

**Dynamic versioning** (recommended over hardcoding): use `setuptools-scm` to derive
the version from git tags:
```toml
[tool.setuptools_scm]   # version = git tag, e.g. v0.1.0
```
Then `git tag v0.1.0 && git push --tags` drives the release version automatically.

### 5.3 Wheel building with `cibuildwheel` (if Cython is added)

Pure-Python: skip this — the single `py3-none-any` wheel works everywhere.

With Cython extensions, add to `.github/workflows/publish.yml`:
```yaml
- uses: pypa/cibuildwheel@v2
  with:
    package-dir: .
    output-dir: dist
  env:
    CIBW_BUILD: "cp311-* cp312-*"   # match the Python versions in the CI test matrix (§6.3)
    CIBW_ARCHS_LINUX: "x86_64"
    CIBW_ARCHS_MACOS: "arm64 x86_64"
```

### 5.4 TestPyPI before production

Always do a dry run on TestPyPI first:
```bash
pip install twine
twine upload --repository testpypi dist/*
# --extra-index-url is required because TestPyPI does not mirror all dependencies
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            foregrounds-diffusion
```
Verify the install works cleanly before uploading to production PyPI.

### 5.5 PyPI publish via GitHub Actions

Use OIDC Trusted Publisher (preferred — no long-lived secret stored in GitHub).
Set this up at pypi.org → Publishing → Add a pending publisher before running the
workflow for the first time. Then add:

```yaml
# .github/workflows/publish.yml
name: Publish to PyPI
on:
  push:
    tags: ["v*"]          # triggers on git tag v0.1.0, v0.2.0, etc.

jobs:
  build-and-publish:
    runs-on: ubuntu-latest
    environment: pypi                    # requires manual approval in GitHub UI
    permissions:
      id-token: write                    # for Trusted Publisher (no token needed)
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }         # needed for setuptools-scm
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install build
      - run: python -m build
      - uses: pypa/gh-action-pypi-publish@release/v1
        # uses OIDC Trusted Publisher — no API token secret required
        # set up at pypi.org → Publishing → Add a pending publisher
```

**Trusted Publisher** (OIDC) is preferred over API tokens — it is more secure
because no long-lived secret is stored in GitHub.

### 5.6 Release workflow

1. Merge all changes to `main`; confirm tests pass
2. `git tag v0.1.0 && git push --tags`
3. GitHub Actions builds sdist + wheel, waits for manual approval in the `pypi`
   environment, then publishes
4. Manually activate the tag version in RTD (readthedocs.org → Versions → activate `v0.1.0`) so versioned docs are built alongside `latest`
5. Create a GitHub Release from the tag with release notes

---

## Phase 6 — CI/CD Pipeline

### 6.1 Current state

No `.github/workflows/` directory exists yet. As part of Phase 1 infrastructure,
create the directory and add a minimal `tests.yml` stub (just `pytest tests/ -v`)
before writing the first unit tests — this ensures tests run in CI from the first
commit. The full workflow below replaces that stub.

### 6.2 Recommended workflow files

```
.github/workflows/
  tests.yml        # run test suite on every push and PR
  lint.yml         # code quality checks on every push and PR
  publish.yml      # build and publish to PyPI on version tag
```

### 6.3 `tests.yml` — test suite on push/PR

```yaml
name: Tests
on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]   # test against multiple Python versions
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v --cov=foregrounds_diffusion --cov-report=xml
      - uses: codecov/codecov-action@v4
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
```

### 6.4 `lint.yml` — code quality on push/PR

```yaml
name: Lint
on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11", cache: pip }
      - run: pip install ruff mypy
      - run: ruff check foregrounds_diffusion/       # fast linter (replaces flake8/isort)
      - run: ruff format --check foregrounds_diffusion/
      - run: mypy foregrounds_diffusion/ --ignore-missing-imports
```

### 6.5 Additional CI/CD improvements (suggested)

The items below are ordered from most to least impactful for a research codebase.

**a) Dependency review on PRs**
```yaml
# Flags PRs that add dependencies with known vulnerabilities
- uses: actions/dependency-review-action@v4
```
Prevents accidentally pulling in a compromised transitive dependency.

**b) Pin dependencies with `pip-compile`**
```bash
pip install pip-tools
pip-compile pyproject.toml --output-file requirements.lock
```
Store `requirements.lock` in the repo. CI installs from the lock file, so the
test environment is 100% reproducible. Add a weekly scheduled workflow to
`pip-compile --upgrade` and open a PR with the diff.

**c) Test result caching**
```yaml
- uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('requirements.lock') }}
```
Cuts CI time by ~60% on cache hits.

**d) Benchmark regression tracking**
Add `pytest-benchmark` and a nightly workflow that runs the profiling harness
from Phase 2.1 on a fixed synthetic dataset. Store results as a GitHub Actions
artifact and fail the workflow if any benchmark regresses by more than 20%.
Prevents optimisation work from being silently undone.

**e) Notebook smoke tests**
```yaml
- run: jupyter nbconvert --to notebook --execute \
         docs/tutorials/06_power_spectra.ipynb \
         --ExecutePreprocessor.timeout=120
```
Run the key tutorial notebooks in CI (without FITS data — mock the data loading)
to catch import errors and broken cells before they reach users on RTD.

**f) Branch protection rules (GitHub settings, not a workflow)**
- Require the `Tests` and `Lint` checks to pass before merging to `main`
- Require at least 1 review for PRs
- Prevent force-push to `main`

**g) Changelog automation with `towncrier`**
Each PR adds a small news fragment (`changes/123.bugfix.md`). On release,
`towncrier build` assembles `CHANGELOG.md` automatically. Eliminates merge
conflicts in a hand-maintained changelog.

**h) Security scanning with `pip-audit`**
```yaml
- run: pip install pip-audit && pip-audit
```
Checks all installed packages against the OSV vulnerability database. Runs in
under 10 seconds and catches issues like the `requests` CVEs.

**i) `pre-commit` hooks (local, mirrors CI lint)**
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.0
    hooks:
      - id: ruff
      - id: ruff-format
```
Catches lint errors locally before they reach CI, keeping the feedback loop tight.

---

## Phase 7 — Codebase Cleanup

Remove legacy code that predates the current package structure. Each file must be
reviewed for salvageable content before deletion — the review findings are recorded
here so the rationale is preserved in git history.

---

### 7.1 Redundant Python scripts (`foregrounds_diffusion/redundant/`)

| File | What it contains | Disposition |
|---|---|---|
| `ds_utils.py` | `apply_stdnorm` (superseded by `preprocessing.py`), `stats` (superseded by `statistics.py`), `renormalize_dm_maps` (post-sampling rescaling logic) | Review `renormalize_dm_maps` — if the rescaling formula is not documented elsewhere, extract a note to `docs/paper_code_inconsistencies.md` before deleting |
| `flatsky.py` | Early flat-sky FFT utilities (`get_lxly`, `cl_to_cl2d`, `make_gaussian_realisation`, etc.) superseded by `flatmaps.py` | Delete — all functions are present in `flatmaps.py` with improved implementations |
| `gen_masks_from_maps.py` | Early Gaussian fitting and mask generation; uses bare `from numpy import *` style; superseded by `masking.py` and `statistics.py` | Delete — no logic absent from the current modules |
| `ps_utils.py` | `apply_maxmin_normalization` (in `preprocessing.py`), `load_all_moments` (loads saved moments NPZ by bandpass centre) | Check whether `load_all_moments` is called anywhere in active notebooks; if not, delete — the function encodes a specific NPZ layout that may have changed |
| `sample.py` | Old sampling script using `channels=3` and `Trainer1D`; model architecture is incompatible with current checkpoints | Delete — superseded by `foregrounds_diffusion/sample.py` |

**Deletion process (per file):**
1. `grep -r "<filename>" . --include="*.py" --include="*.ipynb"` — confirm no active import
2. Read through for any non-obvious logic not present in the current codebase
3. Delete the file; commit with message `chore: remove redundant/<file> — superseded by <module>`

Once all files are removed, delete the `redundant/` directory and remove its
exclusions from `[tool.ruff]` and `[tool.mypy]` in `pyproject.toml`.

---

### 7.2 Old notebooks in `docs/` (outside `docs/tutorials/`)

These predate the tutorial series and were used during initial development. Review
each for content worth preserving in the tutorials or docs before deleting.

| Notebook | Contents | Disposition |
|---|---|---|
| `docs/00_model.ipynb` | Early model loading and data inspection at the start of the project | Delete — covered by tutorials 04 and 05 |
| `docs/01_map_cuts.ipynb` | Raw preprocessing pipeline from cluster FITS files; uses old `ds_utils`/`ps_utils` imports | **Review** — may contain preprocessing parameters (NSIDE, step size, frequency channels) not fully captured in `preprocessing.ipynb`; extract any new details to `docs/paper_code_inconsistencies.md`, then delete |
| `docs/02_visualization-joint.ipynb` | Visualisation of joint CIB+tSZ maps and 2D power spectra | Delete — superseded by tutorials 05 and 06 |
| `docs/03_compute_moments-joint.ipynb` | Moment statistics on joint maps; old import style | Delete — superseded by tutorial 07 |
| `docs/03_compute_moments-sum.ipynb` | Summed moment statistics; uses `scienceplots` style | **Review** — uses `scienceplots` and may have the cleanest version of the summed-moments plot layout; extract plot style to §8 before deleting |
| `docs/05_plots.ipynb` | **Paper figure generation notebook**: Figures 1–end of the Prabhu et al. paper, using real FITS data and trained model outputs | **Do not delete** — move to `docs/tutorials/` as `14_paper_figures.ipynb` and rewrite to use the `foregrounds_diffusion` package API instead of raw `ds_utils`/`ps_utils`; apply §8 plot standards |
| `docs/scratch.ipynb` | Ad-hoc exploratory cells; no coherent structure | Delete — no salvageable content |
| `docs/stack_tsz_based_on_snr.ipynb` | tSZ stacking analysis by SNR bin, comparing AGORA and DDPM outputs | **Review** — may contain the authoritative stacking parameter choices (SNR bins, cutout sizes); confirm these are documented in tutorial 09 before deleting |
| `docs/tutorials/masking.ipynb` | Stray notebook that does not follow the tutorial numbering; applies masking to real HEALPix maps | **Review** — check whether it covers content absent from tutorial 02; if so, merge relevant cells into tutorial 02, then delete |

**Priority order:** review `docs/05_plots.ipynb` first (migrate to tutorial 14); then
`01_map_cuts.ipynb`, `stack_tsz_based_on_snr.ipynb`, `03_compute_moments-sum.ipynb`,
`tutorials/masking.ipynb`; then delete the rest.

---

### 7.3 Tutorial numbering after cleanup

After `docs/05_plots.ipynb` is migrated:

```
docs/tutorials/
  01_halo_catalogue.ipynb
  02_masking.ipynb
  03_patch_extraction.ipynb
  04_model_and_training.ipynb
  05_sampling.ipynb
  06_power_spectra.ipynb
  07_higher_order_stats.ipynb
  08_morphology_and_histograms.ipynb
  09_tsz_stacking.ipynb
  10_peak_minima_counts.ipynb
  11_scattering_transforms.ipynb
  12_minkowski_tensors.ipynb
  13_benchmarks.ipynb
  14_paper_figures.ipynb   ← migrated from docs/05_plots.ipynb
```

Update `docs/notebook_summaries.md` after each migration/deletion.

---

## Phase 8 — Publication-Quality Plots

All figures used in the thesis or paper must meet journal submission standards:
vector-format primary output, colourblind-safe palette, accessible font sizes,
and no `pylab`/`from pylab import *` anti-patterns.

---

### 8.1 Matplotlib style baseline

Create `foregrounds_diffusion/plot_style.py` (not a public API module — imported
only inside notebooks) with a single `apply()` call that sets rcParams once:

```python
import matplotlib as mpl
import matplotlib.pyplot as plt

# Wong (2011) 8-colour palette — the standard colorblind-safe set.
# Distinguishable under deuteranopia, protanopia, and tritanopia.
WONG = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]

def apply(fig_width_pt=246.0, n_cols=1):
    """Set rcParams for publication-quality figures.

    Parameters
    ----------
    fig_width_pt : float
        Journal text width in points (246 pt ≈ MNRAS single column;
        510 pt ≈ MNRAS double column).  Pass the LaTeX \\textwidth.
    n_cols : int
        Number of figure columns (1 or 2).  Width is divided accordingly.
    """
    inches_per_pt = 1.0 / 72.27
    fig_width = fig_width_pt * inches_per_pt * n_cols
    golden = (1 + 5**0.5) / 2  # golden ratio for default height

    mpl.rcParams.update({
        # --- Figure size and DPI ---
        "figure.figsize":       (fig_width, fig_width / golden),
        "figure.dpi":           150,        # screen preview
        "savefig.dpi":          300,        # raster output
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.05,

        # --- Fonts (match LaTeX body font) ---
        "font.family":          "serif",
        "font.serif":           ["Computer Modern Roman", "DejaVu Serif"],
        "text.usetex":          False,      # True if LaTeX is installed locally
        "mathtext.fontset":     "cm",
        "font.size":            9,
        "axes.titlesize":       9,
        "axes.labelsize":       9,
        "xtick.labelsize":      8,
        "ytick.labelsize":      8,
        "legend.fontsize":      8,

        # --- Lines and markers ---
        "lines.linewidth":      1.2,
        "lines.markersize":     4,
        "axes.linewidth":       0.8,
        "xtick.major.width":    0.8,
        "ytick.major.width":    0.8,
        "xtick.minor.width":    0.6,
        "ytick.minor.width":    0.6,
        "xtick.direction":      "in",
        "ytick.direction":      "in",
        "xtick.top":            True,
        "ytick.right":          True,

        # --- Colour cycle ---
        "axes.prop_cycle":      mpl.cycler(color=WONG),

        # --- Legend ---
        "legend.frameon":       False,
        "legend.handlelength":  1.5,

        # --- Layout ---
        "figure.constrained_layout.use": True,
    })
    return WONG
```

Usage in every notebook that produces paper figures:
```python
from foregrounds_diffusion.plot_style import apply, WONG
apply(fig_width_pt=246.0)          # MNRAS single column
# or
apply(fig_width_pt=510.0, n_cols=2)  # MNRAS double column spanning both cols
```

---

### 8.2 Colourmap choices

| Use case | Recommended cmap | Avoid |
|---|---|---|
| CIB intensity map | `"cividis"` (perceptually uniform, colorblind-safe) | `"Blues"`, `"inferno"` |
| tSZ Compton-y map | `"RdBu_r"` centred at zero (diverging, colourblind-safe) | `"hot_r"` |
| Power spectrum / residuals | `"viridis"` | `"jet"`, `"rainbow"` |
| Binary masks / excursion sets | Black/white only | any colour |
| Correlation matrix / covariance | `"coolwarm"` or `"PuOr"` (diverging) | `"seismic"` |

For HEALPix mollview projections, `healpy` uses its own cmap argument — pass
`cmap="cividis"` explicitly rather than accepting the default `"gray"` or `"viridis"`.

---

### 8.3 Save format convention

```python
fig.savefig("plots/paper/fig01_cib_fullsky.pdf")   # primary: vector for LaTeX
fig.savefig("plots/paper/fig01_cib_fullsky.png", dpi=300)  # backup: raster
```

- PDF is the submission format for most journals (MNRAS, A&A, ApJ).
- PNG at 300 dpi is required if the figure contains rasterised content (imshow,
  healpy maps) that does not render well as vector.
- Never commit SVG to the repo (large file sizes; git history bloat).
- All paper figures go under `plots/paper/`; benchmark figures under `plots/benchmarks/`.
- Add `plots/paper/*.png` and `plots/paper/*.pdf` to `.gitignore` — generated files
  should not be committed; the notebooks that generate them are the source of truth.

---

### 8.4 Figures to rewrite

These are the figures in `docs/05_plots.ipynb` (to become tutorial 14) that need
to be ported to the new style:

| Figure | Current issues | Required fixes |
|---|---|---|
| Fig 1 — CIB full-sky mollview | `cmap='Blues'`; no DPI; font sizes hardcoded | Switch to `cmap='cividis'`; apply §8.1 rcParams; save as PDF+PNG |
| Fig 2 — Processed patch example | `cmap='Blues'`; `from pylab import *` | Same cmap fix; remove pylab; use `fig, ax = plt.subplots()` |
| Multifrequency map panel | `cmap='inferno'`; ad-hoc font sizes | Switch to `cmap='cividis'`; use `apply()` |
| CIB/tSZ side-by-side panel | Non-standard cmaps; `cmap='hot_r'` for tSZ | CIB → `cividis`; tSZ → `RdBu_r` centred at zero |
| Power spectrum comparison | `colors = {'CIB': 'royalblue', 'tSZ': 'orangered'}` — not from WONG | Replace with `WONG[5]` (blue), `WONG[6]` (vermillion) |
| Moments comparison plots | `from pylab import *`; hardcoded `fsval` | Apply `apply()` once at top of notebook; remove all explicit `fontsize=` overrides |
| Minkowski functionals | Inconsistent axis formatting | Apply `apply()` and `constrained_layout` |

---

### 8.5 Anti-patterns to remove from all notebooks

- `from pylab import *` — pollutes namespace; replace with explicit `import matplotlib.pyplot as plt`
- `rcParams.update({'font.size': 12})` scattered through cells — consolidate into single `apply()` call
- `plt.figure(figsize=(W, H))` with hardcoded inches — replace with `fig_width_pt` formula in `apply()`
- `#plt.savefig(...)` (commented-out saves) — uncomment or delete; never leave dead save calls
- `clf()` and bare `figure()` calls (pylab relics) — replace with `fig, ax = plt.subplots()`
- `cmap='Blues'`, `cmap='hot_r'`, `cmap='inferno'` for intensity maps — replace per §8.2

---

## Phase 9 — Notebook Variable Naming Consistency

A reader moving through the tutorial series should not have to re-learn variable
names between notebooks. This phase establishes a canonical glossary, documents
every current deviation, and records the per-notebook edits needed.

---

### 9.1 Canonical name glossary

| Object | Canonical name | Type / shape | Notes |
|---|---|---|---|
| Flat-sky map parameters | `flatskymapparams` | `[nx, ny, dx, dy]` — `list[int, int, float, float]` | dx = dy = **1.40625** arcmin exactly (= 6° × 60 / 256); `1.41` in 06 and 07 is a bug |
| AGORA CIB patch stack (full, loaded) | `cib_maps` | `(N, H, W, 1)` channels-last | Renamed `cib_patches` in 03 |
| AGORA tSZ patch stack (full, loaded) | `tsz_maps` | `(N, H, W, 1)` channels-last | Renamed `tsz_patches` in 03 |
| AGORA CIB for stats (channels-first slice) | `agora_cib` | `(N, H, W)` | Sliced from `cib_maps[..., 0].transpose(0, ...)` or channels-first load |
| AGORA tSZ for stats | `agora_tsz` | `(N, H, W)` | Mirror of above |
| DDPM raw sample array (file on disk) | `ddpm_raw` | `(N, 2, H, W)` channels-first | Current name in 06–12; `sample_data` in 01 |
| DDPM CIB channel | `ddpm_cib` | `(N, H, W)` | `ddpm_raw[:, 0]` |
| DDPM tSZ channel | `ddpm_tsz` | `(N, H, W)` | `ddpm_raw[:, 1]` |
| Gaussian baseline array (file on disk) | `gauss_maps` | `(N, 2, H, W)` channels-first | Already consistent in 06–12 |
| Gaussian CIB channel | `gauss_cib` | `(N, H, W)` | `gauss_maps[:, 0]` |
| Gaussian tSZ channel | `gauss_tsz` | `(N, H, W)` | `gauss_maps[:, 1]` |
| Point-source threshold (mJy) | `PTSRC` | `int` (2 or 6) | Already consistent across all notebooks |
| Project root path | `PROJECT_ROOT` | `pathlib.Path` | Already consistent |
| Patches directory | `PATCHES_DIR` | `pathlib.Path` | `PROJECT_ROOT / "data" / "low_pass" / f"{PTSRC}mJy"` — consistent in 06–12 |
| CIB file path | `fpath_cib` | `pathlib.Path` | Deviates in 04 (uses bare string); should use `PATCHES_DIR / filename` |
| tSZ file path | `fpath_tsz` | `pathlib.Path` | Same |
| Checkpoint path | `CHECKPOINT` | `pathlib.Path` | Already consistent in 05 |
| Maximum patches to analyse | `N_MAPS` | `int` | Config constant; `N_MAPS = 500` or similar; replaces bare `N = min(...)` |
| Actual patch count after capping | `n_maps` | `int` | `n_maps = min(N_MAPS, len(agora_cib), len(ddpm_cib), len(gauss_cib))` |
| Multipole array (from `map2cl`) | `el` | `(n_bins,) float` | Matches the `map2cl` return name; `ell` and `el_arr` in 03 are deviations |
| Power spectrum array | `cl` | `(n_bins,) float` | Or `cl_cib`, `cl_cross`, etc. for named spectra |
| Bandpass ℓ-edge pairs | `bp_edges` | `list[tuple[float, float]]` | List of `(lmin, lmax)` pairs; `bandpass_edges` in 07 |
| 2D bandpass filter arrays | `bp_filters` | `list[(H, W) float]` | Already consistent in 07 |
| Threshold array (morphology / peaks) | `thresholds` | `(T,) float` | `thresholds_fixed` in 13 is an unnecessary suffix |
| Map parameters for benchmarks | `flatskymapparams` | Same as above | `params` in 13 is a deviation; rename for consistency |

---

### 9.2 Per-notebook change list

**`03_patch_extraction.ipynb`**
- Rename `cib_patches` → `cib_maps`, `tsz_patches` → `tsz_maps`
- Rename `ell` and `el_arr` → `el` (match `map2cl` return convention)
- Rename `n_maps` → keep as-is (it is already the actual count)

**`04_model_and_training.ipynb`**
- Replace `fpath_cib = f"..."` and `fpath_tsz = f"..."` with `PATCHES_DIR`-relative paths:
  ```python
  PATCHES_DIR = PROJECT_ROOT / "data" / "low_pass" / f"{PTSRC}mJy"
  fpath_cib = PATCHES_DIR / f"CIB_map_150GHz_256_st6_minmax_{PTSRC}mJy_zero_lp.npy"
  fpath_tsz = PATCHES_DIR / f"tSZ3_map_150GHz_256_st6_minmax_{PTSRC}mJy_norm_lp.npy"
  ```

**`05_sampling.ipynb`**
- Replace bare string `"data/low_pass/{PTSRC}mJy/..."` paths with `PATCHES_DIR`-relative paths
- Rename `cib_train` → `cib_maps`, `tsz_train` → `tsz_maps` (only used as rescaling reference here — the name `train` is misleading because these are normalised training patches, not a train/test split)

**`06_power_spectra.ipynb`** ⚠ contains a bug
- **Bug fix:** `flatskymapparams = [256, 256, 1.41, 1.41]` → `[256, 256, 1.40625, 1.40625]`
- Replace `N = min(len(agora_cib), ...)` pattern with two lines:
  ```python
  N_MAPS = 500           # config constant at top of notebook
  n_maps = min(N_MAPS, len(agora_cib), len(ddpm_cib), len(gauss_cib))
  ```
- Rename `el` loop variable to be consistent with API (already `el` here — no change needed)

**`07_higher_order_stats.ipynb`** ⚠ contains a bug
- **Bug fix:** `flatskymapparams = [256, 256, 1.41, 1.41]` → `[256, 256, 1.40625, 1.40625]`
- Rename `bandpass_edges` → `bp_edges` for consistency with glossary
- Replace `N=5` display snippet with `N_MAPS = 5  # display subset` to disambiguate from the count

**`08_morphology_and_histograms.ipynb`**
- Add `N_MAPS` config constant at top; replace implicit `N` usages

**`10_peak_minima_counts.ipynb`**, **`11_scattering_transforms.ipynb`**, **`12_minkowski_tensors.ipynb`**
- These three are already internally consistent with each other; main change is ensuring `N_MAPS` is declared in the config cell (it is, as `N_MAPS`) and that the `n_maps = min(...)` pattern uses lowercase

**`13_benchmarks.ipynb`**
- Rename `params` → `flatskymapparams` (the benchmark fixture that represents the production map size)
- Rename `thresholds_fixed` → `thresholds`

---

### 9.3 Pixel-size bug (priority fix)

`flatskymapparams = [256, 256, 1.41, 1.41]` in notebooks 06 and 07 uses a
rounded pixel size. The exact value is:

```
dx = 6° × 60 arcmin/° / 256 pixels = 360/256 = 1.40625 arcmin/pixel
```

Using `1.41` introduces a 0.27% error in all ℓ values computed via `get_lxly`
(since `lx ∝ 1/dx`). This shifts every ℓ bin by ~3 — negligible for qualitative
plots but inconsistent with the data conventions described in `CLAUDE.md` and with
the value used in all other notebooks, tests, and `conftest.py`.

Fix in both notebooks before any other changes in this phase.

---

### 9.4 Anti-patterns to eliminate

- **Bare capital `N` as both a config limit and a computed count** — use `N_MAPS` (config)
  and `n_maps` (computed) to make the distinction explicit
- **`cib_train` / `tsz_train` for the normalised patch arrays** — the `train` suffix
  implies a train/test split that does not exist here; use `cib_maps` / `tsz_maps`
- **`ell` / `el_arr` mixed with `el`** — the `map2cl` function returns `(el, cl)`;
  unpack with those names consistently so readers can match notebook code to API docs
- **Bare string paths** in 05 (`"data/low_pass/{PTSRC}mJy/..."`) — use `PATCHES_DIR`
  so that changing `PTSRC` or `PROJECT_ROOT` propagates everywhere

---

## Sequencing recommendation

1. ✅ **Tests** — full unit + integration suite across all modules.
2. ✅ **Baseline profiling** — §2.2 sweeps; Figures 1–4 in benchmark notebook.
3. ✅ **Single-core optimisations** — §2.6b–c; Figures 5–9.
4. ✅ **`n_jobs` parallelisation** — §3.2 on two bottleneck functions.
5. ✅ **CI foundation** — `tests.yml` + `lint.yml` (Node 24 actions).
6. ✅ **Numba JIT** — §2.6a; skipped (accumulation < 3% of runtime).
7. ✅ **GPU port** — `map2cl_torch` with equivalence test (§3.3).
8. ✅ **Codebase cleanup** — §7; removed `redundant/` scripts, old `docs/` notebooks, stale root `sample.py`, empty `diffusion.py`; migrated `05_plots.ipynb` → tutorial 14.
9. ✅ **Notebook naming consistency** — §9; `dx=1.41`→1.40625, glossary renames, anti-patterns removed.
10. ✅ **Publication-quality plots** — §8; `plot_style.py` + Wong palette in tutorial 14.
11. ✅ **`n_jobs` on remaining functions + `torch.compile` sampling** — §3.2 full; §2.6f.
12. ✅ **Docstring audit + Sphinx/RTD + PyPI** — §4/§5; RTD live, `v0.1.0`/`v0.1.1` published.
13. ✅ **Test hardening + opt-in `--rescale`** — distinct-channel/non-Gaussian tests; inconsistency #4 flag.
14. ✅ **CI/CD hardening** — §6.5; equivalence gate in CI, docs `-W`, `twine check`, concurrency.

**Remaining (cluster-dependent, see Cluster action plan):**

15. ⚠ **Resolve normalisation #7** — confirm on-disk files + checkpoint training scheme *before* running statistics. ← Next (on cluster login)
16. **Sample → statistics → paper figures** — notebooks 06–09 are the critical path; 10–14 if time permits.
17. **MPI wrapper + eval SLURM array job** — §3.5/3.7; needs `eval.py`; optional (convergence plots).
18. **Multi-node training SLURM script** — §3.6; **deferred post-thesis** (single GPU suffices).
19. **Cython (§2.6g), remaining §3.9 parallel figures (12–15)** — post-thesis / cluster.
