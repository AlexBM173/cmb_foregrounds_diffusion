# Planner-Critic Refinement Log
Started: 2026-06-26T22:17:34Z
Max iterations: 5

## Iteration 1 — Planner (Sonnet)

- §1.1: Added `flatskymapparams_256` and `patch_stack_256` fixtures (required by §2.7 benchmarks)
- §1.3: Corrected split ratios in preprocessing pipeline test; clarified channels-last → channels-first shape
- §2.2: Fixed `bandpass_filter` and `map2cl` signatures to match actual implementations
- §2.4: Added Section 8 skeleton to benchmark notebook (referenced by §3.9 but absent)
- §2.5 Figure 5: Fixed inverted speedup formula (`pre_time / post_time`, not `post_time / pre_time`)
- §2.6b: Added memory explosion warning for (N,T,H,W) binarisation; recommended chunking over N
- §2.6e: Removed `get_lpf_hpf` from C-contiguous enforcement (it takes scalars, not map arrays)
- §2.7: Made benchmark snippet self-contained with imports, `thresholds`, and `lambda x: x` identity
- §3.2: Added `from multiprocessing import cpu_count`; fixed `_parallel_` → `parallel_` dispatch name
- §3.3: Rewrote `map2cl_torch` to fix rfft2 vs fft2 grid shape mismatch in `scatter_add_`
- §3.5: Replaced undefined `_merge` with inline `np.concatenate`; eliminated redundant `array_split`
- §3.7: Fixed venv activation path; flagged `eval.py` as not yet existing
- §3.10: Added `eval.py` creation to implementation order; noted Figure 14 dependency on Phase 2 baselines
- §4.2: Added `autodoc_mock_imports` and `nbsphinx_execute = "never"` to `conf.py` settings
- §4.4: Made step 4 actionable (specify `autodoc_mock_imports` fix); added manual RTD tag activation step
- §5.1: Fixed two cross-references (§2.4 → §2.6g; §4.3 → §5.3)
- §5.3: Extended `cibuildwheel` to `cp311-* cp312-*` to match CI test matrix
- §5.4: Added `--extra-index-url https://pypi.org/simple/` to TestPyPI install command
- §5.5: Removed contradictory API token preamble; OIDC-only publish path now consistent
- §5.6: Added explicit RTD tag activation instruction
- §6.1: Resolved orphaned reference to non-existent `tests.yml` stub

## Iteration 1 — Critic (Opus)

```
ITERATION: 2
SCORES: D1=7/10  D2=6/10  D3=4/10  D4=6/10  D5=6/10  D6=6/10
OVERALL: 5.8/10

CRITICAL ISSUES:

1. [Whole plan / D3] No calendar, no effort estimates, no scope triage. Thesis deadline
   2026-07-01 is FIVE days away. Six full engineering phases with no acknowledgement of
   remaining time or what is cuttable. → Add effort budget, minimum-viable-thesis subset,
   and explicit defer/cut list (§3.5 MPI, §3.6 DeepSpeed, §2.6g Cython, §5 PyPI, §6.5g towncrier).

2. [§2.6/§2.7/§3.3 / D2,D6] No correctness/equivalence tests for any optimised implementation.
   Benchmarks compare timings but never assert outputs match. → Mandate np.allclose/torch.allclose
   test vs reference for every optimisation, in CI, before any benchmark.

3. [§3.3 map2cl_torch / D1,D2] GPU port omits dx_rad**2/(nx*ny) normalisation present in the
   CPU map2cl; returns wrong physical amplitude. → Apply the normalisation factor and add
   equivalence test.

4. [§3.2/§3.5 / D5,D6] Merge strategies incompatible with actual return types:
   - compute_mfs returns tuple (M0,M1,M2) each (N,T), not (N,T,3) array
   - compute_cross_moments returns (moments_out, labels) tuple
   - mean_cls returns already-averaged (el, mean_cl, std_cl) — no per-map spectra exist
   → Correct every row of §3.2 table to real return types; give tuple-aware merge.

MODERATE ISSUES:

1. [§2.2 / D5] map2cl signature still wrong: actual is map2cl(flatskymapparams, flatskymap1,
   flatskymap2=None, binsize=None, minbin=100, maxbin=10000) — not lmin/lmax.
2. [§1.2 / D5] scattering_summary formula wrong: n_features = J + L*J*(J-1)/2, not J + J*(J-1)/2.
3. [§2.6a / D2] Numba mis-targeted — real bottleneck is scipy.ndimage.sobel + binary_erosion
   (can't run in nopython), not the accumulation tail. Demote to "candidate pending profiling".
4. [§3.6 / D4] ZeRO-2 overkill for dim=64 U-Net; launcher oversubscribes (4 tasks × 16 processes).
   Switch to DDP; one launcher per node. Mark §3.6 explicitly optional/deferred.
5. [§3.5 / D4] MPI: padding unnecessary for lowercase scatter; no failure handling → deadlock.
6. [§3.4 / D2] multi_gpu_eval passes CUDA tensor to numpy statistics functions — incompatible.

MINOR ISSUES:

1. [§1.2] apply_stdnorm test needs channels-last input (patch_stack has no channel axis).
2. [§1.2/§4] Minkowski named M0/M1/M2 in plan vs V0/V1/V2 in codebase — reconcile.
3. [§3.8] lfs setstripe only affects new files; note re-copy requirement.
4. [§2.2/§2.3] "O(12·N·B·HW log HW)" misleading — 12 moments are cheap elementwise after 2 FFTs.
5. [§3.6] Note A100 40 vs 80 GB variant for scattering-covariance at N≥100.

SIGNAL: CONTINUE
FOCUS FOR NEXT ITERATION: (1) Make plan deadline-aware: add per-step effort estimates,
minimum-viable-thesis subset, and explicit cut/defer list. (2) Add correctness-equivalence
gate for every optimisation/port; fix map2cl_torch normalisation. (3) Fix §3.2/§3.5 merge
strategies and §2.2 map2cl and §1.2 scattering signatures to match actual source.
```

## Iteration 2 — Planner (Sonnet)

- Added `## Deadline and scope triage` section: 5-day effort budget table, minimum-viable-thesis subset, explicit defer/cut list (§3.5 MPI, §3.6 DeepSpeed, §2.6g Cython, §5 PyPI, §4 RTD, §6.5g towncrier)
- Added correctness/equivalence gate block before §2.6a: np.allclose/torch.allclose patterns for all output types; mandates tests/benchmarks/test_equivalence.py in CI
- Fixed §3.3 `map2cl_torch`: added `dx_rad = math.radians(dx_arcmin/60.)` and `norm = dx_rad**2/(H*W)` normalisation; added equivalence test code block
- Fixed §3.2 table return types: compute_mfs → tuple (M0,M1,M2) each (N,T); compute_cross_moments → tuple (moments_out, labels); mean_cls → not directly parallelisable via chunk-and-concat, use internal joblib instead
- Fixed §2.2 map2cl signature: minbin/maxbin not lmin/lmax
- Fixed §1.2 scattering formula: n_features = J + L*J*(J-1)/2
- §2.6a: Demoted Numba to "candidate pending profiling"; noted sobel/binary_erosion cannot run nopython
- §3.6: Renamed to DEFERRED; switched ZeRO-2 → DDP; added A100 40/80 GB note
- §3.5: Removed unnecessary MPI padding; added try/except + comm.Abort(1) for deadlock prevention
- §3.4: Noted fn must be torch-native; used torch.from_numpy().to(device) pattern
- §1.2: Noted apply_stdnorm needs channels-last input
- §1.2/plan: Reconciled M0/M1/M2 ≡ V0/V1/V2
- §2.3: Corrected O(12·N·B) → O(N·B)
- §3.8: Added lfs setstripe re-copy note with cp command
- §3.1: Updated parallelism table to DDP consistent with §3.6
- §3.10/sequencing: Marked steps 7-14 as post-submission; reordered NumPy vectorisation before Numba

## Iteration 2 — Critic (Opus)

```
ITERATION: 2
SCORES: D1=7/10  D2=6/10  D3=8/10  D4=6/10  D5=7/10  D6=7/10
OVERALL: 6.8/10

CRITICAL ISSUES:

1. [§2.2/§2.3/§2.6d — select_snr_pixels] Plan mischaracterises bottleneck and proposes wrong fix.
   Actual implementation uses scipy.ndimage.maximum_filter (vectorised) — NOT O(K²) pairwise loop.
   True cost is O(N·HW). cKDTree proposal has different semantics from local-maximum selection and
   would fail the correctness gate. → Correct §2.3 scaling to O(N·HW); delete or replace §2.6d.

2. [§3.6 train_slurm_multinode.sh] Double-spawn bug still present: --ntasks-per-node=4 × each task
   running accelerate launch --num_processes 16 = 64 processes/node instead of 16. All 4 tasks on
   a node also share same --machine_rank → corrupt rendezvous. → Change to --ntasks-per-node=1.

MODERATE ISSUES:

1. [§1.2 / D1] No inverse-normalisation/round-trip tests. Post-sampling rescaling is flagged in
   paper_code_inconsistencies.md. → Add test that normalise→denormalise recovers original map.
2. [§3.2/§3.10] Two-array chunking (compute_cross_moments, compute_summed_moments take cib AND tsz)
   not shown. Single-array template will mis-chunk. → Add dual-array zip pattern.
3. [§3.3 / D6] lbin_idx_rfft/bin_counts construction left as prose — not reproducible. → Add
   code building rfft2 ℓ-grid from get_lxly digitised to same bins as CPU map2cl.
4. [§4.3] RTD install pulls full torch/healpy deps — autodoc_mock_imports doesn't stop pip install.
   → Install only [docs] deps; put heavy deps on sys.path with --no-deps.

MINOR ISSUES:

1. [§3.5] module load openmpi/4.1 may not be exact CSD3 string — annotate as placeholder.
2. [§1.2] Scattering shape formula fragile — assert against function output, not re-derived formula.
3. [§2.6e] get_lpf_hpf in both flatmaps.py and preprocessing.py — disambiguate which to use.
4. [§3.6] "fits in A100 40 GB" — CSD3 ampere mostly 80 GB A100s.

SIGNAL: CONTINUE
FOCUS FOR NEXT ITERATION: Fix select_snr_pixels cost model (O(N·HW), drop cKDTree); fix SLURM
ntasks-per-node=1; add inverse-normalisation tests; add dual-array joblib pattern; make lbin_idx
construction concrete.
```

## Iteration 3 — Planner (Sonnet)

- §2.2 stacking bullet: Replaced "separation check loop; scales poorly with N" with maximum_filter description; cost corrected to O(N·HW)
- §2.3 scaling table: select_snr_pixels row changed from O(N·K²) → O(N·HW); serial O(K) → O(HW)
- §2.6d: Deleted cKDTree proposal (wrong semantics); replaced with two real candidates: (1) restrict maximum_filter to SNR-mask bounding box; (2) apply maximum_filter to full (N,H,W) stack with 3D footprint
- §3.6 train_slurm_multinode.sh: Changed --ntasks-per-node=4 → --ntasks-per-node=1; --cpus-per-task 8 → 32; added detailed note explaining double-spawn failure mode
- §1.2 test_preprocessing.py: Added three round-trip tests: (1) min-max normalise→denormalise recovers original within atol=1e-6; (2) stdnorm round-trip; (3) power-spectrum amplitude preservation using renormalize_dm_maps + map2cl
- §3.2: Added "Dual-array functions" block with concrete parallel_cross_moments using zip(cib_chunks, tsz_chunks)
- §3.3 map2cl_torch: Replaced prose with concrete build_lbin_idx_rfft function building rfft2 ℓ-grid from get_lxly, digitising to same bins as CPU map2cl (binsize from lx spacing, minbin=100, maxbin=10000)
- §4.2/§4.3 RTD: Changed to requirements: docs/requirements.txt only; added sys.path.insert(0, ...) to conf.py; expanded autodoc_mock_imports; added numpy/scipy to docs/requirements.txt
- §1.2 scattering test: Changed hard-coded formula assertion to assert against function's own output length
- §2.6e: Added note disambiguating get_lpf_hpf — flatmaps.py line 51 is canonical
- §3.5: Annotated module load openmpi string as placeholder
- §3.6 A100 phrasing: Changed to "fits in a single Ampere GPU"; CSD3 mostly 80 GB; "confirm with nvidia-smi"

## Iteration 3 — Critic (Opus)

```
ITERATION: 3
SCORES: D1=8/10  D2=8/10  D3=9/10  D4=8/10  D5=8/10  D6=8/10
OVERALL: 8.2/10

CRITICAL ISSUES: none

MODERATE ISSUES:

1. [§3.3 build_lbin_idx_rfft / D2,D6] Bin-edge array wrong: plan uses
   np.arange(minbin, maxbin + binsize, binsize) but radial_profile uses
   np.arange(minbin, maxbin, binsize) (no +binsize). Yields bin count differing
   by one → equivalence test fails on shape mismatch at first run.
   → Use np.arange(minbin, maxbin, binsize); set n_bins = len(bins) not len(bins)-1.
   Also note radial_profile divides by NONZERO pixel count (hits), not total count.

2. [§2.6d batched maximum_filter / D2] Proposed footprint side 2*min_sep+1 differs
   from scipy's size=min_separation semantics (different even/odd origin). Also,
   batched version silently assumes pre-normalised stack but reference normalises
   per-map by m.std() BEFORE filtering.
   → Use footprint matching size=min_separation; note N-stack must be pre-normalised.

MINOR ISSUES:

1. [§1.2 renormalize_dm_maps / D6] Round-trip test only valid when train_maps is
   the same realisation that was normalised. Shape contract: dm_maps channels-first
   (N,C,H,W), train_maps channels-last (N,H,W,C) — renormalize_dm_maps transposes.
2. [§2.6e / D5] "scalar filter parameter" wrong; actual is get_lpf_hpf(flatskymapparams,
   lmin_lmax, filter_type=0) where lmin_lmax is a pair, not a scalar.
3. [§1.2 test_moments / D6] compute_cross_moments label assertion: assert exact list
   ['S2aa','S2bb','S2ab','S3aaa','S3bbb','S3aab','S3abb','S4aaaa','S4bbbb',
   'S4aaab','S4aabb','S4abbb'], not just len==12.

SIGNAL: CONTINUE
FOCUS FOR NEXT ITERATION: Fix §3.3 bin-edge (+binsize bug); fix §2.6d footprint
size and pre-normalisation note; tidy three minor actionability notes. Plan should
converge after these land.
```

## Iteration 4 — Planner (Sonnet)

- §1.2 test_moments: Replaced len==12 check with exact label list assertion against confirmed moments.py list; strengthened shape assertion to (N, n_bands, 12)
- §1.2 test_preprocessing renormalize_dm_maps: Added constraint that test requires same realisation as normalised; added shape contract (dm_maps channels-first, train_maps channels-last; renormalize_dm_maps transposes)
- §2.6e get_lpf_hpf: Replaced "scalar filter parameter" with actual signature get_lpf_hpf(flatskymapparams, lmin_lmax, filter_type=0); noted lmin_lmax is scalar for LP/HP, pair for bandpass
- §2.6d batched maximum_filter: Replaced footprint=(1,2*min_sep+1,2*min_sep+1) with size=(1,min_separation,min_separation); added comment that snr_stack_nhw must be pre-normalised per-map
- §3.3 build_lbin_idx_rfft: Changed np.arange(minbin, maxbin+binsize) → np.arange(minbin, maxbin); removed false "matches map2cl edges" comment; n_bins = len(bins) not len(bins)-1; updated sentinel/mask; added note on radial_profile NONZERO pixel hits normalisation

## Iteration 4 — Critic (Opus)

```
ITERATION: 4
SCORES: D1=8/10  D2=9/10  D3=8/10  D4=9/10  D5=9/10  D6=8/10
OVERALL: 8.5/10

CRITICAL ISSUES: none remaining.

MODERATE ISSUES:

1. [§1.2 test_scattering_stats / D1] Test description wrong about input contract. Actual
   signature is scattering_summary(coeffs, scale_idx=None) where coeffs is the dict returned
   by compute_scattering_coefficients (reads coeffs['J'], coeffs['S1'], coeffs['S2']). Plan
   says "call scattering_summary on a single-map batch" — a developer following this literally
   writes a failing test. → Fix: "first call compute_scattering_coefficients(patches) to get
   coeffs, then scattering_summary(coeffs); assert the feature dimension is stable across batch
   sizes."

2. [§1.1 conftest / §3.3 / §2.2 / D5] Pixel-scale convention unreconciled. flatskymapparams
   uses 1.41, map2cl_torch hard-codes dx_arcmin=1.41, but peak_counts.smooth_map defaults to
   pixel_res_arcmin=1.40625. A ~0.3% mismatch across modules should be stated and resolved.
   → Fix: add a note in §1.1 or a "map parameters" subsection fixing the canonical pixel scale
   and explaining the 1.41 vs 1.40625 difference; have smooth_map benchmarks pass
   pixel_res_arcmin=1.41 to match.

3. [§1.2 test_flatmaps / D1] Correlated two-field Gaussian baseline untested. The thesis
   comparison uses the correlated make_gaussian_realisation(cl2=, cl12=) path (SIM_FIELD_1 /
   SIM_FIELD_2_FFT). Plan only tests the single-field branch. → Fix: add a test asserting the
   correlated realisation recovers input auto- and cross-spectra (cl, cl2, cl12) within sample
   variance, and note in scope triage that this is the realisation the thesis comparison uses.

MINOR ISSUES:

1. [§"Deadline and scope triage"] "approximately 5 days from current date" — today is 2026-06-27,
   deadline 2026-07-01, i.e. 4 days. The must/should effort table sums to ~4.5 days against 4
   available with zero buffer. Tighten language and acknowledge no slack remains.
2. [§2.2] cl2map signature: actual is cl2map(flatskymapparams, cl, el=None) (el is optional).
3. [§3.2 table, extract_cutouts] max_cutouts=500 cap applies per chunk in parallel — changes
   results vs serial. Note or exclude from concat-merge list.
4. [§2.2] extract_cutouts signature omits max_cutouts=500.

SIGNAL: CONTINUE
FOCUS FOR NEXT ITERATION: Fix the three moderate issues — they are concrete, code-verifiable
corrections. (1) Fix scattering_summary test contract: it consumes a coeffs dict, not raw
patches. (2) Reconcile the 1.41 vs 1.40625 pixel scale across modules with a canonical note.
(3) Add a correlated two-field Gaussian-baseline test and explicitly state the correlated
(cl2/cl12) realisation is the thesis scientific control. Fix the deadline arithmetic (4 days,
not 5). With these landed the plan should converge.
```

## Iteration 5 — Planner (Sonnet)

- §"Deadline and scope triage": Changed "approximately 5 days" → "4 days; no slack beyond the MVT subset"; changed "Effort estimates (5-day window)" → "Effort estimates (4-day window)"; updated sequencing recommendation to read "4-day thesis window (no slack — MVT subset only)"
- §1.1 Infrastructure: Added canonical pixel-scale convention note — 1.41 arcmin/pixel is the project default in flatskymapparams and map2cl_torch; smooth_map defaults to 1.40625 (6°/256); all test/benchmark calls to smooth_map must pass pixel_res_arcmin=1.41 explicitly
- §1.2 test_flatmaps: Added correlated two-field make_gaussian_realisation test — asserts recovered auto- and cross-spectra (cl, cl2, cl12) within sample variance; notes this is the scientific control path used in the thesis non-Gaussianity comparison
- §1.2 test_scattering_stats: Rewrote scattering_summary bullet to make two-step contract explicit: first call compute_scattering_coefficients(patches) → coeffs dict, then scattering_summary(coeffs, scale_idx=None); feature-count stability check via single-map reference run
- §2.2 cl2map entry: Fixed signature to cl2map(flatskymapparams, cl, el=None) — el is optional
- §3.2 extract_cutouts row: Added note that max_cutouts=500 cap applies per chunk in parallel, yielding up to n_jobs×500 cutouts vs serial 500; fix is to trim post-concatenation or disable the cap

## Iteration 5 — Critic (Opus)

```
ITERATION: 5
SCORES: D1=7/10  D2=8/10  D3=7/10  D4=8/10  D5=8/10  D6=7/10
OVERALL: 7.5/10

CRITICAL ISSUES:

1. [§3.3 map2cl_torch / build_lbin_idx_rfft / D2,D6] GPU port crashes as written.
   build_lbin_idx_rfft sets sentinel lbin_idx[~valid] = n_bins, but map2cl_torch allocates
   cl = torch.zeros(N, n_bins) and scatter_add_ uses that index → RuntimeError (index n_bins
   out of bounds for dim 1 of size n_bins). The equivalence test crashes rather than passes.
   → Allocate cl = torch.zeros(N, n_bins+1), scatter, then return cl[:, :n_bins] / bin_counts;
   or mask sentinel entries out before scattering.

2. [§1.1 pixel-scale convention / D5] Plan adopts 1.41 as canonical to force smooth_map into
   alignment, but 1.41 is the rounded value — the physically exact value is 6°/256 = 1.40625
   arcmin/pixel (smooth_map default). This injects a systematic ~0.27% ℓ-shift and ~0.53%
   amplitude bias into all thesis power spectra. The fix direction is backwards.
   → Verify actual flatskymapparams dx in production eval notebooks; if patches are 6°, adopt
   1.40625 as canonical and pass pixel_res_arcmin=1.40625 to smooth_map explicitly.

3. [§3.10 step 5 / eval.py / D1] eval.py contract only says "run all §2.2 statistics on
   generated samples." The thesis non-Gaussianity claim requires comparing DDPM samples to
   (a) AGORA truth maps and (b) matched Gaussian realisations using the correlated
   make_gaussian_realisation(cl2=, cl12=) path. Both comparisons are absent from the eval.py
   spec. → Expand eval.py to compute and persist the full statistics vector for
   {DDPM samples, AGORA truth, Gaussian baseline} side-by-side.

MODERATE ISSUES:

1. [§2.6a Numba / D2] Proposed JIT target (_accumulate_normals) is already vectorised NumPy
   in morphology.py:67 — executes at C speed over a tiny boundary-pixel set, never ≥30% of
   runtime. Real cost is the Python N×T double loop issuing binary_erosion/sobel per threshold,
   with binary_erosion recomputed for both W012 and W201. → Retarget §2.6a/b at hoisting and
   sharing the per-threshold scipy calls across tensor types.

2. [§3.2 extract_cutouts parallelisation / D6] extract_cutouts coords are global (patch_idx,
   row, col) tuples; chunking maps_nhw invalidates every patch_idx. The max_cutouts note
   addresses a symptom but not the broken indexing. → Drop extract_cutouts from the parallel
   set or specify chunking over coords with full maps_nhw retained per worker.

3. [§1.2 preprocessing round-trip / D1] apply_maxmin_normalization uses global min/max
   (preprocessing.py:15-16); renormalize_dm_maps inverts per channel (lines 56-58). They are
   exact inverses only for single-channel input — for 2-channel CIB+tSZ they are not
   (documented in paper_code_inconsistencies.md §4). Test on a single Gaussian realisation
   hides the production-relevant failure. → Add 2-channel round-trip case asserting the
   mismatch (xfail or documented tolerance).

4. [§3.3 map2cl_torch / D2] cl / bin_counts divides by zero for empty ℓ-bins → inf/nan,
   whereas CPU radial_profile leaves empty bins at 0. With minbin=100 on 256² maps some
   low-ℓ bins can be empty. → Guard with torch.where(bin_counts>0, cl/bin_counts, 0.).

5. [§"Effort estimates" / D3] Budget ~4 days of engineering for even the MVT subset with no
   time for thesis writing, results figures, or defending the non-Gaussianity claim. → State
   which benchmark figures go in which appendix; reserve writing time; or acknowledge this
   engineering is post-submission polish.

MINOR ISSUES:
1. [§3.5 mpi_parallel_eval] np.array_split on rank 0 runs before comm.scatter and outside
   the try; an exception there leaves all ranks blocked. → Wrap chunk construction in try/Abort.
2. [§2.2 / naming] compute_summed_moments returns normalised S3/S4; compute_cross_moments
   returns raw moments — shared S2/S3/S4 labels invite confusion. → Note the difference.
3. [§3.4 multi_gpu_eval] Result ordering via (rank, out) sort under-documented for
   variable-chunk case.

SIGNAL: CONTINUE
FOCUS FOR NEXT ITERATION: Fix the three critical items — the map2cl_torch scatter sentinel
crash, the pixel-scale convention (adopt 1.40625 after verifying production flatskymapparams),
and the eval.py contract to compare DDPM/AGORA/Gaussian side-by-side. Then retarget the
Numba section toward sharing per-threshold scipy calls.
```

## Final Status (Run 1, iterations 1–5)
Stopped after iteration 5 — iteration limit reached
Final scores: D1=7 D2=8 D3=7 D4=8 D5=8 D6=7  Overall=7.5/10
Remaining issues carried forward (Phase 0 not yet written):
- [CRITICAL] §3.3 map2cl_torch: sentinel index n_bins out-of-bounds in scatter_add_ → crashes
- [CRITICAL] §1.1 pixel-scale convention: 1.41 adopted as canonical but physically exact value is 1.40625; wrong direction
- [CRITICAL] §3.10 eval.py: does not specify comparing DDPM samples to AGORA truth and Gaussian baseline (the core thesis comparison)
- [MODERATE] §2.6a Numba: targets already-vectorised W012 accumulation; should target hoisting shared scipy calls per threshold
- [MODERATE] §3.2 extract_cutouts: coord-based indexing breaks when maps_nhw is chunked; should be dropped from parallel set
- [MODERATE] §1.2 preprocessing: 2-channel min-max round-trip is not an exact inverse (production-relevant failure hidden by single-channel test)
- [MODERATE] §3.3 map2cl_torch: division by zero for empty ℓ-bins; needs torch.where guard
- [MODERATE] §"Effort estimates": no thesis writing time budgeted

---

# Run 2 — Phase 0 Focus
Started: 2026-06-28
Max iterations: 3
Focus: Phase 0 DDPM model extensions — ablation design, evaluation protocol, timeline realism, cross-phase dependencies, deadline consistency.

## Iteration 1 — Planner

- Phase 0 header: added explicit baseline_100k run spec; controls enumerated (same seed, data split, augmentation, single kwarg changed)
- §0.2: added quantitative win criterion — training loss ≤ baseline AND power-spectrum χ² vs AGORA not increased (NOTE: critic flags this as invalid cross-objective comparison; fixed in iteration 2)
- §0.3: added KS-test p-value win criterion (NOTE: critic flags p-value ranking as misuse; fixed in iteration 2)
- §0.4: added ≥20% fractional power-spectrum error reduction at ℓ<500 as win criterion
- §0.5: added gating note — combined run not submitted until ablation results reviewed
- §0.6: added Gaussian baseline as step 3b; three-way comparison (DDPM/Gaussian/AGORA) now explicit
- §0.6: added dependency note — eval.py must exist before Phase 0 evaluation begins (day 1–2)
- §0.6: added NPZ results storage format specification
- §0.6: added compute time warning — 2–4 hrs single-core; use --n-jobs 16 via eval_slurm_array.sh
- §0.7: added GPU queue time note (2–24 hr CSD3 wait); instructs submitting baseline+§0.2+§0.4 on day 1
- Triage table: elevated eval.py creation to Must-do
- §1.1: updated canonical pixel scale 1.41 → 1.40625 (exact 6°/256 px)
- §3.3 map2cl_torch: fixed sentinel crash — allocate n_bins+1, slice to n_bins on return
- §3.3 equivalence test: updated dx_arcmin 1.41 → 1.40625
- §3.10 step 5: expanded eval.py CLI spec with --agora-maps, --n-samples; three-way comparison

## Iteration 1 — Critic

```
ITERATION: 1
SCORES: D1=5/10  D2=8/10  D3=4/10  D4=7/10  D5=5/10  D6=6/10
OVERALL: 5.8/10

CRITICAL ISSUES:

1. [Deadline] Plan says July 8 (10 days); critic claims authoritative deadline is July 1 (3 days).
   NOTE: User confirmed July 8 is the real deadline — critic is wrong here; no fix needed.

2. [§0.7] No wall-clock estimate for a 100k-step training run. → Measure steps/sec on pilot job.

3. [§0.2 win criterion] Loss comparison invalid across different objectives (min-SNR reweights loss).
   → Log common unweighted metric (e.g. plain MSE/validation NLL) for both runs.

4. [§0.6] Missing MFs (compute_mfs), tSZ stacking, scattering covariances from evaluation.
   → Add or explicitly justify exclusion.

5. [§0.6] No renormalisation-to-physical-units step before statistics computation.
   → Make renormalize_dm_maps mandatory in eval.py before statistics.

6. [eval.py / §3.2 n_jobs] Must-do eval depends on deferred n_jobs (§3.2).
   → Pull eval.py prereqs out of deferred Phase 3; state single-core fallback runtime.

MODERATE ISSUES:
1. §0.1 — no scalar quality metric defined for "knee" plot
2. §0.3 win criterion — p-value ranking is statistically unsound; use KS distance D instead
3. §0.2 χ² undefined without band-power covariance
4. §0.1 → §0.6 — DDIM knee at 50 steps may not transfer to self-conditioning / min-SNR architectures
5. §3.10 eval.py — under-budgeted at 0.5 day; needs a function-level skeleton

MINOR ISSUES:
1. NPZ stores only beta_W012/theta_W012 but W200/W201 also computed
2. N=200 in §0.3 vs N=500 in protocol — use 500 everywhere
3. §2.2 shows single-field make_gaussian_realisation; correlated path needed
4. sampling_timesteps=50 in training GaussianDiffusion — harmless but should note sampling-only

SIGNAL: CONTINUE
FOCUS FOR NEXT ITERATION: Fix §0.2 win criterion (cross-objective loss invalid); fix §0.3
win criterion (p-value → KS distance); add renormalisation step to §0.6; include MFs/stacking
or justify exclusion; promote eval.py prereqs; add training wall-clock estimate.
```

## Iteration 2 — Planner

- §0 baseline: added training wall-clock estimate (pilot job → ~3 it/s on A100 → 100k steps ≈ 9 hrs; self-conditioning ≈ 11 hrs)
- §0.1: defined scalar knee metric — mean fractional C_ℓ error averaged over ℓ∈[100,10000] and both channels; knee = smallest T_sample within 5% of 1000-step reference
- §0.1: added DDIM knee re-verification note for self-conditioning architecture
- §0.2 win criterion: replaced raw loss condition with unweighted validation MSE on x₀; added diagonal-covariance specification for χ² condition
- §0.3 win criterion: replaced p-value ranking with KS statistic D (lower = closer to AGORA)
- §0.6 step 0 (new): mandatory renormalize_dm_maps before any statistic; cross-references paper_code_inconsistencies.md
- §0.6 step 2: added compute_mfs, compute_scattering_covariance (conditional on kymatio), tSZ stacking (§0.5 only with justification)
- §0.6 figures list: added MF, scattering covariance, tSZ stack entries
- §0.6 dependency: clarified eval.py is single-core; n_jobs is Phase 3
- §0.6 compute time: updated to 3–6 hrs single-core; must run on compute node

## Iteration 2 — Critic

```
ITERATION: 2
SCORES: D1=6/10  D2=8/10  D3=7/10  D4=7/10  D5=6/10  D6=8/10
OVERALL: 7.0/10

CRITICAL:
1. [§0.1 knee] Defined on power spectrum (least sensitive statistic); knee locked in for all
   ablations including morphological metrics. → Redefine knee on most demanding statistic:
   5% plateau must hold simultaneously on C_ℓ, a Minkowski-tensor β summary, and S3/S4 moment.
2. [§0.6 / §0.3 / NPZ] Only beta_W012/theta_W012 stored; W200 and W201 dropped.
   → Store all three tensors; name exact tensor in §0.3 win criterion.

MODERATE:
1. [§0.4] ℓ<500 win criterion collides with minbin=100 floor on 6° field; offset noise affects
   ℓ<100 which is below the measurement floor. → Lower minbin or restate range as [100,500].
2. [§0.3 vs §0.6] N=200 in §0.3 vs N=500 in §0.6. → Reconcile to one value.
3. [§0.7 table] Shows §0.2 and §0.4 as sequential 1d each; narrative says submit together.
   → Clarify table = attention/effort days, GPU wall-clock overlaps.
4. [§3.6] SLURM script missing venv activation. → Add for parity with §3.7.
5. [§0.6 / data location] AGORA data is on SPT cluster; plan never confirms CSD3 staging.
6. [§2.6f] torch.compile on GaussianDiffusion risks recompiles; compile diffusion.model instead.

MINOR: (various labelling, step-count mismatch note, equivalence test scope, shape assertion)

SIGNAL: CONTINUE
FOCUS FOR NEXT ITERATION: Fix the two criticals (knee on most demanding statistic; all three
Minkowski tensors). Fix ℓ<500/minbin collision in §0.4 and N=200/500 inconsistency.
```

## Iteration 3 — Planner

- §0.1 knee: redefined as three-metric simultaneous condition — C_ℓ, W012 β summary, and |S3| cross-moment must all plateau within 5%; largest T_sample across three taken as adopted knee
- §0.3 win criterion: named W012 explicitly; N=200 → N=500
- §0.4 win criterion: ℓ range corrected to [100,500]; added minbin=60 instruction for eval.py; noted ℓ<60 inaccessible
- §0.6 step 1: knee reference updated to three-metric criterion
- §0.6 figures: W200/W201 supplementary panels added; omission requires physical justification
- §0.6 NPZ layout: added beta_W200, theta_W200, beta_W201, theta_W201
- §0.7 table: column renamed "Effort days"; paragraph added explaining GPU parallelism

## Iteration 3 — Critic

```
ITERATION: 3
SCORES: D1=8/10  D2=8/10  D3=9/10  D4=8/10  D5=7/10  D6=7/10
OVERALL: 7.8/10

CRITICAL:
1. [§0.6 / §3.10 eval.py] compute_minkowski_tensors default is tensor_types=('W012',) — plan
   never instructs passing ('W012','W200','W201'). eval.py would silently compute W012 only
   and the NPZ write of beta_W200/W201 would raise KeyError. §2.2 also falsely states
   "loops N × T × 3 types" when the default is one type.
   → Add explicit tensor_types kwarg in §0.6 step 2 and eval.py spec; correct §2.2.

MODERATE:
1. [§0.6 step 0] renormalize_dm_maps has variance_scaling=True by default, applying a
   second variance-match beyond min-max inversion — not documented in plan. Ambiguity
   affects central non-Gaussian amplitude claim. → State and justify variance_scaling setting.
2. [§0.6 NPZ] Layout missing agora_* arrays, MF arrays, scattering arrays, full gaussian_*
   set — cannot produce the promised figures. → Expand layout or document external cache.
3. [§0.6 / data staging] AGORA truth on SPT cluster (/sptlocal/...); CSD3 path never confirmed.
   Live blocker within 10-day window. → Add rsync/staging step and confirm --agora-maps path.
4. [§2.6f] torch.compile(diffusion) still wraps Python sampling loop; should compile
   diffusion.model instead to avoid graph recompiles.

SIGNAL: CONTINUE (iteration limit reached — stopping)
```

## Final Status (Run 2, iterations 1–3)
Stopped after iteration 3 — iteration limit reached
Final scores: D1=8 D2=8 D3=9 D4=8 D5=7 D6=7  Overall=7.8/10

Remaining issues for manual follow-up:
- [CRITICAL] §0.6 / §3.10: eval.py must call compute_minkowski_tensors with tensor_types=('W012','W200','W201') explicitly; §2.2 description of default is wrong
- [MODERATE] §0.6 step 0: renormalize_dm_maps variance_scaling kwarg not specified — affects non-Gaussian amplitude comparisons
- [MODERATE] §0.6 NPZ: missing agora_*, MF, scattering, full gaussian_* arrays needed for promised figures
- [MODERATE] §0.6 / data staging: AGORA truth maps not confirmed staged onto CSD3; live blocker
- [MODERATE] §2.6f: torch.compile target should be diffusion.model not diffusion wrapper



