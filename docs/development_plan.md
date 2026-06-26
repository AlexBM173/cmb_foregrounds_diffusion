# Development Plan

Four-phase plan covering a full test suite, code optimisation, public documentation,
and package distribution. Each phase is largely independent, though Phase 1 (tests)
should precede Phase 3 (optimisations) so regressions are caught automatically, and
Phase 3 (docs) should be reasonably complete before Phase 4 (PyPI).

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
  - `flatskymapparams` — `[64, 64, 1.41, 1.41]` (small maps for speed)
  - `gaussian_patch` — single 64×64 Gaussian realisation
  - `patch_stack` — `(16, 64, 64)` stack of Gaussian patches
  - `binary_map` — 64×64 binary excursion set at a fixed threshold

### 1.2 Unit tests per module

**`test_flatmaps.py`**
- `get_lxly`: shape, dtype, zero at DC
- `map2cl`: output shape, positivity, symmetry under map flip
- `cl2map`: round-trip `cl2map → map2cl` recovers input spectrum within sample variance
- `make_gaussian_realisation`: pixel variance matches input `Cl` amplitude
- `radial_profile`: monotonically spaced bins, correct output shape
- `bandpass_filter`: energy outside band is suppressed

**`test_preprocessing.py`**
- `apply_maxmin_normalization`: output in `[0, 1]`, min=0, max=1
- `apply_stdnorm`: output mean≈0, std≈1
- `get_lpf_hpf`: low-pass kills high-ℓ; high-pass kills low-ℓ
- `augment_images_unique`: output has 8× the input count; no duplicate tensors
- `load_all_moments`: returns correct shape given mock `.npy` files (monkeypatch)

**`test_statistics.py`**
- `gaussian`: callable; correct value at centre
- `moments`: returns 6-tuple; centre estimates correct on a synthetic Gaussian image
- `fitgaussian`: fitted centre within 1 pixel of true centre on a noiseless image
- `stats`: correct min/max/mean/std on known array

**`test_moments.py`**
- `mean_cls`: output shape `(n_bins,)`, values positive
- `mean_cross_cls`: cross-spectrum of independent maps is near zero (within noise)
- `compute_summed_moments`: shape `(N, n_bands, 3)`; Gaussian input gives S3≈0, S4≈0
- `compute_cross_moments`: shape `(N, n_bands, 12)`; labels returned correctly

**`test_morphology.py`**
- `_eigendecompose_2x2`: identity matrix gives β=1, θ=0; known anisotropic tensor gives correct β
- `_tensor_W012`: all-ones binary map gives isotropic tensor (β≈1)
- `_tensor_W200`: circular excursion set gives β≈1
- `compute_minkowski_tensors`: shape `{'W012': {'beta': (N,T), 'theta': (N,T)}}`; β ∈ [0,1]
- `compute_mfs` (requires `quantimpy`): marked `pytest.mark.optional`; M0 decreasing with threshold

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
- `scattering_summary`: output shape `(N, n_features)` where `n_features = J + J*(J-1)/2`

### 1.3 Integration tests

**`test_power_spectrum_roundtrip.py`**
- Generate a Gaussian realisation from a known power-law `Cl`
- Measure `Cl` back with `map2cl`
- Assert recovered spectrum within 20% of input at each ℓ-bin (loose tolerance for small maps)

**`test_preprocessing_pipeline.py`**
- Synthetic `(8, 64, 64, 2)` array through normalisation → augmentation → DataLoader
- Assert augmented count = 64, dtype float32, values in expected range

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
- `map2cl(maps_nhw, mapparams, lmin, lmax, binsize)` — called N times per evaluation; FFT-based
- `cl2map(mapparams, cl, el)` — used in Gaussian baseline generation
- `make_gaussian_realisation(mapparams, cl, el)` — called ~1000× to build baseline
- `bandpass_filter(mapparams, lminmax)` — called once per ℓ-band per evaluation run
- `radial_profile(stack, xy, bin_size, ...)` — called per SNR bin in stacking

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
- `select_snr_pixels(tsz_maps_nhw, snr_min, snr_max, min_separation)` — separation check loop; scales poorly with N
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
| `compute_cross_moments` | O(12 · N · B · HW log HW) | O(B · HW) |
| `compute_minkowski_tensors` | O(N · T · HW) | O(N · T · HW) if vectorised |
| `_tensor_W012` | O(HW) | O(HW) |
| `select_snr_pixels` | O(N · K²) where K = n_peaks | O(K) |
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
y-axis: speedup factor (post_time / pre_time, log scale)
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

**a) Numba JIT — highest priority**

Target: `_tensor_W012`, `_tensor_W200`, `_tensor_W201` pixel-level loops and
the outer `(map, threshold)` loop in `compute_minkowski_tensors`.

```python
import numba

@numba.jit(nopython=True, cache=True)
def _accumulate_normals(rows, cols, gx, gy):
    W = np.zeros((2, 2))
    for i in range(len(rows)):
        nx, ny = gx[rows[i], cols[i]], gy[rows[i], cols[i]]
        norm = np.sqrt(nx*nx + ny*ny)
        if norm > 0:
            nx /= norm; ny /= norm
            W[0,0] += nx*nx; W[0,1] += nx*ny
            W[1,0] += nx*ny; W[1,1] += ny*ny
    return W
```

Use `cache=True` so compilation is skipped on subsequent calls (important in CI).
Warm up the JIT cache with a small dummy call before the benchmark.

**b) NumPy threshold vectorisation**

`compute_minkowski_tensors` and `compute_mfs` loop over T thresholds in Python.
Binarise the entire stack at once:
```python
binary_stack = maps_nhw[:, None, :, :] > thresholds[None, :, None, None]  # (N, T, H, W)
```
Then process each `(n, t)` slice with the JIT kernel. Removes the Python threshold
loop and enables better cache locality.

**c) `mean_cls` / `mean_cross_cls` — pre-compute ℓ-bin mask**

Currently recomputes the ℓ-bin assignment array inside each `map2cl` call.
Compute once outside the loop:
```python
lbin_idx = np.digitize(ell_2d.ravel(), bins)   # computed once
# then reuse across all N maps
```

**d) `select_snr_pixels` — `cKDTree` separation check**

The current separation check is O(K²) in the number of candidate pixels.
Replace with `scipy.spatial.cKDTree` for O(K log K):
```python
from scipy.spatial import cKDTree
tree = cKDTree(candidate_coords)
pairs = tree.query_pairs(r=min_separation)
# remove one member of each conflicting pair
```

**e) Memory layout — C-contiguous enforcement**

Add `maps = np.ascontiguousarray(maps)` at the entry point of `map2cl`,
`get_lpf_hpf`, and `bandpass_filter`. Prevents silent internal copies in
numpy's FFT when arrays arrive in non-standard memory order.

**f) `torch.compile` for sampling**

```python
diffusion = torch.compile(diffusion)   # PyTorch 2.0+
```
Expected 20–40% speedup on repeated forward passes. Add `--no-compile` flag
to `sample.py` to disable for debugging.

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
for statistically robust, reproducible timings:

```python
# tests/benchmarks/test_bench_morphology.py
def test_minkowski_tensors_baseline(benchmark, patch_stack_256):
    benchmark(compute_minkowski_tensors, patch_stack_256, identity, thresholds)

def test_minkowski_tensors_optimised(benchmark, patch_stack_256):
    benchmark(compute_minkowski_tensors_v2, patch_stack_256, identity, thresholds)
```

Run and save a JSON baseline:
```bash
pytest tests/benchmarks/ --benchmark-save=baseline
pytest tests/benchmarks/ --benchmark-compare=baseline --benchmark-compare-fail=mean:20%
```

The `--benchmark-compare-fail` flag makes CI fail if any benchmark regresses
by more than 20% against the saved baseline.

---

## Phase 3 — Documentation and ReadTheDocs

### 3.1 Docstring audit

All public functions should have NumPy-style docstrings covering:
- One-line summary
- `Parameters` section with types and shapes
- `Returns` section with types and shapes
- `Notes` for any non-obvious behaviour (e.g. normalisation conventions, edge cases)

Priority order: `flatmaps` → `preprocessing` → `moments` → `morphology` → `masking`.
`statistics`, `stacking`, `peak_counts` are already reasonably documented.

### 3.2 Sphinx setup

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
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",      # NumPy docstring support
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",       # for ℓ, θ, β notation
    "nbsphinx",                 # render tutorial notebooks
    "sphinx_copybutton",
]
html_theme = "furo"             # clean, mobile-friendly
```

**`docs/requirements.txt`:**
```
sphinx>=7
furo
nbsphinx
sphinx-copybutton
sphinx-autodoc-typehints
```

### 3.3 ReadTheDocs configuration

**`.readthedocs.yaml`** (repo root):
```yaml
version: 2
build:
  os: ubuntu-22.04
  tools:
    python: "3.11"
python:
  install:
    - method: pip
      path: .
      extra_requirements: [docs]
sphinx:
  configuration: docs/conf.py
```

RTD rebuilds automatically on every push to `main` via a GitHub webhook that RTD
installs when you connect the repo. No extra CI step is needed — RTD polls GitHub
or receives the webhook and triggers its own build pipeline.

Add `[docs]` optional dependency group to `pyproject.toml`:
```toml
[project.optional-dependencies]
docs = ["sphinx>=7", "furo", "nbsphinx", "sphinx-copybutton",
        "sphinx-autodoc-typehints", "ipykernel"]
```

### 3.4 ReadTheDocs setup steps

1. Push `.readthedocs.yaml` and `docs/conf.py` to GitHub
2. Go to readthedocs.org → Import project → connect `AlexBM173/cmb_foregrounds_diffusion`
3. Set default branch to `main`; enable "build on every push"
4. Trigger first build; fix any autodoc import errors (common: missing optional deps)
5. Add RTD badge to `README.md`

### 3.5 Content plan

| Page | Source |
|---|---|
| Installation | New `.rst` — venv setup, optional deps |
| Quickstart | New `.rst` — load data, run `mean_cls`, plot |
| Data conventions | Extract from `CLAUDE.md` |
| API reference | Auto-generated from docstrings |
| Tutorials 01–12 | Rendered notebooks via `nbsphinx` |
| Contributing | New `.rst` — how to add modules, run tests |

---

## Phase 4 — Distribution and PyPI

### 4.1 Source distribution and wheels

**Source distribution (sdist):** a `.tar.gz` of the source tree — what pip uses when
no pre-built wheel is available for the target platform.

**Wheel (bdist_wheel):** a pre-built `.whl` archive. For pure-Python packages (no
Cython) this is a single `py3-none-any` wheel. If Cython extensions are added
(Phase 2.4), platform-specific wheels (`linux_x86_64`, `macosx_arm64`, etc.) must
be built separately — use `cibuildwheel` for this (see §4.3).

Build both with:
```bash
pip install build
python -m build          # produces dist/foregrounds_diffusion-*.tar.gz and *.whl
```

### 4.2 `pyproject.toml` audit

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

### 4.3 Wheel building with `cibuildwheel` (if Cython is added)

Pure-Python: skip this — the single `py3-none-any` wheel works everywhere.

With Cython extensions, add to `.github/workflows/publish.yml`:
```yaml
- uses: pypa/cibuildwheel@v2
  with:
    package-dir: .
    output-dir: dist
  env:
    CIBW_BUILD: "cp311-*"
    CIBW_ARCHS_LINUX: "x86_64"
    CIBW_ARCHS_MACOS: "arm64 x86_64"
```

### 4.4 TestPyPI before production

Always do a dry run on TestPyPI first:
```bash
pip install twine
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ foregrounds-diffusion
```
Verify the install works cleanly before uploading to production PyPI.

### 4.5 PyPI publish via GitHub Actions

Create a PyPI API token (pypi.org → Account settings → API tokens), store it as
`PYPI_API_TOKEN` in the GitHub repo secrets, then add:

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

### 4.6 Release workflow

1. Merge all changes to `main`; confirm tests pass
2. `git tag v0.1.0 && git push --tags`
3. GitHub Actions builds sdist + wheel, waits for manual approval in the `pypi`
   environment, then publishes
4. RTD picks up the tag and builds versioned docs (`v0.1.0` alongside `latest`)
5. Create a GitHub Release from the tag with release notes

---

## Phase 5 — CI/CD Pipeline

### 5.1 Current state

The `.github/workflows/tests.yml` stub from Phase 1 covers the basics. The full
pipeline below replaces and extends it.

### 5.2 Recommended workflow files

```
.github/workflows/
  tests.yml        # run test suite on every push and PR
  lint.yml         # code quality checks on every push and PR
  publish.yml      # build and publish to PyPI on version tag
```

### 5.3 `tests.yml` — test suite on push/PR

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

### 5.4 `lint.yml` — code quality on push/PR

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

### 5.5 Additional CI/CD improvements (suggested)

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

## Sequencing recommendation

1. **CI foundation** — `tests.yml` + `lint.yml` + branch protection. Low effort, high value.
2. **Tests** — write `conftest.py` and unit tests for `flatmaps`, `moments`, `morphology`.
3. **Docstring audit** — prerequisite for useful API docs.
4. **Sphinx + RTD skeleton** — get a basic build passing; RTD auto-updates on push from this point.
5. **`pyproject.toml` audit + TestPyPI** — dry-run the publish workflow.
6. **Optimisations** — profile first, then JIT; benchmark CI to guard regressions.
7. **PyPI publish** — tag `v0.1.0`; set up Trusted Publisher; release.
8. **Cython** — only if Numba JIT is insufficient.
9. **Additional CI items** — add dependency pinning, notebook smoke tests, `towncrier`
   incrementally as the project matures.
