# Development Plan

Three-phase plan covering a full test suite, code optimisation, and public documentation.
Each phase is independent and can be started in any order, though Phase 1 (tests) should
ideally precede Phase 3 (optimisations) so regressions are caught automatically.

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

### 1.4 CI (GitHub Actions)

```yaml
# .github/workflows/tests.yml
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v --cov=foregrounds_diffusion --cov-report=xml
      - uses: codecov/codecov-action@v4
```

Add `[dev]` optional dependency group to `pyproject.toml`:
```toml
[project.optional-dependencies]
dev = ["pytest", "pytest-cov", "codecov"]
```

---

## Phase 2 — Code Optimisations

Profile before optimising. Use `cProfile` + `line_profiler` (`@profile`) to identify
hot spots. The candidates below are ranked by expected impact.

### 2.1 Profile first

```python
# Quick profiling harness
import cProfile
cProfile.run('compute_minkowski_tensors(maps, norm_fn, thresholds)', sort='cumtime')
```

### 2.2 Numba JIT — high priority

Target: `morphology.py` inner loops and `peak_counts.py`.

`_tensor_W012`, `_tensor_W200`, `_tensor_W201` each iterate over pixels in Python.
Replace with `@numba.jit(nopython=True)` kernels:

```python
import numba

@numba.jit(nopython=True)
def _accumulate_normals(rows, cols, gx, gy):
    W = np.zeros((2, 2))
    for i in range(len(rows)):
        nx, ny = gx[rows[i], cols[i]], gy[rows[i], cols[i]]
        n = np.array([nx, ny])
        norm = np.sqrt(nx*nx + ny*ny)
        if norm > 0:
            n /= norm
            W += np.outer(n, n)
    return W
```

Also JIT the per-threshold loop in `compute_minkowski_tensors` to avoid Python overhead
across 50+ thresholds × 100+ maps.

### 2.3 Advanced NumPy vectorisation

**`compute_minkowski_tensors`:** The current implementation loops over `(map, threshold)`.
Vectorise over thresholds by binarising the entire stack at once:
```python
# (N, T, H, W) binary array in one shot
binary_stack = maps_nhw[:, None, :, :] > thresholds[None, :, None, None]
```
Then use `np.einsum` for the tensor accumulations where possible.

**`compute_mfs`:** Same threshold vectorisation opportunity.

**`mean_cls` / `mean_cross_cls`:** Pre-compute the ℓ-bin assignment array once
outside the loop rather than recomputing per map.

**`select_snr_pixels`:** Replace the separation check loop with a vectorised
distance-matrix approach using `scipy.spatial.cKDTree`.

### 2.4 Cython — medium priority

For `morphology.py` boundary pixel accumulation if Numba is insufficient:

```
foregrounds_diffusion/
  _morphology_cy.pyx      # Cython extension
  _morphology_cy.pxd      # declarations
```

Build via `pyproject.toml` with `Cython` as a build dependency. Keep the pure-Python
fallback path for environments without a C compiler.

### 2.5 `torch.compile` for sampling

In `sample.py`, wrap the model before sampling:
```python
diffusion = torch.compile(diffusion)   # requires PyTorch 2.0+
```
Expected 20–40% speedup on repeated forward passes with no code changes. Add a
`--no-compile` flag to disable for debugging.

### 2.6 Memory layout

Ensure all arrays passed to FFT routines are C-contiguous:
```python
maps = np.ascontiguousarray(maps)
```
Add this to `map2cl` and `get_lpf_hpf` entry points. Avoids silent copies inside
numpy's FFT implementation.

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

Add `[docs]` optional dependency group to `pyproject.toml`:
```toml
[project.optional-dependencies]
docs = ["sphinx>=7", "furo", "nbsphinx", "sphinx-copybutton",
        "sphinx-autodoc-typehints", "ipykernel"]
```

### 3.4 ReadTheDocs setup steps

1. Push `.readthedocs.yaml` and `docs/conf.py` to GitHub
2. Go to readthedocs.org → Import project → connect `AlexBM173/cmb_foregrounds_diffusion`
3. Set default branch to `main`
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

## Sequencing recommendation

1. **Tests first** — write `conftest.py` and the unit tests for the three most-used
   modules (`flatmaps`, `moments`, `morphology`). Wire up GitHub Actions.
2. **Docstring audit** — fix missing/incomplete docstrings across all modules.
   This is a prerequisite for useful auto-generated API docs.
3. **Sphinx skeleton** — get a basic RTD build passing before adding content.
4. **Optimisations** — profile a full sampling run, then JIT the top-3 hot spots.
   Benchmark before and after with `timeit` to confirm gains.
5. **Cython** — only if Numba JIT is insufficient for the Minkowski tensor loops.
