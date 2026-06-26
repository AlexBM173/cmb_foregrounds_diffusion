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
