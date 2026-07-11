# Contributing

Thanks for your interest in `foregrounds_diffusion`. This file is the quick
start; the full guide (adding modules, docstring style, tutorial conventions,
building the docs) lives at
[docs/guides/contributing.rst](docs/guides/contributing.rst) and renders at
<https://cmb-foregrounds-diffusion.readthedocs.io/en/latest/guides/contributing.html>.

## Development setup

```bash
git clone https://github.com/AlexBM173/cmb_foregrounds_diffusion.git
cd cmb_foregrounds_diffusion
pip install -e ".[dev,docs]"
pre-commit install          # ruff + nbstripout run automatically on commit
```

A CUDA GPU is needed only for training; the statistics, sampling evaluation,
and tests all run CPU-only.

## Running the checks

```bash
pytest tests/                         # full suite (228 tests, ~15 s, CPU-only)
pytest tests/ -m "not optional"       # skip tests needing quantimpy / kymatio
pytest tests/ --cov=foregrounds_diffusion --cov-report=term-missing
ruff check . && ruff format --check . # lint + format (also run by pre-commit)
```

Please make sure `pytest tests/` passes before opening a pull request.

## Conventions at a glance

- **Style**: `ruff` (line length 100). All public functions use **NumPy-style**
  docstrings with `Parameters` and `Returns`, array **shapes** (`shape (N, H, W)`),
  and **units** (`arcmin`, `µK`, `Jy/sr`).
- **Tests**: every module in `foregrounds_diffusion/` has a
  `tests/test_<name>.py`. Mark tests needing an optional dependency with
  `@pytest.mark.optional`.
- **Notebooks**: commit tutorials with **outputs cleared** (the `nbstripout`
  pre-commit hook enforces this). First cell is a single `#` heading `NN — Title`.
- **Config**: the pipeline is config-driven — see
  [docs/guides/configuration.rst](docs/guides/configuration.rst). New settings go
  in `config/validate.py` (schema) *and* `config/default.yaml` (annotated).
- **Docs**: a new module needs an API page in `docs/api/`, a toctree entry, and
  a row in the package tables in `README.md` and `CLAUDE.md`.

## Pull requests

1. Branch off `main`.
2. Keep the change focused; update tests and docs alongside the code.
3. Fill in the PR checklist (tests pass, docstrings, docs, notebook outputs
   cleared).

## Reporting bugs and proposing extensions

Open an issue using the templates. Proposed extensions (new foreground channels,
conditional generation, faster samplers, …) are catalogued under **Future
extensions** in the [README](README.md#future-extensions) — a good place to look
before proposing one.
