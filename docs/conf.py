"""Sphinx configuration for foregrounds_diffusion documentation."""

import os
import sys

# Make the package importable without installing it.
sys.path.insert(0, os.path.abspath(".."))

project = "foregrounds_diffusion"
author = "Alex Blake Martin"
copyright = "2025, Alex Blake Martin"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
]

html_theme = "furo"
html_title = "foregrounds-diffusion"

# Mock heavy optional packages so autodoc does not error on RTD.
autodoc_mock_imports = [
    "torch",
    "torch.fft",
    "torch.multiprocessing",
    "healpy",
    "accelerate",
    "denoising_diffusion_pytorch",
    "numba",
    "quantimpy",
    "kymatio",
    "astropy",
    "astropy.units",
    "einops",
    "ema_pytorch",
    "PIL",
    "pytorch_fid",
    "tqdm",
    "wandb",
]

# Napoleon settings (NumPy-style docstrings).
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_rtype = False

# autodoc defaults.
# FlatCutter uses an astropy @quantity_input decorator that confuses autodoc's
# signature introspection — exclude it from the rendered docs.
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
    "exclude-members": "FlatCutter",
}
autodoc_typehints = "description"

# nbsphinx: never re-execute notebooks on RTD (data not available).
nbsphinx_execute = "never"
nbsphinx_allow_errors = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

html_static_path = ["_static"]
