Contributing
============

This page covers the conventions for adding new modules, writing tests, and
keeping the documentation in sync.

.. contents:: Contents
   :local:
   :depth: 1

----

Development setup
-----------------

Clone the repository and install in editable mode with all development
dependencies:

.. code-block:: bash

    git clone https://github.com/AlexBM173/cmb_foregrounds_diffusion.git
    cd cmb_foregrounds_diffusion
    pip install -e ".[dev,docs]"

Install pre-commit hooks so ruff runs automatically before each commit:

.. code-block:: bash

    pre-commit install

Running tests
-------------

The test suite uses ``pytest``.  Run all tests from the repository root:

.. code-block:: bash

    pytest tests/ -v

Run a single module's tests:

.. code-block:: bash

    pytest tests/test_flatmaps.py -v

Run with coverage:

.. code-block:: bash

    pytest tests/ --cov=foregrounds_diffusion --cov-report=term-missing

Some tests are marked ``optional`` and are skipped when a dependency (e.g.
``quantimpy``, ``kymatio``) is not installed:

.. code-block:: bash

    pytest tests/ -m "not optional"   # skip optional tests
    pytest tests/ -m optional          # run only optional tests

Adding a module
---------------

1. Create ``foregrounds_diffusion/<name>.py``.
2. Write a module-level docstring (see :ref:`docstring-style` below).
3. Add an API page at ``docs/api/<name>.rst`` — copy the structure from an
   existing page such as ``docs/api/flatmaps.rst``.
4. Add the new page to the toctree in ``docs/api/index.rst``.
5. Add a corresponding test file ``tests/test_<name>.py``.
6. Add the module to the package summary table in ``CLAUDE.md`` and
   ``README.md``.

.. _docstring-style:

Docstring style
---------------

All public functions use **NumPy-style** docstrings.  The minimum required
sections are ``Parameters`` and ``Returns``; add ``Notes`` for any non-obvious
behaviour.

.. code-block:: python

    def my_function(maps, threshold, n_jobs=1):
        """One-line summary ending with a period.

        Optional extended description.  Can span multiple paragraphs and
        include references or equations.

        Parameters
        ----------
        maps : ndarray, shape (N, H, W)
            Stack of N flat-sky patches, each H × W pixels.
        threshold : float
            Excursion-set threshold in physical units.
        n_jobs : int, optional
            Number of parallel workers.  ``-1`` uses all available cores.
            Default is 1 (single-threaded).

        Returns
        -------
        result : ndarray, shape (N, T)
            Description of the output.

        Notes
        -----
        Any non-obvious behaviour, edge cases, or references to related
        functions.
        """

Key conventions:

- **Shape annotations**: always write ``shape (N, H, W)`` for arrays, not
  just ``array-like``.
- **Units**: include units in the description (e.g. ``arcmin``, ``µK``,
  ``Jy/sr``).
- **Defaults**: document the default value in the parameter description.
- **Private helpers**: functions whose names start with ``_`` do not need
  full docstrings, but a one-line summary is still helpful.

Adding a tutorial notebook
--------------------------

Tutorials live in ``docs/tutorials/`` and are numbered sequentially.
Notebooks 01–09 cover the core pipeline; 10–14 cover extended statistics and
paper figures.

Conventions:

- The first cell must be a Markdown cell with a single ``#`` heading:
  ``# NN — Title``.
- Subsequent sections use ``##`` headings; subsections use ``###``.
- Do not use ``#`` for descriptive prose — use plain text paragraphs.
- Set all configuration constants (paths, thresholds, ``N_MAPS``) in a
  dedicated code cell near the top, before any computation.
- Use the canonical variable names from ``CLAUDE.md`` (e.g. ``agora_cib``,
  ``ddpm_tsz``, ``flatskymapparams``).
- Commit notebooks **with outputs cleared** — outputs contain large arrays
  that bloat the git history.  Clear with:

  .. code-block:: bash

      jupyter nbconvert --clear-output --inplace docs/tutorials/NN_name.ipynb

- Add the notebook to the toctree in ``docs/index.rst`` under
  ``Tutorials``.

Building the documentation locally
-----------------------------------

.. code-block:: bash

    pip install -e ".[docs]"
    sphinx-build -b html docs/ docs/_build/html

Open ``docs/_build/html/index.html`` in a browser to preview.  The CI
workflow at ``.github/workflows/docs.yml`` runs the same command on every
push to verify the build succeeds before it reaches ReadTheDocs.
