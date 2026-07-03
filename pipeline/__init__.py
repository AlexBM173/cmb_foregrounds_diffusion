"""Pipeline stage modules called by ``run.py``.

Each stage takes a validated :class:`config.PipelineConfig` and a
:class:`pipeline.rundir.RunDir` and writes all outputs there. Import stages
explicitly (``pipeline.train``, ``pipeline.rundir``, …) — this package does
not re-export them, so lightweight consumers (e.g. the ``train.py`` shim)
avoid pulling in the config machinery.
"""
