#!/usr/bin/env python
# coding: utf-8
"""Thin CLI shim — the implementation lives in ``pipeline/train.py``.

Kept so the original commands keep working unchanged::

    accelerate launch train.py --run-name my_run_v1 [--wandb] [--resume]

For config-driven runs use ``python run.py train --config <yaml>`` instead.
"""

from pipeline.train import main

if __name__ == "__main__":
    main()
