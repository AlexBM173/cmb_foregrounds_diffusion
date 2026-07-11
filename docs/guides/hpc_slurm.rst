Running on an HPC cluster (SLURM)
=================================

The config-driven ``run.py`` workflow is the recommended way to drive the
pipeline everywhere; on an HPC cluster you simply wrap it in a SLURM batch
script so the scheduler allocates the GPUs. The batch scripts in
``scripts/slurm/`` are thin wrappers around the same entry points you would run
interactively — nothing about the pipeline is cluster-specific.

Scripts
-------

=================================  =================================================
Script                             Purpose
=================================  =================================================
``scripts/slurm/train.sh``         single-GPU training, 1–12 h wall time
``scripts/slurm/train_test.sh``    100-step smoke test (data + model + checkpoint)
``scripts/slurm/sample.sh``        four-GPU sampling, ~2 h wall time
=================================  =================================================

Portability
-----------

Each script begins with a small block of cluster-specific settings you must
adjust for your site. The repository and virtual-environment locations are
environment variables with sensible defaults, so you can override them without
editing the file:

.. code-block:: bash

   export REPO_DIR="$HOME/cmb_foregrounds_diffusion"   # where you cloned the repo
   export VENV_DIR="$HOME/diffusion_project_env"        # your Python environment

Also review, inside each script, the ``#SBATCH`` directives that name your
account, partition, QOS, and notification email, and the ``module load`` line
that provides CUDA — these are necessarily site-specific.

Submitting a job
----------------

Edit the run variables at the top of the script (``RUN_NAME``, ``USE_WANDB``,
and for sampling ``CHECKPOINT`` / ``OUTPUT`` / ``BATCHES`` / ``BATCH_SIZE`` /
``SAMPLING_TIMESTEPS``), then submit:

.. code-block:: bash

   sbatch scripts/slurm/train_test.sh   # verify the environment first (100 steps)
   sbatch scripts/slurm/train.sh        # full training run
   sbatch scripts/slurm/sample.sh       # sampling once a checkpoint exists

Logs stream to ``logs/<job>_<jobid>.out`` and ``.err``. Checkpoints are written
to ``results/<RUN_NAME>/`` by the legacy training entry point these scripts
invoke; if you want the config-stamped ``runs/<run>/`` layout instead, replace
the ``accelerate launch .../train.py`` line with
``python run.py train --config config/<your_run>.yaml``.

Relationship to ``run.py``
--------------------------

A SLURM job is just a batch context for the same commands. Anything you can do
interactively —

.. code-block:: bash

   python run.py train    --config config/v5_4ch.yaml
   python run.py sample   --config config/v5_4ch.yaml
   python run.py evaluate --config config/v5_4ch.yaml

— you can put inside a batch script; the scheduler only adds the resource
request. For the exact resource shapes used for the reported runs (single A100
for training, four A100s for sampling), see the ``#SBATCH`` headers in the
scripts.
