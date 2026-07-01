Installation
============

Requirements
------------

- Python 3.11+
- A CUDA-capable GPU is required for training; CPU-only is sufficient for
  statistics, sampling evaluation, and running the tutorials.

From source (recommended)
--------------------------

.. code-block:: bash

   git clone https://github.com/AlexBM173/cmb_foregrounds_diffusion
   cd cmb_foregrounds_diffusion
   python -m venv .venv
   source .venv/bin/activate

   # CPU-only PyTorch (avoids pulling in the large CUDA build on machines
   # without a GPU):
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

   # Install the package with development extras:
   pip install -e ".[dev]"

Optional dependencies
---------------------

Some tutorial notebooks require additional packages not installed by default:

.. code-block:: bash

   # Minkowski functionals (tutorial 12):
   pip install quantimpy

   # Scattering transforms (tutorial 11) — choose one:
   pip install kymatio           # CPU/GPU via PyTorch
   # or clone the Cheng et al. (2021) implementation

   # Documentation build (local):
   pip install -e ".[docs]"

GPU training
------------

For training on a CUDA GPU, install the CUDA-enabled PyTorch wheel matching
your driver version before running ``pip install -e .``:

.. code-block:: bash

   # Example for CUDA 12.1:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip install -e .
   accelerate config     # select GPU count and mixed precision
   accelerate launch foregrounds_diffusion/train.py --run-name my_run_v1

SLURM cluster
-------------

Pre-built SLURM scripts are provided at the repo root:

.. code-block:: bash

   # Edit RUN_NAME and USE_WANDB at the top of each script, then:
   sbatch train_slurm.sh    # single GPU, 1–12 h wall time
   sbatch sample_slurm.sh   # 4 GPUs, 2 h wall time
