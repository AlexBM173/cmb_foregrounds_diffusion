Running on Google Cloud + Colab Pro+
====================================

This is the workflow actually used to produce the results: preprocessing on a
Google Compute Engine VM, training and sampling on a Colab Pro+ A100, with all
artifacts staged through a Google Cloud Storage (GCS) bucket. It is written
generically — substitute your own project and bucket names for the
``<your-...>`` placeholders throughout.

The pattern is worth the setup because Colab sessions are time-limited and can
disconnect: keeping the single source of truth in a GCS bucket makes every stage
resumable and lets the (short-lived) Colab GPU be treated as disposable compute.

Prerequisites
-------------

- A Google Cloud project with billing enabled and the Compute Engine and Cloud
  Storage APIs turned on.
- A GCS bucket for artifacts, e.g. ``gs://<your-bucket>``.
- Colab Pro+ (for A100 access and longer/background sessions).
- The ``gcloud``/``gsutil`` CLIs locally, or use them from within Colab.

Bucket layout
-------------

A single bucket holds everything the pipeline reads and writes::

    gs://<your-bucket>/
    ├── patches/<run>/        # preprocessed training .npy arrays (VM output)
    ├── colab/                # the launcher scripts (optional convenience copy)
    └── runs/<run>/           # checkpoints, logs, samples, stats (Colab output)

1. Preprocessing on a Compute Engine VM
---------------------------------------

Preprocessing the full-sky maps is CPU- and memory-bound, not GPU-bound, so a
standard high-memory VM is the right tool.

.. code-block:: bash

   gcloud compute instances create <your-vm> \
       --machine-type=n2-highmem-16 --boot-disk-size=200GB \
       --image-family=debian-12 --image-project=debian-cloud

   gcloud compute ssh <your-vm>
   # on the VM: clone the repo, install deps, pull the raw maps, then run
   python scripts/vm_preprocessing/nb01_run.py            # halo catalogue
   python scripts/vm_preprocessing/nb02_run.py            # CIB + tSZ masking
   python scripts/vm_preprocessing/nb02b_mask_ksz_kappa.py  # kSZ + kappa (4-field)
   python scripts/vm_preprocessing/nb03_run.py            # 2-field patches
   python scripts/vm_preprocessing/nb03b_extract_4ch.py   # 4-field patches

Push the resulting patches to the bucket, then **delete the VM** so it stops
costing money:

.. code-block:: bash

   gcloud storage rsync data/low_pass/2mJy gs://<your-bucket>/patches/<run>
   gcloud compute instances delete <your-vm>

2. Training on a Colab Pro+ A100
--------------------------------

Select a Colab **A100** runtime, authenticate, and launch. The launchers in
``scripts/colab/`` encode the full resumable pattern; a minimal cell is:

.. code-block:: ipython3

   from google.colab import auth; auth.authenticate_user()
   !gcloud config set project <your-project> -q
   !gcloud -q storage cp gs://<your-bucket>/colab/train.sh /content/ && bash /content/train.sh

The launcher script (see ``scripts/colab/train_v5.sh`` for the reference
implementation) does the following, all keyed to the bucket:

- clones the repository and checks out the target branch;
- installs the package and the pinned ``denoising-diffusion-pytorch`` version
  (the checkpoint architecture depends on it);
- ``rsync``\ s the preprocessed patches from ``patches/<run>`` to local disk;
- ``rsync``\ s any existing ``runs/<run>/`` back down and, if a checkpoint is
  present, sets ``RESUME=1`` so training continues instead of restarting;
- starts a **background loop that pushes checkpoints to the bucket every ten
  minutes**, so a disconnect loses at most the current unfinished milestone;
- runs ``python run.py train --config config/<run>.yaml``.

Because state lives in the bucket, recovering from a dropped session is simply
re-running the same cell: it finds the latest checkpoint and resumes.

3. Sampling and evaluation
--------------------------

Sampling follows the same shape with ``scripts/colab/sample_v5.sh``: pull the
checkpoint from ``runs/<run>/checkpoints/``, run
``python run.py sample --config config/<run>.yaml``, and push the samples back
to the bucket. Evaluation is CPU-friendly and can run on the VM, in Colab, or
locally after pulling ``runs/<run>/`` down:

.. code-block:: bash

   gcloud storage rsync --recursive gs://<your-bucket>/runs/<run> runs/<run>
   python run.py evaluate --config config/<run>.yaml

Cost notes
----------

- Delete the preprocessing VM as soon as the patches are in the bucket; it is
  the most expensive idle resource.
- GCS egress is charged, so run evaluation where the data already is when you
  can, rather than pulling full run directories down repeatedly.
- The A100 session is disposable — the resumable pattern means you never pay to
  redo completed training.
