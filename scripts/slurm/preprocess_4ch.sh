#!/bin/bash
#SBATCH --job-name=v5_preproc
#SBATCH --account=mphil-dis-sl2-cpu     # CPU account (GPU one is -gpu). VERIFY: `mybalance`
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=05:00:00
#SBATCH --partition=icelake              # CSD3 CPU partition (use icelake-himem if OOM)
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=apb86@cam.ac.uk

# ---------------------------------------------------------------------------
# End-to-end v5 (4-channel) preprocessing as a BATCH job, so a login-node or
# remote-desktop timeout can't kill it (the failure mode that lost the first
# interactive run). Chain:
#     nb02b  (mask kSZ + kappa)   ->  data/{ksz_150_masked,kappa_masked}.fits
#     nb03b  (extract 4ch patches)->  ${OUT_DIR}/*.npy + norm_params
#     verify_patches (gate)       ->  non-zero exit aborts before training
# With AUTO_TRAIN=true it submits the GPU training job (Wilkes3) on a pass.
#
# Submit from the repo root so logs/ resolves:
#   cd $RDS/repo
#   REPO_DIR=$RDS/repo AUTO_TRAIN=true sbatch scripts/slurm/preprocess_4ch.sh
# ---------------------------------------------------------------------------
set -eo pipefail

# Cluster paths — override via environment. On CSD3: $RDS/repo + ~/diffusion_project_env.
REPO_DIR="${REPO_DIR:-$HOME/cmb_foregrounds_diffusion}"
VENV_DIR="${VENV_DIR:-$HOME/diffusion_project_env}"
# 4-channel patches go to a SEPARATE dir so they don't clobber the 2-channel
# (v4) patches in data/low_pass/2mJy. rerun_4ch.sh + nb03b honour this OUT_DIR.
export OUT_DIR="${OUT_DIR:-${REPO_DIR}/data/low_pass/2mJy_4ch}"
AUTO_TRAIN="${AUTO_TRAIN:-false}"        # true -> sbatch GPU training on gate pass

cd "${REPO_DIR}"
mkdir -p logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"   # healpy map2alm is OpenMP-threaded

# Channel knobs confirmed from the FITS headers (kSZ field0 std 1.9 uK => uK;
# kappa is the col-0 "TEMPERATURE" slot of the [kappa, gamma1, gamma2] raytrace).
export KSZ_INPUT_UNIT="${KSZ_INPUT_UNIT:-uK}"
export KAPPA_FIELD="${KAPPA_FIELD:-0}"

echo "================================================"
echo "v5 4-channel preprocessing"
echo "REPO_DIR : ${REPO_DIR}"
echo "OUT_DIR  : ${OUT_DIR}"
echo "threads  : ${OMP_NUM_THREADS}"
echo "AUTO_TRAIN: ${AUTO_TRAIN}"
echo "SLURM job: ${SLURM_JOB_ID}"
echo "Started  : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================"

# 1+2. Mask kSZ/kappa (nb02b) then extract 4-channel patches (nb03b). rerun_4ch.sh
#      sources the venv, exports PROJECT_ROOT/M500C_THRESHOLD/HALO_CAT/OUT_DIR and
#      runs both scripts. It runs in a subshell, inheriting the exports above.
REPO_DIR="${REPO_DIR}" VENV_DIR="${VENV_DIR}" \
    bash scripts/vm_preprocessing/rerun_4ch.sh

# 3. Gate the extracted patches (denormalise + assert one-sidedness/finiteness).
#    set -e => a non-zero exit here aborts the script (no training on bad data).
echo "=== GATE: verify_patches --data-dir ${OUT_DIR} ==="
source "${VENV_DIR}/bin/activate"
python scripts/vm_preprocessing/verify_patches.py --data-dir "${OUT_DIR}"

echo "=== v5 PREPROCESSING + GATE PASSED @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# 4. Optionally submit GPU training (Wilkes3) with the v5 config. train.sh reads
#    RUN_NAME/CHANNELS/DIM/DATA_DIR/USE_WANDB from the environment.
if [ "${AUTO_TRAIN}" = "true" ]; then
    echo "Submitting v5 training ..."
    REPO_DIR="${REPO_DIR}" DATA_DIR="${OUT_DIR}" \
        RUN_NAME="${RUN_NAME:-v5_clean_4ch}" CHANNELS=4 DIM=96 \
        USE_WANDB="${USE_WANDB:-false}" \
        sbatch scripts/slurm/train.sh
else
    echo "AUTO_TRAIN=false — to train, run:"
    echo "  REPO_DIR=${REPO_DIR} DATA_DIR=${OUT_DIR} RUN_NAME=v5_clean_4ch CHANNELS=4 DIM=96 \\"
    echo "    sbatch scripts/slurm/train.sh"
fi
