#!/bin/bash
#SBATCH --job-name=cmb_diffusion
#SBATCH --account=mphil-dis-sl2-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-12:00:00
#SBATCH --partition=ampere
#SBATCH --qos=gpu1
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=apb86@cam.ac.uk

# ---------------------------------------------------------------------------
# Edit these before each submission
# ---------------------------------------------------------------------------
# These may be set here or overridden from the environment (so preprocess_4ch.sh
# can chain into training with RUN_NAME=... CHANNELS=4 DIM=96 sbatch train.sh).
RUN_NAME="${RUN_NAME:-run_v1}"   # checkpoints saved to results/<RUN_NAME>/
CHANNELS="${CHANNELS:-2}"        # 2 (CIB+tSZ) or 4 (+kSZ+kappa)
DIM="${DIM:-64}"                 # U-Net base width (v4=64, v5=96)
USE_WANDB="${USE_WANDB:-false}"  # set to true to enable Weights & Biases logging

# Cluster-specific paths — override via environment or edit here.
REPO_DIR="${REPO_DIR:-$HOME/cmb_foregrounds_diffusion}"
VENV_DIR="${VENV_DIR:-$HOME/diffusion_project_env}"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/data/low_pass/2mJy}"  # holds the nb03/nb03b .npy patches

# ---------------------------------------------------------------------------

echo "================================================"
echo "Run name  : ${RUN_NAME}"
echo "Channels  : ${CHANNELS}  (dim ${DIM})"
echo "Data dir  : ${DATA_DIR}"
echo "WandB     : ${USE_WANDB}"
echo "SLURM job : ${SLURM_JOB_ID}"
echo "Started   : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================"

# Run from the repo so results/<run> and logs/ resolve to the checkout.
cd "${REPO_DIR}"
mkdir -p logs

module load cuda/11.8
source "${VENV_DIR}/bin/activate"

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

WANDB_FLAG=""
if [ "${USE_WANDB}" = "true" ]; then
    WANDB_FLAG="--wandb"
fi

accelerate launch --num_processes 1 \
    "${REPO_DIR}/train.py" \
    --run-name "${RUN_NAME}" \
    --channels "${CHANNELS}" \
    --dim "${DIM}" \
    --data-dir "${DATA_DIR}" \
    ${WANDB_FLAG}
