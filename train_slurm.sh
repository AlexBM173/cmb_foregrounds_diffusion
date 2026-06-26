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
RUN_NAME="run_v1"   # checkpoints saved to results/<RUN_NAME>/
USE_WANDB=false     # set to true to enable Weights & Biases logging

# ---------------------------------------------------------------------------

echo "================================================"
echo "Run name  : ${RUN_NAME}"
echo "WandB     : ${USE_WANDB}"
echo "SLURM job : ${SLURM_JOB_ID}"
echo "Started   : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================"

mkdir -p logs

module load cuda/11.8
source ~/diffusion_project_env/bin/activate

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

WANDB_FLAG=""
if [ "${USE_WANDB}" = "true" ]; then
    WANDB_FLAG="--wandb"
fi

accelerate launch --num_processes 1 \
    ~/cmb_foregrounds_diffusion/train.py \
    --run-name "${RUN_NAME}" \
    ${WANDB_FLAG}
