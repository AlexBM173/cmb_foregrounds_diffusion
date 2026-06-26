#!/bin/bash
#SBATCH --job-name=cmb_test
#SBATCH --account=mphil-dis-sl2-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --partition=ampere
#SBATCH --qos=gpu1
#SBATCH --output=logs/train_test_%j.out
#SBATCH --error=logs/train_test_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=apb86@cam.ac.uk

# ---------------------------------------------------------------------------
# Quick smoke test — 100 steps, no WandB.
# Verifies data loading, model init, and checkpoint saving work on the cluster.
# ---------------------------------------------------------------------------
RUN_NAME="test_run"
USE_WANDB=false

echo "================================================"
echo "Run name  : ${RUN_NAME}"
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
    --steps 100 \
    --batch-size 4 \
    ${WANDB_FLAG}

echo "================================================"
echo "Finished  : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================"
