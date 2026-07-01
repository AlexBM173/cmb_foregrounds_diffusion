#!/bin/bash
#SBATCH --job-name=cmb_sample
#SBATCH --account=mphil-dis-sl2-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --partition=ampere
#SBATCH --qos=gpu1
#SBATCH --output=logs/sample_%j.out
#SBATCH --error=logs/sample_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=apb86@cam.ac.uk

# ---------------------------------------------------------------------------
# Edit these before each submission
# ---------------------------------------------------------------------------
CHECKPOINT="results/run_v1/model-20.pt"   # path to trained checkpoint
OUTPUT="data/low_pass/2mJy/samples.npy"   # output .npy file
BATCHES=10                                 # number of sampling batches
BATCH_SIZE=16                              # samples per batch (per GPU)
SAMPLING_TIMESTEPS=""                      # leave empty for full DDPM (1000 steps); set e.g. 250 for DDIM
USE_WANDB=false                            # set to true to enable WandB logging

# ---------------------------------------------------------------------------

echo "================================================"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output    : ${OUTPUT}"
echo "Samples   : $((BATCHES * BATCH_SIZE * 4)) (${BATCHES} batches × ${BATCH_SIZE} × 4 GPUs)"
echo "Timesteps : ${SAMPLING_TIMESTEPS:-1000 (DDPM)}"
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

DDIM_FLAG=""
if [ -n "${SAMPLING_TIMESTEPS}" ]; then
    DDIM_FLAG="--sampling-timesteps ${SAMPLING_TIMESTEPS}"
fi

accelerate launch --multi_gpu --num_processes 4 \
    ~/cmb_foregrounds_diffusion/foregrounds_diffusion/sample.py \
    --checkpoint "${CHECKPOINT}" \
    --batches "${BATCHES}" \
    --batch-size "${BATCH_SIZE}" \
    --output "${OUTPUT}" \
    ${DDIM_FLAG} \
    ${WANDB_FLAG}
