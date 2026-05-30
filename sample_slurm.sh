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

module load cuda/11.8
source ~/diffusion_project_env/bin/activate

accelerate launch --multi_gpu --num_processes 4 \
    ~/cmb_foregrounds_diffusion/sample.py \
    --checkpoint results/model-20.pt \
    --batches 10 \
    --batch-size 16 \
    --output data/low_pass/2mJy/new_samples_cib_tsz_2mJy_zero_norm_6x6_w_au_lp.npy
