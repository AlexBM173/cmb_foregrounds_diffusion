#!/bin/bash
#SBATCH --job-name=cmb_diffusion
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

mkdir -p ~/cmb_foregrounds_diffusion/logs

module load cuda/11.8
source ~/diffusion_project_env/bin/activate

accelerate launch --multi_gpu --num_processes 4 \
    ~/cmb_foregrounds_diffusion/train.py
