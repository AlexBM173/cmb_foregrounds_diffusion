#!/bin/bash
#SBATCH --job-name=cmb_diffusion
#SBATCH --account=mphil-dis-sl2-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-12:00:00
#SBATCH --partition=ampere
#SBATCH --qos=gpu1
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

module load cuda/11.8
source ~/diffusion_project_env/bin/activate

accelerate launch --multi_gpu --num_processes 4 \
    ~/cmb_foregrounds_diffusion/train.py
