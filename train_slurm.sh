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

module load cuda/11.8
source ~/diffusion_project_env/bin/activate

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

accelerate launch --num_processes 1 \
    ~/cmb_foregrounds_diffusion/train.py