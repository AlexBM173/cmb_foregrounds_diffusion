# Denoising Diffusion Probabilistic Models for Extragalactic Foregrounds from AGORA

This repository implements a denoising diffusion probabilistic model (DDPM) pipeline to generate realistic AGORA map patches, incorporating point-source masked Cosmic Infrared Background (CIB) and cluster masked thermal Sunyaev-Zeldovich (tSZ) maps. The model is trained to reproduce statistical features of simulated sky patches.

## Overview

- **Data**: AGORA maps with point sources masked at 2mJy threshold. Zeroed-out pixels represent masked regions.
- **Preprocessing**: 
  - High-frequency suppression via sharp mode cutoff (`l > 7000`) to avoid aliasing.
  - Negative pixel values from filtering artifacts are zeroed out.
- **Patching**: 
  - Patches of size 6°×6° projected to 256×256 pixel Cartesian grids.
  - Centered on a grid defined by step size of 6° adjusted for equal angular separation in galactic coordinates.

## Data Location

Maps are produced by Srini and are located at: /sptlocal/analysis/ymap/sims/mdpl2/data/v0.7/bahamas80_scal1.000/mask_radio_cib_2.0mjy/cib(tsz)

## Training
Training is handled using `huggingface-accelerate` by running the script `train.py`:
accelerate launch train.py

The training script:
Loads preprocessed maps from data/low_pass/{ptsrc}mJy/
Stacks CIB and tSZ maps into a 2-channel tensor: (N, 2, 256, 256)
Augments with 90°, 180°, 270° rotations and horizontal flips
Trains a U-Net-based DDPM model with flash attention

## Sampling
The trained model generates synthetic CIB and tSZ map pairs that resemble the original astrophysical simulations and preserve the correct cross-correlations. These samples are useful for data augmentation, uncertainty estimation, and testing cosmological inference pipelines.
New samples can be generated using `sample.py`, which loads a trained checkpoint and produces batches of correlated CIB–tSZ pairs:
accelerate launch sample.py

## Requirements
* Python 3+
** Denoising-diffusion (https://github.com/lucidrains/denoising-diffusion-pytorch) 
