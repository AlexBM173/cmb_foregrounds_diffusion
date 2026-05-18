"""foregrounds_diffusion — DDPM pipeline for extragalactic CMB foregrounds.

Public API
----------
flatmaps
    Flat-sky Fourier utilities: power-spectrum conversion, map generation,
    radial profiling, and polarisation rotation.

preprocessing
    Data normalisation, HEALPix patch extraction, Fourier filtering,
    masking, and train/val/test splitting.

statistics
    2D Gaussian fitting and summary statistics.

get_cluster_source_mask_for_agora
    Apodised cluster and point-source mask generation for AGORA MDPL2 maps.

diffusion
    Model definition (U-Net + GaussianDiffusion).

train
    Training entry point (run via ``accelerate launch train.py``).

sample
    Sampling entry point (run via ``accelerate launch sample.py``).
"""

from foregrounds_diffusion import flatmaps, preprocessing, statistics
from foregrounds_diffusion import get_cluster_source_mask_for_agora

__all__ = [
    "flatmaps",
    "preprocessing",
    "statistics",
    "get_cluster_source_mask_for_agora",
]