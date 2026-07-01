"""foregrounds_diffusion — DDPM pipeline for extragalactic CMB foregrounds.

Public API
----------
flatmaps
    Flat-sky Fourier utilities: power-spectrum conversion, map generation,
    radial profiling, and polarisation rotation.

preprocessing
    Data normalisation, HEALPix patch extraction, Fourier filtering,
    and train/val/test splitting.

statistics
    2D Gaussian fitting and summary statistics.

moments
    Power-spectrum helpers (mean_cls, mean_cross_cls) and higher-order
    statistics (compute_summed_moments, compute_cross_moments).

morphology
    Minkowski functionals (compute_mfs) and Minkowski tensors
    (compute_minkowski_tensors).

stacking
    tSZ cluster stacking utilities (select_snr_pixels, extract_cutouts).

masking
    Flat-sky peak masks and HEALPix cluster/point-source masks for AGORA
    MDPL2 maps.

diffusion
    Model definition (U-Net + GaussianDiffusion).

train
    Training entry point (run via ``accelerate launch train.py``).

sample
    Sampling entry point (run via ``accelerate launch sample.py``).
"""

from foregrounds_diffusion import (
    flatmaps,
    masking,
    moments,
    morphology,
    preprocessing,
    stacking,
    statistics,
)

__all__ = [
    "flatmaps",
    "preprocessing",
    "statistics",
    "moments",
    "morphology",
    "stacking",
    "masking",
]
