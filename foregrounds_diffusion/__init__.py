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

train
    Training entry point (repository-root script ``train.py``; run via
    ``accelerate launch train.py``).

sample
    Sampling entry point and reusable helpers (``build_model``,
    ``load_checkpoint``, ``sample``); run via
    ``accelerate launch foregrounds_diffusion/sample.py``.
"""

__all__ = [
    "flatmaps",
    "preprocessing",
    "statistics",
    "moments",
    "morphology",
    "stacking",
    "masking",
    "peak_counts",
    "scattering_stats",
    "plot_style",
]
