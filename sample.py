#!/usr/bin/env python
# coding: utf-8
"""Generate CIB–tSZ map pairs from a trained diffusion model checkpoint.

Usage
-----
    accelerate launch sample.py [--checkpoint PATH] [--batches N] [--batch-size B] [--output PATH]

The script loads a trained U-Net DDPM checkpoint, generates *batches* × *batch_size*
correlated CIB–tSZ patches, and saves them as a single ``.npy`` array.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from denoising_diffusion_pytorch import GaussianDiffusion, Unet


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT = "results/model-14.pt"
DEFAULT_OUTPUT = "data/low_pass/2mJy/new_samples_cib_tsz_2mJy_zero_norm_6x6_w_au_lp.npy"
DEFAULT_BATCHES = 5
DEFAULT_BATCH_SIZE = 16


def build_model(channels: int = 2) -> GaussianDiffusion:
    """Instantiate the U-Net and wrap it in a GaussianDiffusion object.

    Parameters
    ----------
    channels : int
        Number of map channels (default 2: CIB + tSZ).

    Returns
    -------
    GaussianDiffusion
        Un-trained diffusion model ready for weight loading.
    """
    unet = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=channels,
        flash_attn=True,
    )
    return GaussianDiffusion(unet, image_size=256, timesteps=1000)


def load_checkpoint(diffusion: GaussianDiffusion,
                    checkpoint_path: str,
                    accelerator: Accelerator) -> GaussianDiffusion:
    """Load model weights from *checkpoint_path* into *diffusion*.

    Parameters
    ----------
    diffusion : GaussianDiffusion
        Model instance to populate.
    checkpoint_path : str
        Path to the ``.pt`` checkpoint produced by the Trainer.
    accelerator : Accelerator
        HuggingFace Accelerator (used to unwrap the model if needed).

    Returns
    -------
    GaussianDiffusion
        Model with weights loaded, on the correct device.
    """
    device = accelerator.device
    data = torch.load(checkpoint_path, map_location=device)
    unwrapped = accelerator.unwrap_model(diffusion)
    unwrapped.load_state_dict(data['model'])
    return unwrapped


def sample(diffusion: GaussianDiffusion,
           accelerator: Accelerator,
           num_batches: int = DEFAULT_BATCHES,
           batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
    """Draw samples from the trained diffusion model.

    Parameters
    ----------
    diffusion : GaussianDiffusion
        Trained diffusion model.
    accelerator : Accelerator
        HuggingFace Accelerator instance.
    num_batches : int
        Number of sampling batches.
    batch_size : int
        Samples per batch.

    Returns
    -------
    ndarray, shape (num_batches * batch_size, C, H, W)
        Generated map patches in channels-first layout.
    """
    sampled_batches = []
    for i in range(num_batches):
        print(f"Sampling batch {i + 1}/{num_batches} "
              f"({(i / num_batches) * 100:.0f}% complete)")
        with torch.no_grad():
            batch = accelerator.gather(
                diffusion.sample(batch_size=batch_size).detach())
        sampled_batches.append(batch.cpu().numpy())

    return np.concatenate(sampled_batches, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Sample from a trained DDPM foreground model.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--batches", type=int, default=DEFAULT_BATCHES,
                        help="Number of sampling batches")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Samples per batch")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output .npy file path")
    parser.add_argument("--channels", type=int, default=2,
                        help="Number of map channels (default: 2 for CIB+tSZ)")
    args = parser.parse_args()

    accelerator = Accelerator(split_batches=True, mixed_precision='fp16')

    print(f"Loading checkpoint: {args.checkpoint}")
    diffusion = build_model(channels=args.channels)
    diffusion = diffusion.to(accelerator.device)
    diffusion = load_checkpoint(diffusion, args.checkpoint, accelerator)

    print(f"Generating {args.batches * args.batch_size} samples …")
    all_samples = sample(diffusion, accelerator,
                         num_batches=args.batches,
                         batch_size=args.batch_size)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, all_samples)
    print(f"Saved {all_samples.shape[0]} samples → {output_path}")


if __name__ == "__main__":
    main()