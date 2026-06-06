#!/usr/bin/env python
# coding: utf-8
import os
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer1D, Dataset1D

RES   = 256
PTSRC = 2

DATA_DIR = Path(f"docs/tutorials/data/low_pass/{PTSRC}mJy")
cib_maps = np.load(DATA_DIR / f"CIB_map_150GHz_{RES}_st6_zscore_{PTSRC}mJy_lp.npy")
tsz_maps = np.load(DATA_DIR / f"tSZ3_map_150GHz_{RES}_st6_zscore_{PTSRC}mJy_lp.npy")
cut_maps = np.concatenate([cib_maps, tsz_maps], axis=-1).transpose(0, 3, 1, 2)
print(f"Loaded {len(cut_maps)} patches, shape {cut_maps.shape}")

rng           = np.random.default_rng(seed=42)
indices       = rng.permutation(len(cut_maps))
train_indices = indices[:int(0.8 * len(cut_maps))]
training_images = torch.tensor(cut_maps[train_indices], dtype=torch.float32)
print(f"Training set: {len(training_images)} patches")

def augment_images_unique(images):
    r1 = torch.rot90(images, k=1, dims=(2, 3))
    r2 = torch.rot90(images, k=2, dims=(2, 3))
    r3 = torch.rot90(images, k=3, dims=(2, 3))
    flips = [torch.flip(x, dims=[3]) for x in [images, r1, r2, r3]]
    return torch.cat([images, r1, r2, r3] + flips, dim=0)

augmented_images = augment_images_unique(training_images)
print(f"After augmentation: {len(augmented_images)} patches")

model = Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=2, flash_attn=True)
diffusion = GaussianDiffusion(model, image_size=256, timesteps=1000)

dataset = Dataset1D(augmented_images)

trainer = Trainer1D(
    diffusion,
    dataset=dataset,
    train_batch_size=16,
    train_lr=1e-4,
    train_num_steps=100000,
    save_and_sample_every=5000,
    gradient_accumulate_every=2,
    ema_decay=0.995,
    mixed_precision_type='bf16',
)

# Override dataloader after init with num_workers=0 to prevent hanging
trainer.dl = cycle(
    trainer.accelerator.prepare(
        data.DataLoader(
            dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
    )
)

trainer.train()