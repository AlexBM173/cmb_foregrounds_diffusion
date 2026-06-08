#!/usr/bin/env python
# coding: utf-8
import argparse
import json
import os
from datetime import datetime, timezone
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer1D, Dataset1D

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Train CMB foreground diffusion model.")
parser.add_argument(
    "--run-name",
    default=None,
    metavar="NAME",
    help="Name for this training run (also read from $RUN_NAME env var). "
         "Checkpoints and logs are saved to results/<NAME>/.",
)
parser.add_argument("--ptsrc",      type=int,   default=2,      help="Point-source threshold in mJy (default: 2)")
parser.add_argument("--res",        type=int,   default=256,    help="Map resolution in pixels (default: 256)")
parser.add_argument("--steps",      type=int,   default=100000, help="Training steps (default: 100000)")
parser.add_argument("--batch-size", type=int,   default=16,     help="Batch size per GPU (default: 16)")
parser.add_argument("--lr",         type=float, default=1e-4,   help="Learning rate (default: 1e-4)")
args = parser.parse_args()

# Run name: CLI flag takes priority, then $RUN_NAME env var
run_name = args.run_name or os.environ.get("RUN_NAME", "").strip()
if not run_name:
    parser.error(
        "--run-name is required (or set the RUN_NAME environment variable).\n"
        "  CLI:   accelerate launch train.py --run-name my_run_v1\n"
        "  SLURM: set RUN_NAME=\"my_run_v1\" in train_slurm.sh"
    )

RES   = args.res
PTSRC = args.ptsrc

RESULTS_DIR = Path("results") / run_name
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

model     = Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=2, flash_attn=True)
diffusion = GaussianDiffusion(model, image_size=256, timesteps=1000, auto_normalize=False)
dataset   = Dataset1D(augmented_images)

trainer = Trainer1D(
    diffusion,
    dataset=dataset,
    train_batch_size=args.batch_size,
    train_lr=args.lr,
    train_num_steps=args.steps,
    save_and_sample_every=5000,
    gradient_accumulate_every=2,
    ema_decay=0.995,
    mixed_precision_type='bf16',
    results_folder=str(RESULTS_DIR),
)

# Override dataloader after init with num_workers=0 to prevent hanging
trainer.dl = cycle(
    trainer.accelerator.prepare(
        data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
    )
)

# ---------------------------------------------------------------------------
# Log run config
# ---------------------------------------------------------------------------

run_config = {
    "run_name":       run_name,
    "started":        datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "slurm_job_id":   os.environ.get("SLURM_JOB_ID"),
    "results_dir":    str(RESULTS_DIR),
    "data_dir":       str(DATA_DIR),
    "ptsrc_mJy":      PTSRC,
    "resolution":     RES,
    "n_train":        len(training_images),
    "n_augmented":    len(augmented_images),
    "train_steps":    args.steps,
    "batch_size":     args.batch_size,
    "lr":             args.lr,
}
with open(RESULTS_DIR / "run_config.json", "w") as f:
    json.dump(run_config, f, indent=2)

print(f"\nRun '{run_name}'  →  {RESULTS_DIR}/")
print(json.dumps(run_config, indent=2))
print()

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

trainer.train()
