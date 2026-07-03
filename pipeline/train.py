#!/usr/bin/env python
# coding: utf-8
"""Train the CIB+tSZ diffusion model.

This module is the implementation behind two equivalent entry points::

    accelerate launch train.py --run-name my_run_v1        # root shim
    python run.py train --config config/my_run.yaml        # config-driven

The CLI surface is unchanged from the original root ``train.py``; the
``--data-dir`` and ``--results-dir`` options exist so the config-driven
runner can redirect inputs and outputs into ``runs/<run_name>/``.
"""

import argparse
import json
import os
from datetime import datetime, timezone
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
from denoising_diffusion_pytorch import Dataset1D, GaussianDiffusion, Trainer1D, Unet
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import num_to_groups
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Trainer with WandB loss logging
# ---------------------------------------------------------------------------


class WandbTrainer1D(Trainer1D):
    """Trainer1D with optional per-step wandb loss and sample image logging."""

    def __init__(self, *args, use_wandb: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_wandb = use_wandb

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        log = self.use_wandb and accelerator.is_main_process

        with tqdm(
            initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process
        ) as pbar:
            while self.step < self.train_num_steps:
                self.model.train()
                total_loss = 0.0

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)
                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                    self.accelerator.backward(loss)

                pbar.set_description(f"loss: {total_loss:.4f}")

                if log:
                    import wandb

                    wandb.log({"train/loss": total_loss}, step=self.step)

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()
                self.opt.zero_grad()
                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        milestone = self.step // self.save_and_sample_every

                        if self.num_samples > 0:
                            self.ema.ema_model.eval()

                            with torch.no_grad():
                                batches = num_to_groups(self.num_samples, self.batch_size)
                                all_samples = torch.cat(
                                    [self.ema.ema_model.sample(batch_size=n) for n in batches],
                                    dim=0,
                                )

                            sample_path = self.results_folder / f"sample-{milestone}.pt"
                            torch.save(all_samples, str(sample_path))

                        self.save(milestone)

                        if log and self.num_samples > 0:
                            import wandb

                            samples_np = all_samples.cpu().numpy()
                            n_show = min(8, len(samples_np))
                            wandb.log(
                                {
                                    "samples/cib": [
                                        wandb.Image(samples_np[i, 0]) for i in range(n_show)
                                    ],
                                    "samples/tsz": [
                                        wandb.Image(samples_np[i, 1]) for i in range(n_show)
                                    ],
                                    "checkpoint": milestone,
                                },
                                step=self.step,
                            )

                pbar.update(1)

        accelerator.print("training complete")


def augment_images_unique(images):
    """Return the 8× dihedral augmentation of a channels-first image stack.

    Parameters
    ----------
    images : Tensor, shape (N, C, H, W)
        Input image batch.

    Returns
    -------
    Tensor, shape (8N, C, H, W)
        The four rotations of each image plus their mirror images.
    """
    r1 = torch.rot90(images, k=1, dims=(2, 3))
    r2 = torch.rot90(images, k=2, dims=(2, 3))
    r3 = torch.rot90(images, k=3, dims=(2, 3))
    flips = [torch.flip(x, dims=[3]) for x in [images, r1, r2, r3]]
    return torch.cat([images, r1, r2, r3] + flips, dim=0)


def main(argv: list[str] | None = None):
    """Run the training CLI.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments. ``None`` (default) reads ``sys.argv``,
        preserving the original CLI behaviour; passing a list allows the
        pipeline runner (``run.py``) to call this entry point directly.
    """
    parser = argparse.ArgumentParser(description="Train CMB foreground diffusion model.")
    parser.add_argument(
        "--run-name",
        default=None,
        metavar="NAME",
        help="Name for this training run (also read from $RUN_NAME env var). "
        "Checkpoints and logs are saved to results/<NAME>/.",
    )
    parser.add_argument(
        "--ptsrc", type=int, default=2, help="Point-source threshold in mJy (default: 2)"
    )
    parser.add_argument(
        "--res", type=int, default=256, help="Map resolution in pixels (default: 256)"
    )
    parser.add_argument(
        "--steps", type=int, default=100000, help="Training steps (default: 100000)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size per GPU (default: 16)"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument(
        "--save-every",
        type=int,
        default=5000,
        metavar="N",
        help="Save a checkpoint (and sample) every N steps (default: 5000)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=25,
        metavar="M",
        help="Samples to generate at each checkpoint milestone; must have an integer "
        "square root; 0 disables milestone sampling (default: 25)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from the latest model-*.pt in results/<run-name>/ (also set via RESUME=1 env var)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        metavar="DIR",
        help="Directory holding the training .npy patch files "
        "(default: docs/tutorials/data/low_pass/<ptsrc>mJy)",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        metavar="DIR",
        help="Directory for checkpoints and logs (default: results/<run-name>)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases logging (also set via WANDB=1 env var)",
    )
    args = parser.parse_args(argv)

    # Run name: CLI flag takes priority, then $RUN_NAME env var
    run_name = args.run_name or os.environ.get("RUN_NAME", "").strip()
    if not run_name:
        parser.error(
            "--run-name is required (or set the RUN_NAME environment variable).\n"
            "  CLI:   accelerate launch train.py --run-name my_run_v1\n"
            '  SLURM: set RUN_NAME="my_run_v1" in train_slurm.sh'
        )

    use_wandb = args.wandb or os.environ.get("WANDB", "").strip() in ("1", "true", "yes")
    resume = args.resume or os.environ.get("RESUME", "").strip() in ("1", "true", "yes")

    RES = args.res
    PTSRC = args.ptsrc

    RESULTS_DIR = Path(args.results_dir) if args.results_dir else Path("results") / run_name
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------

    DATA_DIR = (
        Path(args.data_dir) if args.data_dir else Path(f"docs/tutorials/data/low_pass/{PTSRC}mJy")
    )
    cib_maps = np.load(DATA_DIR / f"CIB_map_150GHz_{RES}_st6_zscore_{PTSRC}mJy_lp.npy")
    tsz_maps = np.load(DATA_DIR / f"tSZ3_map_150GHz_{RES}_st6_zscore_{PTSRC}mJy_lp.npy")
    cut_maps = np.concatenate([cib_maps, tsz_maps], axis=-1).transpose(0, 3, 1, 2)
    print(f"Loaded {len(cut_maps)} patches, shape {cut_maps.shape}")

    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(len(cut_maps))
    train_indices = indices[: int(0.8 * len(cut_maps))]
    training_images = torch.tensor(cut_maps[train_indices], dtype=torch.float32)
    print(f"Training set: {len(training_images)} patches")

    augmented_images = augment_images_unique(training_images)
    print(f"After augmentation: {len(augmented_images)} patches")

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------

    model = Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=2, flash_attn=True)
    diffusion = GaussianDiffusion(model, image_size=256, timesteps=1000, auto_normalize=False)
    dataset = Dataset1D(augmented_images)

    trainer = WandbTrainer1D(
        diffusion,
        dataset=dataset,
        train_batch_size=args.batch_size,
        train_lr=args.lr,
        train_num_steps=args.steps,
        save_and_sample_every=args.save_every,
        num_samples=args.num_samples,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        mixed_precision_type="bf16",
        results_folder=str(RESULTS_DIR),
        use_wandb=use_wandb,
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

    # -----------------------------------------------------------------------
    # Resume
    # -----------------------------------------------------------------------

    resume_step = 0
    if resume:
        checkpoints = sorted(
            RESULTS_DIR.glob("model-*.pt"), key=lambda p: int(p.stem.split("-")[-1])
        )
        if not checkpoints:
            raise SystemExit(
                f"--resume: no model-*.pt checkpoints found in {RESULTS_DIR}/. "
                "Restore them first, or drop --resume to start from scratch."
            )
        milestone = int(checkpoints[-1].stem.split("-")[-1])
        trainer.load(milestone)
        resume_step = trainer.step
        trainer.accelerator.print(f"Resumed from {checkpoints[-1]} (step {trainer.step})")

    # -----------------------------------------------------------------------
    # Log run config
    # -----------------------------------------------------------------------

    run_config = {
        "run_name": run_name,
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "results_dir": str(RESULTS_DIR),
        "data_dir": str(DATA_DIR),
        "ptsrc_mJy": PTSRC,
        "resolution": RES,
        "n_train": len(training_images),
        "n_augmented": len(augmented_images),
        "train_steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "save_every": args.save_every,
        "num_samples": args.num_samples,
        "resumed_from_step": resume_step,
        "wandb": use_wandb,
    }
    with open(RESULTS_DIR / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"\nRun '{run_name}'  →  {RESULTS_DIR}/")
    print(json.dumps(run_config, indent=2))
    print()

    # -----------------------------------------------------------------------
    # WandB
    # -----------------------------------------------------------------------

    if use_wandb and trainer.accelerator.is_main_process:
        import wandb

        wandb.init(
            project="cmb_foregrounds_diffusion",
            name=run_name,
            id=run_name,
            resume="allow",
            config=run_config,
        )

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------

    trainer.train()

    if use_wandb and trainer.accelerator.is_main_process:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
