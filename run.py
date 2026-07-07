#!/usr/bin/env python
# coding: utf-8
"""Config-driven pipeline runner for the CIB+tSZ diffusion project.

One YAML file (see ``config/default.yaml``) drives every stage::

    python run.py preprocess --config config/my_run.yaml
    python run.py train      --config config/my_run.yaml
    python run.py sample     --config config/my_run.yaml
    python run.py evaluate   --config config/my_run.yaml
    python run.py report     --config config/my_run.yaml
    python run.py all        --config config/my_run.yaml

Every stage validates the config, creates ``runs/<run_name>/`` with a config
copy, its SHA256 hash, and the git commit, then runs with all outputs inside
that directory. Multi-GPU use is transparent: launch any stage with
``accelerate launch run.py <stage> --config <yaml>``.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from config import ConfigError, PipelineConfig, load_config
from pipeline.rundir import RunDir, create_run_dir, git_commit_hash, sha256_of_file

# Architecture and optimiser settings currently fixed in pipeline/train.py and
# foregrounds_diffusion/sample.py. Threading them through from the config is a
# v0.2 item; until then a config that asks for anything else must fail loudly
# rather than silently train a different model than it describes.
_FIXED_SETTINGS = {
    # model.dim is honoured (threaded to train.py/sample.py via --dim).
    "model.dim_mults": (lambda c: list(c.model.dim_mults), [1, 2, 4, 8]),
    # model.channels is honoured (2 or 4); enforced in pipeline/train.py.
    "model.flash_attn": (lambda c: c.model.flash_attn, True),
    "model.timesteps": (lambda c: c.model.timesteps, 1000),
    "model.noise_schedule": (lambda c: c.model.noise_schedule, "sigmoid"),
    "model.objective": (lambda c: c.model.objective, "pred_v"),
    "model.auto_normalize": (lambda c: c.model.auto_normalize, False),
    "training.gradient_accumulate_every": (lambda c: c.training.gradient_accumulate_every, 2),
    "training.ema_decay": (lambda c: c.training.ema_decay, 0.995),
    "training.mixed_precision": (lambda c: c.training.mixed_precision, "bf16"),
    "training.lr_scheduler": (lambda c: c.training.lr_scheduler, "none"),
    "training.warmup_steps": (lambda c: c.training.warmup_steps, 0),
    "preprocessing.normalisation": (lambda c: c.preprocessing.normalisation, "zscore"),
}


def _check_fixed_settings(cfg: PipelineConfig) -> None:
    """Reject configs that request settings the scripts do not yet honour.

    Parameters
    ----------
    cfg : PipelineConfig
        Validated configuration.

    Raises
    ------
    ConfigError
        Listing every field whose value differs from the current fixed
        implementation, so the config never claims settings the run
        would not actually use.
    """
    mismatches = [
        f"  {name}: config says {getter(cfg)!r}, implementation is fixed at {expected!r}"
        for name, (getter, expected) in _FIXED_SETTINGS.items()
        if getter(cfg) != expected
    ]
    if mismatches:
        raise ConfigError(
            "these fields are fixed in the current implementation (configurable in v0.2):\n"
            + "\n".join(mismatches)
        )


def _patch_files(cfg: PipelineConfig) -> list[Path]:
    """Return the expected training patch file paths for all model channels.

    Parameters
    ----------
    cfg : PipelineConfig
        Validated configuration.

    Returns
    -------
    list of Path
        Paths of the per-channel ``.npy`` patch files (CIB, tSZ, then kSZ,
        kappa for a 4-channel run) inside the resolved patches directory,
        following the notebook 03 / nb03b naming convention. kappa has no
        frequency tag — lensing convergence is achromatic.
    """
    d = cfg.resolve_patches_dir()
    res, ptsrc = cfg.data.res, int(cfg.preprocessing.point_source_mjy)
    templates = [
        "CIB_map_150GHz_{res}_st6_zscore_{ptsrc}mJy_lp.npy",
        "tSZ3_map_150GHz_{res}_st6_zscore_{ptsrc}mJy_lp.npy",
        "kSZ_map_150GHz_{res}_st6_zscore_{ptsrc}mJy_lp.npy",
        "kappa_map_{res}_st6_zscore_{ptsrc}mJy_lp.npy",
    ]
    return [d / t.format(res=res, ptsrc=ptsrc) for t in templates[: cfg.model.channels]]


def _export_wandb_env(cfg: PipelineConfig, run: RunDir) -> None:
    """Export WandB project/entity/tags via environment variables.

    ``wandb.init`` reads ``WANDB_ENTITY`` and ``WANDB_TAGS`` from the
    environment, which lets the stage scripts stay wandb-optional while the
    runner attaches provenance (git commit, config hash) to every run.

    Parameters
    ----------
    cfg : PipelineConfig
        Validated configuration.
    run : RunDir
        Run directory (source of the config hash).
    """
    if not cfg.wandb.enabled:
        return
    os.environ["WANDB_PROJECT"] = cfg.wandb.project
    if cfg.wandb.entity:
        os.environ["WANDB_ENTITY"] = cfg.wandb.entity
    tags = list(cfg.wandb.tags)
    tags.append(f"commit-{git_commit_hash()[:12]}")
    config_copy = run.root / "config.yaml"
    if config_copy.exists():
        tags.append(f"config-{sha256_of_file(config_copy)[:12]}")
    os.environ["WANDB_TAGS"] = ",".join(tags)


def _banner(stage: str, cfg: PipelineConfig, run: RunDir) -> None:
    """Print a one-look summary of what is about to run."""
    print(f"[run.py {stage}] run_name={cfg.run_name}  run_dir={run.root}")
    print(f"[run.py {stage}] git={git_commit_hash()[:12]}  config={cfg.source}")


def stage_preprocess(cfg: PipelineConfig, run: RunDir, dry_run: bool = False) -> None:
    """Check for training-ready patches; direct to notebooks 01–03 if absent.

    The script refactor of the preprocessing notebooks is scheduled (Tier 2);
    until it lands, this stage verifies inputs and tells the user exactly
    what to run, instead of pretending to have preprocessed anything.

    Parameters
    ----------
    cfg : PipelineConfig
        Validated configuration.
    run : RunDir
        Run directory.
    dry_run : bool
        Print what would happen without touching anything.
    """
    _banner("preprocess", cfg, run)
    patch_files = _patch_files(cfg)
    if all(p.is_file() for p in patch_files):
        print(
            f"[run.py preprocess] training patches present in {patch_files[0].parent} "
            "— nothing to do"
        )
        return
    missing = ", ".join(p.name for p in patch_files if not p.is_file())
    raise SystemExit(
        f"[run.py preprocess] missing training patches in {patch_files[0].parent}: {missing}.\n"
        "Scripted preprocessing is not implemented yet — run the tutorial notebooks\n"
        "01_halo_catalogue → 02_masking → 03_patch_extraction (docs/tutorials/), and for\n"
        "kSZ/kappa the scripts/vm_preprocessing/nb02b + nb03b chain, then set\n"
        "data.patches_dir in the config to that output directory."
    )


def stage_train(cfg: PipelineConfig, run: RunDir, dry_run: bool = False) -> None:
    """Train the diffusion model with outputs in ``<run_dir>/checkpoints``.

    Parameters
    ----------
    cfg : PipelineConfig
        Validated configuration.
    run : RunDir
        Run directory.
    dry_run : bool
        Print the equivalent ``train.py`` invocation and return.
    """
    _check_fixed_settings(cfg)
    _banner("train", cfg, run)
    patch_files = _patch_files(cfg)
    missing = [p for p in patch_files if not p.is_file()]
    if missing:
        names = ", ".join(p.name for p in missing)
        raise SystemExit(
            f"[run.py train] training patches not found in {patch_files[0].parent}: {names} — "
            "run `run.py preprocess` first (or set data.patches_dir)."
        )

    t = cfg.training
    argv = [
        "--run-name",
        cfg.run_name,
        "--steps",
        str(t.train_num_steps),
        "--batch-size",
        str(t.batch_size),
        "--lr",
        str(t.lr),
        "--save-every",
        str(t.save_and_sample_every),
        "--num-samples",
        str(t.milestone_num_samples),
        "--ptsrc",
        str(int(cfg.preprocessing.point_source_mjy)),
        "--res",
        str(cfg.data.res),
        "--channels",
        str(cfg.model.channels),
        "--dim",
        str(cfg.model.dim),
        "--data-dir",
        str(patch_files[0].parent),
        "--results-dir",
        str(run.checkpoints),
    ]
    if t.resume_from_checkpoint:
        argv.append("--resume")
    if cfg.wandb.enabled:
        argv.append("--wandb")

    if dry_run:
        print("[run.py train] would run: train.py " + " ".join(argv))
        return
    _export_wandb_env(cfg, run)
    from pipeline.train import main as train_main

    train_main(argv)


def stage_sample(
    cfg: PipelineConfig, run: RunDir, checkpoint: str | None = None, dry_run: bool = False
) -> None:
    """Generate samples from the latest (or given) checkpoint into ``samples/``.

    Parameters
    ----------
    cfg : PipelineConfig
        Validated configuration.
    run : RunDir
        Run directory.
    checkpoint : str or None
        Explicit checkpoint path; ``None`` uses the latest
        ``model-*.pt`` in ``<run_dir>/checkpoints``.
    dry_run : bool
        Print the equivalent ``sample.py`` invocation and return.
    """
    _check_fixed_settings(cfg)
    _banner("sample", cfg, run)
    if checkpoint is None:
        candidates = sorted(
            run.checkpoints.glob("model-*.pt"), key=lambda p: int(p.stem.split("-")[-1])
        )
        if not candidates:
            raise SystemExit(
                f"[run.py sample] no model-*.pt in {run.checkpoints} — train first, "
                "or pass --checkpoint PATH."
            )
        checkpoint = str(candidates[-1])

    s = cfg.sampling
    batches = math.ceil(s.num_samples / s.batch_size)
    output = run.samples / f"samples_{cfg.run_name}.npy"
    argv = [
        "--checkpoint",
        checkpoint,
        "--batches",
        str(batches),
        "--batch-size",
        str(s.batch_size),
        "--output",
        str(output),
        "--channels",
        str(cfg.model.channels),
        "--dim",
        str(cfg.model.dim),
    ]
    if s.ddim_steps is not None:
        argv += ["--sampling-timesteps", str(s.ddim_steps)]
    if s.compile:
        argv.append("--compile")
    if s.rescale_cib is not None:
        argv += ["--rescale-cib", str(s.rescale_cib)]
    if s.rescale_tsz is not None:
        argv += ["--rescale-tsz", str(s.rescale_tsz)]
    if cfg.wandb.enabled:
        argv.append("--wandb")

    if dry_run:
        print("[run.py sample] would run: sample.py " + " ".join(argv))
        return
    _export_wandb_env(cfg, run)
    from foregrounds_diffusion.sample import main as sample_main

    sample_main(argv)


def stage_evaluate(cfg: PipelineConfig, run: RunDir, dry_run: bool = False) -> None:
    """Compute the configured statistics (implementation arriving in Tier 2)."""
    _banner("evaluate", cfg, run)
    try:
        from pipeline.evaluate import main as evaluate_main
    except ImportError:
        raise SystemExit(
            "[run.py evaluate] pipeline/evaluate.py is not implemented yet (Tier 2, "
            "in progress) — run the statistics notebooks 06–09 (docs/tutorials/) meanwhile."
        ) from None
    evaluate_main(cfg, run, dry_run=dry_run)


def stage_report(cfg: PipelineConfig, run: RunDir, dry_run: bool = False) -> None:
    """Generate ``report.md`` for the run (implementation arriving in Tier 2)."""
    _banner("report", cfg, run)
    try:
        from pipeline.report import main as report_main
    except ImportError:
        raise SystemExit(
            "[run.py report] pipeline/report.py is not implemented yet (Tier 2, in progress)."
        ) from None
    report_main(cfg, run, dry_run=dry_run)


def stage_slurm(cfg: PipelineConfig, run: RunDir, stage: str, submit: bool) -> None:
    """Generate a SLURM job script (deferred to v0.2)."""
    raise SystemExit(
        "[run.py slurm] job-script generation is deferred to v0.2: this project migrated "
        "off the HPC (no cluster access), so generated scripts could not be tested. The "
        "slurm config section is preserved for future cluster users; see the reference "
        "scripts train_slurm.sh / sample_slurm.sh."
    )


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline CLI.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        Process exit code (0 = success).
    """
    parser = argparse.ArgumentParser(
        description="Config-driven pipeline runner (see config/default.yaml).",
    )
    sub = parser.add_subparsers(dest="stage", required=True)
    stages = ["preprocess", "train", "sample", "evaluate", "report", "all", "slurm"]
    for name in stages:
        sp = sub.add_parser(name, help=f"run the {name} stage")
        sp.add_argument("--config", required=True, help="Path to the pipeline YAML config")
        sp.add_argument(
            "--dry-run",
            action="store_true",
            default=False,
            help="Print what would run without executing it",
        )
        if name == "sample":
            sp.add_argument(
                "--checkpoint",
                default=None,
                help="Checkpoint to sample from (default: latest in <run_dir>/checkpoints)",
            )
        if name == "slurm":
            sp.add_argument(
                "--stage",
                dest="slurm_stage",
                default="train",
                help="Pipeline stage the job script should run",
            )
            sp.add_argument(
                "--submit", action="store_true", default=False, help="sbatch the generated script"
            )
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
        run = create_run_dir(cfg)
        if args.stage == "preprocess":
            stage_preprocess(cfg, run, dry_run=args.dry_run)
        elif args.stage == "train":
            stage_train(cfg, run, dry_run=args.dry_run)
        elif args.stage == "sample":
            stage_sample(cfg, run, checkpoint=args.checkpoint, dry_run=args.dry_run)
        elif args.stage == "evaluate":
            stage_evaluate(cfg, run, dry_run=args.dry_run)
        elif args.stage == "report":
            stage_report(cfg, run, dry_run=args.dry_run)
        elif args.stage == "slurm":
            stage_slurm(cfg, run, stage=args.slurm_stage, submit=args.submit)
        elif args.stage == "all":
            stage_preprocess(cfg, run, dry_run=args.dry_run)
            stage_train(cfg, run, dry_run=args.dry_run)
            stage_sample(cfg, run, dry_run=args.dry_run)
            stage_evaluate(cfg, run, dry_run=args.dry_run)
            stage_report(cfg, run, dry_run=args.dry_run)
    except ConfigError as exc:
        print(f"[run.py] invalid config: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
