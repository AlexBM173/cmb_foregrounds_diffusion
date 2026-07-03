#!/usr/bin/env python
# coding: utf-8
"""Run directory creation and provenance stamping.

Every pipeline invocation works inside ``<output.base_dir>/<run_name>/``::

    runs/<run_name>/
      config.yaml          # copy of the config used
      config_hash.txt      # SHA256 of config.yaml for reproducibility
      git_commit.txt       # git commit hash at the time of the run
      data/
        patches/           # extracted .npy patch files
        masks/             # saved mask files
      checkpoints/         # model-N.pt files
      samples/             # generated sample arrays
      stats/               # per-statistic result files
      plots/               # figures
      logs/                # job logs
      report.md            # auto-generated run summary (run.py report)

The directory tree is idempotent: re-running a stage reuses the existing
structure, and the config copy is only compared — never silently replaced —
if one is already present with different content.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from config import ConfigError, PipelineConfig

SUBDIRS = (
    "data/patches",
    "data/masks",
    "checkpoints",
    "samples",
    "stats",
    "plots",
    "logs",
)


@dataclass(frozen=True)
class RunDir:
    """Resolved paths inside a run directory.

    Attributes
    ----------
    root : Path
        The run directory itself, ``<base_dir>/<run_name>``.
    """

    root: Path

    @property
    def data(self) -> Path:
        """Directory for preprocessing outputs (norm params, split indices)."""
        return self.root / "data"

    @property
    def patches(self) -> Path:
        """Directory for extracted ``.npy`` patch files."""
        return self.root / "data" / "patches"

    @property
    def masks(self) -> Path:
        """Directory for saved mask files."""
        return self.root / "data" / "masks"

    @property
    def checkpoints(self) -> Path:
        """Directory for ``model-N.pt`` training checkpoints."""
        return self.root / "checkpoints"

    @property
    def samples(self) -> Path:
        """Directory for generated sample arrays."""
        return self.root / "samples"

    @property
    def stats(self) -> Path:
        """Directory for per-statistic result files."""
        return self.root / "stats"

    @property
    def plots(self) -> Path:
        """Directory for figures."""
        return self.root / "plots"

    @property
    def logs(self) -> Path:
        """Directory for job logs."""
        return self.root / "logs"

    @property
    def report(self) -> Path:
        """Path of the auto-generated markdown report."""
        return self.root / "report.md"


def sha256_of_file(path: Path) -> str:
    """Return the SHA256 hex digest of a file's bytes.

    Parameters
    ----------
    path : Path
        File to hash.

    Returns
    -------
    str
        64-character lowercase hex digest.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_commit_hash(repo_root: Path | None = None) -> str:
    """Return the current git commit hash, or ``"unknown"`` outside a repo.

    Parameters
    ----------
    repo_root : Path or None
        Directory to run git in; ``None`` uses the current working directory.

    Returns
    -------
    str
        Full commit hash, with a ``-dirty`` suffix if the working tree has
        uncommitted changes, or ``"unknown"`` if git is unavailable.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"{commit}-dirty" if status else commit
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def create_run_dir(cfg: PipelineConfig) -> RunDir:
    """Create (or reuse) the run directory for *cfg* and stamp provenance.

    Copies the source config into the run directory, writes its SHA256 to
    ``config_hash.txt`` and the current git commit to ``git_commit.txt``.
    Idempotent: an existing identical config copy is left untouched.

    Parameters
    ----------
    cfg : PipelineConfig
        Validated configuration (``cfg.source`` must point at the YAML file).

    Returns
    -------
    RunDir
        Resolved paths for the run.

    Raises
    ------
    ConfigError
        If the run directory already holds a *different* config — reusing a
        run name with changed settings would corrupt provenance; pick a new
        ``run_name`` instead.
    """
    run = RunDir(root=cfg.run_dir())
    for sub in SUBDIRS:
        (run.root / sub).mkdir(parents=True, exist_ok=True)

    config_copy = run.root / "config.yaml"
    if cfg.source is not None:
        source_text = Path(cfg.source).read_text()
        if config_copy.exists() and config_copy.read_text() != source_text:
            raise ConfigError(
                f"{config_copy} already exists with different content — run "
                f"{cfg.run_name!r} was created from another config. Use a new "
                f"run_name rather than mixing configurations in one run."
            )
        config_copy.write_text(source_text)
        (run.root / "config_hash.txt").write_text(sha256_of_file(config_copy) + "\n")

    (run.root / "git_commit.txt").write_text(git_commit_hash() + "\n")
    return run
