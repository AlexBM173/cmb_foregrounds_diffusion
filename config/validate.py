#!/usr/bin/env python
# coding: utf-8
"""Load and validate pipeline configuration YAML files.

The configuration schema is documented field-by-field in
``config/default.yaml``. This module turns a YAML file into a typed
:class:`PipelineConfig` object, raising :class:`ConfigError` with a clear,
path-qualified message for every unknown key, missing field, or incompatible
setting.

Usage
-----
As a library::

    from config import load_config
    cfg = load_config("config/default.yaml")

From the command line::

    python config/validate.py config/default.yaml
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Statistics understood by `run.py evaluate`, and the keys each accepts.
KNOWN_STATISTICS = {
    "power_spectrum": {"lmin", "lmax", "binsize", "n_maps"},
    "cross_spectrum": {"lmin", "lmax", "binsize", "n_maps"},
    "moments": {"n_bands", "lmin", "lmax", "n_maps", "noise_tiers"},
    "cross_moments": {"n_bands", "lmin", "lmax", "n_maps", "noise_tiers"},
    "minkowski_functionals": {"n_thresholds", "threshold_min", "threshold_max", "n_maps"},
    "minkowski_tensors": {
        "n_thresholds",
        "threshold_min",
        "threshold_max",
        "tensor_types",
        "n_maps",
    },
    "pixel_histograms": {"n_bins", "cib_range", "tsz_range", "smooth_sigma", "n_maps"},
    "peak_counts": {
        "smoothing_fwhm_arcmin",
        "threshold_min",
        "threshold_max",
        "n_thresholds",
        "n_maps",
    },
    "minima_counts": {
        "smoothing_fwhm_arcmin",
        "threshold_min",
        "threshold_max",
        "n_thresholds",
        "n_maps",
    },
    "scattering_transforms": {"J", "L", "n_maps", "covariance", "device"},
    "tsz_stacking": {"snr_bins", "cutout_pix", "n_maps"},
    # 4-channel extension: stack a cross-field on tSZ-SNR-selected clusters
    "kappa_on_tsz_stacking": {"snr_bins", "cutout_pix", "n_maps"},
    "ksz_stacking": {"snr_bins", "cutout_pix", "n_maps"},
}

_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class ConfigError(ValueError):
    """Raised when a configuration file is invalid, with a path-qualified message."""


def _require(condition: bool, message: str) -> None:
    """Raise :class:`ConfigError` with *message* unless *condition* holds.

    Parameters
    ----------
    condition : bool
        The invariant that must be true.
    message : str
        Error message, prefixed with the YAML path of the offending field.
    """
    if not condition:
        raise ConfigError(message)


def _check_keys(mapping: dict, allowed: set[str], section: str) -> None:
    """Reject unknown keys in a config section.

    Parameters
    ----------
    mapping : dict
        The parsed YAML section.
    allowed : set of str
        Keys the section accepts.
    section : str
        Dotted YAML path of the section, used in the error message.
    """
    unknown = set(mapping) - allowed
    _require(
        not unknown,
        f"{section}: unknown key(s) {sorted(unknown)} — allowed keys are {sorted(allowed)}",
    )


@dataclass
class OutputConfig:
    """Where run artefacts are written (``output`` section)."""

    base_dir: str = "runs"


@dataclass
class DataConfig:
    """Raw map paths and patch geometry (``data`` section)."""

    cib_map: str = "data/agora_len_mag_cibmap_act_150ghz.fits"
    tsz_map: str = "data/agora_ltszNG_bahamas80_bnd_unb_1.0e+12_1.0e+18_lensed.fits"
    halo_catalogue: str = "data/halo_catalogue/halo_catalogue_m500gt3e14.npz"
    frequency_ghz: int = 150
    nside_in: int = 8192
    nside_out: int = 2048
    patch_deg: float = 6.0
    step_deg: float = 6.0
    res: int = 256
    gal_cut_deg: float = 20.0
    pole_cut_deg: float = 6.0
    train_size: float = 0.8
    val_size: float = 0.1
    test_size: float = 0.1
    seed: int = 42
    patches_dir: str | None = None


@dataclass
class LowpassConfig:
    """Low-pass filter settings (``preprocessing.lowpass``)."""

    type: str = "sharp"
    ell_max: int = 7000


@dataclass
class ClusterMaskConfig:
    """Apodised cluster mask settings (``preprocessing.cluster_mask``)."""

    enabled: bool = True
    m500c_min: float = 3.0e14
    theta500_multiplier: float = 3.0


@dataclass
class PreprocessingConfig:
    """Masking, filtering, and normalisation settings (``preprocessing`` section)."""

    lowpass: LowpassConfig = field(default_factory=LowpassConfig)
    normalisation: str = "zscore"
    point_source_mjy: float = 2
    cluster_mask: ClusterMaskConfig = field(default_factory=ClusterMaskConfig)
    inpainting: str = "gaussian_noise"
    augmentation: bool = True


@dataclass
class ModelConfig:
    """U-Net / diffusion architecture (``model`` section)."""

    dim: int = 64
    dim_mults: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    channels: int = 2
    flash_attn: bool = True
    timesteps: int = 1000
    noise_schedule: str = "sigmoid"
    objective: str = "pred_v"
    auto_normalize: bool = False


@dataclass
class TrainingConfig:
    """Optimisation settings (``training`` section)."""

    batch_size: int = 16
    lr: float = 1.0e-4
    lr_scheduler: str = "none"
    warmup_steps: int = 0
    train_num_steps: int = 100000
    gradient_accumulate_every: int = 2
    ema_decay: float = 0.995
    mixed_precision: str = "bf16"
    save_and_sample_every: int = 5000
    milestone_num_samples: int = 25
    resume_from_checkpoint: bool = False
    num_gpus: int = 1


@dataclass
class SamplingConfig:
    """Sample generation settings (``sampling`` section)."""

    num_samples: int = 640
    batch_size: int = 16
    ddim_steps: int | None = 250
    output_format: str = "npy"
    compile: bool = False
    rescale_cib: float | None = None
    rescale_tsz: float | None = None


@dataclass
class EvaluationConfig:
    """Which statistics to compute and their parameters (``evaluation`` section)."""

    n_jobs: int = 8
    ilc_noise_file: str = "data/ilc/ilc_weights_residuals_agora_fg_model.npy"
    noise_seed: int = 42
    statistics: list[str] = field(
        default_factory=lambda: [
            "power_spectrum",
            "cross_spectrum",
            "moments",
            "cross_moments",
            "minkowski_functionals",
            "pixel_histograms",
            "tsz_stacking",
        ]
    )
    params: dict[str, dict] = field(default_factory=dict)


@dataclass
class WandbConfig:
    """Weights & Biases logging settings (``wandb`` section)."""

    enabled: bool = False
    project: str = "cmb_foregrounds_diffusion"
    entity: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class SlurmConfig:
    """Cluster job settings (``slurm`` section; job-script generation is v0.2)."""

    partition: str = "ampere"
    qos: str | None = None
    account: str | None = None
    num_gpus: int = 1
    mem: str = "64G"
    time: str = "12:00:00"
    mail_user: str | None = None


@dataclass
class PipelineConfig:
    """Fully validated pipeline configuration.

    Attributes
    ----------
    run_name : str
        Label for this run; artefacts go to ``<output.base_dir>/<run_name>/``.
    source : Path or None
        The YAML file this configuration was loaded from, if any.
    """

    run_name: str = "example_run"
    output: OutputConfig = field(default_factory=OutputConfig)
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    source: Path | None = None

    def run_dir(self) -> Path:
        """Return the run directory ``<output.base_dir>/<run_name>``."""
        return Path(self.output.base_dir) / self.run_name

    def resolve_patches_dir(self) -> Path:
        """Return the directory holding training-ready ``.npy`` patch files.

        ``data.patches_dir`` when set, else ``<run_dir>/data/patches``.
        """
        if self.data.patches_dir is not None:
            return Path(self.data.patches_dir)
        return self.run_dir() / "data" / "patches"


def _build_section(cls, mapping: dict | None, section: str):
    """Instantiate dataclass *cls* from a YAML *mapping*, rejecting unknown keys.

    Parameters
    ----------
    cls : type
        A flat dataclass (no nested dataclass fields).
    mapping : dict or None
        Parsed YAML for this section; ``None`` yields all defaults.
    section : str
        Dotted YAML path for error messages.

    Returns
    -------
    cls
        Populated dataclass instance.
    """
    mapping = mapping or {}
    _require(
        isinstance(mapping, dict), f"{section}: expected a mapping, got {type(mapping).__name__}"
    )
    allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    _check_keys(mapping, allowed, section)
    return cls(**mapping)


def _parse(raw: dict, source: Path | None) -> PipelineConfig:
    """Assemble a :class:`PipelineConfig` from parsed YAML, without value checks."""
    _require(isinstance(raw, dict), "config root: expected a mapping")
    top_allowed = {
        "run_name",
        "output",
        "data",
        "preprocessing",
        "model",
        "training",
        "sampling",
        "evaluation",
        "wandb",
        "slurm",
    }
    _check_keys(raw, top_allowed, "config root")

    pre_raw = dict(raw.get("preprocessing") or {})
    _require(isinstance(pre_raw, dict), "preprocessing: expected a mapping")
    lowpass = _build_section(LowpassConfig, pre_raw.pop("lowpass", None), "preprocessing.lowpass")
    cluster = _build_section(
        ClusterMaskConfig, pre_raw.pop("cluster_mask", None), "preprocessing.cluster_mask"
    )
    _check_keys(
        pre_raw,
        {"normalisation", "point_source_mjy", "inpainting", "augmentation"},
        "preprocessing",
    )
    preprocessing = PreprocessingConfig(lowpass=lowpass, cluster_mask=cluster, **pre_raw)

    eval_raw = dict(raw.get("evaluation") or {})
    _require(isinstance(eval_raw, dict), "evaluation: expected a mapping")
    n_jobs = eval_raw.pop("n_jobs", 8)
    ilc_noise_file = eval_raw.pop(
        "ilc_noise_file", "data/ilc/ilc_weights_residuals_agora_fg_model.npy"
    )
    noise_seed = eval_raw.pop("noise_seed", 42)
    statistics = eval_raw.pop("statistics", None)
    params = {}
    for name, stat_cfg in eval_raw.items():
        _require(
            name in KNOWN_STATISTICS,
            f"evaluation.{name}: unknown statistic — known statistics are "
            f"{sorted(KNOWN_STATISTICS)}",
        )
        stat_cfg = stat_cfg or {}
        _check_keys(stat_cfg, KNOWN_STATISTICS[name], f"evaluation.{name}")
        params[name] = dict(stat_cfg)
    evaluation = EvaluationConfig(
        n_jobs=n_jobs, ilc_noise_file=ilc_noise_file, noise_seed=noise_seed, params=params
    )
    if statistics is not None:
        evaluation.statistics = list(statistics)

    return PipelineConfig(
        run_name=raw.get("run_name", "example_run"),
        output=_build_section(OutputConfig, raw.get("output"), "output"),
        data=_build_section(DataConfig, raw.get("data"), "data"),
        preprocessing=preprocessing,
        model=_build_section(ModelConfig, raw.get("model"), "model"),
        training=_build_section(TrainingConfig, raw.get("training"), "training"),
        sampling=_build_section(SamplingConfig, raw.get("sampling"), "sampling"),
        evaluation=evaluation,
        wandb=_build_section(WandbConfig, raw.get("wandb"), "wandb"),
        slurm=_build_section(SlurmConfig, raw.get("slurm"), "slurm"),
        source=source,
    )


def validate(cfg: PipelineConfig) -> PipelineConfig:
    """Check all field values and cross-field constraints of *cfg*.

    Parameters
    ----------
    cfg : PipelineConfig
        Parsed configuration to check.

    Returns
    -------
    PipelineConfig
        The same object, if valid.

    Raises
    ------
    ConfigError
        On the first invalid or incompatible setting found.
    """
    _require(
        isinstance(cfg.run_name, str) and _RUN_NAME_RE.match(cfg.run_name) is not None,
        f"run_name: must match [A-Za-z0-9._-]+ (got {cfg.run_name!r})",
    )
    _require(bool(cfg.output.base_dir), "output.base_dir: must be non-empty")

    d = cfg.data
    split_sum = d.train_size + d.val_size + d.test_size
    _require(
        abs(split_sum - 1.0) < 1e-6,
        f"data: train/val/test sizes must sum to 1 (got {split_sum:g})",
    )
    for name, val in [
        ("train_size", d.train_size),
        ("val_size", d.val_size),
        ("test_size", d.test_size),
    ]:
        _require(0.0 <= val <= 1.0, f"data.{name}: must be in [0, 1] (got {val})")
    for name, val in [("nside_in", d.nside_in), ("nside_out", d.nside_out)]:
        _require(
            isinstance(val, int) and val > 0 and (val & (val - 1)) == 0,
            f"data.{name}: must be a positive power of two (got {val})",
        )
    _require(d.nside_out <= d.nside_in, "data.nside_out: cannot exceed data.nside_in")
    _require(d.res > 0, f"data.res: must be positive (got {d.res})")
    _require(d.patch_deg > 0, f"data.patch_deg: must be positive (got {d.patch_deg})")
    _require(d.step_deg > 0, f"data.step_deg: must be positive (got {d.step_deg})")

    p = cfg.preprocessing
    _require(
        p.lowpass.type in {"sharp", "cosine", "wiener"},
        f"preprocessing.lowpass.type: must be sharp|cosine|wiener (got {p.lowpass.type!r})",
    )
    _require(p.lowpass.ell_max > 0, "preprocessing.lowpass.ell_max: must be positive")
    _require(
        p.normalisation in {"zscore", "minmax"},
        f"preprocessing.normalisation: must be zscore|minmax (got {p.normalisation!r})",
    )
    _require(
        p.inpainting in {"gaussian_noise"},
        f"preprocessing.inpainting: must be gaussian_noise (got {p.inpainting!r})",
    )
    _require(p.point_source_mjy > 0, "preprocessing.point_source_mjy: must be positive")
    if isinstance(p.cluster_mask.m500c_min, str):
        # YAML 1.1 parses exponents without a sign (e.g. 3.0e14) as strings.
        try:
            p.cluster_mask.m500c_min = float(p.cluster_mask.m500c_min)
        except ValueError:
            raise ConfigError(
                f"preprocessing.cluster_mask.m500c_min: not a number "
                f"(got {p.cluster_mask.m500c_min!r}; write exponents as 3.0e+14)"
            ) from None
    _require(p.cluster_mask.m500c_min > 0, "preprocessing.cluster_mask.m500c_min: must be positive")
    _require(
        p.cluster_mask.theta500_multiplier > 0,
        "preprocessing.cluster_mask.theta500_multiplier: must be positive",
    )

    m = cfg.model
    _require(m.dim > 0, f"model.dim: must be positive (got {m.dim})")
    _require(
        isinstance(m.dim_mults, list)
        and m.dim_mults
        and all(isinstance(x, int) and x > 0 for x in m.dim_mults),
        f"model.dim_mults: must be a non-empty list of positive ints (got {m.dim_mults!r})",
    )
    _require(m.channels > 0, f"model.channels: must be positive (got {m.channels})")
    _require(m.timesteps > 0, f"model.timesteps: must be positive (got {m.timesteps})")
    _require(
        m.noise_schedule in {"linear", "cosine", "sigmoid"},
        f"model.noise_schedule: must be linear|cosine|sigmoid (got {m.noise_schedule!r})",
    )
    _require(
        m.objective in {"pred_v", "pred_noise", "pred_x0"},
        f"model.objective: must be pred_v|pred_noise|pred_x0 (got {m.objective!r})",
    )

    t = cfg.training
    _require(t.batch_size >= 1, f"training.batch_size: must be ≥ 1 (got {t.batch_size})")
    _require(t.lr > 0, f"training.lr: must be positive (got {t.lr})")
    _require(
        t.lr_scheduler in {"none", "cosine"},
        f"training.lr_scheduler: must be none|cosine (got {t.lr_scheduler!r})",
    )
    _require(t.warmup_steps >= 0, "training.warmup_steps: must be ≥ 0")
    _require(t.train_num_steps >= 1, "training.train_num_steps: must be ≥ 1")
    _require(t.gradient_accumulate_every >= 1, "training.gradient_accumulate_every: must be ≥ 1")
    _require(0.0 < t.ema_decay < 1.0, f"training.ema_decay: must be in (0, 1) (got {t.ema_decay})")
    _require(
        t.mixed_precision in {"no", "fp16", "bf16"},
        f"training.mixed_precision: must be no|fp16|bf16 (got {t.mixed_precision!r})",
    )
    _require(t.save_and_sample_every >= 1, "training.save_and_sample_every: must be ≥ 1")
    _require(
        t.milestone_num_samples >= 0
        and math.isqrt(max(t.milestone_num_samples, 0)) ** 2 == t.milestone_num_samples,
        f"training.milestone_num_samples: must be a perfect square or 0 "
        f"(got {t.milestone_num_samples}) — the diffusion Trainer arranges "
        f"milestone samples in a square grid",
    )
    _require(t.num_gpus >= 1, "training.num_gpus: must be ≥ 1")

    s = cfg.sampling
    _require(s.num_samples >= 1, f"sampling.num_samples: must be ≥ 1 (got {s.num_samples})")
    _require(s.batch_size >= 1, f"sampling.batch_size: must be ≥ 1 (got {s.batch_size})")
    if s.ddim_steps is not None:
        _require(
            1 <= s.ddim_steps <= m.timesteps,
            f"sampling.ddim_steps: must be in [1, model.timesteps={m.timesteps}] "
            f"(got {s.ddim_steps})",
        )
    _require(
        s.output_format == "npy",
        f"sampling.output_format: only 'npy' is supported (got {s.output_format!r}; "
        f"fits/h5 are planned for v0.2)",
    )
    for name, val in [("rescale_cib", s.rescale_cib), ("rescale_tsz", s.rescale_tsz)]:
        if val is not None:
            _require(val > 0, f"sampling.{name}: must be positive when set (got {val})")

    e = cfg.evaluation
    _require(e.n_jobs != 0, "evaluation.n_jobs: must be a positive worker count or -1 (all cores)")
    for name in e.statistics:
        _require(
            name in KNOWN_STATISTICS,
            f"evaluation.statistics: unknown statistic {name!r} — known statistics are "
            f"{sorted(KNOWN_STATISTICS)}",
        )
    for name, stat_cfg in e.params.items():
        n_maps = stat_cfg.get("n_maps")
        if n_maps is not None:
            _require(n_maps >= 1, f"evaluation.{name}.n_maps: must be ≥ 1 (got {n_maps})")

    w = cfg.wandb
    if w.enabled:
        _require(bool(w.project), "wandb.project: must be non-empty when wandb.enabled is true")

    return cfg


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate a pipeline configuration from a YAML file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    PipelineConfig
        Validated configuration object.

    Raises
    ------
    ConfigError
        If the file does not exist, is not valid YAML, or fails validation.
    """
    path = Path(path)
    _require(path.is_file(), f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML — {exc}") from exc
    return validate(_parse(raw or {}, source=path))


def main(argv: list[str] | None = None) -> int:
    """Validate a configuration file from the command line.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        Process exit code (0 = valid).
    """
    parser = argparse.ArgumentParser(description="Validate a pipeline configuration YAML.")
    parser.add_argument("config", help="Path to the YAML configuration file")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"INVALID: {exc}")
        return 1
    print(f"OK: {args.config} is valid (run_name={cfg.run_name!r}, run_dir={cfg.run_dir()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
