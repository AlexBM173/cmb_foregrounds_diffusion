"""Tests for the YAML config system (config/) and run-directory provenance."""

from pathlib import Path

import pytest
import yaml

from config import ConfigError, load_config
from pipeline.rundir import create_run_dir, sha256_of_file

DEFAULT_YAML = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def _deep_update(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into *base* and return *base*."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def write_config(tmp_path: Path, overrides: dict | None = None, name: str = "cfg.yaml") -> Path:
    """Write default.yaml with *overrides* applied to a temp file and return its path."""
    raw = yaml.safe_load(DEFAULT_YAML.read_text())
    if overrides:
        _deep_update(raw, overrides)
    path = tmp_path / name
    path.write_text(yaml.safe_dump(raw))
    return path


def test_default_config_valid():
    cfg = load_config(DEFAULT_YAML)
    assert cfg.run_name == "example_run"
    assert cfg.model.dim_mults == [1, 2, 4, 8]
    assert cfg.model.auto_normalize is False
    assert "power_spectrum" in cfg.evaluation.statistics
    assert cfg.evaluation.params["tsz_stacking"]["cutout_pix"] == 40


def test_missing_file():
    with pytest.raises(ConfigError, match="not found"):
        load_config("nonexistent.yaml")


def test_unknown_top_level_key(tmp_path):
    path = write_config(tmp_path, {"trainnig": {}})
    with pytest.raises(ConfigError, match="unknown key.*trainnig"):
        load_config(path)


def test_unknown_section_key(tmp_path):
    path = write_config(tmp_path, {"training": {"batchsize": 8}})
    with pytest.raises(ConfigError, match="training.*batchsize"):
        load_config(path)


def test_split_must_sum_to_one(tmp_path):
    path = write_config(tmp_path, {"data": {"train_size": 0.9}})
    with pytest.raises(ConfigError, match="sum to 1"):
        load_config(path)


def test_bad_normalisation(tmp_path):
    path = write_config(tmp_path, {"preprocessing": {"normalisation": "robust"}})
    with pytest.raises(ConfigError, match="zscore|minmax"):
        load_config(path)


def test_bad_run_name(tmp_path):
    path = write_config(tmp_path, {"run_name": "bad/name"})
    with pytest.raises(ConfigError, match="run_name"):
        load_config(path)


def test_unknown_statistic_in_list(tmp_path):
    path = write_config(tmp_path, {"evaluation": {"statistics": ["power_spectrum", "bispectrum"]}})
    with pytest.raises(ConfigError, match="bispectrum"):
        load_config(path)


def test_unknown_statistic_section(tmp_path):
    path = write_config(tmp_path, {"evaluation": {"bispectrum": {"n_maps": 10}}})
    with pytest.raises(ConfigError, match="bispectrum"):
        load_config(path)


def test_unknown_statistic_param(tmp_path):
    path = write_config(tmp_path, {"evaluation": {"power_spectrum": {"nmaps": 10}}})
    with pytest.raises(ConfigError, match="power_spectrum.*nmaps"):
        load_config(path)


def test_milestone_samples_must_be_square(tmp_path):
    path = write_config(tmp_path, {"training": {"milestone_num_samples": 10}})
    with pytest.raises(ConfigError, match="perfect square"):
        load_config(path)
    assert (
        load_config(
            write_config(tmp_path, {"training": {"milestone_num_samples": 16}}, name="ok.yaml")
        ).training.milestone_num_samples
        == 16
    )


def test_milestone_samples_zero_allowed(tmp_path):
    path = write_config(tmp_path, {"training": {"milestone_num_samples": 0}})
    assert load_config(path).training.milestone_num_samples == 0


def test_ddim_exceeds_timesteps(tmp_path):
    path = write_config(tmp_path, {"sampling": {"ddim_steps": 2000}})
    with pytest.raises(ConfigError, match="ddim_steps"):
        load_config(path)


def test_output_format_h5_rejected(tmp_path):
    path = write_config(tmp_path, {"sampling": {"output_format": "h5"}})
    with pytest.raises(ConfigError, match="npy"):
        load_config(path)


def test_rundir_provenance(tmp_path):
    path = write_config(
        tmp_path, {"run_name": "prov_test", "output": {"base_dir": str(tmp_path / "runs")}}
    )
    cfg = load_config(path)
    run = create_run_dir(cfg)

    for sub in ("data/patches", "data/masks", "checkpoints", "samples", "stats", "plots", "logs"):
        assert (run.root / sub).is_dir()
    assert (run.root / "config.yaml").read_text() == path.read_text()
    stored_hash = (run.root / "config_hash.txt").read_text().strip()
    assert stored_hash == sha256_of_file(run.root / "config.yaml")
    assert (run.root / "git_commit.txt").read_text().strip()

    # Idempotent for the identical config
    create_run_dir(cfg)


def test_rundir_rejects_conflicting_config(tmp_path):
    base = {"run_name": "conflict_test", "output": {"base_dir": str(tmp_path / "runs")}}
    cfg_a = load_config(write_config(tmp_path, base, name="a.yaml"))
    create_run_dir(cfg_a)

    conflicting = dict(base)
    conflicting["training"] = {"lr": 5.0e-5}
    cfg_b = load_config(write_config(tmp_path, conflicting, name="b.yaml"))
    with pytest.raises(ConfigError, match="different content"):
        create_run_dir(cfg_b)
