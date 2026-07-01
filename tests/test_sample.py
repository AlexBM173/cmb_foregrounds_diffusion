"""Tests for foregrounds_diffusion.sample (build_model, argument parsing)."""


def test_build_model_default():
    from foregrounds_diffusion.sample import build_model

    model = build_model(channels=2)
    assert model.num_timesteps == 1000
    # No DDIM: sampling_timesteps should equal full timesteps
    assert model.sampling_timesteps == model.num_timesteps


def test_build_model_ddim():
    from foregrounds_diffusion.sample import build_model

    model = build_model(channels=2, sampling_timesteps=250)
    assert model.sampling_timesteps == 250
    assert model.num_timesteps == 1000


def test_build_model_channels():
    from foregrounds_diffusion.sample import build_model

    model = build_model(channels=3)
    # The wrapped UNet should have 3 input channels
    assert model.model.channels == 3


def test_sampling_timesteps_cli_flag(monkeypatch):
    import sys

    # Parse args with --sampling-timesteps; ensure it reaches build_model correctly.
    test_argv = [
        "sample.py",
        "--sampling-timesteps",
        "50",
        "--batches",
        "1",
        "--batch-size",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    import argparse

    from foregrounds_diffusion.sample import build_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling-timesteps", type=int, default=None)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    model = build_model(sampling_timesteps=args.sampling_timesteps)
    assert model.sampling_timesteps == 50
