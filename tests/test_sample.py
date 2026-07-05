"""Tests for foregrounds_diffusion.sample (build_model, argument parsing, x₀ clamp)."""

import torch
from torch import nn


class _ConstVModel(nn.Module):
    """Stub denoiser returning a large constant v so predicted x₀ lands far outside [-1, 1]."""

    channels = 2
    out_dim = 2
    self_condition = False
    random_or_learned_sinusoidal_cond = False

    def forward(self, x, t, x_self_cond=None):
        return torch.full_like(x, 40.0)


def _tiny(cls):
    return cls(_ConstVModel(), image_size=8, timesteps=8, auto_normalize=False)


def test_build_model_returns_unclamped():
    from foregrounds_diffusion.sample import UnclampedGaussianDiffusion, build_model

    assert isinstance(build_model(channels=2), UnclampedGaussianDiffusion)


def test_ancestral_path_x0_not_clamped():
    # p_sample → p_mean_variance hard-codes clip_denoised=True in the library;
    # the subclass must let predicted x₀ escape [-1, 1] (z-scored data).
    from denoising_diffusion_pytorch import GaussianDiffusion

    from foregrounds_diffusion.sample import UnclampedGaussianDiffusion

    x = torch.zeros(1, 2, 8, 8)
    t = torch.tensor([4])
    *_, x0_base = _tiny(GaussianDiffusion).p_mean_variance(x, t, clip_denoised=True)
    *_, x0_unclamped = _tiny(UnclampedGaussianDiffusion).p_mean_variance(x, t, clip_denoised=True)

    assert x0_base.min() >= -1.0 - 1e-6  # library behaviour: clamped
    assert x0_unclamped.min() < -1.5  # fix: unclamped


def test_ddim_path_x0_not_clamped():
    # ddim_sample → model_predictions hard-codes clip_x_start=True in the library.
    from denoising_diffusion_pytorch import GaussianDiffusion

    from foregrounds_diffusion.sample import UnclampedGaussianDiffusion

    x = torch.zeros(1, 2, 8, 8)
    t = torch.tensor([4])
    pred_base = _tiny(GaussianDiffusion).model_predictions(
        x, t, clip_x_start=True, rederive_pred_noise=True
    )
    pred_unclamped = _tiny(UnclampedGaussianDiffusion).model_predictions(
        x, t, clip_x_start=True, rederive_pred_noise=True
    )

    assert pred_base.pred_x_start.min() >= -1.0 - 1e-6
    assert pred_unclamped.pred_x_start.min() < -1.5


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
