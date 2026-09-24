"""U-Net shapes, gradient flow and the full separator wrapper."""
from __future__ import annotations

import pytest
import torch

from src.config import ModelConfig, STFTConfig
from src.models.separation import SpectrogramSeparator
from src.models.unet import UNet


@pytest.mark.parametrize("depth,base", [(3, 8), (5, 8)])
def test_unet_output_shape(depth: int, base: int) -> None:
    model = UNet(in_channels=2, n_sources=4, out_per_source=4, base_channels=base,
                 depth=depth)
    x = torch.randn(2, 2, 1025, 64)
    y = model(x)
    assert y.shape == (2, 16, 1025, 64)


def test_unet_handles_odd_shapes_and_skip_sizes() -> None:
    model = UNet(in_channels=2, n_sources=4, out_per_source=2, base_channels=8, depth=4)
    for shape in [(1, 2, 513, 33), (1, 2, 1025, 17)]:
        y = model(torch.randn(*shape))
        assert y.shape[-2:] == shape[-2:]


def test_unet_parameters_and_gradients() -> None:
    model = UNet(base_channels=16, depth=4)
    n_params = sum(p.numel() for p in model.parameters())
    assert 1e5 < n_params < 2e7
    out = model(torch.randn(1, 2, 257, 32))
    out.square().mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_separator_forward_shapes_and_consistency() -> None:
    stft_cfg = STFTConfig(n_fft=1024, hop_length=256)
    model_cfg = ModelConfig(base_channels=8, depth=3)
    model = SpectrogramSeparator(stft_cfg, model_cfg, channels=2, n_sources=4)
    mixture = torch.randn(2, 2, 44100)
    out = model(mixture)
    assert out.masks.shape[:3] == (2, 4, 2)
    assert out.spectra.shape[1] == 4
    assert out.waveforms.shape == (2, 4, 2, 44100)
    # mixture consistency: the four estimates must sum back to the mixture
    assert torch.allclose(out.waveforms.sum(dim=1), mixture, atol=1e-4)
    # cIRM masks are produced with tanh, so they must live in [-1, 1]
    assert float(out.masks.real.detach().abs().max()) <= 1.0
    assert float(out.masks.imag.detach().abs().max()) <= 1.0


def test_cirm_masks_are_bounded() -> None:
    stft_cfg = STFTConfig(n_fft=512, hop_length=128)
    model = SpectrogramSeparator(stft_cfg, ModelConfig(base_channels=8, depth=3))
    masks, _ = model.predict_masks(torch.randn(1, 2, 8000))
    assert masks.real.abs().max() <= 1.0 + 1e-6
    assert masks.imag.abs().max() <= 1.0 + 1e-6
