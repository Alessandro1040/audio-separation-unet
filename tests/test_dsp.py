"""STFT/iSTFT correctness - the foundation everything else rests on."""
from __future__ import annotations

import pytest
import torch

from src.config import STFTConfig
from src.dsp import (
    istft,
    log_magnitude,
    mixture_consistency,
    normalize_features,
    split_complex_maps,
    stack_complex_maps,
    stft,
)


@pytest.mark.parametrize("n_fft,hop", [(2048, 512), (1024, 256), (512, 128)])
def test_stft_istft_roundtrip(n_fft: int, hop: int) -> None:
    cfg = STFTConfig(n_fft=n_fft, hop_length=hop)
    torch.manual_seed(0)
    x = torch.randn(2, 2, 3 * 44100) * 0.1
    y = istft(stft(x, cfg), cfg, x.shape[-1])
    assert y.shape == x.shape
    assert torch.allclose(x, y, atol=1e-4), (x - y).abs().max()


def test_roundtrip_non_multiple_of_hop() -> None:
    cfg = STFTConfig(n_fft=2048, hop_length=512)
    for n in (44100, 44100 + 137, 5000):
        x = torch.randn(2, n) * 0.1
        y = istft(stft(x, cfg), cfg, n)
        assert y.shape == x.shape
        assert torch.allclose(x, y, atol=1e-4)


def test_features_are_normalized() -> None:
    cfg = STFTConfig()
    x = torch.randn(3, 2, 44100 * 2) * 0.2
    feat = normalize_features(log_magnitude(stft(x, cfg)))
    assert torch.allclose(feat.mean(dim=(1, 2, 3)), torch.zeros(3), atol=1e-4)
    assert torch.allclose(feat.std(dim=(1, 2, 3)), torch.ones(3), atol=1e-2)


def test_complex_map_roundtrip() -> None:
    spec = torch.randn(2, 4, 2, 8, 5, dtype=torch.float32) + 1j * torch.randn(
        2, 4, 2, 8, 5, dtype=torch.float32
    )
    flat = stack_complex_maps(spec)
    assert flat.shape == (2, 16, 8, 5)
    back = split_complex_maps(flat, 4, 2)
    assert torch.allclose(spec, back)


def test_mixture_consistency_sums_to_mixture() -> None:
    est = torch.randn(2, 4, 2, 1000)
    mix = torch.randn(2, 2, 1000)
    fixed = mixture_consistency(est, mix)
    assert torch.allclose(fixed.sum(dim=1), mix, atol=1e-5)
