"""The complete separator: waveform -> U-Net -> masks -> separated waveforms.

    mixture waveform
      -> STFT                    "take a photo of the sound"
      -> log-magnitude image     (B, 2, F, T)
      -> U-Net                   "segment the photo per instrument"
      -> complex masks           (B, S, 2, F, T)
      -> mask * mixture spectrum "cut the sources out of the photo"
      -> iSTFT                   "print the pictures back to sound"

`separate_long` adds sliding-window overlap-add so that arbitrary song lengths and
16 GB of unified memory can coexist.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..config import ModelConfig, STFTConfig
from ..dsp import (
    istft,
    log_magnitude,
    mixture_consistency,
    normalize_features,
    split_complex_maps,
    stft,
)
from .unet import UNet


@dataclass
class Separation:
    """Everything a caller may want from one forward pass."""

    masks: torch.Tensor       # (B, S, ch, F, T) complex
    spectra: torch.Tensor     # (B, S, ch, F, T) complex
    waveforms: torch.Tensor   # (B, S, ch, samples)
    mixture_spec: torch.Tensor  # (B, ch, F, T) complex


class SpectrogramSeparator(nn.Module):
    """U-Net mask estimator wrapped around the STFT/iSTFT pair."""

    def __init__(self, stft_cfg: STFTConfig, model_cfg: ModelConfig,
                 channels: int = 2, n_sources: int = 4) -> None:
        super().__init__()
        self.stft_cfg = stft_cfg
        self.channels = channels
        self.n_sources = n_sources
        # cIRM mode predicts (real, imag) per source/channel, a magnitude mask only a gain
        out_per_source = channels * 2 if model_cfg.mask == "tanh" else channels
        self.unet = UNet(
            in_channels=channels,
            n_sources=n_sources,
            out_per_source=out_per_source,
            base_channels=model_cfg.base_channels,
            depth=model_cfg.depth,
            norm=model_cfg.norm,
            up_mode=model_cfg.up_mode,
            mask=model_cfg.mask,
            dropout=model_cfg.dropout,
        )

    def predict_masks(self, mixture: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Waveform (B, ch, n) -> (masks (B,S,ch,F,T) complex, mixture spectrogram)."""
        mix_spec = stft(mixture, self.stft_cfg)
        features = normalize_features(log_magnitude(mix_spec))
        flat_masks = self.unet(features)
        if self.unet.mask == "sigmoid":
            # a real gain g applied to both parts: (g + 0i) * X
            gains = flat_masks.reshape(
                flat_masks.shape[0], self.n_sources, self.channels, *flat_masks.shape[-2:]
            )
            masks = torch.complex(gains, torch.zeros_like(gains))
        else:
            masks = split_complex_maps(flat_masks, self.n_sources, self.channels)
        return masks, mix_spec

    def forward(self, mixture: torch.Tensor, length: int | None = None,
                consistent: bool = True) -> Separation:
        """mixture: (B, ch, samples) -> estimates, with mixture consistency."""
        length = length or mixture.shape[-1]
        masks, mix_spec = self.predict_masks(mixture)
        spectra = masks * mix_spec.unsqueeze(1)
        b, s, ch, f, t = spectra.shape
        waveforms = istft(
            spectra.reshape(b * s, ch, f, t), self.stft_cfg, length
        ).reshape(b, s, ch, length)
        if consistent:
            waveforms = mixture_consistency(waveforms, mixture)
        return Separation(masks=masks, spectra=spectra, waveforms=waveforms,
                          mixture_spec=mix_spec)


@torch.no_grad()
def separate_long(
    model: SpectrogramSeparator,
    mixture: torch.Tensor,
    sample_rate: int = 44100,
    chunk_seconds: float = 10.0,
    overlap: float = 0.5,
) -> torch.Tensor:
    """Separate an arbitrarily long (ch, samples) waveform by sliding-window overlap-add.

    The U-Net is fully convolutional, but a whole 5-minute spectrogram does not fit in
    memory, so the track is cut into overlapping pieces whose waveforms are cross-faded
    with a raised-cosine window.
    """
    ch, n = mixture.shape
    chunk = int(chunk_seconds * sample_rate)
    hop = max(1, int(chunk * (1.0 - overlap)))
    n_sources = model.n_sources

    out = torch.zeros(n_sources, ch, n + chunk, device=mixture.device, dtype=mixture.dtype)
    weight = torch.zeros(n + chunk, device=mixture.device, dtype=mixture.dtype)
    fade = torch.hann_window(chunk, periodic=False, device=mixture.device).clamp_min(1e-3)

    for start in range(0, max(n, 1), hop):
        piece = mixture[:, start : start + chunk]
        if piece.shape[-1] < chunk:
            piece = torch.nn.functional.pad(piece, (0, chunk - piece.shape[-1]))
        est = model(piece.unsqueeze(0), length=chunk).waveforms[0]  # (S, ch, chunk)
        end = start + chunk
        out[:, :, start:end] += est * fade
        weight[start:end] += fade
        if end >= n:
            break

    out = out[:, :, :n] / weight[:n].clamp_min(1e-3)
    return out  # (S, ch, samples)
