"""The RoFormer separator: waveform -> band-split RoFormer -> masks -> separated waveforms.

    mixture waveform
      -> STFT (2048/512)                 the primary "view"
      -> STFT (1024/512)                 the second resolution (multi-resolution input)
      -> MelBandRoFormer                 mel bands + band/time RoPE attention
      -> complex masks (cIRM, tanh)      one per source and channel
      -> mask * mixture spectrum
      -> iSTFT                           the same exact inverse as the U-Net path

`RoFormerSeparator` deliberately exposes the same members as
`src/models/separation.py::SpectrogramSeparator` - `n_sources`, `stft_cfg`,
`predict_masks`, `forward(mixture, length, consistent)` returning the same `Separation`
dataclass - so `separate_long`, `SeparationLoss` and the analysis code work with either
architecture without a single change.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import STFTConfig
from ..config_roformer import RoFormerModelConfig
from ..dsp import istft, mixture_consistency, stft
from .roformer import MelBandRoFormer
from .separation import Separation


class RoFormerSeparator(nn.Module):
    """Mask estimator around the STFT/iSTFT pair, with a MelBandRoFormer inside."""

    def __init__(
        self,
        stft_cfg: STFTConfig,
        model_cfg: RoFormerModelConfig,
        channels: int = 2,
        n_sources: int = 4,
        sample_rate: int = 44100,
    ) -> None:
        super().__init__()
        self.stft_cfg = stft_cfg
        self.model_cfg = model_cfg
        self.channels = channels
        self.n_sources = n_sources
        self.sample_rate = sample_rate
        # the second resolution keeps the hop of the primary one, so the two views have
        # the same time grid apart from the framing shift of a different window length
        self.extra_stft_cfg = (
            STFTConfig(n_fft=model_cfg.n_fft_extra, hop_length=stft_cfg.hop_length,
                       win_length=None, center=stft_cfg.center, window=stft_cfg.window)
            if model_cfg.n_fft_extra
            else None
        )
        self.net = MelBandRoFormer(
            n_bins=stft_cfg.n_bins,
            channels=channels,
            n_sources=n_sources,
            sample_rate=sample_rate,
            n_fft=stft_cfg.n_fft,
            dim=model_cfg.dim,
            depth=model_cfg.depth,
            heads=model_cfg.heads,
            ff_mult=model_cfg.ff_mult,
            n_bands=model_cfg.n_bands,
            band_overlap=model_cfg.band_overlap,
            fmin=model_cfg.fmin,
            fmax=model_cfg.fmax,
            max_band_bins=model_cfg.max_band_bins,
            n_fft_extra=model_cfg.n_fft_extra,
            mask=model_cfg.mask,
            per_channel_mask=model_cfg.per_channel_mask,
            dropout=model_cfg.dropout,
        )

    def predict_masks(self, mixture: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Waveform (B, ch, n) -> (masks (B, S, ch, F, T) complex, mixture spectrogram)."""
        mix_spec = stft(mixture, self.stft_cfg)
        extra_spec = stft(mixture, self.extra_stft_cfg) if self.extra_stft_cfg else None
        masks = self.net(mix_spec, extra_spec)
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
