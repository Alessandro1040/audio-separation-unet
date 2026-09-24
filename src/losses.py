"""Source-separation losses.

Two complementary terms are used, both functionals of what the model actually emits:

1. `waveform`: L1 between the reconstructed waveform and the target - the loss that
   correlates best with perceptual quality (Demucs-style systems use exactly this).
2. `magnitude`: L1 between compressed magnitudes (|X|^0.3). Compression puts quiet
   details (reverb tails, cymbals, breath) on an equal footing with loud onsets;
   without it the waveform term dominates and quiet passages are under-fitted.

The magnitude term is computed on the *complex spectrograms* the masks produced, so it
costs nothing extra (no second forward STFT of the estimate).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import LossConfig, STFTConfig
from .dsp import stft
from .metrics import si_sdr_per_source


@dataclass
class LossBreakdown:
    total: torch.Tensor
    waveform: torch.Tensor
    magnitude: torch.Tensor
    si_sdr: torch.Tensor | None = None

    def as_dict(self) -> dict[str, float]:
        out = {
            "loss": float(self.total.detach()),
            "loss_wave": float(self.waveform.detach()),
            "loss_mag": float(self.magnitude.detach()),
        }
        if self.si_sdr is not None:
            out["loss_sdr"] = float(self.si_sdr.detach())
        return out


class SeparationLoss(nn.Module):
    """Hybrid waveform + compressed-magnitude loss summed over all stems."""

    def __init__(self, cfg: LossConfig, stft_cfg: STFTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.stft_cfg = stft_cfg

    def forward(
        self,
        estimates: torch.Tensor,        # (B, S, ch, samples)
        targets: torch.Tensor,          # (B, S, ch, samples)
        est_spec: torch.Tensor | None = None,   # (B, S, ch, F, T) complex
        tgt_spec: torch.Tensor | None = None,   # (B, S, ch, F, T) complex
    ) -> LossBreakdown:
        device = estimates.device
        wave_loss = torch.zeros((), device=device)
        mag_loss = torch.zeros((), device=device)

        if self.cfg.waveform > 0:
            wave_loss = (estimates - targets).abs().mean()

        if self.cfg.magnitude > 0:
            if est_spec is None:
                b, s, ch, n = estimates.shape
                flat = stft(estimates.reshape(b * s, ch, n), self.stft_cfg)
                est_spec = flat.reshape(b, s, ch, *flat.shape[-2:])
            if tgt_spec is None:
                b, s, ch, n = targets.shape
                flat = stft(targets.reshape(b * s, ch, n), self.stft_cfg)
                tgt_spec = flat.reshape(b, s, ch, *flat.shape[-2:])
            c = self.cfg.magnitude_compress
            est_mag = est_spec.abs().clamp_min(1e-8).pow(c)
            tgt_mag = tgt_spec.abs().clamp_min(1e-8).pow(c)
            mag_loss = (est_mag - tgt_mag).abs().mean()

        total = self.cfg.waveform * wave_loss + self.cfg.magnitude * mag_loss
        sdr_loss = None
        if self.cfg.si_sdr > 0:
            # -SI-SDR/10 per stem: directly optimises the reported metric without
            # caring about absolute level (scale-invariant), which is exactly what a
            # mask-based model controls well.
            sdr_loss = -si_sdr_per_source(estimates, targets).mean() / 10.0
            total = total + self.cfg.si_sdr * sdr_loss
        return LossBreakdown(total=total, waveform=wave_loss, magnitude=mag_loss,
                             si_sdr=sdr_loss)

