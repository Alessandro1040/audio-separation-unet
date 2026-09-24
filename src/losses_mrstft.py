"""Multi-resolution STFT loss for the Mel-band RoFormer path.

Why this loss instead of the U-Net's single-resolution one: the RoFormer predicts a mask
per *mel band*, so a single STFT resolution can only ever see one frequency/time trade-off
at a time. Averaging the spectral L1 at four window lengths (2048/1024/512/256, i.e. 46 ms
down to 5.8 ms at 44.1 kHz) tells the model at once that a 5 ms attack matters *and* that
a 46 Hz-spaced harmonic series matters - which is exactly the complaint the multi-resolution
loss of the MDX23 winners was designed to answer (see also Yamamoto et al., *Parallel
Waveform Generation*, 2020, for the same loss in a vocoder).

The linear term is L1 on the magnitudes; the log term is L1 on log magnitudes. Both are
needed: the linear term dominates the loud onsets, the log term keeps quiet detail (reverb
tails, cymbals, breath) from being ignored - the same reasoning as the U-Net's compressed
magnitude term, but at every resolution at once.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import STFTConfig
from .config_roformer import RoFormerLossConfig
from .dsp import stft
from .losses import LossBreakdown
from .metrics import si_sdr_per_source


def _resolution_cfgs(cfg: RoFormerLossConfig) -> list[STFTConfig]:
    """One `STFTConfig` per resolution, 75 % overlap, Hann window (the standard choice)."""
    return [
        STFTConfig(n_fft=n_fft, hop_length=max(1, n_fft // 4), center=True, window="hann")
        for n_fft in cfg.mrstft_ffts
    ]


class MRSTFTLoss(nn.Module):
    """Waveform L1 + multi-resolution spectral loss, summed over all stems."""

    def __init__(self, cfg: RoFormerLossConfig, stft_cfg: STFTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.stft_cfg = stft_cfg            # primary resolution (the mask domain)
        self.resolutions = _resolution_cfgs(cfg)

    def _spectral_term(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """estimate/target: (N, samples) -> mean over the resolutions."""
        total = torch.zeros((), device=estimate.device)
        for res in self.resolutions:
            est = stft(estimate, res).abs()
            ref = stft(target, res).abs()
            total = total + (est - ref).abs().mean()
            total = total + self.cfg.mrstft_log_weight * (
                torch.log(est.clamp_min(1e-5)) - torch.log(ref.clamp_min(1e-5))
            ).abs().mean()
        return total / max(len(self.resolutions), 1)

    def forward(
        self,
        estimates: torch.Tensor,        # (B, S, ch, samples)
        targets: torch.Tensor,          # (B, S, ch, samples)
        est_spec: torch.Tensor | None = None,   # accepted for interface parity with SeparationLoss
        tgt_spec: torch.Tensor | None = None,
    ) -> LossBreakdown:
        device = estimates.device
        wave_loss = torch.zeros((), device=device)
        single = torch.zeros((), device=device)          # the U-Net-style magnitude term
        multi = torch.zeros((), device=device)           # the multi-resolution term

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
            single = (est_mag - tgt_mag).abs().mean()

        if self.cfg.mrstft > 0:
            b, s, ch, n = estimates.shape
            multi = self._spectral_term(
                estimates.reshape(b * s * ch, n), targets.reshape(b * s * ch, n)
            )

        total = (self.cfg.waveform * wave_loss
                 + self.cfg.magnitude * single
                 + self.cfg.mrstft * multi)
        sdr_loss = None
        if self.cfg.si_sdr > 0:
            sdr_loss = -si_sdr_per_source(estimates, targets).mean() / 10.0
            total = total + self.cfg.si_sdr * sdr_loss
        # the logged "magnitude" column is the sum of the two spectral terms, so it stays
        # comparable with the U-Net's log (where it is a single-resolution magnitude L1)
        return LossBreakdown(total=total, waveform=wave_loss, magnitude=single + multi,
                             si_sdr=sdr_loss)
