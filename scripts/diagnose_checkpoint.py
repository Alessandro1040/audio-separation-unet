#!/usr/bin/env python3
"""Diagnose a checkpoint: how good are the live weights vs the EMA ones vs references?

    python scripts/diagnose_checkpoint.py checkpoints/last.pt --chunks 8 --device cpu

Reports SI-SDR and the training losses on fixed validation chunks for

    live weights | EMA weights | oracle ideal mask | zero output | mixture as estimate

The last four are what makes the number interpretable: the oracle mask is the ceiling of
any magnitude-mask model, "zero" and "mixture" are the trivial baselines that a model
has to beat.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import STEMS, Config, _SECTIONS, _dataclass_from_dict  # noqa: E402
from src.data.chunks import MusdbEvalChunks, collate_chunks, validation_tracks  # noqa: E402
from src.dsp import istft, stft  # noqa: E402
from src.losses import SeparationLoss  # noqa: E402
from src.metrics import si_sdr_per_source  # noqa: E402
from src.models.separation import SpectrogramSeparator  # noqa: E402
from src.train import pick_device  # noqa: E402


def config_from_checkpoint(ckpt: dict) -> Config:
    cfg = Config()
    for section, cls in _SECTIONS.items():
        values = dict(ckpt["config"].get(section) or {})
        if section == "data" and "aug" in values:
            from src.config import AugmentConfig

            cfg.data.aug = _dataclass_from_dict(AugmentConfig, values.pop("aug"))
        setattr(cfg, section, _dataclass_from_dict(cls, values))
    return cfg


def oracle_estimate(mixture, stems, stft_cfg):
    """Ideal ratio mask (per source) applied to the true complex mixture spectrum.

    Shapes: stems (B, S, ch, F, T), mixture (B, ch, F, T) -> masks (B, S, ch, F, T);
    the mixture spectrum is broadcast over the source axis with `unsqueeze(1)`.
    """
    spec = stft(stems, stft_cfg)
    mix_spec = stft(mixture, stft_cfg)
    masks = spec.abs() / spec.abs().sum(dim=1, keepdim=True).clamp_min(1e-8)
    est_spec = masks * mix_spec.unsqueeze(1)
    b, s, ch, f, t = est_spec.shape
    return istft(est_spec.reshape(b * s, ch, f, t), stft_cfg,
                 mixture.shape[-1]).reshape(b, s, ch, -1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("checkpoint")
    p.add_argument("--chunks", type=int, default=8)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = pick_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = config_from_checkpoint(ckpt)
    cfg.data.num_workers = 0
    tracks = validation_tracks(cfg.data)
    dataset = MusdbEvalChunks(cfg.data, tracks=tracks, chunks_per_track=1)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, range(min(args.chunks, len(dataset)))),
        batch_size=cfg.train.batch_size, collate_fn=collate_chunks,
    )
    loss_fn = SeparationLoss(cfg.loss, cfg.stft)

    model = SpectrogramSeparator(cfg.stft, cfg.model, cfg.data.channels, len(STEMS))
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    ema = None
    if ckpt.get("ema"):
        ema = SpectrogramSeparator(cfg.stft, cfg.model, cfg.data.channels, len(STEMS))
        ema.load_state_dict(ckpt["ema"])
        ema = ema.to(device).eval()

    print(f"checkpoint {args.checkpoint} (step {ckpt.get('step')}), device={device}, "
          f"{min(args.chunks, len(dataset))} chunks of {cfg.data.chunk_seconds}s")
    print(f"{'estimator':<22} " + " ".join(f"{s[:6]:>7}" for s in STEMS) + f" {'mean':>7}"
          + f" {'loss':>7}")

    def report(name: str, waves: torch.Tensor, stems: torch.Tensor) -> None:
        sdr = si_sdr_per_source(waves, stems)
        loss = float(loss_fn(waves, stems).total)
        print(f"{name:<22} " + " ".join(f"{float(v):>7.2f}" for v in sdr)
              + f" {float(sdr.mean()):>7.2f} {loss:>7.3f}")

    with torch.no_grad():
        for batch in loader:
            stems = batch["stems"].to(device)
            mixture = batch["mixture"].to(device)
            n = mixture.shape[-1]

            if ema is not None:
                report("EMA weights", ema(mixture, length=n).waveforms, stems)
            out = model(mixture, length=n)
            report("live weights", out.waveforms, stems)
            report("oracle ideal mask", oracle_estimate(mixture, stems, cfg.stft), stems)
            report("mixture as estimate",
                   torch.ones_like(stems) * mixture.unsqueeze(1), stems)
            report("zero output", torch.zeros_like(stems), stems)
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
