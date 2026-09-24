#!/usr/bin/env python3
"""Is the model *separating*, or only changing the volume? Measure it.

    python scripts/diagnose_masks.py checkpoints/best.pt --chunks 4 --device cpu

Two questions are answered on deterministic validation chunks:

1. **How good are the estimates?** SI-SDR per source for the network, for three
   references any "gain" can reach, and for the ceiling of a mask model:

       mixture as estimate   the mixture itself. SI-SDR is scale invariant, so this *is*
                             the "best possible volume change" - a model that only
                             re-levels its inputs can never beat this row, it can only
                             match it.
       best static gain      the same thing computed explicitly, as a sanity check.
       best fader            per source, the optimal time-varying gain (50 ms frames) on
                             the mixture: the strongest thing "volume automation" can do.
       oracle ideal mask     ideal ratio mask on the true mixture spectrum: the ceiling.

   A separator that "only changes the volumes" scores like `mixture`/`fader`.

2. **What is inside the mask?** A two-way decomposition of each predicted mask m(F, T)
   (mean magnitude over channels, per source), into

       mean |m|        the average gain (what a volume change looks like)
       cv              std/mean of |m|: 0 means a constant gain
       F share         variance from the frequency profile alone  -> fixed EQ / tilt
       T share         variance from the time profile alone       -> a fader
       F x T share     variance from the interaction              -> real separation:
                       only this can lift one instrument out of a bin another occupies
       low/high        where the mask keeps energy (a bass mask must be low-heavy)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import STEMS, Config, _SECTIONS, _dataclass_from_dict  # noqa: E402
from src.config import AugmentConfig  # noqa: E402
from src.data.chunks import MusdbEvalChunks, collate_chunks, validation_tracks  # noqa: E402
from src.data.musdb import find_tracks  # noqa: E402
from src.dsp import istft, stft  # noqa: E402
from src.metrics import si_sdr_per_source  # noqa: E402
from src.models.separation import SpectrogramSeparator  # noqa: E402


def config_from_checkpoint(ckpt: dict) -> Config:
    cfg = Config()
    for section, cls in _SECTIONS.items():
        values = dict(ckpt["config"].get(section) or {})
        if section == "data" and "aug" in values:
            cfg.data.aug = _dataclass_from_dict(AugmentConfig, values.pop("aug"))
        setattr(cfg, section, _dataclass_from_dict(cls, values))
    return cfg


def oracle_estimate(mixture: torch.Tensor, stems: torch.Tensor,
                    cfg: Config) -> torch.Tensor:
    """Ideal ratio mask per source applied to the true complex mixture spectrum."""
    spec = stft(stems, cfg.stft)
    mix_spec = stft(mixture, cfg.stft)
    masks = spec.abs() / spec.abs().sum(dim=1, keepdim=True).clamp_min(1e-8)
    est_spec = masks * mix_spec.unsqueeze(1)
    b, s, ch, f, t = est_spec.shape
    return istft(est_spec.reshape(b * s, ch, f, t), cfg.stft,
                 mixture.shape[-1]).reshape(b, s, ch, -1)


def fader_estimate(mixture: torch.Tensor, stems: torch.Tensor,
                   frame_samples: int) -> torch.Tensor:
    """Per-source *time-varying* gain on the mixture: the best any fader can do."""
    out = torch.zeros_like(stems)
    n = mixture.shape[-1]
    for start in range(0, n, frame_samples):
        mix = mixture[..., start : start + frame_samples].unsqueeze(1)
        tgt = stems[..., start : start + frame_samples]
        num = (mix * tgt).sum(dim=(-1, -2), keepdim=True)
        den = (mix * mix).sum(dim=(-1, -2), keepdim=True).clamp_min(1e-12)
        out[..., start : start + frame_samples] = (num / den) * mix
    return out


def best_static_gain(mixture: torch.Tensor, stems: torch.Tensor) -> torch.Tensor:
    """Per-source constant gain on the mixture (the 'volume only' estimator)."""
    mix = mixture.unsqueeze(1)
    num = (mix * stems).sum(dim=(-1, -2), keepdim=True)
    den = (mix * mix).sum(dim=(-1, -2), keepdim=True).clamp_min(1e-12)
    return (num / den) * mix


def mask_statistics(masks: torch.Tensor, sample_rate: int) -> dict[str, dict[str, float]]:
    """Two-way decomposition of each predicted mask magnitude, per source."""
    m = masks.abs().mean(dim=2)[0]                     # (S, F, T)
    freqs = torch.linspace(0, sample_rate / 2, m.shape[1])
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(STEMS):
        mask = m[i]
        total_var = float(mask.var(unbiased=False))
        f_profile = mask.mean(dim=-1) - mask.mean()
        t_profile = mask.mean(dim=-2) - mask.mean()
        residual = mask - mask.mean() - f_profile[:, None] - t_profile[None, :]
        energy = mask.pow(2).sum().clamp_min(1e-12)
        out[name] = {
            "mean": float(mask.mean()),
            "cv": float(mask.std(unbiased=False) / mask.mean().clamp_min(1e-8)),
            "var_freq": float(f_profile.var(unbiased=False) / max(total_var, 1e-12)),
            "var_time": float(t_profile.var(unbiased=False) / max(total_var, 1e-12)),
            "var_interaction": float(residual.var(unbiased=False) / max(total_var, 1e-12)),
            "energy_low": float(mask[freqs < 200].pow(2).sum() / energy),
            "energy_high": float(mask[freqs > 2000].pow(2).sum() / energy),
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("checkpoint")
    p.add_argument("--config", default=None,
                   help="use the data section of this config (e.g. a synthetic root)")
    p.add_argument("--chunks", type=int, default=4)
    p.add_argument("--chunks-per-track", type=int, default=1,
                   help="chunks taken from each song (spread over its length)")
    p.add_argument("--subset", default="valid", choices=("valid", "test"),
                   help="'valid' = the held-out training songs, 'test' = MUSDB18 test")
    p.add_argument("--device", default="cpu")
    p.add_argument("--fader-ms", type=float, default=50.0)
    args = p.parse_args()

    from src.config import load_config
    from src.train import pick_device

    device = pick_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = config_from_checkpoint(ckpt)
    if args.config:
        cfg.data = load_config(args.config).data
    cfg.data.num_workers = 0

    tracks = (find_tracks(cfg.data.root, "test") if args.subset == "test"
              else validation_tracks(cfg.data))
    dataset = MusdbEvalChunks(cfg.data, tracks=tracks,
                              chunks_per_track=args.chunks_per_track)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, range(min(args.chunks, len(dataset)))),
        batch_size=1, collate_fn=collate_chunks,
    )

    model = SpectrogramSeparator(cfg.stft, cfg.model, cfg.data.channels, len(STEMS))
    model.load_state_dict(ckpt.get("ema") or ckpt.get("model") or ckpt.get("state_dict"))
    model = model.to(device).eval()

    frame = max(1, int(args.fader_ms / 1000.0 * cfg.data.sample_rate))
    sdr_sum: dict[str, torch.Tensor] = {}
    mask_sum: dict[str, dict[str, float]] = {}
    n_chunks = 0
    with torch.no_grad():
        for batch in loader:
            stems = batch["stems"].to(device)
            mixture = batch["mixture"].to(device)
            out = model(mixture, length=mixture.shape[-1])
            estimators = {
                "model": out.waveforms,
                "mixture as estimate": mixture.unsqueeze(1).expand_as(stems),
                "best static gain": best_static_gain(mixture, stems),
                "best fader": fader_estimate(mixture, stems, frame),
                "oracle ideal mask": oracle_estimate(mixture, stems, cfg),
            }
            for name, estimate in estimators.items():
                sdr = si_sdr_per_source(estimate, stems).cpu()
                sdr_sum[name] = sdr_sum.get(name, torch.zeros_like(sdr)) + sdr
            for name, values in mask_statistics(out.masks, cfg.data.sample_rate).items():
                acc = mask_sum.setdefault(name, {})
                for key, value in values.items():
                    acc[key] = acc.get(key, 0.0) + value
            n_chunks += 1

    print(f"checkpoint {args.checkpoint} (step {ckpt.get('step')}), device={device}, "
          f"{n_chunks} chunks of {cfg.data.chunk_seconds}s, fader frame "
          f"{args.fader_ms:.0f} ms")
    print("\nSI-SDR per source (dB): what the network does, and what pure volume changes do")
    print(f"{'estimator':<22} " + " ".join(f"{s[:6]:>7}" for s in STEMS) + f" {'mean':>7}")
    for name, total in sdr_sum.items():
        sdr = total / n_chunks
        print(f"{name:<22} " + " ".join(f"{float(v):>7.2f}" for v in sdr)
              + f" {float(sdr.mean()):>7.2f}")

    print("\npredicted mask: F/T/FxT are shares of its variance")
    print(f"{'stem':<8} {'mean|m|':>8} {'cv':>6} {'F share':>8} {'T share':>8} "
          f"{'FxT':>6} {'low':>6} {'high':>6}")
    for name in STEMS:
        v = {k: value / n_chunks for k, value in mask_sum[name].items()}
        print(f"{name:<8} {v['mean']:8.3f} {v['cv']:6.2f} {v['var_freq']:8.2f} "
              f"{v['var_time']:8.2f} {v['var_interaction']:6.2f} "
              f"{v['energy_low']:6.1%} {v['energy_high']:6.1%}")
    print("\nA mask that only re-levels the mixture has cv ~ 0; a mask that really "
          "separates\nputs its variance in the FxT term (the only term that can lift a "
          "source out of a bin).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
