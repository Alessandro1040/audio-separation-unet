#!/usr/bin/env python3
"""Upper bound of mask-based separation: the *oracle* ideal ratio mask.

    python scripts/oracle_mask_baseline.py --limit 2

For every time-frequency bin the ideal mask is `|S_s| / sum_j |S_j|`, computed from the
ground-truth stems. Applying it to the true complex mixture spectrum gives the best
result any *magnitude-mask* model could hope for on that song.

Why this script exists: it is an end-to-end self-test of the DSP and metric code on real
44.1 kHz data. If the STFT framing, the seek alignment or the BSS-Eval call were wrong,
these numbers would collapse; a correct implementation gives roughly 8-12 dB SDR. It is
also the honest ceiling to compare a trained model against.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import STEMS  # noqa: E402
from src.data.musdb import _TrackCache, find_tracks, load_stems  # noqa: E402
from src.dsp import istft, stft  # noqa: E402
from src.metrics import bss_eval_track  # noqa: E402


def oracle_estimate(stems: torch.Tensor, stft_cfg, n_sources: int) -> torch.Tensor:
    """(n_sources, ch, samples) -> ideal-ratio-mask estimate, same shape."""
    mixture = stems.sum(dim=0)                                   # (ch, samples)
    spec = stft(stems, stft_cfg)                                 # (S, ch, F, T)
    mix_spec = stft(mixture, stft_cfg)                           # (ch, F, T)
    magnitude = spec.abs()
    total = magnitude.sum(dim=0, keepdim=True).clamp_min(1e-8)    # (1, ch, F, T)
    masks = magnitude / total
    estimate_spec = masks * mix_spec.unsqueeze(0)
    s, ch, f, t = estimate_spec.shape
    return istft(estimate_spec.reshape(s * ch, f, t), stft_cfg,
                 mixture.shape[-1]).reshape(s, ch, -1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default="data/musdb18")
    p.add_argument("--subset", default="test")
    p.add_argument("--limit", type=int, default=2)
    p.add_argument("--out", default="runs/oracle_bss.json")
    args = p.parse_args()

    # reuse the trained configuration's STFT so the numbers are comparable
    from src.config import load_config

    cfg = load_config("configs/unet_musdb18.yaml")
    cache = _TrackCache(capacity=2)
    results = []
    for track in find_tracks(args.root, args.subset)[: args.limit]:
        stems = load_stems(track, cache, cfg.data.sample_rate, cfg.data.channels)
        estimate = oracle_estimate(stems, cfg.stft, len(STEMS))
        metrics = bss_eval_track(stems.numpy().astype(np.float64),
                                 estimate.numpy().astype(np.float64),
                                 cfg.data.sample_rate)
        results.append(metrics.sdr)
        print(f"  {track.name[:32]:<32} oracle SDR={metrics.sdr:6.2f} dB  "
              f"SIR={metrics.sir:6.2f} dB  SAR={metrics.sar:6.2f} dB")
    summary = {"estimator": "oracle ideal ratio mask", "tracks": len(results),
               "median_sdr": float(np.median(results))}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"median oracle SDR: {summary['median_sdr']:.2f} dB "
          f"(upper bound for magnitude-mask models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
