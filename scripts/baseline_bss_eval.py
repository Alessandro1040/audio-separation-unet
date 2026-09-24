#!/usr/bin/env python3
"""BSS-Eval of the trivial baseline: use the mixture as the estimate of every source.

    python scripts/baseline_bss_eval.py --limit 8

Any separator worth its salt has to beat this by a wide margin: this is the "do nothing"
reference that says how much of the mixture energy belongs to each source on average.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import STEMS  # noqa: E402
from src.data.musdb import _TrackCache, find_tracks, load_stems  # noqa: E402
from src.metrics import bss_eval_track  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default="data/musdb18")
    p.add_argument("--subset", default="test")
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--max-seconds", type=float, default=None,
                   help="evaluate only the first N seconds of each song (faster)")
    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument("--out", default="runs/baseline_bss.json")
    p.add_argument("--fast", action="store_true",
                   help="chunk-average SI-SDR of the mixture estimator (analytic, fast)")
    args = p.parse_args()

    if args.fast:
        from src.evaluate import si_sdr_chunks

        tracks = find_tracks(args.root, args.subset)[: args.limit]
        cache = _TrackCache(capacity=2)
        rows = []
        for track in tracks:
            reference = load_stems(track, cache, args.sample_rate, 2)
            if args.max_seconds:
                reference = reference[..., : int(args.max_seconds * args.sample_rate)]
            mixture = reference.sum(dim=0)
            estimate = mixture.unsqueeze(0).expand_as(reference).clone()
            sdr = si_sdr_chunks(estimate, reference, args.sample_rate)
            rows.append(float(sdr.mean()))
            print(f"  {track.name[:32]:<32} mixture-estimator SI-SDR="
                  f"{float(sdr.mean()):6.2f} dB", flush=True)
        summary = {"estimator": "mixture (SI-SDR)", "tracks": len(tracks),
                   "median_si_sdr": float(np.median(rows))}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"median SI-SDR of the trivial baseline: {summary['median_si_sdr']:.2f} dB")
        return 0

    tracks = find_tracks(args.root, args.subset)[: args.limit]
    cache = _TrackCache(capacity=2)
    sdrs = []
    for track in tracks:
        reference = load_stems(track, cache, args.sample_rate, 2).numpy()
        if args.max_seconds:
            reference = reference[..., : int(args.max_seconds * args.sample_rate)]
        mixture = reference.sum(axis=0)
        estimate = np.stack([mixture] * len(STEMS)).astype(np.float64)
        metrics = bss_eval_track(reference.astype(np.float64), estimate, args.sample_rate)
        sdrs.append(metrics.sdr)
        print(f"  {track.name[:32]:<32} SDR={metrics.sdr:6.2f} dB  "
              f"SIR={metrics.sir:6.2f} dB  SAR={metrics.sar:6.2f} dB")
    summary = {"estimator": "mixture", "tracks": len(tracks),
               "median_sdr": float(np.median(sdrs))}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"median SDR of the trivial 'mixture' baseline: {summary['median_sdr']:.2f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
