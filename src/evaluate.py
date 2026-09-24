#!/usr/bin/env python3
"""Evaluate a checkpoint with BSS-Eval (SDR / SIR / SAR) on MUSDB18.

    # quick look (10 songs)
    python -m src.evaluate --checkpoint checkpoints/best.pt --limit 10
    # full official test set, writing every estimate to disk
    python -m src.evaluate --checkpoint checkpoints/best.pt --save-stems outputs/test

Metrics are computed frame-wise on 1-second windows and reported as the median over
frames and songs - the same protocol as the official `museval` package, so the numbers
can be compared with the MUSDB18 literature.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .config import STEMS
from .data.musdb import find_tracks
from .data.chunks import _TrackCache, load_stems
from .metrics import aggregate_bss, bss_eval_track, bss_eval_track_per_source, \
    si_sdr_per_source
from .models.separation import separate_long
from .separate import load_model
from .train import pick_device


def evaluate_track(model, cfg, track, device, chunk_seconds: float = 10.0,
                   window: float = 1.0, hop: float = 1.0, cache=None,
                   max_seconds: float | None = None,
                   fast: bool = False) -> dict:
    """Metrics of one song: BSS-Eval, or (fast=True) chunk-average SI-SDR.

    `fast` exists because BSS-Eval is CPU-bound (it solves a 512-tap filter
    decomposition per 1-second frame, per source); chunk SI-SDR is ~20x cheaper and is
    the right tool to look at many songs quickly. Both are reported in the same units
    (dB), but only BSS-Eval is comparable with the MUSDB18 literature.
    """
    cache = cache if cache is not None else _TrackCache(capacity=1)
    reference = load_stems(track, cache, cfg.data.sample_rate, cfg.data.channels)
    if max_seconds is not None:                 # evaluate only the head of the song
        reference = reference[..., : int(max_seconds * cfg.data.sample_rate)]
    mixture = reference.sum(dim=0).to(device)
    estimates = separate_long(model, mixture, cfg.data.sample_rate, chunk_seconds,
                              0.5).detach().cpu()
    if fast:
        return {"fast_sdr": si_sdr_chunks(estimates, reference, cfg.data.sample_rate),
                "reference": reference, "estimate": estimates,
                "mixture": mixture.cpu()}
    metrics = bss_eval_track(
        np.asarray(reference.numpy(), dtype=np.float64),
        np.asarray(estimates.numpy(), dtype=np.float64),
        cfg.data.sample_rate,
        window=window,
        hop=hop,
    )
    return {"metrics": metrics, "reference": reference, "estimate": estimates,
            "mixture": mixture.cpu()}


def si_sdr_chunks(estimates: torch.Tensor, reference: torch.Tensor,
                  sample_rate: int, chunk_seconds: float = 5.0) -> torch.Tensor:
    """Mean SI-SDR per source over non-overlapping `chunk_seconds` chunks."""
    n = reference.shape[-1]
    chunk = int(chunk_seconds * sample_rate)
    scores = []
    for start in range(0, max(n - chunk + 1, 1), chunk):
        est = estimates[..., start : start + chunk]
        ref = reference[..., start : start + chunk]
        if ref.shape[-1] < chunk:
            break
        scores.append(si_sdr_per_source(est.unsqueeze(0), ref.unsqueeze(0)))
    if not scores:
        return torch.full((reference.shape[0],), float("nan"))
    return torch.stack(scores).mean(dim=0)



def evaluate_checkpoint(
    checkpoint: str | Path,
    cfg_overrides: dict | None = None,
    root: str | None = None,
    subset: str = "test",
    limit: int | None = None,
    device_name: str = "auto",
    save_stems: str | Path | None = None,
    save_references: bool = False,
    chunk_seconds: float = 10.0,
    window: float = 1.0,
    hop: float = 1.0,
    max_seconds: float | None = None,
    fast: bool = False,
    out_dir: str | Path = "runs/eval",
) -> dict:
    device = pick_device(device_name)
    model, cfg = load_model(checkpoint, device)
    if root:
        cfg.data.root = root
    for key, value in (cfg_overrides or {}).items():
        section, _, name = key.partition(".")
        setattr(getattr(cfg, section), name, value)

    tracks = find_tracks(cfg.data.root, subset)
    if limit:
        tracks = tracks[:limit]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] {len(tracks)} {subset} songs, checkpoint={checkpoint}, device={device}",
          flush=True)

    per_track: list[dict] = []
    cache = _TrackCache(capacity=2)
    started = time.time()
    rows: list[dict] = []
    fast_sdrs: list[list[float]] = []
    for i, track in enumerate(tracks, 1):
        result = evaluate_track(model, cfg, track, device, chunk_seconds, window, hop,
                                cache, max_seconds, fast)
        if fast:
            sdr = result["fast_sdr"]
            fast_sdrs.append([float(v) for v in sdr])
            rows.append({"track": track.name, "si_sdr_mean": float(sdr.mean()),
                         **{f"si_sdr_{name}": float(sdr[j])
                            for j, name in enumerate(STEMS)}})
            print(f"[eval] {i:>3}/{len(tracks)} {track.name:<20} "
                  f"SI-SDR mean={float(sdr.mean()):6.2f} dB  "
                  + " ".join(f"{s[:4]}={float(sdr[j]):5.2f}"
                             for j, s in enumerate(STEMS)), flush=True)
            if save_stems:
                dest = Path(save_stems) / track.name
                dest.mkdir(parents=True, exist_ok=True)
                for j, stem in enumerate(STEMS):
                    sf.write(dest / f"{stem}.wav",
                             result["estimate"][j].numpy().T, cfg.data.sample_rate)
                sf.write(dest / "mixture.wav", result["mixture"].numpy().T,
                         cfg.data.sample_rate)
            continue
        metrics = result["metrics"]
        per_source = bss_eval_track_per_source(
            np.asarray(result["reference"].numpy(), dtype=np.float64),
            np.asarray(result["estimate"].numpy(), dtype=np.float64),
            cfg.data.sample_rate, STEMS, window, hop,
        )
        per_track.append(per_source or {name: metrics for name in STEMS})
        rows.append({"track": track.name, "sdr": metrics.sdr, "sir": metrics.sir,
                     "sar": metrics.sar, "frames": metrics.n_frames})
        print(f"[eval] {i:>3}/{len(tracks)} {track.name:<20} "
              f"SDR={metrics.sdr:6.2f} dB  SIR={metrics.sir:6.2f} dB  "
              f"SAR={metrics.sar:6.2f} dB  ({metrics.n_frames} frames)", flush=True)

        if save_stems:
            dest = Path(save_stems) / track.name
            dest.mkdir(parents=True, exist_ok=True)
            sr = cfg.data.sample_rate
            for j, stem in enumerate(STEMS):
                sf.write(dest / f"{stem}.wav", result["estimate"][j].numpy().T, sr)
                if save_references:
                    sf.write(dest / f"{stem}_reference.wav",
                             result["reference"][j].T.numpy().T, sr)
            sf.write(dest / "mixture.wav", result["mixture"].numpy().T, sr)

    if rows and "si_sdr_mean" in rows[0]:
        summary = {
            "checkpoint": str(checkpoint), "subset": subset, "tracks": len(tracks),
            "metric": "chunk-average SI-SDR (5 s chunks, whole song)",
            "per_track_si_sdr": {row["track"]: row["si_sdr_mean"] for row in rows},
            "median_si_sdr": float(np.median([row["si_sdr_mean"] for row in rows])),
            "seconds_per_track": (time.time() - started) / max(len(tracks), 1),
        }
        if fast_sdrs:
            arr = np.asarray(fast_sdrs)
            summary["per_source_si_sdr"] = {
                name: float(np.median(arr[:, j])) for j, name in enumerate(STEMS)
            }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        with (out_dir / "per_track.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[eval] median SI-SDR over {len(tracks)} songs: "
              f"{summary['median_si_sdr']:.2f} dB  "
              f"({summary['seconds_per_track']:.1f} s/song)", flush=True)
        return summary

    summary = {
        "checkpoint": str(checkpoint), "subset": subset, "tracks": len(tracks),
        "per_track_sdr": {row["track"]: row["sdr"] for row in rows},
        "median_sdr": float(np.median([r["sdr"] for r in rows])),
        "median_sir": float(np.median([r["sir"] for r in rows])),
        "median_sar": float(np.median([r["sar"] for r in rows])),
        "seconds_per_track": (time.time() - started) / max(len(tracks), 1),
    }
    summary["per_source"] = aggregate_bss(per_track, STEMS)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out_dir / "per_track.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["track"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[eval] median over tracks: SDR={summary['median_sdr']:.2f} dB  "
          f"SIR={summary['median_sir']:.2f} dB  SAR={summary['median_sar']:.2f} dB",
          flush=True)
    print(f"[eval] wrote {out_dir / 'summary.json'}", flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--root", default=None, help="dataset root (default: from checkpoint)")
    p.add_argument("--subset", default="test", choices=["test", "train"])
    p.add_argument("--limit", type=int, default=None, help="evaluate only N songs")
    p.add_argument("--device", default="auto")
    p.add_argument("--chunk-seconds", type=float, default=10.0)
    p.add_argument("--window", type=float, default=1.0, help="BSS-Eval window (s)")
    p.add_argument("--hop", type=float, default=1.0, help="BSS-Eval hop (s)")
    p.add_argument("--max-seconds", type=float, default=None,
                   help="evaluate only the first N seconds of each song (faster)")
    p.add_argument("--fast", action="store_true",
                   help="chunk-average SI-SDR instead of BSS-Eval (much faster)")
    p.add_argument("--save-stems", default=None)
    p.add_argument("--save-references", action="store_true",
                   help="also write the ground-truth stems next to the estimates")
    p.add_argument("--out-dir", default="runs/eval")
    args = p.parse_args(argv)

    evaluate_checkpoint(
        args.checkpoint, root=args.root, subset=args.subset, limit=args.limit,
        device_name=args.device, save_stems=args.save_stems,
        save_references=args.save_references,
        chunk_seconds=args.chunk_seconds, window=args.window, hop=args.hop,
        max_seconds=args.max_seconds, fast=args.fast, out_dir=args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
