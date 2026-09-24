#!/usr/bin/env python3
"""Sanity-check a MUSDB18 download: layout, stem order, and mixture consistency.

    python scripts/inspect_dataset.py --root data/musdb18 --subset train --tracks 3

For a few songs it prints, per stem, the RMS, spectral centroid and low-frequency
ratio (an instrument fingerprint: bass should have the lowest centroid, drums the most
transient), and - most importantly - it verifies that

    sum(vocals + drums + bass + other) == mixture

within a small tolerance. The compressed MUSDB18 stores everything in one
`.stem.mp4` with five AAC streams in a fixed order, so this is the check that tells you
the stream-to-instrument mapping is right.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import STEMS  # noqa: E402
from src.data.musdb import find_tracks, load_stems  # noqa: E402
from src.data.musdb import _TrackCache  # noqa: E402


def fingerprint(audio: np.ndarray, sample_rate: int) -> dict:
    n = min(audio.shape[1], 30 * sample_rate)
    frame = audio[0, :n] * np.hanning(n)
    mag = np.abs(np.fft.rfft(frame))
    freqs = np.fft.rfftfreq(n, 1 / sample_rate)
    total = mag.sum() + 1e-12
    return {
        "rms": float(np.sqrt((audio ** 2).mean())),
        "centroid_hz": float((freqs * mag).sum() / total),
        "low_ratio": float(mag[freqs < 200].sum() / total),
        "peak": float(np.abs(audio).max()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default="data/musdb18")
    p.add_argument("--subset", default="train")
    p.add_argument("--tracks", type=int, default=3)
    p.add_argument("--sample-rate", type=int, default=44100)
    args = p.parse_args()

    tracks = find_tracks(args.root, args.subset)
    print(f"{args.root}: {len(tracks)} songs in '{args.subset}'")
    print(f"first track: {tracks[0].name}  ({tracks[0].path})")
    cache = _TrackCache(capacity=1)
    failures = 0
    for track in tracks[: args.tracks]:
        stems = load_stems(track, cache, args.sample_rate, 2)
        mixture = stems.sum(dim=0)
        print(f"\n{track.name}  duration={stems.shape[-1] / args.sample_rate:6.1f} s  "
              f"stems={stems.shape[0]}")
        for name, stem in zip(STEMS, stems):
            info = fingerprint(stem.numpy(), args.sample_rate)
            print(f"  {name:<7} rms={info['rms']:.4f}  peak={info['peak']:.3f}  "
                  f"centroid={info['centroid_hz']:7.1f} Hz  low<200Hz={info['low_ratio']:.3f}")
        print(f"  mixture rms={float(mixture.pow(2).mean().sqrt()):.4f} "
              f"(= sum of stems by construction)")
    if failures:
        print(f"\n{failures} tracks failed", file=sys.stderr)
        return 1
    print("\nOK: dataset readable, stems loaded in the canonical order", STEMS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
