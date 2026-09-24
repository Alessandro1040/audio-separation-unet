#!/usr/bin/env python3
"""Separate a song with a trained Mel-band RoFormer.

    python -m src.separate_roformer --checkpoint checkpoints_roformer/best.pt \
        --input song.mp3 --out outputs/song_roformer --also-mixture

Same output contract as `python -m src.separate` (the U-Net path): one 44.1 kHz stereo wav
per stem plus `mixture.wav`, produced by the same sliding-window overlap-add
(`src.models.separation.separate_long`), which works on any model exposing `n_sources` and
a `forward(mixture, length=...)` returning `.waveforms`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .config import STEMS
from .config_roformer import RoFormerConfig
from .dsp import load_audio
from .models.roformer_separation import RoFormerSeparator
from .models.separation import separate_long
from .train import pick_device
from .train_roformer import load_roformer_checkpoint


def separate_file(model: RoFormerSeparator, cfg: RoFormerConfig, path: str | Path,
                  device: torch.device, chunk_seconds: float = 10.0,
                  overlap: float = 0.5) -> torch.Tensor:
    """(n_stems, channels, samples) estimates for one audio file."""
    mixture = load_audio(path, cfg.data.sample_rate, cfg.data.channels).to(device)
    return separate_long(model, mixture, cfg.data.sample_rate, chunk_seconds, overlap)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True, help="audio file (wav/mp3/flac/...)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--device", default="auto")
    p.add_argument("--weights", default="ema", choices=["ema", "live", "auto"],
                   help="ema (default) = the moving-average weights, live = raw weights")
    p.add_argument("--chunk-seconds", type=float, default=10.0)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--also-mixture", action="store_true",
                   help="write the input mixture next to the stems")
    args = p.parse_args(argv)

    device = pick_device(args.device)
    model, cfg = load_roformer_checkpoint(args.checkpoint, device, weights=args.weights)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    estimates = separate_file(model, cfg, args.input, device, args.chunk_seconds,
                              args.overlap)
    sr = cfg.data.sample_rate
    for i, stem in enumerate(STEMS):
        audio = estimates[i].detach().cpu().numpy().T
        sf.write(out_dir / f"{stem}.wav", audio, sr)
        print(f"wrote {out_dir / f'{stem}.wav'}  rms={float(np.sqrt((audio ** 2).mean())):.4f}")
    if args.also_mixture:
        mixture = load_audio(args.input, sr, cfg.data.channels).numpy().T
        sf.write(out_dir / "mixture.wav", mixture, sr)
        print(f"wrote {out_dir / 'mixture.wav'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
