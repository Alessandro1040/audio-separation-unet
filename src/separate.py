#!/usr/bin/env python3
"""Separate a song with a trained U-Net.

    python -m src.separate --checkpoint checkpoints/best.pt --input song.mp3 \
        --out outputs/song --stems vocals drums bass other

The result is one 44.1 kHz stereo wav per stem plus `mixture.wav`, ready to listen to.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .config import Config, STEMS, load_config
from .dsp import load_audio
from .models.separation import SpectrogramSeparator, separate_long
from .train import build_model, pick_device


def load_model(checkpoint: str | Path, device: torch.device,
               strict_config: bool = True) -> tuple[SpectrogramSeparator, Config]:
    """Rebuild the model exactly as it was trained.

    Understands both a training checkpoint (`model` / `ema` + optimizer state) and the
    small inference-only export produced by `scripts/export_inference_checkpoint.py`
    (`state_dict`, optionally float16).
    """
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = config_from_payload(ckpt)
    state = ckpt.get("state_dict") or ckpt.get("ema") or ckpt.get("model")
    if state is None:
        raise KeyError(f"{checkpoint} contains no weights (keys: {list(ckpt)})")
    if all(t.dtype == torch.float16 for t in state.values()
           if torch.is_floating_point(t)):
        state = {k: (v.float() if torch.is_floating_point(v) else v)
                 for k, v in state.items()}
    model = SpectrogramSeparator(cfg.stft, cfg.model, channels=cfg.data.channels,
                                 n_sources=len(STEMS))
    model.load_state_dict(state)
    return model.to(device).eval(), cfg


def config_from_payload(ckpt: dict) -> Config:
    """Resolve the `Config` embedded in a checkpoint (training or inference format)."""
    if not ckpt.get("config"):
        raise KeyError("checkpoint has no stored config")
    from .config import _SECTIONS, _dataclass_from_dict

    cfg = Config()
    for section, cls in _SECTIONS.items():
        values = dict(ckpt["config"].get(section) or {})
        if section == "data" and "aug" in values:
            from .config import AugmentConfig

            cfg.data.aug = _dataclass_from_dict(AugmentConfig, values.pop("aug"))
        setattr(cfg, section, _dataclass_from_dict(cls, values))
    return cfg


def separate_file(model: SpectrogramSeparator, cfg: Config, path: str | Path,
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
    p.add_argument("--chunk-seconds", type=float, default=10.0)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--also-mixture", action="store_true",
                   help="write the input mixture next to the stems")
    args = p.parse_args(argv)

    device = pick_device(args.device)
    model, cfg = load_model(args.checkpoint, device)
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
