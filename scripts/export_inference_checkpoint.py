#!/usr/bin/env python3
"""Export a small, self-contained inference checkpoint from a training checkpoint.

    python scripts/export_inference_checkpoint.py checkpoints/best.pt \
        --out models/unet_musdb18_ema.pt

Training checkpoints keep the optimizer state (two Adam moments) and the *live* weights,
which makes them ~160 MB. For inference only the EMA weights and the config are needed:
this writes them in float32 (~42 MB, so it fits in a plain GitHub repository without
Git LFS) together with the metrics measured for that checkpoint.

`--weights live` exports the raw weights instead. The EMA is the better choice for a long
run, but with `ema_decay: 0.999` a short run's average still contains a lot of the random
initialisation, and then the live weights can win by several dB (`scripts/
diagnose_checkpoint.py` reports both).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def pick_weights(ckpt: dict, choice: str) -> tuple[dict, str]:
    """Return (state_dict, label) for the requested weights."""
    has_ema = bool(ckpt.get("ema"))
    if choice == "ema" and not has_ema:
        raise SystemExit("this checkpoint has no EMA weights; use --weights live")
    use_live = choice == "live" or (choice == "auto" and not has_ema)
    return (ckpt["model"], "live") if use_live else (ckpt["ema"], "ema")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("checkpoint")
    p.add_argument("--out", required=True)
    p.add_argument("--metrics", default=None,
                   help="optional JSON file with evaluation numbers to embed")
    p.add_argument("--weights", default="auto", choices=("auto", "ema", "live"),
                   help="which weights to export: 'auto' = EMA if the checkpoint has "
                        "them, otherwise the live weights")
    p.add_argument("--half", action="store_true",
                   help="store the weights in float16 (halves the size, cast back to "
                        "float32 at load time)")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    weights, label = pick_weights(ckpt, args.weights)
    if args.half:
        weights = {k: (v.half() if torch.is_floating_point(v) else v)
                   for k, v in weights.items()}
    payload = {
        "format": "spectrogram-unet-separator/1",
        "weights": label,
        "dtype": "float16" if args.half else "float32",
        "state_dict": weights,
        "config": ckpt["config"],
        "stems": ckpt.get("stems", ["vocals", "drums", "bass", "other"]),
        "step": ckpt.get("step"),
        "validation_si_sdr_db": ckpt.get("best_sdr"),
        "dataset": "MUSDB18 (train split, 86 songs; 14 held out for validation)",
    }
    if args.metrics and Path(args.metrics).exists():
        payload["evaluation"] = json.loads(Path(args.metrics).read_text())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out} ({size_mb:.1f} MB, weights={payload['weights']}, "
          f"dtype={payload['dtype']}, step={payload['step']})")
    if size_mb > 95:
        print("WARNING: over GitHub's 100 MB file limit; use --half or Git LFS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
