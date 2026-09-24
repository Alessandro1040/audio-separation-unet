#!/usr/bin/env python3
"""Am I running the right RoFormer checkpoint, and is it actually separating?

    python scripts/check_roformer.py --checkpoint checkpoints_roformer/best.pt
    python scripts/check_roformer.py --checkpoint ... --input my_song.mp3

The RoFormer counterpart of `scripts/check_separation.py`: it prints the numbers that tell a
working separator apart from a model that only re-levelled the mixture.

    checkpoint      which architecture / step / how many mel bands are on disk
    mean sum of masks   ~1.0 = calibrated mask set (the four masks add up to ~1 per bin)
    stem similarity     ~0.6 = separating; ~0.9+ = all four stems are the same audio
    band tilt           the bass mask must be heavier below 200 Hz than above 2 kHz

Nothing here needs the MUSDB18 download: by default it runs on the repository's own sample.
"""
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import STEMS  # noqa: E402
from src.dsp import load_audio  # noqa: E402
from src.train import pick_device  # noqa: E402
from src.train_roformer import ARCH, load_roformer_checkpoint  # noqa: E402

SAMPLE = "samples/Al James - Schoolboy Facination - mixture (input).mp3"
BANDS = {"low_<200Hz": (0.0, 200.0), "mid_200Hz-2kHz": (200.0, 2000.0),
         "high_>2kHz": (2000.0, 1e9)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", default=SAMPLE, help="audio file to analyse")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--seconds", type=float, default=10.0, help="how much audio to analyse")
    p.add_argument("--out", default=None, help="output folder (default: a temp dir)")
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="check-roformer-"))
    device = pick_device(args.device)
    model, cfg = load_roformer_checkpoint(args.checkpoint, device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "unknown"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"commit           : {commit}")
    print(f"checkpoint       : {args.checkpoint}")
    print(f"                   arch={ckpt.get('arch', '?')} (expected '{ARCH}') | "
          f"step {ckpt.get('step')} | params {n_params / 1e6:.2f}M | "
          f"bands {model.net.n_bands} x {model.net.band_width} bins | "
          f"dim {cfg.model.dim} | extra_n_fft {cfg.model.n_fft_extra or 'off'}")

    mixture = load_audio(args.input, cfg.data.sample_rate, cfg.data.channels)
    n = min(int(args.seconds * cfg.data.sample_rate), mixture.shape[-1])
    chunk = mixture[..., :n].unsqueeze(0).to(device)
    with torch.no_grad():
        masks, _ = model.predict_masks(chunk)
        estimates = model(chunk, length=n).waveforms[0]
    masks = masks[0].detach().cpu()                     # (S, ch, F, T)
    n_sums = float(masks.sum(dim=0).abs().mean())
    print(f"\nmean sum of masks: {n_sums:.2f}"
          "   (~1.0 = calibrated mask set, <0.8 = re-levelling)")

    freqs = torch.linspace(0.0, cfg.data.sample_rate / 2.0, masks.shape[-2])
    per_band = {}
    for name, (lo, hi) in BANDS.items():
        sel = (freqs >= lo) & (freqs < hi)
        per_band[name] = masks[:, :, sel, :].abs().mean(dim=(1, 2, 3))
    print("\nmean |mask| per band")
    print(f"{'stem':<10}" + "".join(f"{k:>16}" for k in BANDS))
    for i, stem in enumerate(STEMS):
        print(f"{stem:<10}" + "".join(f"{float(per_band[k][i]):>16.2f}" for k in BANDS))
    bass = per_band["low_<200Hz"][STEMS.index("bass")] / \
        per_band["high_>2kHz"][STEMS.index("bass")].clamp_min(1e-6)
    print(f"\nbass low/high ratio: {float(bass):.2f}"
          "   (on MUSDB a bass mask should be > 1: low-heavy)")

    audio: dict[str, np.ndarray] = {}
    for i, stem in enumerate(STEMS):
        data = estimates[i].cpu().numpy().T
        sf.write(out / f"{stem}.wav", data, cfg.data.sample_rate)
        audio[stem] = data
    sim = [float(np.dot(audio[x].ravel(), audio[y].ravel()) /
                 (np.linalg.norm(audio[x]) * np.linalg.norm(audio[y]) + 1e-12))
           for x, y in itertools.combinations(STEMS, 2)]
    print(f"stem similarity  : {float(np.mean(sim)):.2f}"
          "   (~0.6 = separating, ~0.9 = the same audio four times)")
    print(f"\nstems for listening: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
