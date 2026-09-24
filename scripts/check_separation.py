#!/usr/bin/env python3
"""Am I running the right checkpoint, and is it actually separating?

    python scripts/check_separation.py                    # the 24 s example in samples/
    python scripts/check_separation.py --input my_song.mp3

It separates the input, then prints the numbers that tell a working model apart from one that
only re-levelled the mixture:

    commit / step / validation SI-SDR   which checkpoint is on disk right now
    mean sum of masks                   ~1.0 = calibrated mask set, ~0.74 = re-levelling
    stem similarity                     ~0.6 = separating, ~0.88 = the step-1200 checkpoint
                                        (its worst pair was 0.98, i.e. the same audio)
    bass mask low/high                  a bass mask must be heavier below 200 Hz

Nothing here needs the MUSDB18 download: it runs on the repository's own sample.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SAMPLE = "samples/Al James - Schoolboy Facination - mixture (input).mp3"
DEFAULT_CKPT = "models/unet_musdb18_ema.pt"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", default=SAMPLE, help="audio file to analyse")
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--out", default=None, help="output folder (default: a temp dir)")
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="check-sep-"))

    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "unknown"
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"commit           : {commit}")
    print(f"checkpoint       : {args.checkpoint}")
    print(f"                   step {ckpt.get('step')} | validation SI-SDR "
          f"{ckpt.get('validation_si_sdr_db', float('nan')):.2f} dB | "
          f"{ckpt.get('weights')} {ckpt.get('dtype')}")
    if (ckpt.get("step") or 0) < 2250:
        print("                   *** this is older than the retrained checkpoint (step 2250) ***")

    cmd = [sys.executable, "-m", "src.analyze", "--checkpoint", str(args.checkpoint),
           "--input", str(args.input), "--out", str(out), "--no-figures",
           "--bss-seconds", "1"]
    subprocess.run(cmd, cwd=repo, check=True)

    stems = {s: sf.read(out / f"{s}.wav", always_2d=True)[0]
             for s in ("vocals", "drums", "bass", "other")}
    sim = [float(np.dot(stems[a].ravel(), stems[b].ravel()) /
                 (np.linalg.norm(stems[a]) * np.linalg.norm(stems[b]) + 1e-12))
           for a, b in itertools.combinations(stems, 2)]
    stats = json.loads((out / "report.json").read_text())["mask_statistics"]
    bass = stats["mean_abs_mask_per_frequency_band"]["bass"]
    print(f"\nmean sum of masks: {stats['mean_sum_of_masks']:.2f}"
          "   (~1.0 = calibrated mask set, ~0.74 = re-levelling)")
    print(f"stem similarity  : {float(np.mean(sim)):.2f}"
          "   (~0.6 = separating, ~0.88 = the step-1200 checkpoint)")
    print(f"bass low/high    : {bass['low_<200Hz']:.2f} / {bass['high_>2kHz']:.2f}"
          "   (a bass mask must be heavier at low)")
    print(f"\nstems for listening: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
