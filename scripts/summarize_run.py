#!/usr/bin/env python3
"""Print a compact table of a training run from its CSV log.

    python scripts/summarize_run.py runs/unet/log.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import STEMS  # noqa: E402


def load_rows(path: Path) -> list[dict]:
    with path.open() as fh:
        return [row for row in csv.DictReader(fh)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("log", nargs="?", default="runs/unet/log.csv")
    p.add_argument("--every", type=int, default=1, help="keep every Nth validation row")
    args = p.parse_args()

    rows = load_rows(Path(args.log))
    train_rows = [r for r in rows if r.get("loss")]
    valid_rows = [r for r in rows if r.get("sdr_mean")]

    print(f"{args.log}: {len(train_rows)} training rows, {len(valid_rows)} validations\n")
    header = f"{'step':>7} {'loss':>7} {'wave':>7} {'mag':>7} {'s/step':>7} " + \
             " ".join(f"{s[:5]:>7}" for s in STEMS) + f" {'mean':>7}"
    print(header)
    print("-" * len(header))
    for i, row in enumerate(valid_rows):
        if i % args.every:
            continue
        step = int(row["step"])
        near = min(train_rows, key=lambda r: abs(int(r["step"]) - step))
        sdr = [float(row[f"sdr_{s}"]) for s in STEMS]
        print(f"{step:>7} {float(near['loss']):>7.4f} {float(near['loss_wave']):>7.4f} "
              f"{float(near['loss_mag']):>7.4f} {float(near['sec_per_step']):>7.2f} "
              + " ".join(f"{v:>7.2f}" for v in sdr) + f" {float(row['sdr_mean']):>7.2f}")

    if valid_rows:
        best = max(valid_rows, key=lambda r: float(r["sdr_mean"]))
        print(f"\nbest validation mean SI-SDR: {float(best['sdr_mean']):.2f} dB "
              f"at step {best['step']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
