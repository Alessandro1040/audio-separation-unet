#!/usr/bin/env python3
"""Write the procedural dataset used for smoke tests (MUSDB18 directory layout)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.synthetic import generate_dataset  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=Path("data/synthetic"))
    p.add_argument("--train", type=int, default=4)
    p.add_argument("--test", type=int, default=2)
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--sample-rate", type=int, default=22050)
    args = p.parse_args()
    root = generate_dataset(args.out, args.train, args.test, args.seconds,
                            args.sample_rate)
    print(f"wrote synthetic dataset to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
