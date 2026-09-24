#!/usr/bin/env python3
"""Measure step time and memory of the real training configuration on this machine.

    python scripts/benchmark_model.py --chunk-seconds 6 --batches 2 4 6 8

Useful before a long run: it reports seconds/step (hence samples/s) and the peak
memory, so the batch size can be chosen without guessing.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.dsp import stft  # noqa: E402
from src.losses import SeparationLoss  # noqa: E402
from src.train import build_model, pick_device  # noqa: E402


def bench(cfg, device, batch: int, chunk_seconds: float, steps: int = 6) -> float | None:
    model = build_model(cfg, device)
    loss_fn = SeparationLoss(cfg.loss, cfg.stft)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    n = int(chunk_seconds * cfg.data.sample_rate)
    stems = torch.randn(batch, 4, cfg.data.channels, n, device=device) * 0.1
    mixture = stems.sum(dim=1)

    times = []
    try:
        for i in range(steps):
            if device.type == "mps":
                torch.mps.synchronize()
            t0 = time.time()
            out = model(mixture, length=n)
            b, s, ch, _ = stems.shape
            tgt = stft(stems.reshape(b * s, ch, n), cfg.stft)
            tgt = tgt.reshape(b, s, ch, *tgt.shape[-2:])
            loss = loss_fn(out.waveforms, stems, est_spec=out.spectra, tgt_spec=tgt)
            optimizer.zero_grad(set_to_none=True)
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if device.type == "mps":
                torch.mps.synchronize()
            if i > 0:  # skip the first (warm-up / allocation) step
                times.append(time.time() - t0)
    except RuntimeError as exc:
        print(f"  batch={batch:>2}  FAILED: {str(exc)[:90]}")
        del model, optimizer, stems, mixture
        if device.type == "mps":
            torch.mps.empty_cache()
        return None

    mean = sum(times) / max(len(times), 1)
    per_second_of_audio = mean / (batch * chunk_seconds)
    mem = ""
    if device.type == "mps":
        mem = f"  peak_alloc={torch.mps.driver_allocated_memory() / 1e9:5.2f} GB"
    print(f"  batch={batch:>2}  {mean * 1000:7.1f} ms/step  "
          f"{batch * chunk_seconds / mean:6.2f} s_audio/s{mem}")
    del model, optimizer, stems, mixture
    if device.type == "mps":
        torch.mps.empty_cache()
    return mean


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/unet_musdb18.yaml")
    p.add_argument("--batches", type=int, nargs="+", default=[2, 4, 6, 8])
    p.add_argument("--chunk-seconds", type=float, default=None)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="config override, e.g. --set stft.hop_length=1024")
    args = p.parse_args()

    overrides: dict[str, object] = {}
    for item in args.set:
        key, _, value = item.partition("=")
        try:
            parsed: object = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        overrides[key] = parsed
    cfg = load_config(args.config, **overrides)
    if args.chunk_seconds is not None:
        cfg.data.chunk_seconds = args.chunk_seconds
    device = pick_device(cfg.train.device)
    n_params = sum(p.numel() for p in build_model(cfg, device).parameters())
    frames = int(cfg.data.chunk_seconds * cfg.data.sample_rate / cfg.stft.hop_length) + 1
    print(f"device={device}  model={n_params / 1e6:.2f}M params  chunk={cfg.data.chunk_seconds}s"
          f"  n_fft={cfg.stft.n_fft} hop={cfg.stft.hop_length} "
          f"bins={cfg.stft.n_fft // 2 + 1} frames={frames} "
          f"base={cfg.model.base_channels} depth={cfg.model.depth}")
    for batch in args.batches:
        bench(cfg, device, batch, cfg.data.chunk_seconds, args.steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
