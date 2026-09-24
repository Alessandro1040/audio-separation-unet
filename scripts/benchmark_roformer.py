#!/usr/bin/env python3
"""Measure step time and memory of the Mel-band RoFormer on this machine.

    python scripts/benchmark_roformer.py --batches 1 2 4 --chunk-seconds 3

The counterpart of `scripts/benchmark_model.py` for the second architecture: it reports
seconds/step and the allocator peak, so the config to use for a long run is chosen from
measurements instead of guesses. On an Apple M5 / 16 GB the numbers in
`configs/roformer_musdb18.yaml` come from this script.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config_roformer import load_roformer_config  # noqa: E402
from src.losses_mrstft import MRSTFTLoss  # noqa: E402
from src.train import pick_device  # noqa: E402
from src.train_roformer import build_model  # noqa: E402


def bench(cfg, device, batch: int, chunk_seconds: float, steps: int = 6) -> float | None:
    model = build_model(cfg, device)
    loss_fn = MRSTFTLoss(cfg.loss, cfg.stft)
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
            out = model(mixture, length=n, consistent=cfg.loss.consistency)
            loss = loss_fn(out.waveforms, stems, est_spec=out.spectra)
            optimizer.zero_grad(set_to_none=True)
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if device.type == "mps":
                torch.mps.synchronize()
            if i > 0:  # skip the first (allocation / Metal shader compilation) step
                times.append(time.time() - t0)
    except RuntimeError as exc:
        print(f"  batch={batch:>2}  FAILED: {str(exc)[:90]}")
        del model, optimizer, stems, mixture
        if device.type == "mps":
            torch.mps.empty_cache()
        return None

    mean = sum(times) / max(len(times), 1)
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
    p.add_argument("--config", default="configs/roformer_musdb18.yaml")
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--chunk-seconds", type=float, default=None)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="config override, e.g. --set model.dim=192")
    args = p.parse_args()

    overrides: dict[str, object] = {}
    for item in args.set:
        key, _, value = item.partition("=")
        try:
            parsed: object = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        overrides[key] = parsed
    cfg = load_roformer_config(args.config, **overrides)
    if args.chunk_seconds is not None:
        cfg.data.chunk_seconds = args.chunk_seconds
    device = pick_device(cfg.train.device)
    model = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    frames = int(cfg.data.chunk_seconds * cfg.data.sample_rate / cfg.stft.hop_length) + 1
    print(f"device={device}  model={n_params / 1e6:.2f}M params  chunk={cfg.data.chunk_seconds}s"
          f"  n_fft={cfg.stft.n_fft} hop={cfg.stft.hop_length} "
          f"bins={cfg.stft.n_bins} frames={frames}  bands={model.net.n_bands} "
          f"band_width={model.net.band_width} dim={cfg.model.dim} depth={cfg.model.depth} "
          f"extra_n_fft={cfg.model.n_fft_extra or 'off'}")
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    for batch in args.batches:
        bench(cfg, device, batch, cfg.data.chunk_seconds, args.steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
