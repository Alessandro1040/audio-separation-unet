"""Time each stage of a training step on the chosen device."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.dsp import istft, stft  # noqa: E402
from src.models.separation import SpectrogramSeparator  # noqa: E402


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def timeit(label, fn, device, n=3):
    fn()
    sync(device)
    t0 = time.time()
    for _ in range(n):
        out = fn()
    sync(device)
    dt = (time.time() - t0) / n
    print(f"  {label:<42} {dt * 1000:9.1f} ms", flush=True)
    return dt, out


def main() -> int:
    cfg = load_config("configs/unet_musdb18.yaml")
    device = torch.device(sys.argv[1] if len(sys.argv) > 1 else "mps")
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    chunk = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
    n = int(chunk * cfg.data.sample_rate)
    print(f"device={device} batch={batch} chunk={chunk}s samples={n}")

    mixture = torch.randn(batch, 2, n, device=device) * 0.1
    stems = torch.randn(batch, 4, 2, n, device=device) * 0.1
    model = SpectrogramSeparator(cfg.stft, cfg.model).to(device).train()

    print("--- forward pieces (no grad)")
    with torch.no_grad():
        timeit("stft(mixture)", lambda: stft(mixture, cfg.stft), device)
        spec = stft(mixture, cfg.stft)
        timeit("istft(spec)  [B,ch,F,T]", lambda: istft(spec, cfg.stft, n), device)
        s4 = stft(stems.reshape(batch * 4, 2, n), cfg.stft)
        timeit("istft(stems) [B*4,ch,F,T]", lambda: istft(s4, cfg.stft, n), device)
        timeit("unet forward", lambda: model.predict_masks(mixture), device)
        timeit("full forward", lambda: model(mixture, length=n), device)

    print("--- full training step (grad)")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.L1Loss()

    def step():
        out = model(mixture, length=n)
        loss = loss_fn(out.waveforms, stems) + loss_fn(out.spectra.abs(), torch.zeros_like(out.spectra.real))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss

    try:
        timeit("forward + backward + step", step, device, n=2)
    except RuntimeError as exc:
        print(f"  step FAILED: {str(exc)[:120]}")
    if device.type == "mps":
        print(f"  driver_allocated={torch.mps.driver_allocated_memory() / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
