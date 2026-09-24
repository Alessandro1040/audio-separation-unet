"""Training loop for the Mel-band RoFormer (`python -m src.train_roformer`).

It intentionally mirrors `src/train.py`: same data pipeline, same augmentation, same
warm-up + cosine schedule, same EMA, same log columns and the same checkpoint format
(`model` / `ema` / `optimizer` / `step` / `config`), plus an `arch` marker that tells the
two model families apart. Everything that is architecture independent is *imported* from
`src.train` instead of copied - `lr_at`, `EMA`, `evaluate_chunks`, `build_loaders`,
`pick_device` - so a fix in one trainer is a fix in both, and the two runs stay comparable.

What is genuinely different:

  * the model is `RoFormerSeparator` (band split + RoPE attention);
  * the loss is `MRSTFTLoss` (multi-resolution STFT + waveform L1);
  * the checkpoint records `arch: melband_roformer`.

A fair A/B against the U-Net means running this file and `src/train.py` with the same
`data`/`train`/`loss` sections - `scripts/ab_roformer_vs_unet.sh` does exactly that.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .config import STEMS
from .config_roformer import (
    RoFormerConfig,
    config_from_payload,
    load_roformer_config,
    save_roformer_config,
)
from .losses_mrstft import MRSTFTLoss
from .models.roformer_separation import RoFormerSeparator
from .train import EMA, build_loaders, evaluate_chunks, lr_at, pick_device

ARCH = "melband_roformer"


def build_model(cfg: RoFormerConfig, device: torch.device) -> RoFormerSeparator:
    model = RoFormerSeparator(
        cfg.stft, cfg.model, channels=cfg.data.channels, n_sources=len(STEMS),
        sample_rate=cfg.data.sample_rate,
    )
    return model.to(device)


def load_roformer_checkpoint(path: str | Path, device: torch.device,
                             weights: str = "ema") -> tuple[RoFormerSeparator, RoFormerConfig]:
    """Rebuild a RoFormer (and its config) from a checkpoint written by this file.

    `weights`: "ema" (the moving average, the default everywhere else in the repository),
    "live" (the raw trained weights) or "auto" (whichever the checkpoint has).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("arch") not in (None, ARCH):
        raise KeyError(f"{path} holds a '{ckpt.get('arch')}' checkpoint, not a RoFormer")
    cfg = config_from_payload(ckpt.get("config") or {})
    state = ckpt.get("ema") if weights in ("ema", "auto") else ckpt.get("model")
    if state is None:
        state = ckpt.get("model") or ckpt.get("state_dict")
    if state is None:
        raise KeyError(f"{path} contains no weights (keys: {list(ckpt)})")
    model = build_model(cfg, torch.device("cpu"))
    model.load_state_dict(state)
    return model.to(device).eval(), cfg


def _log_row(writer: csv.writer, row: dict) -> None:
    writer.writerow(row)
    message = "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in row.items() if k != "step")
    print(f"[train] step {row['step']:>6}  {message}", flush=True)


def _validate_and_checkpoint(model, ema, optimizer, valid_loader, loss_fn, cfg, device,
                             step, writer, log_file, ckpt_dir, best_sdr) -> dict[str, float]:
    """Validation, log row and the two checkpoints - the same format `src/train.py` writes."""
    eval_model = ema.shadow if ema is not None else model
    metrics = evaluate_chunks(eval_model, valid_loader, loss_fn, cfg, device)
    print(f"[valid] step {step:>6}  " + "  ".join(
        f"{k}={v:.2f} dB" if k.startswith("sdr") else f"{k}={v:.4f}"
        for k, v in metrics.items()), flush=True)
    writer.writerow({"step": step, "epoch": step // max(cfg.train.steps_per_epoch, 1),
                     **metrics})
    log_file.flush()
    state = {
        "model": model.state_dict(),
        "ema": ema.shadow.state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_sdr": max(best_sdr, metrics["sdr_mean"]),
        "config": asdict(cfg),
        "stems": STEMS,
        "arch": ARCH,
    }
    torch.save(state, Path(ckpt_dir) / "last.pt")
    if metrics["sdr_mean"] > best_sdr:
        torch.save(state, Path(ckpt_dir) / "best.pt")
        print(f"[valid] new best mean SI-SDR {metrics['sdr_mean']:.2f} dB "
              f"-> {Path(ckpt_dir) / 'best.pt'}", flush=True)
    return metrics


def train(cfg: RoFormerConfig, resume: str | Path | None = None) -> Path:
    """Run the whole training schedule; returns the path of the best checkpoint."""
    torch.manual_seed(cfg.train.seed)
    device = pick_device(cfg.train.device)
    run_dir = Path(cfg.train.out_dir)
    ckpt_dir = Path(cfg.train.ckpt_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_roformer_config(cfg, run_dir / "config.yaml")

    model = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    n_bands = model.net.n_bands
    print(f"[train] device={device} arch={ARCH} params={n_params / 1e6:.2f}M "
          f"bands={n_bands} band_width={model.net.band_width} chunk={cfg.data.chunk_seconds}s",
          flush=True)

    loss_fn = MRSTFTLoss(cfg.loss, cfg.stft)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    # same policy as the U-Net trainer: autocast only on CUDA (MPS autocast is unreliable
    # with complex-valued kernels, and this model works on complex spectra throughout)
    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    if cfg.train.amp and not use_amp:
        print(f"[train] AMP requested but disabled for device '{device.type}'", flush=True)
    autocast = torch.autocast(device_type=device.type, dtype=torch.float16,
                              enabled=use_amp)
    ema = EMA(model, cfg.train.ema_decay) if cfg.train.ema_decay > 0 else None
    train_loader, valid_loader = build_loaders(cfg)

    total_steps = cfg.train.epochs * cfg.train.steps_per_epoch
    step = 0
    start_step = 0
    best_sdr = -1e9
    if resume is not None:
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        if ema is not None and ckpt.get("ema") is not None:
            ema.shadow.load_state_dict(ckpt["ema"])
        if ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        step = int(ckpt.get("step", 0))
        start_step = step
        best_sdr = float(ckpt.get("best_sdr", -1e9))
        print(f"[train] resumed from {resume} at step {step} "
              f"(best SI-SDR {best_sdr:.2f} dB)", flush=True)
    log_path = run_dir / "log.csv"
    new_log = not log_path.exists()
    log_file = log_path.open("a", newline="")
    writer = csv.DictWriter(
        log_file,
        fieldnames=["step", "epoch", "loss", "loss_wave", "loss_mag", "lr",
                    "sec_per_step", *[f"sdr_{s}" for s in STEMS], "sdr_mean", "val_loss"],
        extrasaction="ignore",
    )
    if new_log:
        writer.writeheader()

    model.train()
    started = time.time()
    running: dict[str, float] = {}
    data_iter = iter(train_loader)

    while step < total_steps:
        if cfg.train.max_hours is not None and \
                (time.time() - started) / 3600.0 >= cfg.train.max_hours:
            print("[train] time budget reached, stopping cleanly", flush=True)
            break
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        t0 = time.time()
        mixture = batch["mixture"].to(device)
        stems = batch["stems"].to(device)
        lr = lr_at(step, total_steps, cfg, start_step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        with autocast:
            out = model(mixture, length=mixture.shape[-1],
                        consistent=cfg.loss.consistency)
            loss = loss_fn(out.waveforms, stems, est_spec=out.spectra)
        optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        if cfg.train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(model)

        step += 1
        for key, value in loss.as_dict().items():
            running[key] = running.get(key, 0.0) + value
        running["sec_per_step"] = running.get("sec_per_step", 0.0) + (time.time() - t0)

        if step % cfg.train.log_every == 0:
            row = {k: v / cfg.train.log_every for k, v in running.items()}
            row["lr"] = lr
            row["step"] = step
            row["epoch"] = step // max(cfg.train.steps_per_epoch, 1)
            _log_row(writer, row)
            log_file.flush()
            running = {}

        if step % cfg.train.val_every == 0 or step == total_steps:
            metrics = _validate_and_checkpoint(
                model, ema, optimizer, valid_loader, loss_fn, cfg, device, step,
                writer, log_file, ckpt_dir, best_sdr,
            )
            best_sdr = max(best_sdr, metrics["sdr_mean"])

    log_file.close()
    elapsed = (time.time() - started) / 3600
    print(f"[train] done: {step} steps in {elapsed:.2f} h, "
          f"best mean SI-SDR {best_sdr:.2f} dB", flush=True)
    (run_dir / "summary.json").write_text(json.dumps(
        {"arch": ARCH, "steps": step, "hours": elapsed, "best_sdr": best_sdr,
         "params": n_params, "bands": n_bands, "device": str(device)}, indent=2))
    return ckpt_dir / "best.pt"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                        help="override a config value, e.g. --set train.batch_size=2")
    args = parser.parse_args(argv)

    overrides: dict[str, object] = {}
    for item in args.set:
        key, _, value = item.partition("=")
        try:
            parsed: object = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        overrides[key] = parsed
    cfg = load_roformer_config(args.config, **overrides)
    train(cfg, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
