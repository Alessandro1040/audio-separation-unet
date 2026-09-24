"""Training loop: train the U-Net to segment spectrograms into instrument masks.

Recipe (tuned for a laptop-class GPU / Apple MPS):
  * random 6-second chunks with heavy augmentation (only ~100 training songs!)
  * complex-ratio masks predicted by a 5-level U-Net, mixture consistency enforced
  * hybrid loss: waveform L1 + L1 on compressed magnitudes
  * AdamW with warm-up + cosine decay, gradient clipping, EMA of the weights
  * model selection on validation SI-SDR (14 held-out songs), checkpointing, resume
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import Config, STEMS, load_config, save_config
from .data.chunks import (
    MusdbEvalChunks,
    MusdbTrainIterable,
    collate_chunks,
    validation_tracks,
)
from .dsp import stft
from .losses import SeparationLoss
from .metrics import si_sdr_per_source
from .models.separation import SpectrogramSeparator


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(cfg: Config, device: torch.device) -> SpectrogramSeparator:
    model = SpectrogramSeparator(
        cfg.stft, cfg.model, channels=cfg.data.channels, n_sources=len(STEMS)
    )
    return model.to(device)


class EMA:
    """Exponential moving average of the weights - a cheap, reliable quality boost."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)


def lr_at(step: int, total: int, cfg: Config) -> float:
    """Linear warm-up, then cosine decay to 5% of the peak learning rate."""
    warmup = max(cfg.train.warmup_steps, 1)
    if step < warmup:
        return cfg.train.lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(total - warmup, 1))
    return cfg.train.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress)))


def build_loaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
    train_set = MusdbTrainIterable(cfg.data, seed=cfg.train.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_chunks,
        prefetch_factor=2 if cfg.data.num_workers > 0 else None,
        persistent_workers=cfg.data.num_workers > 0,
        pin_memory=False,
    )
    valid_tracks_list = validation_tracks(cfg.data)
    per_track = max(1, cfg.train.val_chunks // max(len(valid_tracks_list), 1))
    valid_set = MusdbEvalChunks(
        cfg.data, tracks=valid_tracks_list, chunks_per_track=per_track
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=cfg.train.batch_size,
        num_workers=0,
        collate_fn=collate_chunks,
    )
    return train_loader, valid_loader


@torch.no_grad()
def evaluate_chunks(
    model: SpectrogramSeparator,
    loader: DataLoader,
    loss_fn: SeparationLoss,
    cfg: Config,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Validation loss and SI-SDR per source over deterministic chunks."""
    model.eval()
    sdr_sum = torch.zeros(len(STEMS))
    n_batches = 0
    loss_sum = 0.0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        mixture = batch["mixture"].to(device)
        stems = batch["stems"].to(device)
        out = model(mixture, length=mixture.shape[-1],
                    consistent=cfg.loss.consistency)
        sdr_sum += si_sdr_per_source(out.waveforms, stems).cpu()
        loss_sum += float(loss_fn(out.waveforms, stems).total)
        n_batches += 1
    model.train()
    sdr = (sdr_sum / max(n_batches, 1)).tolist()
    result = {f"sdr_{name}": value for name, value in zip(STEMS, sdr)}
    result["sdr_mean"] = float(sum(sdr) / len(sdr))
    result["val_loss"] = loss_sum / max(n_batches, 1)
    return result



def _log_row(writer: csv.writer, row: dict) -> None:
    writer.writerow(row)
    message = "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in row.items() if k != "step")
    print(f"[train] step {row['step']:>6}  {message}", flush=True)


def _validate_and_checkpoint(model, ema, optimizer, valid_loader, loss_fn, cfg, device,
                             step, writer, log_file, ckpt_dir, best_sdr) -> dict[str, float]:
    """Run validation, append it to the log and checkpoint (last + best)."""
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
    }
    torch.save(state, Path(ckpt_dir) / "last.pt")
    if metrics["sdr_mean"] > best_sdr:
        torch.save(state, Path(ckpt_dir) / "best.pt")
        print(f"[valid] new best mean SI-SDR {metrics['sdr_mean']:.2f} dB "
              f"-> {Path(ckpt_dir) / 'best.pt'}", flush=True)
    return metrics


def train(cfg: Config, resume: str | Path | None = None) -> Path:
    """Run the whole training schedule; returns the path of the best checkpoint."""
    torch.manual_seed(cfg.train.seed)
    device = pick_device(cfg.train.device)
    run_dir = Path(cfg.train.out_dir)
    ckpt_dir = Path(cfg.train.ckpt_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    model = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] device={device} params={n_params / 1e6:.2f}M "
          f"stems={STEMS} chunk={cfg.data.chunk_seconds}s", flush=True)

    loss_fn = SeparationLoss(cfg.loss, cfg.stft)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    # Mixed precision is only wired for CUDA (where it gives a large speed-up); on MPS
    # autocast is still unreliable for complex-valued kernels, so it stays off there.
    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    if cfg.train.amp and not use_amp:
        print(f"[train] AMP requested but disabled for device '{device.type}'", flush=True)
    autocast = torch.autocast(device_type=device.type, dtype=torch.float16,
                              enabled=use_amp)
    ema = EMA(model, cfg.train.ema_decay) if cfg.train.ema_decay > 0 else None
    train_loader, valid_loader = build_loaders(cfg)

    total_steps = cfg.train.epochs * cfg.train.steps_per_epoch
    step = 0
    best_sdr = -1e9
    if resume is not None:
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        if ema is not None and ckpt.get("ema") is not None:
            ema.shadow.load_state_dict(ckpt["ema"])
        if ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        step = int(ckpt.get("step", 0))
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
        lr = lr_at(step, total_steps, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        with autocast:
            out = model(mixture, length=mixture.shape[-1],
                        consistent=cfg.loss.consistency)
            b, s, ch, n = stems.shape
            tgt_spec = stft(stems.reshape(b * s, ch, n), cfg.stft)
            tgt_spec = tgt_spec.reshape(b, s, ch, *tgt_spec.shape[-2:])
            loss = loss_fn(out.waveforms, stems, est_spec=out.spectra,
                           tgt_spec=tgt_spec)
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
        {"steps": step, "hours": elapsed, "best_sdr": best_sdr,
         "params": n_params, "device": str(device)}, indent=2))
    return ckpt_dir / "best.pt"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                        help="override a config value, e.g. --set train.batch_size=8")
    args = parser.parse_args(argv)

    overrides: dict[str, object] = {}
    for item in args.set:
        key, _, value = item.partition("=")
        try:
            parsed: object = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        overrides[key] = parsed
    cfg = load_config(args.config, **overrides)
    train(cfg, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

