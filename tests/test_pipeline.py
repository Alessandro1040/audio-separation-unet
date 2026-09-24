"""End-to-end smoke test: train a few steps, then separate audio with the result.

This is the test that catches breakage in the parts unit tests cannot see: the training
loop, checkpointing/resuming, and the sliding-window inference path.
"""
from __future__ import annotations

import csv
import json

import soundfile as sf
import torch

from src.config import Config, load_config
from src.data.synthetic import STEMS, generate_dataset
from src.dsp import load_audio
from src.evaluate import evaluate_checkpoint
from src.models.separation import separate_long
from src.separate import load_model, separate_file
from src.train import build_model, pick_device, train


def _smoke_config(tmp_path) -> Config:
    root = generate_dataset(tmp_path / "synthetic", n_train=4, n_test=1, seconds=3.0,
                            sample_rate=22050)
    cfg = load_config("configs/unet_smoke.yaml")
    cfg.data.root = str(root)
    cfg.data.cache_tracks = 2
    cfg.train.out_dir = str(tmp_path / "runs")
    cfg.train.ckpt_dir = str(tmp_path / "ckpt")
    cfg.train.device = "cpu"          # keep tests deterministic and machine independent
    cfg.train.epochs = 1
    cfg.train.steps_per_epoch = 6
    cfg.train.warmup_steps = 2
    cfg.train.val_every = 6
    cfg.train.val_chunks = 1
    cfg.train.log_every = 3
    cfg.train.ema_decay = 0.9
    return cfg


def test_training_smoke_writes_checkpoints_and_improves(tmp_path) -> None:
    cfg = _smoke_config(tmp_path)
    best = train(cfg)
    assert best.exists()
    assert (tmp_path / "ckpt" / "last.pt").exists()
    with (tmp_path / "runs" / "log.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    losses = [float(row["loss"]) for row in rows if row["loss"]]
    assert len(losses) >= 2
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
    assert any(row["sdr_mean"] for row in rows), "no validation row was written"
    summary = json.loads((tmp_path / "runs" / "summary.json").read_text())
    assert summary["steps"] == 6


def test_resume_continues_from_checkpoint(tmp_path) -> None:
    cfg = _smoke_config(tmp_path)
    train(cfg)
    cfg.train.steps_per_epoch = 12
    best = train(cfg, resume=tmp_path / "ckpt" / "last.pt")
    assert best.exists()
    summary = json.loads((tmp_path / "runs" / "summary.json").read_text())
    assert summary["steps"] == 12


def test_separate_file_end_to_end(tmp_path) -> None:
    cfg = _smoke_config(tmp_path)
    train(cfg)
    device = pick_device("cpu")
    model, loaded_cfg = load_model(tmp_path / "ckpt" / "best.pt", device)

    track_dir = next((tmp_path / "synthetic" / "test").iterdir())
    mixture_path = track_dir / "mixture.wav"
    out_dir = tmp_path / "stems"
    separate_file(model, loaded_cfg, mixture_path, device, chunk_seconds=1.0, overlap=0.5)
    estimates = separate_long(
        model,
        load_audio(mixture_path, loaded_cfg.data.sample_rate,
                   loaded_cfg.data.channels),
        loaded_cfg.data.sample_rate, chunk_seconds=1.0, overlap=0.5,
    )
    assert estimates.shape[0] == len(STEMS)
    mixture = load_audio(mixture_path, loaded_cfg.data.sample_rate,
                         loaded_cfg.data.channels)
    # sliding-window overlap-add must reconstruct shapes and stay bounded
    assert estimates.shape[-1] == mixture.shape[-1]
    assert torch.isfinite(estimates).all()
    assert float(estimates.abs().max()) < 10.0

    # the same model through the CLI wrapper writes four wav files
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-m", "src.separate", "--checkpoint",
         str(tmp_path / "ckpt" / "best.pt"), "--input", str(mixture_path),
         "--out", str(out_dir), "--device", "cpu", "--chunk-seconds", "1.0"],
        check=True, cwd=".", capture_output=True,
    )
    for stem in STEMS:
        audio, sr = sf.read(out_dir / f"{stem}.wav")
        assert sr == loaded_cfg.data.sample_rate
        assert audio.shape[1] == 2


def test_evaluation_runs_on_a_checkpoint(tmp_path) -> None:
    cfg = _smoke_config(tmp_path)
    train(cfg)
    summary = evaluate_checkpoint(
        tmp_path / "ckpt" / "best.pt", root=str(cfg.data.root), subset="test",
        device_name="cpu", chunk_seconds=1.0, window=1.0, hop=1.0,
        out_dir=tmp_path / "eval",
    )
    assert summary["tracks"] == 1
    assert "vocals" in summary["per_source"]
    assert (tmp_path / "eval" / "per_track.csv").exists()
