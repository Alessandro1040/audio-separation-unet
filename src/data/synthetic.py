"""A tiny procedural dataset that mimics the MUSDB18 directory layout.

Purpose: unit tests and smoke tests must run *without* the 4.7 GB MUSDB18 download,
but they should exercise the exact same code path (directory layout, ffmpeg decode,
stem loading, chunking, training loop). So this module writes

    <root>/train/<song-name>/{vocals,drums,bass,other,mixture}.wav
    <root>/test/<song-name>/{vocals,drums,bass,other,mixture}.wav

with four clearly different "instruments":
  vocals - a vibrato harmonic stack (formant-ish, mid frequencies)
  drums  - noise bursts with sharp attack and short decay (broadband transients)
  bass   - low-frequency sine/square tones
  other  - slow chords (steady harmonics, no transients)
That is enough signal for a U-Net to visibly learn something in a couple of minutes,
which makes it ideal for end-to-end tests.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

STEMS = ("vocals", "drums", "bass", "other")


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _vocals(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / sr
    f0 = rng.uniform(180.0, 350.0)
    vibrato = 1.0 + 0.02 * np.sin(2 * np.pi * rng.uniform(4.0, 6.0) * t)
    phase = 2 * np.pi * f0 * np.cumsum(vibrato) / sr
    y = np.zeros(n, dtype=np.float64)
    for k, amp in enumerate([1.0, 0.6, 0.4, 0.25, 0.15], start=1):
        y += amp * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
    env = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.2, 0.5) * t)
    return 0.25 * y * env


def _drums(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    y = np.zeros(n, dtype=np.float64)
    hits = int(n / sr * 2)  # two hits per second
    for _ in range(max(hits, 1)):
        start = rng.integers(0, max(n - sr // 8, 1))
        length = int(sr * rng.uniform(0.03, 0.15))
        env = np.exp(-np.linspace(0, 12, length))
        burst = rng.standard_normal(length) * env
        y[start : start + length] += burst
    return 0.35 * y


def _bass(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / sr
    f0 = rng.uniform(45.0, 110.0)
    y = np.sign(np.sin(2 * np.pi * f0 * t)) * 0.3 + np.sin(2 * np.pi * f0 * t)
    gate = (np.sin(2 * np.pi * rng.uniform(0.4, 1.0) * t) > 0).astype(np.float64)
    return 0.3 * y * (0.3 + 0.7 * gate)


def _other(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / sr
    y = np.zeros(n, dtype=np.float64)
    for f in rng.uniform(300.0, 1400.0, size=3):
        y += np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    return 0.12 * y * (0.6 + 0.4 * np.sin(2 * np.pi * 0.1 * t))


_GENERATORS = {"vocals": _vocals, "drums": _drums, "bass": _bass, "other": _other}


def make_song(seconds: float, sr: int, seed: int) -> dict[str, np.ndarray]:
    """One synthetic song: a dict stem -> (samples, 2) float32 array."""
    n = int(seconds * sr)
    song: dict[str, np.ndarray] = {}
    for i, stem in enumerate(STEMS):
        rng = _rng(seed * 17 + i)
        mono = _GENERATORS[stem](n, sr, rng)
        pan = rng.uniform(-0.6, 0.6)
        left = mono * np.sqrt((1 - pan) / 2)
        right = mono * np.sqrt((1 + pan) / 2)
        song[stem] = np.stack([left, right], axis=1).astype(np.float32)
    return song


def generate_dataset(root: str | Path, n_train: int = 4, n_test: int = 2,
                     seconds: float = 6.0, sample_rate: int = 44100) -> Path:
    """Write a synthetic dataset with the MUSDB18 layout; returns its root."""
    root = Path(root)
    for subset, count in (("train", n_train), ("test", n_test)):
        for k in range(count):
            seed = (0 if subset == "train" else 1000) + k
            name = f"{subset}{k:02d}"
            out = root / subset / name
            out.mkdir(parents=True, exist_ok=True)
            song = make_song(seconds, sample_rate, seed)
            mixture = sum(song[s] for s in STEMS)
            peak = float(np.abs(mixture).max()) + 1e-9
            gain = 0.7 / peak if peak > 0.7 else 1.0
            for stem in STEMS:
                sf.write(out / f"{stem}.wav", song[stem] * gain, sample_rate,
                         subtype="FLOAT")
            sf.write(out / "mixture.wav", mixture * gain, sample_rate, subtype="FLOAT")
    return root
