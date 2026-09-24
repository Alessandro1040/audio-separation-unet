"""MUSDB18 reader: decode stems, sample random chunks, augment, hand out waveforms.

MUSDB18 (https://sigsep.github.io/datasets/musdb.html) is the reference dataset for
music source separation: 150 songs, 4 stems each (vocals, drums, bass, other).
`train` has 100 songs, `test` has 50. Two flavours exist on Zenodo:

* MUSDB18      - AAC-compressed `.mp4` stems (4.7 GB) - default, decoded via ffmpeg
* MUSDB18-HQ   - uncompressed `.wav` stems (22.7 GB) - used automatically if present

The loader decodes a whole song with ffmpeg (fast, C code) and keeps it in an LRU
cache in RAM as float16, so cutting random training chunks costs nothing.
"""
from __future__ import annotations

import subprocess
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..config import STEMS

#: MUSDB18-HQ uses one wav per stem; the compressed release packs the whole song in a
#: single `.stem.mp4` with five AAC streams, mixture first: the naming/order convention
#: of the musdb/stempeg tools is mixture, drums, bass, other, vocals.
MUSDB_STEM_STREAM_ORDER = ("drums", "bass", "other", "vocals")


@dataclass(frozen=True)
class Track:
    """A MUSDB18 song: `path` is either its folder (HQ) or its `.stem.mp4` (compressed)."""

    name: str
    subset: str
    path: Path


def _single_stem_file(track: Track, stem: str) -> Path | None:
    for suffix in (".wav", ".mp4"):
        candidate = track.path / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _multi_stream_file(track: Track) -> Path | None:
    if track.path.is_file() and track.path.suffix == ".mp4":
        return track.path
    matches = sorted(track.path.glob("*.stem.mp4"))
    return matches[0] if matches else None


def stem_source(track: Track, stem: str) -> tuple[Path, int | None]:
    """Where to read `stem` from: (file, stream index) - index None means whole file."""
    direct = _single_stem_file(track, stem)
    if direct is not None:
        return direct, None
    multi = _multi_stream_file(track)
    if multi is not None:
        if stem not in MUSDB_STEM_STREAM_ORDER:
            raise KeyError(f"'{stem}' is not part of MUSDB18; expected {STEMS}")
        return multi, 1 + MUSDB_STEM_STREAM_ORDER.index(stem)
    raise FileNotFoundError(f"no audio for stem '{stem}' in {track.path}")


def mixture_source(track: Track) -> tuple[Path, int | None]:
    """Where to read the *mixture* from (stream 0 of a `.stem.mp4`, or `mixture.wav`)."""
    direct = track.path / "mixture.wav"
    if direct.exists():
        return direct, None
    multi = _multi_stream_file(track)
    if multi is not None:
        return multi, 0
    raise FileNotFoundError(f"no mixture found in {track.path}")


def find_tracks(root: str | Path, subset: str) -> list[Track]:
    """All songs of a MUSDB18 subset (`train` / `test`), sorted by name.

    Supports both releases: `train/<song>/vocals.wav` (MUSDB18-HQ) and
    `train/<song>.stem.mp4` (compressed MUSDB18).
    """
    root = Path(root)
    subset_dir = root / subset
    if not subset_dir.is_dir():
        raise FileNotFoundError(
            f"{subset_dir} not found - run scripts/download_musdb.py first"
        )
    tracks: list[Track] = []
    for entry in sorted(subset_dir.iterdir()):
        if entry.is_dir():
            if any(_single_stem_file(Track(entry.name, subset, entry), s) is not None
                   for s in STEMS):
                tracks.append(Track(entry.name, subset, entry))
        elif entry.suffix == ".mp4":          # compressed release
            name = entry.name[: -len(".stem.mp4")] if entry.name.endswith(".stem.mp4") \
                else entry.stem
            tracks.append(Track(name, subset, entry))
    if not tracks:
        raise FileNotFoundError(f"no MUSDB18 tracks found in {subset_dir}")
    return tracks


def split_train_valid(
    tracks: list[Track], valid_tracks: int = 14
) -> tuple[list[Track], list[Track]]:
    """Hold out the first `valid_tracks` songs (by name) for validation.

    Keeping the split deterministic means validation numbers stay comparable
    between runs and checkpoints.
    """
    valid = tracks[:valid_tracks]
    train = tracks[valid_tracks:]
    if not train:
        raise ValueError("validation split consumed the whole training set")
    return train, valid


def _stem_path(track: Track, stem: str) -> Path:
    wav = track.path / f"{stem}.wav"
    return wav if wav.exists() else track.path / f"{stem}.mp4"


def _decode_ffmpeg(path: Path, sample_rate: int, channels: int,
                   stream: int | None = None, start: float | None = None,
                   duration: float | None = None) -> np.ndarray:
    """Decode any file ffmpeg understands to float32 PCM (channels, samples).

    `stream` selects one audio stream of a multi-stream `.stem.mp4` (how the compressed
    MUSDB18 release stores the four stems plus a mixture). `start`/`duration` decode only
    a slice of the song - the single most important speed-up of the whole data pipeline:
    a random 5-second training crop never needs the other 4 minutes of the track.
    """
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(path)]
    if stream is not None:
        cmd += ["-map", f"0:a:{stream}"]
    cmd += [
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", str(channels), "-ar", str(sample_rate),
    ]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    out = subprocess.run(cmd + ["-"], stdout=subprocess.PIPE, check=True).stdout
    audio = np.frombuffer(out, dtype="<f4")
    return audio.reshape(-1, channels).T.copy()


class _TrackCache:
    """Smallest useful thing: an LRU cache of decoded songs (float16) per worker."""

    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, capacity)
        self._items: OrderedDict[str, np.ndarray] = OrderedDict()
        self._durations: dict[str, float | None] = {}

    def get(self, track: Track, stems: tuple[str, ...], sample_rate: int,
            channels: int, start: float | None = None,
            duration: float | None = None) -> np.ndarray:
        key = f"{track.subset}/{track.name}"
        if start is not None:
            key += f"@{start:.3f}"
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key]
        wavs = []
        for stem in stems:
            path, stream = stem_source(track, stem)
            wavs.append(_decode_ffmpeg(path, sample_rate, channels, stream, start,
                                       duration))
        n = min(w.shape[1] for w in wavs)
        audio = np.stack([w[:, :n] for w in wavs]).astype(np.float16)  # (S, ch, n)
        self._items[key] = audio
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)
        return audio

    def duration(self, track: Track) -> float | None:
        """Length in seconds of a song (cached), used to pick a random segment."""
        if track.name in self._durations:
            return self._durations[track.name]
        path, _ = stem_source(track, MUSDB_STEM_STREAM_ORDER[0])
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            stdout=subprocess.PIPE, check=True).stdout.decode().strip()
        try:
            value = float(out.splitlines()[0])
        except (ValueError, IndexError):
            value = None
        self._durations[track.name] = value
        return value


def load_stems(
    track: Track,
    cache: _TrackCache,
    sample_rate: int = 44100,
    channels: int = 2,
    stems: tuple[str, ...] = STEMS,
    start: float | None = None,
    duration: float | None = None,
) -> torch.Tensor:
    """Decoded stems of `track` as a float32 tensor (n_stems, channels, samples)."""
    audio = cache.get(track, stems, sample_rate, channels, start, duration)
    return torch.from_numpy(audio.astype(np.float32))
