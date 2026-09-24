"""Random-chunk datasets over MUSDB18, with the augmentation used during training."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from ..config import AugmentConfig, DataConfig
from .musdb import Track, _TrackCache, find_tracks, load_stems, split_train_valid


@dataclass
class Chunk:
    """One training/validation example: the stems and the mixture they sum to."""

    stems: torch.Tensor        # (S, ch, samples)
    mixture: torch.Tensor      # (ch, samples)
    name: str

    @property
    def n_stems(self) -> int:
        return self.stems.shape[0]

    @property
    def n_samples(self) -> int:
        return self.stems.shape[-1]


def training_tracks(cfg: DataConfig) -> list[Track]:
    return split_train_valid(find_tracks(cfg.root, "train"), cfg.valid_tracks)[0]


def validation_tracks(cfg: DataConfig) -> list[Track]:
    return split_train_valid(find_tracks(cfg.root, "train"), cfg.valid_tracks)[1]


def _augment(
    stems: torch.Tensor,
    aug: AugmentConfig,
    rng: random.Random,
    other_stems: torch.Tensor | None,
) -> torch.Tensor:
    """Apply the training augmentations to a (S, ch, samples) tensor of stems.

    Music separation datasets are tiny (100 songs), so aggressive augmentation is the
    main regulariser: random gains, channel swaps, polarity flips, dropping a source
    and - most useful of all - swapping a stem for the same stem of another song, which
    teaches the model to be robust to unseen *combinations* of instruments.
    """
    stems = stems.clone()
    n_stems = stems.shape[0]

    if other_stems is not None and rng.random() < aug.remix_p:
        stems = other_stems.clone()  # remix: a completely different song is the truth

    if other_stems is not None and aug.stem_swap_p > 0:
        n = min(stems.shape[-1], other_stems.shape[-1])
        for i in range(n_stems):
            if rng.random() < aug.stem_swap_p:
                # The *same* stem index of the donor song. A random index here would
                # relabel the targets (the drums of another song landing in the `vocals`
                # slot): nothing in the mixture tells the network which instrument was
                # substituted, so the only loss-minimising answer is the conditional
                # mean - a per-source gain on the mixture. That is exactly the failure
                # mode this augmentation must not create.
                stems[i, :, :n] = other_stems[i, :, :n]

    if aug.gain_db > 0:
        gains = torch.tensor(
            [10 ** (rng.uniform(-aug.gain_db, aug.gain_db) / 20) for _ in range(n_stems)]
        )
        stems = stems * gains[:, None, None]

    if aug.channel_swap_p > 0:
        for i in range(n_stems):
            if rng.random() < aug.channel_swap_p:
                stems[i] = stems[i].flip(0)

    if aug.polarity_p > 0:
        for i in range(n_stems):
            if rng.random() < aug.polarity_p:
                stems[i] = -stems[i]

    if aug.drop_source_p > 0:
        for i in range(n_stems):
            if rng.random() < aug.drop_source_p:
                stems[i] = 0.0

    return stems


def _random_chunk(stems: torch.Tensor, n_samples: int, rng: random.Random) -> torch.Tensor:
    """Cut a chunk of `n_samples` at a random position (repeating if too short)."""
    total = stems.shape[-1]
    if total <= n_samples:
        reps = math.ceil(n_samples / max(total, 1))
        stems = stems.repeat(1, 1, reps)
        total = stems.shape[-1]
    start = rng.randrange(0, total - n_samples + 1)
    return stems[..., start : start + n_samples]

class MusdbTrainIterable(IterableDataset):
    """Endless stream of augmented chunks drawn from the training songs.

    Decoding is the bottleneck (the songs are compressed AAC), so the loader does *not*
    decode a song per sample. Instead, per worker:

    * songs are visited in a shuffled order, one at a time;
    * for each visit only a random `segment_seconds` slice of the song is decoded;
    * `chunks_per_load` random crops are cut from that slice (plus heavy augmentation);
    * the last few decoded slices are kept in a small "recent" pool, which is where the
      stem-swap augmentation takes its replacement stems from - so cross-song mixing
      costs nothing extra.

    That is what keeps the data pipeline well ahead of the GPU (~100x fewer decodes).
    """

    def __init__(self, cfg: DataConfig, chunk_samples: int | None = None,
                 seed: int = 1234, tracks: list[Track] | None = None) -> None:
        self.cfg = cfg
        self.chunk_samples = chunk_samples or int(cfg.chunk_seconds * cfg.sample_rate)
        self.seed = seed
        self.tracks = tracks if tracks is not None else training_tracks(cfg)
        self.cache = _TrackCache(cfg.cache_tracks)
        self._epoch = 0
        self._order: list[Track] = []
        self._cursor = 0
        self._current: torch.Tensor | None = None
        self._current_name = ""
        self._current_left = 0
        self._recent: list[torch.Tensor] = []

    def set_epoch(self, epoch: int) -> None:
        """Change the random stream (called by the trainer between epochs)."""
        self._epoch = epoch
        self._order = []
        self._cursor = 0
        self._current = None
        self._current_left = 0

    def _worker_rng(self) -> random.Random:
        info = get_worker_info()
        worker_id = 0 if info is None else info.id
        return random.Random(self.seed * 1_000_003 + self._epoch * 1_009 + worker_id)

    def __iter__(self):
        rng = self._worker_rng()
        while True:
            yield self._sample(rng)

    def _next_track(self, rng: random.Random) -> tuple[torch.Tensor, str]:
        """Decode (a segment of) the next song in the shuffled order."""
        if not self._order or self._cursor >= len(self._order):
            self._order = list(self.tracks)
            rng.shuffle(self._order)
            self._cursor = 0
        track = self._order[self._cursor]
        self._cursor += 1
        start = duration = None
        if self.cfg.segment_seconds > 0:
            length = self.cache.duration(track)
            duration = self.cfg.segment_seconds
            if length and length > duration:
                start = rng.uniform(0.0, length - duration)
        stems = load_stems(track, self.cache, self.cfg.sample_rate, self.cfg.channels,
                           start=start, duration=duration)
        if stems.shape[-1] < self.chunk_samples:      # very short segment: repeat it
            reps = math.ceil(self.chunk_samples / max(stems.shape[-1], 1))
            stems = stems.repeat(1, 1, reps)
        self._recent.insert(0, stems)
        del self._recent[3:]                          # keep the last three songs only
        return stems, track.name

    def _sample(self, rng: random.Random) -> Chunk:
        if self._current is None or self._current_left <= 0:
            self._current, self._current_name = self._next_track(rng)
            self._current_left = max(1, self.cfg.chunks_per_load)
        self._current_left -= 1

        n = self.chunk_samples
        stems = _random_chunk(self._current, n, rng)

        other = None
        if (self.cfg.aug.remix_p > 0 or self.cfg.aug.stem_swap_p > 0) \
                and len(self._recent) > 1:
            other = _random_chunk(rng.choice(self._recent[1:]), n, rng)

        stems = _augment(stems, self.cfg.aug, rng, other)
        return Chunk(stems=stems, mixture=stems.sum(dim=0), name=self._current_name)


class MusdbEvalChunks(Dataset):
    """Deterministic chunks used for validation and quick listening tests.

    Every song contributes `chunks_per_track` chunks spread evenly over its length,
    so the metric is stable and reproducible across runs.
    """

    def __init__(self, cfg: DataConfig, chunk_samples: int | None = None,
                 tracks: list[Track] | None = None, chunks_per_track: int = 4) -> None:
        self.cfg = cfg
        self.chunk_samples = chunk_samples or int(cfg.chunk_seconds * cfg.sample_rate)
        self.tracks = tracks if tracks is not None else validation_tracks(cfg)
        self.chunks_per_track = chunks_per_track
        self.cache = _TrackCache(capacity=2)

    def __len__(self) -> int:
        return len(self.tracks) * self.chunks_per_track

    def __getitem__(self, index: int) -> Chunk:
        track = self.tracks[index // self.chunks_per_track]
        k = index % self.chunks_per_track
        stems = load_stems(track, self.cache, self.cfg.sample_rate, self.cfg.channels)
        total = stems.shape[-1]
        n = min(self.chunk_samples, total)
        usable = max(total - n, 1)
        start = int(round(usable * (k + 0.5) / self.chunks_per_track))
        chunk = stems[..., start : start + n]
        if chunk.shape[-1] < self.chunk_samples:  # pad the very short tail
            chunk = torch.nn.functional.pad(
                chunk, (0, self.chunk_samples - chunk.shape[-1])
            )
        return Chunk(stems=chunk, mixture=chunk.sum(dim=0), name=track.name)


def collate_chunks(batch: list[Chunk]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "stems": torch.stack([c.stems for c in batch]),
        "mixture": torch.stack([c.mixture for c in batch]),
        "name": [c.name for c in batch],
    }
