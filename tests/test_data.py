"""Dataset layer: MUSDB18 layout, deterministic validation chunks, augmentation."""
from __future__ import annotations

import random

import torch

from src.config import DataConfig
from src.data.chunks import MusdbEvalChunks, MusdbTrainIterable, collate_chunks
from src.data.musdb import find_tracks, load_stems, split_train_valid
from src.data.synthetic import STEMS, generate_dataset


def _make_dataset(tmp_path, n_train=5, n_test=2, seconds=3.0, sr=22050):
    root = generate_dataset(tmp_path / "synthetic", n_train, n_test, seconds, sr)
    return root


def test_find_tracks_and_split(tmp_path) -> None:
    root = _make_dataset(tmp_path)
    train = find_tracks(root, "train")
    test = find_tracks(root, "test")
    assert len(train) == 5 and len(test) == 2
    assert train[0].name == "train00"
    train_split, valid_split = split_train_valid(train, valid_tracks=2)
    assert len(valid_split) == 2 and len(train_split) == 3
    assert {t.name for t in valid_split} & {t.name for t in train_split} == set()


def test_load_stems_shapes_and_dtype(tmp_path) -> None:
    root = _make_dataset(tmp_path)
    track = find_tracks(root, "train")[0]
    from src.data.musdb import _TrackCache

    cache = _TrackCache(capacity=1)
    stems = load_stems(track, cache, 22050, 2)
    assert stems.shape[0] == len(STEMS)
    assert stems.shape[1] == 2
    assert stems.dtype == torch.float32
    # the mixture written by the generator is the sum of the stems
    assert abs(stems.sum(dim=0).abs().max() - abs(stems).sum(dim=0).abs().max()) < 1.0


def test_train_iterable_yields_consistent_chunks(tmp_path) -> None:
    cfg = DataConfig(root=str(_make_dataset(tmp_path)), sample_rate=22050,
                     chunk_seconds=1.0, valid_tracks=1, num_workers=0,
                     cache_tracks=2)
    dataset = MusdbTrainIterable(cfg, seed=1)
    it = iter(dataset)
    for _ in range(5):
        chunk = next(it)
        assert chunk.stems.shape == (4, 2, 22050)
        assert torch.allclose(chunk.mixture, chunk.stems.sum(dim=0), atol=1e-5)
        assert chunk.name.startswith("train")


def test_augmentation_changes_signal_but_keeps_shape(tmp_path) -> None:
    cfg = DataConfig(root=str(_make_dataset(tmp_path)), sample_rate=22050,
                     chunk_seconds=1.0, valid_tracks=1, cache_tracks=2)
    cfg.aug.remix_p = 1.0          # force the strongest augmentation
    cfg.aug.gain_db = 12.0
    cfg.aug.stem_swap_p = 1.0
    dataset = MusdbTrainIterable(cfg, seed=2)
    chunks = [dataset._sample(random.Random(i)) for i in range(4)]
    assert all(c.stems.shape == (4, 2, 22050) for c in chunks)
    assert not torch.allclose(chunks[0].stems, chunks[1].stems)


def test_eval_chunks_are_deterministic(tmp_path) -> None:
    cfg = DataConfig(root=str(_make_dataset(tmp_path)), sample_rate=22050,
                     chunk_seconds=1.0, valid_tracks=1, cache_tracks=2)
    dataset = MusdbEvalChunks(cfg, chunks_per_track=2)
    assert len(dataset) == 2  # 1 validation song x 2 chunks
    first = dataset[0]
    second = dataset[0]
    assert torch.equal(first.stems, second.stems)
    other = dataset[1]
    assert not torch.equal(first.stems, other.stems)


def test_collate(tmp_path) -> None:
    cfg = DataConfig(root=str(_make_dataset(tmp_path)), sample_rate=22050,
                     chunk_seconds=1.0, valid_tracks=1, cache_tracks=2)
    dataset = MusdbEvalChunks(cfg, chunks_per_track=1)
    batch = collate_chunks([dataset[0], dataset[0]])
    assert batch["stems"].shape == (2, 4, 2, 22050)
    assert batch["mixture"].shape == (2, 2, 22050)
    assert len(batch["name"]) == 2
