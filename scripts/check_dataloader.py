"""Smoke-test the real-data loader: decode a batch of MUSDB18 chunks and time it."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import STEMS, load_config  # noqa: E402
from src.data.chunks import (  # noqa: E402
    MusdbEvalChunks,
    MusdbTrainIterable,
    collate_chunks,
)


def verify_segment_alignment(cfg) -> int:
    """Check that a `-ss` segment of the 4 stems still adds up to the mixture stream.

    Training decodes short random segments of each song for speed. That is only valid
    if the seek lands on the *same* time offset in every stream of the `.stem.mp4`;
    otherwise the "mixture" would silently become a collage of different moments. The
    check compares `sum(stems)` with the mixture stream of the container (stream 0) for
    a few random segments.
    """
    from src.data.musdb import stem_source, mixture_source, _decode_ffmpeg

    tracks = MusdbTrainIterable(cfg.data, seed=0).tracks
    worst = 1.0
    for track in tracks[:3]:
        for start in (12.0, 37.5, 61.25):
            decoded = []
            for stem in STEMS:
                path, stream = stem_source(track, stem)
                decoded.append(_decode_ffmpeg(path, cfg.data.sample_rate,
                                              cfg.data.channels, stream, start=start,
                                              duration=10.0))
            stems = np.stack(decoded)
            path, stream = mixture_source(track)
            mixture = _decode_ffmpeg(path, cfg.data.sample_rate, cfg.data.channels,
                                     stream, start=start, duration=10.0)
            n = min(stems.shape[-1], mixture.shape[-1])
            ref = mixture[:, :n].ravel()
            est = stems[:, :, :n].sum(axis=0).ravel()
            corr = float(np.corrcoef(ref, est)[0, 1])
            err = float(np.sqrt(((ref - est) ** 2).mean()) / (np.sqrt((ref ** 2).mean()) + 1e-12))
            worst = min(worst, corr)
            print(f"  {track.name[:28]:<28} start={start:5.1f}s  corr={corr:.4f}  "
                  f"rel_err={err:.3f}")
    print(f"worst correlation between sum(stems) and the mixture stream: {worst:.4f}")
    if worst < 0.95:
        print("ALIGNMENT PROBLEM: stems come from different time offsets", file=sys.stderr)
        return 1
    print("OK: segment seeking stays aligned across streams")
    return 0


def main() -> int:
    cfg = load_config("configs/unet_musdb18.yaml")
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        cfg.data.root = sys.argv[1]
    if "--verify-alignment" in sys.argv:
        return verify_segment_alignment(cfg)

    print(f"root={cfg.data.root} chunk={cfg.data.chunk_seconds}s "
          f"workers={cfg.data.num_workers} cache={cfg.data.cache_tracks}")

    dataset = MusdbTrainIterable(cfg.data, seed=0)
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size,
                        num_workers=cfg.data.num_workers, collate_fn=collate_chunks,
                        prefetch_factor=2, persistent_workers=True)
    it = iter(loader)
    t0 = time.time()
    batches = 0
    for _ in range(8):
        batch = next(it)
        batches += 1
    dt = (time.time() - t0) / batches
    print(f"train: {batches} batches in {dt * batches:.2f}s -> {dt * 1000:.0f} ms/batch "
          f"({cfg.train.batch_size * cfg.data.chunk_seconds / dt:.1f} s_audio/s of data)")
    print("  stems", tuple(batch["stems"].shape), batch["stems"].dtype,
          "mixture", tuple(batch["mixture"].shape))
    assert torch.allclose(batch["mixture"], batch["stems"].sum(dim=1), atol=1e-4)
    print("  mixture == sum(stems) OK; names:", batch["name"][:2])

    valid = MusdbEvalChunks(cfg.data, chunks_per_track=1)
    chunk = valid[0]
    print(f"valid: {len(valid)} chunks, first from '{chunk.name}' "
          f"shape={tuple(chunk.stems.shape)}")
    total_mb = sum(t.path.stat().st_size for t in dataset.tracks) / 1e6
    print(f"train songs={len(dataset.tracks)} ({total_mb:.0f} MB of compressed audio)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
