#!/usr/bin/env python3
"""Compare two checkpoints on the same validation chunk: one figure, side by side.

    python scripts/compare_checkpoints.py checkpoints/best_step1200.pt \
        checkpoints/fixed/best.pt --out figures/masks_before_after.png

The left column is the first checkpoint, the right column the second; each row is one source
and the top strip is the mixture spectrogram both models saw. This is how the augmentation
bug in the first checkpoint was made visible: with mislabelled training targets the predicted
masks are flat (a static frequency profile), while a working model puts its mask energy on
the percussive onsets (`drums`) or in the low band (`bass`).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import STEMS, load_config  # noqa: E402
from src.data.chunks import MusdbEvalChunks, validation_tracks  # noqa: E402
from src.separate import load_model  # noqa: E402
from src.train import pick_device  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("old", help="checkpoint shown in the left column")
    p.add_argument("new", help="checkpoint shown in the right column")
    p.add_argument("--out", default="figures/masks_before_after.png")
    p.add_argument("--config", default="configs/unet_musdb18.yaml")
    p.add_argument("--track-index", type=int, default=0,
                   help="which held-out validation song to use")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--labels", default=None,
                   help="comma-separated column titles (default: the file names)")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    device = pick_device(args.device)
    cfg = load_config(args.config)
    tracks = validation_tracks(cfg.data)
    dataset = MusdbEvalChunks(cfg.data, tracks=tracks[args.track_index:args.track_index + 1],
                              chunks_per_track=1)
    chunk = dataset[0]
    seconds = min(args.seconds, chunk.mixture.shape[-1] / cfg.data.sample_rate)
    mixture = chunk.mixture[..., : int(seconds * cfg.data.sample_rate)]
    print(f"chunk: {chunk.name}  {seconds:.1f} s")

    images, mix_mag, steps = {}, None, {}
    for key, path in (("old", args.old), ("new", args.new)):
        model, _ = load_model(path, device)
        with torch.no_grad():
            masks, spec = model.predict_masks(mixture.unsqueeze(0))
        images[key] = masks[0].abs().mean(dim=1).cpu().numpy()
        mix_mag = spec[0].abs().mean(dim=0).cpu().numpy()
        steps[key] = torch.load(path, map_location="cpu",
                                weights_only=False).get("step")
        print(f"{Path(path).name}: step {steps[key]}, mean |mask| = "
              f"{[round(float(images[key][i].mean()), 3) for i in range(len(STEMS))]}")

    if args.labels:
        titles = dict(zip(("old", "new"), args.labels.split(",")))
    else:
        titles = {"old": f"{Path(args.old).name} (step {steps['old']})",
                  "new": f"{Path(args.new).name} (step {steps['new']})"}

    extent = [0, images["old"].shape[-1] * cfg.stft.hop_length / cfg.data.sample_rate,
              0, cfg.data.sample_rate / 2000]
    fig = plt.figure(figsize=(12, 11))
    grid = fig.add_gridspec(len(STEMS) + 1, 2, hspace=0.55, wspace=0.08)
    ax = fig.add_subplot(grid[0, :])
    im = ax.imshow(20 * np.log10(mix_mag.clip(1e-8)), origin="lower", aspect="auto",
                   extent=extent, cmap="magma")
    ax.set_title("mixture spectrogram (what both models see)")
    ax.set_ylabel("freq (kHz)")
    fig.colorbar(im, ax=ax, format="%d dB", pad=0.01)

    for i, stem in enumerate(STEMS):
        for j, key in enumerate(("old", "new")):
            ax = fig.add_subplot(grid[i + 1, j])
            im = ax.imshow(images[key][i], origin="lower", aspect="auto", extent=extent,
                           cmap="viridis", vmin=0, vmax=1)
            if i == 0:
                ax.set_title(titles[key])
            if j == 1:
                fig.colorbar(im, ax=ax, pad=0.01)
            if j == 0:
                ax.set_ylabel(f"{stem}\nfreq (kHz)")
            if i == len(STEMS) - 1:
                ax.set_xlabel("time (s)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
