#!/usr/bin/env bash
# Controlled A/B: same data, same STFT, same loss, same schedule, same number of steps,
# same validation songs - only the architecture differs (U-Net vs Mel-band RoFormer).
#
#     DEV=mps STEPS=150 ./scripts/ab_roformer_vs_unet.sh
#     DEV=mps STEPS=600 TAG=600 ./scripts/ab_roformer_vs_unet.sh    # a second, longer point
#
# It writes runs/ab_unet${TAG}/ and runs/ab_roformer${TAG}/ (logs + summary.json) and
# prints the two best validation SI-SDRs at the end. Read docs/ROFORMER.md before drawing
# conclusions: a short run measures *learning speed*, not the ceiling of either architecture.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
DEV="${DEV:-mps}"
STEPS="${STEPS:-150}"
TAG="${TAG:-}"
BATCH="${BATCH:-2}"
CHUNK="${CHUNK:-3.0}"
VAL_CHUNKS="${VAL_CHUNKS:-8}"
WORKERS="${WORKERS:-2}"

echo "== U-Net side (configs/unet_musdb18.yaml, architecture unchanged) =="
"$PYTHON" -m src.train --config configs/unet_musdb18.yaml \
    --set train.epochs=1 --set "train.steps_per_epoch=${STEPS}" \
    --set "train.batch_size=${BATCH}" --set "data.chunk_seconds=${CHUNK}" \
    --set train.warmup_steps=50 --set "train.val_every=${STEPS}" \
    --set "train.val_chunks=${VAL_CHUNKS}" --set train.ema_decay=0.99 \
    --set train.max_hours=100 --set "data.num_workers=${WORKERS}" \
    --set "train.out_dir=runs/ab_unet${TAG}" --set "train.ckpt_dir=checkpoints/ab_unet${TAG}" \
    --set "train.device=${DEV}"

echo
echo "== RoFormer side (configs/roformer_ab.yaml, matched settings) =="
"$PYTHON" -m src.train_roformer --config configs/roformer_ab.yaml \
    --set "train.steps_per_epoch=${STEPS}" --set "train.batch_size=${BATCH}" \
    --set "data.chunk_seconds=${CHUNK}" --set "train.val_chunks=${VAL_CHUNKS}" \
    --set "data.num_workers=${WORKERS}" --set "train.device=${DEV}" \
    --set "train.out_dir=runs/ab_roformer${TAG}" \
    --set "train.ckpt_dir=checkpoints/ab_roformer${TAG}" \
    --set "train.val_every=${STEPS}"

echo
echo "== result =="
"$PYTHON" - <<PY
import json
from pathlib import Path

rows = []
for name, path in (("unet", "runs/ab_unet${TAG}/summary.json"),
                   ("roformer", "runs/ab_roformer${TAG}/summary.json")):
    data = json.loads(Path(path).read_text())
    rows.append((name, data))
    print(f"{name:<9} steps={data['steps']:<5} hours={data['hours']:.2f} "
          f"params={data['params'] / 1e6:.2f}M  best mean SI-SDR = {data['best_sdr']:.2f} dB")
unet, rof = rows[0][1]["best_sdr"], rows[1][1]["best_sdr"]
delta = rof - unet
print(f"\ndelta (roformer - unet) = {delta:+.2f} dB at an equal step budget"
      f"  ->  {'RoFormer ahead' if delta > 0 else 'U-Net ahead'}"
      " (short runs say nothing about the ceiling: see docs/ROFORMER.md)")
PY
