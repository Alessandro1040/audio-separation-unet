#!/usr/bin/env bash
# Second phase, run unattended after the first evaluation:
#   1. wait for runs/final/STATUS to contain ALL_DONE
#   2. resume training for a bounded amount of time (fresh, shorter cosine schedule)
#   3. evaluate the new best checkpoint into runs/final2 + outputs/test_eval2
# Launch with:  nohup scripts/continue_training_chain.sh > runs/chain.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"

EXTRA_HOURS=${EXTRA_HOURS:-0.8}
LIMIT=${LIMIT:-8}

echo "waiting for the first evaluation to finish..."
for _ in $(seq 1 360); do
    grep -q ALL_DONE runs/final/STATUS 2>/dev/null && break
    sleep 20
done
echo "resuming training at $(date +%H:%M:%S)"

# a shorter cosine horizon than the original 20k-step recipe: the new schedule anneals
# the learning rate to its floor exactly when this phase stops
"$PYTHON" -m src.train --config configs/unet_musdb18.yaml --resume checkpoints/last.pt \
    --set train.steps_per_epoch=400 --set train.epochs=10 --set train.val_every=400 \
    --set train.max_hours="$EXTRA_HOURS" >> runs/train3.log 2>&1

mkdir -p runs/final2
echo "evaluating $(date +%H:%M:%S)" > runs/final2/STATUS
"$PYTHON" -m src.evaluate --checkpoint checkpoints/best.pt --subset test --limit "$LIMIT" \
    --save-stems outputs/test_eval2 --out-dir runs/final2/eval > runs/final2/eval.log 2>&1
"$PYTHON" scripts/summarize_run.py runs/unet/log.csv > runs/final2/training_curve.txt 2>&1
echo "ALL_DONE $(date +%H:%M:%S)" >> runs/final2/STATUS
