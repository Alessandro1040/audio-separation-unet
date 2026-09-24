#!/usr/bin/env bash
# Wait for training to finish, then produce the final artefacts:
#   runs/final/eval/       BSS-Eval of the best checkpoint on the MUSDB18 test set
#   outputs/test_eval/     separated stems for listening
#   runs/final/baseline*   BSS-Eval of the trivial "mixture" baseline (reference point)
#   runs/final/SUMMARY.txt training curve, results, test-suite result
# Launch with:  nohup scripts/run_final_eval.sh > runs/final_eval.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
mkdir -p runs/final
echo "START $(date +%H:%M:%S)" > runs/final/STATUS

LIMIT=${LIMIT:-8}
TARGET_STEP=${TARGET_STEP:-4400}
MAX_WAIT_MIN=${MAX_WAIT_MIN:-360}

# 1. wait for training: the step target must be reached AND the process must be gone
#    (debounced, so that a restart in the middle of the run is not mistaken for the end)
deadline=$(( $(date +%s) + MAX_WAIT_MIN * 60 ))
gone=0
while true; do
    step=$(tail -1 runs/unet/log.csv 2>/dev/null | cut -d, -f1)
    step=${step:-0}
    if pgrep -f "src.train" > /dev/null; then gone=0; else gone=$((gone + 1)); fi
    if [ "$step" -ge "$TARGET_STEP" ] && [ "$gone" -ge 3 ]; then
        echo "target step $step reached, training stopped" >> runs/final/STATUS
        break
    fi
    if [ "$(date +%s)" -gt "$deadline" ]; then
        echo "WAIT_TIMEOUT at step $step" >> runs/final/STATUS
        break
    fi
    sleep 20
done
echo "TRAIN_DONE $(date +%H:%M:%S) step=$step" >> runs/final/STATUS

# 2. training curve
"$PYTHON" scripts/summarize_run.py runs/unet/log.csv > runs/final/training_curve.txt 2>&1

# 3. BSS-Eval of the trained model (with stems saved for listening)
"$PYTHON" -m src.evaluate --checkpoint checkpoints/best.pt --subset test --limit "$LIMIT" \
    --save-stems outputs/test_eval --out-dir runs/final/eval > runs/final/eval.log 2>&1
echo "EVAL_DONE $(date +%H:%M:%S)" >> runs/final/STATUS

# 4. the trivial baseline on the same songs, for context
"$PYTHON" scripts/baseline_bss_eval.py --limit "$LIMIT" --out runs/final/baseline_bss.json \
    > runs/final/baseline.log 2>&1
echo "BASELINE_DONE $(date +%H:%M:%S)" >> runs/final/STATUS

# 5. test suite on the frozen code
"$PYTHON" -m pytest tests/ -q -p no:warnings > runs/final/tests.log 2>&1
echo "TESTS_DONE $(date +%H:%M:%S)" >> runs/final/STATUS

# 6. combine everything into one readable summary
{
    echo "training curve"
    echo "=============="
    cat runs/final/training_curve.txt
    echo
    echo "trained model vs trivial baseline on MUSDB18 test set (median SDR)"
    echo "=============================================================="
    "$PYTHON" - <<'PY'
import json
from pathlib import Path
try:
    model = json.loads(Path("runs/final/eval/summary.json").read_text())
    print(f"model   : SDR={model['median_sdr']:6.2f} dB  SIR={model['median_sir']:6.2f} dB "
          f"SAR={model['median_sar']:6.2f} dB  ({model['tracks']} songs)")
except Exception as exc:                     # pragma: no cover
    print("model   :", exc)
try:
    base = json.loads(Path("runs/final/baseline_bss.json").read_text())
    print(f"baseline: SDR={base['median_sdr']:6.2f} dB  (mixture used as every source)")
except Exception as exc:                     # pragma: no cover
    print("baseline:", exc)
PY
    echo
    echo "test suite: $(tail -1 runs/final/tests.log)"
} > runs/final/SUMMARY.txt

echo "ALL_DONE $(date +%H:%M:%S)" >> runs/final/STATUS

