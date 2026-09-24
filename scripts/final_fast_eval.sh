#!/usr/bin/env bash
# Final evaluation of whatever the last fine-tune produced:
#   runs/fast2/eval/      chunk-average SI-SDR of the model on 10 test songs
#   runs/fast2/baseline.json  the same metric for the trivial mixture estimator
#   runs/fast2/SUMMARY.txt    per-song comparison table
#   outputs/test_eval2/   separated stems of the final model (for listening)
# Launch with:  nohup scripts/final_fast_eval.sh > runs/final_fast_eval.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
LIMIT=${LIMIT:-10}
MAX_WAIT_MIN=${MAX_WAIT_MIN:-90}

mkdir -p runs/fast2
echo "START $(date +%H:%M:%S)" > runs/fast2/STATUS
deadline=$(( $(date +%s) + MAX_WAIT_MIN * 60 ))
gone=0
while true; do
    if pgrep -f "src.train" > /dev/null; then gone=0; else gone=$((gone + 1)); fi
    [ "$gone" -ge 3 ] && break
    [ "$(date +%s)" -gt "$deadline" ] && break
    sleep 20
done
echo "TRAIN_DONE $(date +%H:%M:%S)" >> runs/fast2/STATUS

"$PYTHON" -m src.evaluate --checkpoint checkpoints/best.pt --subset test --limit "$LIMIT" \
    --fast --save-stems outputs/test_eval2 --out-dir runs/fast2/eval \
    > runs/fast2/eval.log 2>&1
echo "EVAL_DONE $(date +%H:%M:%S)" >> runs/fast2/STATUS

"$PYTHON" scripts/baseline_bss_eval.py --limit "$LIMIT" --fast \
    --out runs/fast2/baseline.json > runs/fast2/baseline.log 2>&1
echo "BASELINE_DONE $(date +%H:%M:%S)" >> runs/fast2/STATUS

"$PYTHON" - <<'PY' > runs/fast2/SUMMARY.txt
import json
from pathlib import Path

model = json.loads(Path("runs/fast2/eval/summary.json").read_text())
base = json.loads(Path("runs/fast2/baseline.json").read_text())
print("chunk-average SI-SDR (5 s chunks, whole song) on the MUSDB18 test split")
print("=" * 72)
print(f"{'song':<38} {'U-Net':>8} {'mixture':>9} {'delta':>8}")
wins = 0
for song, value in model["per_track_si_sdr"].items():
    print(f"{song[:38]:<38} {value:8.2f} {'-':>9} {'-':>8}")
print(f"\nmedian over {model['tracks']} songs")
print(f"  U-Net              : {model['median_si_sdr']:6.2f} dB")
print(f"  mixture-estimator  : {base['median_si_sdr']:6.2f} dB")
print(f"  improvement        : {model['median_si_sdr'] - base['median_si_sdr']:+6.2f} dB")
print("\nper-source median SI-SDR of the U-Net:")
for name, value in model.get("per_source_si_sdr", {}).items():
    print(f"  {name:<8} {value:6.2f} dB")
print(f"\nseconds per song: {model['seconds_per_track']:.1f}")
PY
cat runs/fast2/SUMMARY.txt
echo "ALL_DONE $(date +%H:%M:%S)" >> runs/fast2/STATUS
