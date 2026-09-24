#!/usr/bin/env bash
# Phase-1 finalisation: BSS-Eval of the best checkpoint, trivial baseline, test suite.
# LIMIT songs keeps the (CPU-bound) mir_eval BSS-Eval within a sane run time.
set -uo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
LIMIT=${LIMIT:-5}
MAX_SECONDS=${MAX_SECONDS:-}
mkdir -p runs/final
EVAL_EXTRA=""
if [ -n "$MAX_SECONDS" ]; then
    EVAL_EXTRA="--max-seconds $MAX_SECONDS"
fi

"$PYTHON" -m src.evaluate --checkpoint checkpoints/best.pt --subset test --limit "$LIMIT" \
    $EVAL_EXTRA --save-stems outputs/test_eval --out-dir runs/final/eval \
    > runs/final/eval.log 2>&1
echo "EVAL_DONE $(date +%H:%M:%S)" >> runs/final/STATUS

"$PYTHON" scripts/baseline_bss_eval.py --limit "$LIMIT" --max-seconds "$MAX_SECONDS" \
    --out runs/final/baseline_bss.json > runs/final/baseline.log 2>&1
echo "BASELINE_DONE $(date +%H:%M:%S)" >> runs/final/STATUS

"$PYTHON" scripts/summarize_run.py runs/unet/log.csv > runs/final/training_curve.txt 2>&1
"$PYTHON" -m pytest tests/ -q -p no:warnings > runs/final/tests.log 2>&1
echo "TESTS_DONE $(date +%H:%M:%S)" >> runs/final/STATUS

{
    echo "training curve"
    echo "=============="
    cat runs/final/training_curve.txt
    echo
    echo "trained model vs trivial baseline, MUSDB18 test set (median over songs)"
    echo "===================================================================="
    "$PYTHON" - <<'PY'
import json
from pathlib import Path
for label, path in (("model   ", "runs/final/eval/summary.json"),
                    ("baseline", "runs/final/baseline_bss.json")):
    try:
        data = json.loads(Path(path).read_text())
        extra = ""
        if "median_sir" in data:
            extra = f"  SIR={data['median_sir']:6.2f} dB  SAR={data['median_sar']:6.2f} dB"
        print(f"{label}: SDR={data['median_sdr']:6.2f} dB{extra}  "
              f"({data['tracks']} songs)")
    except Exception as exc:                 # pragma: no cover
        print(f"{label}: {exc}")
PY
    echo
    echo "test suite: $(tail -1 runs/final/tests.log)"
} > runs/final/SUMMARY.txt
cat runs/final/SUMMARY.txt

echo "ALL_DONE $(date +%H:%M:%S)" >> runs/final/STATUS
