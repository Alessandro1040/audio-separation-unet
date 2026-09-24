#!/usr/bin/env bash
# End-to-end smoke test: synthetic data -> training -> separation -> evaluation.
# Runs in about a minute on CPU and needs no dataset download.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"

echo "== unit tests =="
"$PYTHON" -m pytest tests/ -q -p no:warnings

echo "== procedural dataset =="
"$PYTHON" scripts/make_synthetic_data.py --out data/synthetic --train 6 --test 2 \
    --seconds 6 --sample-rate 22050

echo "== train a few steps =="
"$PYTHON" -m src.train --config configs/unet_smoke.yaml

echo "== separate one song =="
"$PYTHON" -m src.separate --checkpoint checkpoints/smoke/best.pt \
    --input data/synthetic/test/test00/mixture.wav \
    --out outputs/smoke_test/test00 --device cpu --chunk-seconds 2.0 --also-mixture

echo "== BSS-Eval =="
"$PYTHON" -m src.evaluate --checkpoint checkpoints/smoke/best.pt --root data/synthetic \
    --subset test --device cpu --chunk-seconds 2.0 --out-dir runs/smoke_eval

echo
echo "all good - stems in outputs/smoke_test/test00, metrics in runs/smoke_eval"
