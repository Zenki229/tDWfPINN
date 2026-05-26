#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$ROOT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR=${1:-outputs/test_${TIMESTAMP}}
mkdir -p "$OUT_DIR"

echo "[test] writing outputs to $OUT_DIR"
python scripts/test_generalized_gj.py \
  --device cuda \
  --alpha 1.5 \
  --dtype float64 \
  --out-csv "$OUT_DIR/test_generalized_gj.csv"
