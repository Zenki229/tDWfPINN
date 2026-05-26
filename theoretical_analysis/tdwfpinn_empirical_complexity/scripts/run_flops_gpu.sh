#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$ROOT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR=${1:-outputs/flops_${TIMESTAMP}}
mkdir -p "$OUT_DIR"

echo "[flops] writing outputs to $OUT_DIR"
python scripts/run_flops_sweep.py \
  --device cuda \
  --alpha 1.5 \
  --dtype float64 \
  --warmup \
  --repeats 1 \
  --out-csv "$OUT_DIR/flops_sweep.csv"

python scripts/plot_flops_orders.py \
  --csv "$OUT_DIR/flops_sweep.csv" \
  --metric backward_profiler_flops \
  --out-dir "$OUT_DIR/flops_plots"
