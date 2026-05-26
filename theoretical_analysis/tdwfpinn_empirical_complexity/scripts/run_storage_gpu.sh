#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$ROOT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR=${1:-outputs/storage_${TIMESTAMP}}
mkdir -p "$OUT_DIR"

echo "[storage] writing outputs to $OUT_DIR"
python scripts/run_storage_sweep.py \
  --device cuda \
  --alpha 1.5 \
  --dtype float64 \
  --warmup \
  --repeats 3 \
  --out-csv "$OUT_DIR/storage_sweep.csv"

python scripts/plot_storage_orders.py \
  --csv "$OUT_DIR/storage_sweep.csv" \
  --metric graph_peak_delta_bytes \
  --out-dir "$OUT_DIR/storage_plots"
