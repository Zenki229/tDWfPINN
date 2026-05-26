#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$ROOT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_ROOT=${1:-outputs/run_${TIMESTAMP}}
mkdir -p "$OUT_ROOT"

echo "[all] writing outputs to $OUT_ROOT"
bash scripts/run_test_gpu.sh "$OUT_ROOT/test"
bash scripts/run_storage_gpu.sh "$OUT_ROOT/storage"
bash scripts/run_flops_gpu.sh "$OUT_ROOT/flops"
