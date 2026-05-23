#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper for the unified PyTorch runner.
#
# Usage:
#   bash scripts/run_pytorch_2d_cases.sh all GJ-I,GJ-II,MC-I,MC-II
#   STEPS=5000 GJ_QUAD=64 MC_QUAD=640 bash scripts/run_pytorch_2d_cases.sh lshape GJ-II
#
# This wrapper preserves the old 2D positional arguments. Common schedule,
# logging, batch, quadrature, RAD, model, W&B, and Python variables are handled
# by scripts/run_pytorch_cases.sh.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

CASE_NAME="${1:-all}"
METHODS_CSV="${2:-GJ-I,GJ-II,MC-I,MC-II}"

if [[ "${CASE_NAME}" == "all" ]]; then
  CASES_ARG="2d"
else
  CASES_ARG="${CASE_NAME}"
fi

exec bash "${SCRIPT_DIR}/run_pytorch_cases.sh" "${CASES_ARG}" "${METHODS_CSV}"
