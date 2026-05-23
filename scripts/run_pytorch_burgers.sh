#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper for the unified PyTorch runner.
#
# Usage:
#   bash scripts/run_pytorch_burgers.sh
#   bash scripts/run_pytorch_burgers.sh 1.5 GJ-II
#   ALPHAS=1.25,1.75 METHODS=GJ-I,GJ-II STEPS=5000 bash scripts/run_pytorch_burgers.sh
#
# Positional arguments are kept from the old Burgers launcher:
#   $1  Comma-separated Burgers alphas. Default: ${ALPHAS:-1.25,1.5,1.75}.
#   $2  Comma-separated methods. Default: ${METHODS:-GJ-II}.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

ALPHAS_CSV="${1:-${ALPHAS:-1.25,1.5,1.75}}"
METHODS_CSV="${2:-${METHODS:-GJ-II}}"

export ALPHAS="${ALPHAS_CSV}"

exec bash "${SCRIPT_DIR}/run_pytorch_cases.sh" burgers "${METHODS_CSV}"
