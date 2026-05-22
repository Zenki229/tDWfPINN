#!/usr/bin/env bash
set -Eeuo pipefail

# Run one PDE case with one or more quadrature methods.
#
# This script is useful for SLURM/PBS array jobs or for manually rerunning a
# failed case. It writes independent true/sol/abs-error panels and a per-case
# timing table.
#
# Positional arguments:
#   $1 CASE      PDE case: forward, burgers, irregular_hole, or lshape.
#   $2 METHODS   Optional comma-separated methods. Default: GJ-I,GJ-II,MC-I,MC-II.
#
# Examples:
#   bash scripts/server/run_stress_case.sh forward
#   bash scripts/server/run_stress_case.sh lshape GJ-I,GJ-II
#   ALPHAS=1.25,1.5,1.75 bash scripts/server/run_stress_case.sh lshape
#   CASE=irregular_hole METHODS=MC-I,MC-II STEPS=5000 bash scripts/server/run_stress_case.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# ENV_NAME: conda environment name. Set USE_CONDA=0 if already active.
ENV_NAME="${ENV_NAME:-sciml}"
USE_CONDA="${USE_CONDA:-1}"
if [[ "${USE_CONDA}" == "1" ]]; then
  CONDA_BASE="$(conda info --base)"
  if command -v cygpath >/dev/null 2>&1; then
    CONDA_BASE="$(cygpath -u "${CONDA_BASE}")"
  fi
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

# CASE: PDE case to run.
CASE="${1:-${CASE:-forward}}"

# METHODS: quadrature methods.
# GJ-I/GJ-II use Gauss-Jacobi quadrature.
# MC-I/MC-II use Monte Carlo quadrature.
# Suffix I/II selects Type-I or Type-II transformed derivative.
METHODS="${2:-${METHODS:-GJ-I,GJ-II,MC-I,MC-II}}"

# STEPS: optimizer updates. Use 5000 for one paper-style timing epoch.
STEPS="${STEPS:-5000}"

# ALPHAS: comma-separated fractional orders. Use ALPHAS=1.5 for a single alpha.
ALPHAS="${ALPHAS:-1.25,1.5,1.75}"

# Quadrature sizes. Quick stress default is 3; larger paper-scale examples are
# GJ_QUAD=64 and MC_QUAD=640.
GJ_QUAD="${GJ_QUAD:-3}"
MC_QUAD="${MC_QUAD:-3}"

# Batch sizes.
BATCH_IN_1D="${BATCH_IN_1D:-8}"
BATCH_IN_2D="${BATCH_IN_2D:-4}"
BATCH_BD="${BATCH_BD:-2}"
BATCH_INIT="${BATCH_INIT:-2}"

# Network size for stress testing.
HIDDEN_DIM="${HIDDEN_DIM:-16}"
NUM_LAYERS="${NUM_LAYERS:-2}"

# Plot resolution for generated figures.
GRID_1D="${GRID_1D:-80}"
GRID_2D="${GRID_2D:-90}"

# Optional 2D time slices, e.g. TIME_SLICES=0.25,0.5,1.0.
TIME_SLICES="${TIME_SLICES:-}"

RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
OUTDIR="${OUTDIR:-outputs/stress_results/${RUN_ID}/${CASE}}"
mkdir -p "${OUTDIR}"

echo "Running stress case"
echo "  CASE=${CASE}"
echo "  METHODS=${METHODS}"
echo "  ALPHAS=${ALPHAS}"
echo "  STEPS=${STEPS}"
echo "  OUTDIR=${OUTDIR}"

IFS=',' read -r -a ALPHA_ARRAY <<< "${ALPHAS}"
SUMMARY_ARGS=()
FAILED_RUNS=()
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-1}"

for alpha_value in "${ALPHA_ARRAY[@]}"; do
  alpha_safe="${alpha_value//./p}"
  part_dir="${OUTDIR}/alpha${alpha_safe}"
  mkdir -p "${part_dir}"

  cmd=(
    python scripts/run_stress_tests.py
    --cases "${CASE}"
    --methods "${METHODS}"
    --alpha "${alpha_value}"
    --steps "${STEPS}"
    --gj-quad "${GJ_QUAD}"
    --mc-quad "${MC_QUAD}"
    --batch-in-1d "${BATCH_IN_1D}"
    --batch-in-2d "${BATCH_IN_2D}"
    --batch-bd "${BATCH_BD}"
    --batch-init "${BATCH_INIT}"
    --hidden-dim "${HIDDEN_DIM}"
    --num-layers "${NUM_LAYERS}"
    --grid-1d "${GRID_1D}"
    --grid-2d "${GRID_2D}"
    --outdir "${part_dir}"
  )

  if [[ -n "${TIME_SLICES}" ]]; then
    cmd+=(--time-slices "${TIME_SLICES}")
  fi

  if "${cmd[@]}" 2>&1 | tee "${part_dir}/run.log"; then
    SUMMARY_ARGS+=(--summary "${part_dir}/summary.csv")
  else
    FAILED_RUNS+=("alpha${alpha_value}")
    echo "WARNING: failed CASE=${CASE}, alpha=${alpha_value}" | tee -a "${OUTDIR}/failed_runs.log"
    if [[ "${CONTINUE_ON_FAIL}" != "1" ]]; then
      exit 1
    fi
  fi
done

if [[ "${#SUMMARY_ARGS[@]}" -gt 0 ]]; then
  python scripts/combine_stress_results.py \
    --outdir "${OUTDIR}" \
    --cases "${CASE}" \
    --methods "${METHODS}" \
    --alphas "${ALPHAS}" \
    "${SUMMARY_ARGS[@]}"
else
  echo "No successful runs; nothing to combine." >&2
  exit 1
fi

python - <<'PY' "${OUTDIR}/summary.csv" "${OUTDIR}/timing_axes.csv"
import csv
import sys
from pathlib import Path

summary = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = list(csv.DictReader(summary.open(newline="")))
groups = []
for row in rows:
    key = (row["case"], row["alpha"])
    if key not in groups:
        groups.append(key)

with out.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["case", "alpha", "type_i_seconds", "type_ii_seconds", "gj_seconds", "mc_seconds"],
    )
    writer.writeheader()
    for case, alpha in groups:
        items = [row for row in rows if row["case"] == case and row["alpha"] == alpha]
        def avg(key, value):
            vals = [float(row["elapsed_seconds"]) for row in items if row[key] == value]
            return sum(vals) / len(vals) if vals else float("nan")
        writer.writerow({
            "case": case,
            "alpha": alpha,
            "type_i_seconds": f"{avg('type', 'I'):.8f}",
            "type_ii_seconds": f"{avg('type', 'II'):.8f}",
            "gj_seconds": f"{avg('quadrature', 'GJ'):.8f}",
            "mc_seconds": f"{avg('quadrature', 'MC'):.8f}",
        })
PY

echo "Done."
echo "OUTDIR=${OUTDIR}"
echo "Main figures:"
echo "  ${OUTDIR}/abs_error_preview.png"
echo "  ${OUTDIR}/timing_seconds.png"
echo "Tables:"
echo "  ${OUTDIR}/summary.csv"
echo "  ${OUTDIR}/timing_seconds_pivot.csv"
echo "  ${OUTDIR}/timing_axes.csv"

if [[ "${#FAILED_RUNS[@]}" -gt 0 ]]; then
  echo "Failed runs: ${FAILED_RUNS[*]}" >&2
  exit 2
fi
