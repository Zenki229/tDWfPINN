#!/usr/bin/env bash
set -Eeuo pipefail

# Run the full 5000-step stress suite for all PDEs and all quadrature methods.
#
# Output:
#   outputs/stress_results/<RUN_ID>/
#     summary.csv                 full table with loss, relative error, timing, and figure paths
#     timing_seconds_pivot.csv    seconds per 5000 steps for GJ-I/GJ-II/MC-I/MC-II
#     timing_axes.csv             average Type-I/Type-II and GJ/MC timings
#     timing_seconds.png          timing bar chart
#     abs_error_preview.png       overview image of abs-error panels
#     parts/<case>_<method>/      per-run figures and logs
#
# Typical use:
#   bash scripts/server/run_stress_all.sh
#
# Run only selected cases/methods with positional arguments:
#   bash scripts/server/run_stress_all.sh lshape GJ-I,GJ-II,MC-I,MC-II
#
# Useful overrides:
#   ENV_NAME=sciml bash scripts/server/run_stress_all.sh
#   STEPS=5000 GJ_QUAD=64 MC_QUAD=640 bash scripts/server/run_stress_all.sh
#   CASES=forward,burgers METHODS=GJ-II,MC-II bash scripts/server/run_stress_all.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# ENV_NAME: conda environment name. Set USE_CONDA=0 if the environment is
# already active or if the server uses venv/module loading instead of conda.
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

# XLA_PYTHON_CLIENT_PREALLOCATE=false prevents JAX from grabbing most GPU
# memory before the run starts. Keep it false for shared servers.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

# RUN_ID: output folder name. Default is a timestamp.
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
OUTDIR="${OUTDIR:-outputs/stress_results/${RUN_ID}}"
mkdir -p "${OUTDIR}/parts"

# CASES: comma-separated PDE cases. The first positional argument overrides
# the CASES environment variable when provided.
# Valid values: forward, burgers, irregular_hole, lshape.
CASES="${1:-${CASES:-forward,burgers,irregular_hole,lshape}}"

# METHODS: comma-separated quadrature methods. The second positional argument
# overrides the METHODS environment variable when provided.
# GJ-I  = Gauss-Jacobi quadrature + Type-I transformed derivative.
# GJ-II = Gauss-Jacobi quadrature + Type-II transformed derivative.
# MC-I  = Monte Carlo quadrature + Type-I transformed derivative.
# MC-II = Monte Carlo quadrature + Type-II transformed derivative.
METHODS="${2:-${METHODS:-GJ-I,GJ-II,MC-I,MC-II}}"

# STEPS: optimizer updates per (PDE, method). The paper-style timing epoch is
# 5000 steps, so STEPS=5000 gives one timing measurement per run.
STEPS="${STEPS:-5000}"

# GJ_QUAD / MC_QUAD: number of quadrature points/samples used inside each
# fractional derivative evaluation. For quick stress tests use 3. For paper
# scale, typical values are GJ_QUAD=64 and MC_QUAD=640.
GJ_QUAD="${GJ_QUAD:-3}"
MC_QUAD="${MC_QUAD:-3}"

# Batch sizes. 2D residuals are more expensive, so the default interior batch
# is smaller for irregular_hole/lshape.
BATCH_IN_1D="${BATCH_IN_1D:-8}"
BATCH_IN_2D="${BATCH_IN_2D:-4}"
BATCH_BD="${BATCH_BD:-2}"
BATCH_INIT="${BATCH_INIT:-2}"

# Network size used for stress tests. Increase these for actual training.
HIDDEN_DIM="${HIDDEN_DIM:-16}"
NUM_LAYERS="${NUM_LAYERS:-2}"

# Plot grids used after training to generate true/sol/abs-error panels.
GRID_1D="${GRID_1D:-80}"
GRID_2D="${GRID_2D:-90}"

# Optional comma-separated time slices for 2D plots, e.g. TIME_SLICES=0.5,1.0.
TIME_SLICES="${TIME_SLICES:-}"

# CONTINUE_ON_FAIL=1 keeps the sweep running if one configuration fails.
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-1}"

IFS=',' read -r -a CASE_ARRAY <<< "${CASES}"
IFS=',' read -r -a METHOD_ARRAY <<< "${METHODS}"

SUMMARY_ARGS=()
FAILED_RUNS=()

for case_name in "${CASE_ARRAY[@]}"; do
  for method_name in "${METHOD_ARRAY[@]}"; do
    safe_method="${method_name//-/_}"
    part_dir="${OUTDIR}/parts/${case_name}_${safe_method}"
    mkdir -p "${part_dir}"

    echo "============================================================"
    echo "case=${case_name}, method=${method_name}, steps=${STEPS}"
    echo "outdir=${part_dir}"
    echo "============================================================"

    cmd=(
      python scripts/run_stress_tests.py
      --cases "${case_name}"
      --methods "${method_name}"
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
      FAILED_RUNS+=("${case_name}/${method_name}")
      echo "WARNING: failed case=${case_name}, method=${method_name}" | tee -a "${OUTDIR}/failed_runs.log"
      if [[ "${CONTINUE_ON_FAIL}" != "1" ]]; then
        exit 1
      fi
    fi
  done
done

if [[ "${#SUMMARY_ARGS[@]}" -gt 0 ]]; then
  python scripts/combine_stress_results.py \
    --outdir "${OUTDIR}" \
    --cases "${CASES}" \
    --methods "${METHODS}" \
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
cases = []
for row in rows:
    if row["case"] not in cases:
        cases.append(row["case"])

with out.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["case", "type_i_seconds", "type_ii_seconds", "gj_seconds", "mc_seconds"],
    )
    writer.writeheader()
    for case in cases:
        items = [row for row in rows if row["case"] == case]
        def avg(key, value):
            vals = [float(row["elapsed_seconds"]) for row in items if row[key] == value]
            return sum(vals) / len(vals) if vals else float("nan")
        writer.writerow({
            "case": case,
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
