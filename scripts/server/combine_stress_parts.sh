#!/usr/bin/env bash
set -Eeuo pipefail

# Combine stress-test summaries produced by several independent jobs.
#
# Use this after launching per-case or per-method jobs with run_stress_case.sh
# or with your cluster scheduler. The script scans OUTDIR recursively for
# summary.csv files, combines them, and writes aggregate figures/tables.
#
# Example:
#   OUTDIR=outputs/stress_results/2026-05-21_server bash scripts/server/combine_stress_parts.sh

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

# OUTDIR: root directory containing per-run subfolders with summary.csv files.
OUTDIR="${OUTDIR:-outputs/stress_results/latest}"

# CASES/METHODS define display order in the merged tables and plots.
CASES="${CASES:-forward,burgers,irregular_hole,lshape}"
METHODS="${METHODS:-GJ-I,GJ-II,MC-I,MC-II}"
ALPHAS="${ALPHAS:-1.25,1.5,1.75}"

if [[ ! -d "${OUTDIR}" ]]; then
  echo "OUTDIR does not exist: ${OUTDIR}" >&2
  exit 1
fi

SUMMARY_ARGS=()
while IFS= read -r summary_path; do
  # Skip an already combined top-level summary to avoid duplicate rows.
  if [[ "${summary_path}" == "${OUTDIR}/summary.csv" ]]; then
    continue
  fi
  SUMMARY_ARGS+=(--summary "${summary_path}")
done < <(find "${OUTDIR}" -mindepth 2 -name summary.csv | sort)

if [[ "${#SUMMARY_ARGS[@]}" -eq 0 ]]; then
  echo "No nested summary.csv files found under ${OUTDIR}" >&2
  exit 1
fi

python scripts/combine_stress_results.py \
  --outdir "${OUTDIR}" \
  --cases "${CASES}" \
  --methods "${METHODS}" \
  --alphas "${ALPHAS}" \
  "${SUMMARY_ARGS[@]}"

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

echo "Combined stress results:"
echo "  ${OUTDIR}/summary.csv"
echo "  ${OUTDIR}/timing_seconds_pivot.csv"
echo "  ${OUTDIR}/timing_axes.csv"
echo "  ${OUTDIR}/timing_seconds.png"
echo "  ${OUTDIR}/abs_error_preview.png"
