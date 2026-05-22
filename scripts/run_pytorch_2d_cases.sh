#!/usr/bin/env bash
set -euo pipefail

# Run the PyTorch 2D irregular-domain cases and save plots plus timing.csv.
#
# Usage:
#   bash scripts/run_pytorch_2d_cases.sh all GJ-I,GJ-II,MC-I,MC-II
#   STEPS=5000 GJ_QUAD=64 MC_QUAD=640 bash scripts/run_pytorch_2d_cases.sh lshape GJ-II
#
# Positional arguments:
#   $1  Case name: irregular_hole, lshape, or all. Default: all.
#   $2  Comma-separated methods: GJ-I,GJ-II,MC-I,MC-II. Default: all four.
#
# Environment variables:
#   STEPS               Adam update steps. Default: 5000.
#   STEPS_PER_EPOCH     Adam updates per hybrid epoch. Default: 5000.
#   USE_LBFGS           Enable Adam + L-BFGS hybrid schedule. Default: 0.
#   LBFGS_MAX_ITER      L-BFGS iterations per hybrid epoch. Default: 10.
#   TIMING_EPOCH_STEPS  Steps per timing row. Default: 5000.
#   DOMAIN_BATCH        Interior collocation batch size. Default: 64.
#   BOUNDARY_BATCH      Boundary batch size. Default: 16.
#   INITIAL_BATCH       Initial-condition batch size. Default: 16.
#   GJ_QUAD             Gauss-Jacobi quadrature nodes. Default: 64.
#   MC_QUAD             Monte Carlo samples. Default: 640.
#   HIDDEN_DIM          MLP hidden width. Default: 64.
#   NUM_LAYERS          MLP hidden layers. Default: 4.
#   PLOT_GRID           Irregular-hole plotting grid. L-shape uses its reference grid. Default: 80.
#   WANDB_MODE          W&B mode: disabled, offline, or online. Default: disabled.
#   PYTHON              Python executable. Default: python.

CASE_NAME="${1:-all}"
METHODS_CSV="${2:-GJ-I,GJ-II,MC-I,MC-II}"

STEPS="${STEPS:-5000}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-5000}"
USE_LBFGS="${USE_LBFGS:-0}"
LBFGS_MAX_ITER="${LBFGS_MAX_ITER:-10}"
TIMING_EPOCH_STEPS="${TIMING_EPOCH_STEPS:-5000}"
DOMAIN_BATCH="${DOMAIN_BATCH:-64}"
BOUNDARY_BATCH="${BOUNDARY_BATCH:-16}"
INITIAL_BATCH="${INITIAL_BATCH:-16}"
GJ_QUAD="${GJ_QUAD:-64}"
MC_QUAD="${MC_QUAD:-640}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
NUM_LAYERS="${NUM_LAYERS:-4}"
PLOT_GRID="${PLOT_GRID:-80}"
WANDB_MODE="${WANDB_MODE:-disabled}"
PYTHON_BIN="${PYTHON:-python}"

if [[ "${CASE_NAME}" == "all" ]]; then
  CASES=("irregular_hole" "lshape")
else
  CASES=("${CASE_NAME}")
fi

IFS=',' read -r -a METHODS <<< "${METHODS_CSV}"

for case in "${CASES[@]}"; do
  for method in "${METHODS[@]}"; do
    run_stamp="$(date +%Y-%m-%d_%H-%M-%S)"
    run_dir="outputs/pytorch_2d/${case}/${method}/${run_stamp}"
    echo "Running case=${case}, method=${method}, steps=${STEPS}, output=${run_dir}"

    cmd=(
      "${PYTHON_BIN}" src/train.py
      "pde=${case}" \
      plot=matplotlib \
      "wandb.mode=${WANDB_MODE}" \
      "pde.method=${method}" \
      "trainer.max_steps=${STEPS}" \
      "trainer.steps_per_epoch=${STEPS_PER_EPOCH}" \
      "trainer.timing.epoch_steps=${TIMING_EPOCH_STEPS}" \
      "trainer.batch_size.domain=${DOMAIN_BATCH}" \
      "trainer.batch_size.boundary=${BOUNDARY_BATCH}" \
      "trainer.batch_size.initial=${INITIAL_BATCH}" \
      trainer.rad.use=false \
      "pde.gauss_jacobi_params.nums=${GJ_QUAD}" \
      "pde.gj_params.nums=${GJ_QUAD}" \
      "pde.monte_carlo_params.nums=${MC_QUAD}" \
      "model.hidden_dim=${HIDDEN_DIM}" \
      "model.num_layers=${NUM_LAYERS}" \
      "pde.plot_grid=${PLOT_GRID}" \
      "hydra.run.dir=${run_dir}"
    )
    if [[ "${USE_LBFGS}" == "1" ]]; then
      cmd+=(
        optimizer.lbfgs.use=true
        "optimizer.lbfgs.max_iter=${LBFGS_MAX_ITER}"
      )
    fi
    "${cmd[@]}"
  done
done
