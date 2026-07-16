#!/usr/bin/env bash
set -euo pipefail

# Unified PyTorch runner for 1D and 2D tDWfPINN cases.
#
# Usage:
#   bash scripts/run_pytorch_cases.sh 2d GJ-I,GJ-II
#   bash scripts/run_pytorch_cases.sh burgers GJ-II
#   ALPHAS=1.25,1.5 STEPS=5000 bash scripts/run_pytorch_cases.sh burgers GJ-II
#   CASES=dw_forward,burgers METHODS=GJ-II bash scripts/run_pytorch_cases.sh
#
# Positional arguments:
#   $1  Cases: dw_forward, burgers, irregular_hole, lshape, 1d, 2d, or all.
#       Default: ${CASES:-2d}.
#   $2  Methods: GJ-I,GJ-II,MC-I,MC-II. Default: ${METHODS:-GJ-I,GJ-II,MC-I,MC-II}.
#
# Common environment variables:
#   STEPS, STEPS_PER_EPOCH, USE_LBFGS, LBFGS_MAX_ITER
#   TIMING_EPOCH_STEPS, LOSS_LOG_EVERY, EVAL_EVERY_STEPS, EVAL_EVERY_EPOCHS
#   DOMAIN_BATCH, BOUNDARY_BATCH, INITIAL_BATCH
#   RAD_USE, RAD_RATIO, RAD_DOMAIN_BATCH   (RAD only resamples interior/domain points)
#   GJ_QUAD, MC_QUAD, HIDDEN_DIM, NUM_LAYERS
#   PLOT_GRID          Manufactured-reference grid for irregular_hole only.
#   ALPHAS              Burgers/L-shape reference alphas. Default: 1.25,1.5,1.75.
#   FORWARD_ALPHAS      Optional dw_forward alpha sweep. Default: use config alpha.
#   DATA_DIR            Directory containing burgers_*.npz. Default: data.
#   PLOT_BACKEND        Plot backend. Default: matplotlib.
#   PLOT_JPG            Optional plot.jpg override.
#   WANDB_MODE          W&B mode. Default: disabled.
#   WANDB_PROJECT       W&B project. Default: tDWfPINN.
#   PYTHON              Python executable. Default: python.
#   DRY_RUN             Print commands without running. Default: 0.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

CASES_INPUT="${1:-${CASES:-2d}}"
METHODS_INPUT="${2:-${METHODS:-GJ-I,GJ-II,MC-I,MC-II}}"

STEPS="${STEPS:-5000}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-5000}"
USE_LBFGS="${USE_LBFGS:-0}"
LBFGS_MAX_ITER="${LBFGS_MAX_ITER:-10}"
TIMING_EPOCH_STEPS="${TIMING_EPOCH_STEPS:-5000}"
LOSS_LOG_EVERY="${LOSS_LOG_EVERY:-100}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-5000}"
EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-1}"
RAD_USE="${RAD_USE:-0}"
RAD_RATIO="${RAD_RATIO:-0.8}"
RAD_DOMAIN_BATCH="${RAD_DOMAIN_BATCH:-1000}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
NUM_LAYERS="${NUM_LAYERS:-4}"
PLOT_GRID="${PLOT_GRID:-80}"
ALPHAS_CSV="${ALPHAS:-1.25,1.5,1.75}"
FORWARD_ALPHAS_CSV="${FORWARD_ALPHAS:-}"
DATA_DIR="${DATA_DIR:-data}"
PLOT_BACKEND="${PLOT_BACKEND:-matplotlib}"
WANDB_MODE="${WANDB_MODE:-disabled}"
WANDB_PROJECT="${WANDB_PROJECT:-tDWfPINN}"
PYTHON_BIN="${PYTHON:-python}"
DRY_RUN="${DRY_RUN:-0}"

trim() {
  local value="$*"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

to_lower() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

to_upper() {
  printf '%s' "$1" | tr '[:lower:]' '[:upper:]'
}

append_unique_csv() {
  local csv="$1"
  local value="$2"
  if [[ -z "${csv}" ]]; then
    printf '%s' "${value}"
  elif [[ ",${csv}," == *",${value},"* ]]; then
    printf '%s' "${csv}"
  else
    printf '%s,%s' "${csv}" "${value}"
  fi
}

bool_value() {
  local normalized
  normalized="$(to_lower "$1")"
  case "${normalized}" in
    1|true|yes|y|on) printf 'true' ;;
    0|false|no|n|off) printf 'false' ;;
    *)
      printf 'Invalid boolean value: %s\n' "$1" >&2
      exit 1
      ;;
  esac
}

normalize_cases() {
  local input="$1"
  local normalized='' token token_lower
  IFS=',' read -r -a case_tokens <<< "${input}"
  for token in "${case_tokens[@]}"; do
    token="$(trim "${token}")"
    token_lower="$(to_lower "${token}")"
    case "${token_lower}" in
      1d|all_1d|all-1d)
        normalized="$(append_unique_csv "${normalized}" "dw_forward")"
        normalized="$(append_unique_csv "${normalized}" "burgers")"
        ;;
      2d|all_2d|all-2d)
        normalized="$(append_unique_csv "${normalized}" "irregular_hole")"
        normalized="$(append_unique_csv "${normalized}" "lshape")"
        ;;
      all)
        normalized="$(append_unique_csv "${normalized}" "dw_forward")"
        normalized="$(append_unique_csv "${normalized}" "burgers")"
        normalized="$(append_unique_csv "${normalized}" "irregular_hole")"
        normalized="$(append_unique_csv "${normalized}" "lshape")"
        ;;
      forward|dw|dw_forward)
        normalized="$(append_unique_csv "${normalized}" "dw_forward")"
        ;;
      burgers|irregular_hole|lshape)
        normalized="$(append_unique_csv "${normalized}" "${token_lower}")"
        ;;
      *)
        printf 'Unsupported case: %s\n' "${token}" >&2
        exit 1
        ;;
    esac
  done
  printf '%s' "${normalized}"
}

normalize_methods() {
  local input="$1"
  local normalized='' token method
  if [[ "$(to_lower "${input}")" == "all" ]]; then
    printf 'GJ-I,GJ-II,MC-I,MC-II'
    return
  fi
  IFS=',' read -r -a method_tokens <<< "${input}"
  for token in "${method_tokens[@]}"; do
    method="$(trim "${token}")"
    method="$(to_upper "${method}")"
    case "${method}" in
      GJ-I|GJ-II|MC-I|MC-II)
        normalized="$(append_unique_csv "${normalized}" "${method}")"
        ;;
      *)
        printf 'Unsupported method: %s\n' "${token}" >&2
        exit 1
        ;;
    esac
  done
  printf '%s' "${normalized}"
}

alpha_token() {
  local alpha="$1"
  case "${alpha}" in
    1.25|1.250) printf '125' ;;
    1.5|1.50|1.500) printf '150' ;;
    1.75|1.750) printf '175' ;;
    *)
      printf 'Unsupported Burgers alpha: %s\n' "${alpha}" >&2
      printf 'Available reference data: 1.25, 1.5, 1.75\n' >&2
      exit 1
      ;;
  esac
}

lshape_alpha_token() {
  local alpha="$1"
  case "${alpha}" in
    1.25|1.250) printf '1p25' ;;
    1.5|1.50|1.500) printf '1p50' ;;
    1.75|1.750) printf '1p75' ;;
    *)
      printf 'Unsupported lshape alpha: %s\n' "${alpha}" >&2
      printf 'Available reference data: 1.25, 1.5, 1.75\n' >&2
      exit 1
      ;;
  esac
}

case_default() {
  local case_name="$1"
  local field="$2"
  case "${case_name}:${field}" in
    burgers:domain) printf '1000' ;;
    burgers:boundary) printf '100' ;;
    burgers:initial) printf '100' ;;
    burgers:gj) printf '80' ;;
    burgers:mc) printf '80' ;;
    *:domain) printf '64' ;;
    *:boundary) printf '16' ;;
    *:initial) printf '16' ;;
    *:gj) printf '64' ;;
    *:mc) printf '640' ;;
    *)
      printf 'Unknown default field: %s\n' "${field}" >&2
      exit 1
      ;;
  esac
}

value_or_case_default() {
  local env_name="$1"
  local case_name="$2"
  local field="$3"
  local value="${!env_name-}"
  if [[ -n "${value}" ]]; then
    printf '%s' "${value}"
  else
    case_default "${case_name}" "${field}"
  fi
}

print_command() {
  printf 'Command:'
  printf ' %q' "$@"
  printf '\n'
}

run_one() {
  local case_name="$1"
  local method="$2"
  local alpha="${3:-}"
  local domain_batch boundary_batch initial_batch gj_quad mc_quad
  local run_stamp run_dir datafile token rad_use_bool use_lbfgs_bool dry_run_bool

  domain_batch="$(value_or_case_default DOMAIN_BATCH "${case_name}" domain)"
  boundary_batch="$(value_or_case_default BOUNDARY_BATCH "${case_name}" boundary)"
  initial_batch="$(value_or_case_default INITIAL_BATCH "${case_name}" initial)"
  gj_quad="$(value_or_case_default GJ_QUAD "${case_name}" gj)"
  mc_quad="$(value_or_case_default MC_QUAD "${case_name}" mc)"
  rad_use_bool="$(bool_value "${RAD_USE}")"
  use_lbfgs_bool="$(bool_value "${USE_LBFGS}")"
  dry_run_bool="$(bool_value "${DRY_RUN}")"

  run_stamp="$(date +%Y-%m-%d_%H-%M-%S)"
  case "${case_name}" in
    burgers)
      token="$(alpha_token "${alpha}")"
      datafile="${DATA_DIR}/burgers_${token}.npz"
      if [[ ! -f "${datafile}" ]]; then
        printf 'Missing reference data: %s\n' "${datafile}" >&2
        exit 1
      fi
      run_dir="outputs/pytorch_burgers/alpha${alpha}/${method}/${run_stamp}"
      ;;
    dw_forward)
      if [[ -n "${alpha}" ]]; then
        run_dir="outputs/pytorch_1d/dw_forward/alpha${alpha}/${method}/${run_stamp}"
      else
        run_dir="outputs/pytorch_1d/dw_forward/${method}/${run_stamp}"
      fi
      ;;
    lshape)
      if [[ -n "${alpha}" ]]; then
        token="$(lshape_alpha_token "${alpha}")"
        datafile="${DATA_DIR}/lshape/lshape_reference_alpha${token}.npz"
        if [[ ! -f "${datafile}" ]]; then
          printf 'Missing reference data: %s\n' "${datafile}" >&2
          exit 1
        fi
        run_dir="outputs/pytorch_2d/lshape/alpha${alpha}/${method}/${run_stamp}"
      else
        run_dir="outputs/pytorch_2d/lshape/${method}/${run_stamp}"
      fi
      ;;
    irregular_hole)
      run_dir="outputs/pytorch_2d/${case_name}/${method}/${run_stamp}"
      ;;
  esac

  printf 'Running case=%s' "${case_name}"
  if [[ -n "${alpha}" ]]; then
    printf ', alpha=%s' "${alpha}"
  fi
  printf ', method=%s, steps=%s, rad=%s, output=%s\n' \
    "${method}" "${STEPS}" "${rad_use_bool}" "${run_dir}"

  cmd=(
    "${PYTHON_BIN}" src/train.py
    "pde=${case_name}"
    "plot=${PLOT_BACKEND}"
    "wandb.mode=${WANDB_MODE}"
    "wandb.project=${WANDB_PROJECT}"
    "pde.method=${method}"
    "trainer.max_steps=${STEPS}"
    "trainer.steps_per_epoch=${STEPS_PER_EPOCH}"
    "trainer.timing.epoch_steps=${TIMING_EPOCH_STEPS}"
    "trainer.loss_log_every_steps=${LOSS_LOG_EVERY}"
    "trainer.eval_every_steps=${EVAL_EVERY_STEPS}"
    "trainer.eval_every_epochs=${EVAL_EVERY_EPOCHS}"
    "trainer.batch_size.domain=${domain_batch}"
    "trainer.batch_size.boundary=${boundary_batch}"
    "trainer.batch_size.initial=${initial_batch}"
    "trainer.rad.use=${rad_use_bool}"
    "trainer.rad.ratio=${RAD_RATIO}"
    "trainer.rad.batch.domain=${RAD_DOMAIN_BATCH}"
    "pde.gauss_jacobi_params.nums=${gj_quad}"
    "pde.gj_params.nums=${gj_quad}"
    "pde.monte_carlo_params.nums=${mc_quad}"
    "model.hidden_dim=${HIDDEN_DIM}"
    "model.num_layers=${NUM_LAYERS}"
    "hydra.run.dir=${run_dir}"
  )

  if [[ "${case_name}" == "burgers" ]]; then
    cmd+=("pde.alpha=${alpha}" "pde.datafile=${datafile}")
  elif [[ "${case_name}" == "dw_forward" && -n "${alpha}" ]]; then
    cmd+=("pde.alpha=${alpha}")
  elif [[ "${case_name}" == "lshape" && -n "${alpha}" ]]; then
    cmd+=("pde.alpha=${alpha}" "pde.reference_data=${datafile}")
  fi

  if [[ "${case_name}" == "irregular_hole" ]]; then
    cmd+=("pde.plot_grid=${PLOT_GRID}")
  fi

  if [[ -n "${PLOT_JPG:-}" ]]; then
    cmd+=("plot.jpg=${PLOT_JPG}")
  fi

  if [[ "${use_lbfgs_bool}" == "true" ]]; then
    cmd+=(optimizer.lbfgs.use=true "optimizer.lbfgs.max_iter=${LBFGS_MAX_ITER}")
  fi

  if [[ "${dry_run_bool}" == "true" ]]; then
    print_command "${cmd[@]}"
    return
  fi

  "${cmd[@]}"
}

CASES_CSV="$(normalize_cases "${CASES_INPUT}")"
METHODS_CSV="$(normalize_methods "${METHODS_INPUT}")"
IFS=',' read -r -a CASES_LIST <<< "${CASES_CSV}"
IFS=',' read -r -a METHODS_LIST <<< "${METHODS_CSV}"
IFS=',' read -r -a BURGERS_ALPHAS <<< "${ALPHAS_CSV}"

FORWARD_ALPHAS=()
if [[ -n "${FORWARD_ALPHAS_CSV}" ]]; then
  IFS=',' read -r -a FORWARD_ALPHAS <<< "${FORWARD_ALPHAS_CSV}"
fi

for case_name in "${CASES_LIST[@]}"; do
  for method in "${METHODS_LIST[@]}"; do
    if [[ "${case_name}" == "burgers" ]]; then
      for alpha_raw in "${BURGERS_ALPHAS[@]}"; do
        run_one "${case_name}" "${method}" "$(trim "${alpha_raw}")"
      done
    elif [[ "${case_name}" == "lshape" && "${#BURGERS_ALPHAS[@]}" -gt 0 ]]; then
      for alpha_raw in "${BURGERS_ALPHAS[@]}"; do
        run_one "${case_name}" "${method}" "$(trim "${alpha_raw}")"
      done
    elif [[ "${case_name}" == "dw_forward" && "${#FORWARD_ALPHAS[@]}" -gt 0 ]]; then
      for alpha_raw in "${FORWARD_ALPHAS[@]}"; do
        run_one "${case_name}" "${method}" "$(trim "${alpha_raw}")"
      done
    else
      run_one "${case_name}" "${method}"
    fi
  done
done
