#!/usr/bin/env bash
set -euo pipefail

# Interactive front-end for scripts/run_pytorch_cases.sh.
# The unified runner owns the Hydra overrides; this file only collects
# validated parameters and exports them.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${SCRIPT_DIR}/run_pytorch_cases.sh"

if [[ ! -f "${RUNNER}" ]]; then
  printf 'Error: runner not found: %s\n' "${RUNNER}" >&2
  exit 1
fi

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  BOLD=$'\033[1m'
  DIM=$'\033[2m'
  RED=$'\033[31m'
  GREEN=$'\033[32m'
  CYAN=$'\033[36m'
  RESET=$'\033[0m'
else
  BOLD=''
  DIM=''
  RED=''
  GREEN=''
  CYAN=''
  RESET=''
fi

CASE_OPTIONS=("lshape" "irregular_hole" "2d" "burgers" "dw_forward" "1d" "all")
METHOD_OPTIONS=("GJ-I" "GJ-II" "MC-I" "MC-II")

trim() {
  local value="$*"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

note() {
  printf '%s%s%s\n' "${DIM}" "$*" "${RESET}"gbi
}

ok() {
  printf '%s[ok]%s %s\n' "${GREEN}" "${RESET}" "$*"
}

section() {
  printf '\n%s%s%s\n' "${BOLD}" "$1" "${RESET}"
  printf '%s\n' '----------------------------------------'
}

is_uint() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

is_positive_int() {
  is_uint "$1" && (( "$1" > 0 ))
}

is_nonempty() {
  [[ -n "$1" ]]
}

is_ratio() {
  [[ "$1" =~ ^(0|0\.[0-9]+|1|1\.0+)$ ]]
}

is_alpha_list() {
  local input="$1"
  local token
  IFS=',' read -r -a tokens <<< "${input}"
  for token in "${tokens[@]}"; do
    token="$(trim "${token}")"
    case "${token}" in
      1.25|1.250|1.5|1.50|1.500|1.75|1.750) ;;
      *) return 1 ;;
    esac
  done
  return 0
}

read_or_exit() {
  local input
  if ! read -r input; then
    printf '\n'
    exit 130
  fi
  printf '%s' "${input}"
}

assign_var() {
  local name="$1"
  local value="$2"
  printf -v "${name}" '%s' "${value}"
}

ask_value() {
  local var_name="$1"
  local default="$2"
  local label="$3"
  local validator="$4"
  local hint="${5:-}"
  local input value

  while true; do
    printf '\n%s%s%s\n' "${CYAN}" "${label}" "${RESET}"
    printf '%sDefault:%s %s' "${DIM}" "${RESET}" "${default}"
    if [[ -n "${hint}" ]]; then
      printf ' %s(%s)%s' "${DIM}" "${hint}" "${RESET}"
    fi
    printf '\n> '

    input="$(read_or_exit)"
    input="$(trim "${input}")"
    if [[ -z "${input}" || "${input}" == "pass" ]]; then
      value="${default}"
    else
      value="${input}"
    fi

    if "${validator}" "${value}"; then
      assign_var "${var_name}" "${value}"
      ok "${var_name}=${value}"
      return
    fi

    printf '%sInvalid value:%s %s\n' "${RED}" "${RESET}" "${value}"
  done
}

choose_one() {
  local var_name="$1"
  local default="$2"
  local label="$3"
  shift 3
  local options=("$@")
  local input candidate i

  while true; do
    printf '\n%s%s%s\n' "${CYAN}" "${label}" "${RESET}"
    for i in "${!options[@]}"; do
      printf '  %d) %s\n' "$((i + 1))" "${options[$i]}"
    done
    printf '%sDefault:%s %s\n' "${DIM}" "${RESET}" "${default}"
    printf '> '

    input="$(read_or_exit)"
    input="$(trim "${input}")"
    if [[ -z "${input}" || "${input}" == "pass" ]]; then
      candidate="${default}"
    elif is_positive_int "${input}" && (( input >= 1 && input <= ${#options[@]} )); then
      candidate="${options[$((input - 1))]}"
    else
      candidate="${input}"
    fi

    for i in "${!options[@]}"; do
      if [[ "${candidate}" == "${options[$i]}" ]]; then
        assign_var "${var_name}" "${candidate}"
        ok "${var_name}=${candidate}"
        return
      fi
    done

    printf '%sInvalid choice:%s %s\n' "${RED}" "${RESET}" "${candidate}"
  done
}

normalize_method_token() {
  local token="$1"
  local index
  token="$(trim "${token}")"

  if is_positive_int "${token}" && (( token >= 1 && token <= ${#METHOD_OPTIONS[@]} )); then
    index=$((token - 1))
    printf '%s' "${METHOD_OPTIONS[$index]}"
    return 0
  fi

  token="${token^^}"
  case "${token}" in
    GJ-I|GJ-II|MC-I|MC-II)
      printf '%s' "${token}"
      return 0
      ;;
  esac

  return 1
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

choose_methods() {
  local default="$1"
  local input token method normalized

  while true; do
    printf '\n%sSelect integration methods%s\n' "${CYAN}" "${RESET}"
    for i in "${!METHOD_OPTIONS[@]}"; do
      printf '  %d) %s\n' "$((i + 1))" "${METHOD_OPTIONS[$i]}"
    done
    note 'Use comma-separated numbers or names, for example: 1,2 or GJ-I,MC-II. Use all for every method.'
    printf '%sDefault:%s %s\n' "${DIM}" "${RESET}" "${default}"
    printf '> '

    input="$(read_or_exit)"
    input="$(trim "${input}")"
    if [[ -z "${input}" || "${input}" == "pass" ]]; then
      METHODS_CSV="${default}"
      ok "METHODS_CSV=${METHODS_CSV}"
      return
    fi

    if [[ "${input,,}" == "all" ]]; then
      METHODS_CSV="GJ-I,GJ-II,MC-I,MC-II"
      ok "METHODS_CSV=${METHODS_CSV}"
      return
    fi

    normalized=''
    IFS=',' read -r -a tokens <<< "${input}"
    for token in "${tokens[@]}"; do
      if ! method="$(normalize_method_token "${token}")"; then
        normalized=''
        break
      fi
      normalized="$(append_unique_csv "${normalized}" "${method}")"
    done

    if [[ -n "${normalized}" ]]; then
      METHODS_CSV="${normalized}"
      ok "METHODS_CSV=${METHODS_CSV}"
      return
    fi

    printf '%sInvalid method list:%s %s\n' "${RED}" "${RESET}" "${input}"
  done
}

ask_yes_no() {
  local var_name="$1"
  local default="$2"
  local label="$3"
  local input value

  while true; do
    printf '\n%s%s%s\n' "${CYAN}" "${label}" "${RESET}"
    if [[ "${default}" == "1" ]]; then
      printf '%sDefault:%s yes\n> ' "${DIM}" "${RESET}"
    else
      printf '%sDefault:%s no\n> ' "${DIM}" "${RESET}"
    fi

    input="$(read_or_exit)"
    input="$(trim "${input}")"
    if [[ -z "${input}" || "${input}" == "pass" ]]; then
      value="${default}"
    else
      case "${input,,}" in
        y|yes|1|true|on) value="1" ;;
        n|no|0|false|off) value="0" ;;
        *)
          printf '%sInvalid choice:%s %s\n' "${RED}" "${RESET}" "${input}"
          continue
          ;;
      esac
    fi

    assign_var "${var_name}" "${value}"
    ok "${var_name}=${value}"
    return
  done
}

case_uses_burgers() {
  [[ "${CASES_ARG}" == "burgers" || "${CASES_ARG}" == "1d" || "${CASES_ARG}" == "all" ]]
}

count_csv_items() {
  local csv="$1"
  local -a items
  IFS=',' read -r -a items <<< "${csv}"
  printf '%d' "${#items[@]}"
}

case_units() {
  local alpha_count=0
  if [[ -n "${ALPHAS_CSV:-}" ]]; then
    alpha_count="$(count_csv_items "${ALPHAS_CSV}")"
  fi

  case "${CASES_ARG}" in
    burgers) printf '%d' "${alpha_count}" ;;
    dw_forward|lshape|irregular_hole) printf '1' ;;
    1d) printf '%d' "$((1 + alpha_count))" ;;
    2d) printf '2' ;;
    all) printf '%d' "$((3 + alpha_count))" ;;
    *) printf '1' ;;
  esac
}

print_row() {
  printf '  %-22s %s\n' "$1" "$2"
}

print_summary() {
  local method_count total_runs
  method_count="$(count_csv_items "${METHODS_CSV}")"
  total_runs=$(( $(case_units) * method_count ))

  section 'Run Summary'
  print_row 'Cases' "${CASES_ARG}"
  if case_uses_burgers; then
    print_row 'Burgers alphas' "${ALPHAS_CSV}"
  fi
  print_row 'Methods' "${METHODS_CSV}"
  print_row 'Estimated runs' "${total_runs}"
  printf '\n'
  print_row 'STEPS' "${STEPS}"
  print_row 'STEPS_PER_EPOCH' "${STEPS_PER_EPOCH}"
  print_row 'USE_LBFGS' "${USE_LBFGS}"
  print_row 'LBFGS_MAX_ITER' "${LBFGS_MAX_ITER}"
  print_row 'TIMING_EPOCH_STEPS' "${TIMING_EPOCH_STEPS}"
  print_row 'LOSS_LOG_EVERY' "${LOSS_LOG_EVERY}"
  print_row 'EVAL_EVERY_STEPS' "${EVAL_EVERY_STEPS}"
  print_row 'EVAL_EVERY_EPOCHS' "${EVAL_EVERY_EPOCHS}"
  printf '\n'
  print_row 'DOMAIN_BATCH' "${DOMAIN_BATCH}"
  print_row 'BOUNDARY_BATCH' "${BOUNDARY_BATCH}"
  print_row 'INITIAL_BATCH' "${INITIAL_BATCH}"
  print_row 'RAD_USE' "${RAD_USE}"
  print_row 'RAD_RATIO' "${RAD_RATIO}"
  print_row 'RAD_DOMAIN_BATCH' "${RAD_DOMAIN_BATCH}"
  print_row 'RAD_BOUNDARY_BATCH' "${RAD_BOUNDARY_BATCH}"
  print_row 'RAD_INITIAL_BATCH' "${RAD_INITIAL_BATCH}"
  printf '\n'
  print_row 'GJ_QUAD' "${GJ_QUAD}"
  print_row 'MC_QUAD' "${MC_QUAD}"
  print_row 'HIDDEN_DIM' "${HIDDEN_DIM}"
  print_row 'NUM_LAYERS' "${NUM_LAYERS}"
  print_row 'PLOT_GRID' "${PLOT_GRID}"
  print_row 'WANDB_MODE' "${WANDB_MODE}"
  print_row 'PYTHON' "${PYTHON_BIN}"
}

print_replay_command() {
  cat <<EOF
cd "${ROOT_DIR}"
STEPS="${STEPS}" \\
STEPS_PER_EPOCH="${STEPS_PER_EPOCH}" \\
USE_LBFGS="${USE_LBFGS}" \\
LBFGS_MAX_ITER="${LBFGS_MAX_ITER}" \\
TIMING_EPOCH_STEPS="${TIMING_EPOCH_STEPS}" \\
LOSS_LOG_EVERY="${LOSS_LOG_EVERY}" \\
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS}" \\
EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS}" \\
DOMAIN_BATCH="${DOMAIN_BATCH}" \\
BOUNDARY_BATCH="${BOUNDARY_BATCH}" \\
INITIAL_BATCH="${INITIAL_BATCH}" \\
RAD_USE="${RAD_USE}" \\
RAD_RATIO="${RAD_RATIO}" \\
RAD_DOMAIN_BATCH="${RAD_DOMAIN_BATCH}" \\
RAD_BOUNDARY_BATCH="${RAD_BOUNDARY_BATCH}" \\
RAD_INITIAL_BATCH="${RAD_INITIAL_BATCH}" \\
GJ_QUAD="${GJ_QUAD}" \\
MC_QUAD="${MC_QUAD}" \\
HIDDEN_DIM="${HIDDEN_DIM}" \\
NUM_LAYERS="${NUM_LAYERS}" \\
PLOT_GRID="${PLOT_GRID}" \\
ALPHAS="${ALPHAS_CSV:-}" \\
WANDB_MODE="${WANDB_MODE}" \\
PYTHON="${PYTHON_BIN}" \\
bash scripts/run_pytorch_cases.sh "${CASES_ARG}" "${METHODS_CSV}"
EOF
}

confirm_and_run() {
  local input

  while true; do
    printf '\n%sAction%s\n' "${BOLD}" "${RESET}"
    printf '  Enter: run now\n'
    printf '  p:     print replay command and exit\n'
    printf '  q:     quit\n'
    printf '> '

    input="$(read_or_exit)"
    input="$(trim "${input}")"
    case "${input,,}" in
      '')
        return 0
        ;;
      p|print)
        printf '\n'
        print_replay_command
        exit 0
        ;;
      q|quit|n|no)
        printf 'Cancelled.\n'
        exit 0
        ;;
      *)
        printf '%sInvalid action:%s %s\n' "${RED}" "${RESET}" "${input}"
        ;;
    esac
  done
}

main() {
  local batch_domain_default batch_boundary_default batch_initial_default gj_default mc_default
  if [[ -t 1 && "${TERM:-}" != "dumb" ]]; then
    clear || true
  fi

  printf '%s========================================%s\n' "${BOLD}" "${RESET}"
  printf '%s  tDWfPINN PyTorch Run Selector%s\n' "${BOLD}" "${RESET}"
  printf '%s========================================%s\n' "${BOLD}" "${RESET}"
  note 'Press Enter or type pass to keep a default value. Press Ctrl+C to cancel.'

  section '1. Target'
  choose_one CASES_ARG "lshape" "Select PDE target" "${CASE_OPTIONS[@]}"
  if case_uses_burgers; then
    ask_value ALPHAS_CSV "1.5" "Burgers alpha list" is_alpha_list "available: 1.25,1.5,1.75"
  else
    ALPHAS_CSV=""
  fi
  choose_methods "GJ-II"

  if [[ "${CASES_ARG}" == "burgers" ]]; then
    batch_domain_default="1000"
    batch_boundary_default="100"
    batch_initial_default="100"
    gj_default="80"
    mc_default="80"
  else
    batch_domain_default="64"
    batch_boundary_default="16"
    batch_initial_default="16"
    gj_default="64"
    mc_default="640"
  fi

  section '2. Training Schedule'
  ask_value STEPS "5000" "Adam update steps" is_positive_int
  ask_yes_no USE_LBFGS "0" "Enable Adam + L-BFGS hybrid training?"
  ask_value STEPS_PER_EPOCH "${STEPS}" "Adam steps per hybrid epoch" is_positive_int
  if [[ "${USE_LBFGS}" == "1" ]]; then
    ask_value LBFGS_MAX_ITER "10" "L-BFGS max_iter per hybrid epoch" is_positive_int
  else
    LBFGS_MAX_ITER="10"
    note 'Skipping L-BFGS max_iter because hybrid training is disabled.'
  fi
  ask_value TIMING_EPOCH_STEPS "${STEPS}" "Timing row interval in Adam steps" is_positive_int
  ask_value LOSS_LOG_EVERY "100" "Loss logging interval in Adam steps" is_uint "0 disables periodic loss logs"
  ask_value EVAL_EVERY_STEPS "${STEPS}" "Adam-only evaluation interval in steps" is_uint "0 means final evaluation only"
  ask_value EVAL_EVERY_EPOCHS "1" "Hybrid evaluation interval in epochs" is_uint "0 means final evaluation only"

  section '3. Sampling And RAD'
  ask_value DOMAIN_BATCH "${batch_domain_default}" "Interior collocation batch size" is_positive_int
  ask_value BOUNDARY_BATCH "${batch_boundary_default}" "Boundary batch size" is_positive_int
  ask_value INITIAL_BATCH "${batch_initial_default}" "Initial-condition batch size" is_positive_int
  ask_yes_no RAD_USE "0" "Enable residual-adaptive sampling (RAD)?"
  ask_value RAD_RATIO "0.8" "RAD replacement ratio" is_ratio "0 to 1"
  ask_value RAD_DOMAIN_BATCH "1000" "RAD interior candidate batch size" is_positive_int
  ask_value RAD_BOUNDARY_BATCH "2" "RAD boundary candidate batch size" is_positive_int
  ask_value RAD_INITIAL_BATCH "2" "RAD initial candidate batch size" is_positive_int
  ask_value GJ_QUAD "${gj_default}" "Gauss-Jacobi quadrature nodes" is_positive_int
  ask_value MC_QUAD "${mc_default}" "Monte Carlo samples" is_positive_int

  section '4. Model And Output'
  ask_value HIDDEN_DIM "64" "MLP hidden width" is_positive_int
  ask_value NUM_LAYERS "4" "MLP hidden layers" is_positive_int
  ask_value PLOT_GRID "80" "2D plot grid" is_positive_int
  choose_one WANDB_MODE "disabled" "W&B mode" "disabled" "offline" "online"
  ask_value PYTHON_BIN "${PYTHON:-python}" "Python executable" is_nonempty

  print_summary
  confirm_and_run

  export STEPS STEPS_PER_EPOCH USE_LBFGS LBFGS_MAX_ITER
  export TIMING_EPOCH_STEPS LOSS_LOG_EVERY EVAL_EVERY_STEPS EVAL_EVERY_EPOCHS
  export DOMAIN_BATCH BOUNDARY_BATCH INITIAL_BATCH
  export RAD_USE RAD_RATIO RAD_DOMAIN_BATCH RAD_BOUNDARY_BATCH RAD_INITIAL_BATCH
  export GJ_QUAD MC_QUAD HIDDEN_DIM NUM_LAYERS PLOT_GRID WANDB_MODE
  export ALPHAS="${ALPHAS_CSV:-}"
  export PYTHON="${PYTHON_BIN}"

  cd "${ROOT_DIR}"
  bash "${RUNNER}" "${CASES_ARG}" "${METHODS_CSV}"
}

main "$@"
