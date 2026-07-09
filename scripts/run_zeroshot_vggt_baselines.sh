#!/usr/bin/env bash
set -euo pipefail

# Safe defaults: run only the plain VGGT baseline unless METHODS is overridden.
# Example override:
#   METHODS="vggt naive_causal" bash scripts/run_zeroshot_vggt_baselines.sh
DEFAULT_METHODS=(vggt)
if [[ -n "${METHODS:-}" ]]; then
  # shellcheck disable=SC2206
  METHODS_ARR=(${METHODS})
else
  METHODS_ARR=("${DEFAULT_METHODS[@]}")
fi

# Table 1 zero-shot inputs.  The default active set is the 7-Scenes VGGT
# validation target.  Extend this list deliberately before full sweeps.
DATASETS=(
  "7scenes_sfm        dataset/7scenes/7scenes_sfm"
  "7scenes_random100  dataset/7scenes/7scenes_random100_noisy"
  "7scenes_lt3m       dataset/7scenes/7scenes_lt3m_noisy"

  # "scannet_sift       dataset/scannet/scannet_sift_noisy"
  # "scannet_random100  dataset/scannet/scannet_random100_noisy"
  # "scannet_lt3m       dataset/scannet/scannet_lt3m_noisy"

  # "metropolis_8line   dataset/metropolis/metropolis_8line_noisy"
  # "metropolis_16line  dataset/metropolis/metropolis_16line_noisy"
  # "metropolis_32line  dataset/metropolis/metropolis_32line_noisy"
)

OUTPUT_ROOT="${OUTPUT_ROOT:-output/tab1_zeroshot_vggt_baselines}"
CONDA_ENV="${CONDA_ENV-capa}"
COMPUTE_OPW="${COMPUTE_OPW:-0}"
DRY_RUN="${DRY_RUN:-0}"
GMFLOW_CKPT="${GMFLOW_CKPT:-}"
GMFLOW_REPO="${GMFLOW_REPO:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-capa}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "${CONDA_ENV}" ]]; then
  PYTHON_CMD=(conda run --no-capture-output -n "${CONDA_ENV}" python)
else
  PYTHON_CMD=(python)
fi

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

config_for_method() {
  local method="$1"
  case "${method}" in
    vggt)
      echo "config/vggt_baseline.yaml"
      ;;
    naive_causal)
      echo "config/vggt_ln_stream_naive_causal.yaml"
      ;;
    streamvggt)
      echo "config/streamvggt_baseline.yaml"
      ;;
    retrievevggt)
      echo "config/retrievevggt_baseline.yaml"
      ;;
    vggt_omega)
      echo "config/vggt_omega_baseline.yaml"
      ;;
    depth_anything3)
      echo "config/depth_anything3_baseline.yaml"
      ;;
    *)
      echo "Unknown method: ${method}" >&2
      return 2
      ;;
  esac
}

is_optional_config_method() {
  case "$1" in
    vggt_omega|depth_anything3)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

if [[ "${COMPUTE_OPW}" == "1" && -z "${GMFLOW_CKPT}" ]]; then
  echo "ERROR: COMPUTE_OPW=1 requires GMFLOW_CKPT=/path/to/gmflow_sintel-0c07dcb3.pth" >&2
  exit 2
fi

TOTAL_TASKS=$((${#METHODS_ARR[@]} * ${#DATASETS[@]}))
TASK_IDX=0
SCRIPT_START_TS=$(date +%s)

echo "Zero-shot VGGT-family baseline sweep"
echo "  methods      : ${METHODS_ARR[*]}"
echo "  datasets     : ${#DATASETS[@]}"
echo "  total        : ${TOTAL_TASKS} run slot(s)"
echo "  output       : ${OUTPUT_ROOT}"
echo "  conda env    : ${CONDA_ENV:-<none>}"
echo "  cuda devices : ${CUDA_VISIBLE_DEVICES}"
echo "  compute OPW  : ${COMPUTE_OPW}"
echo "  dry run      : ${DRY_RUN}"
if [[ "${COMPUTE_OPW}" == "1" ]]; then
  echo "  gmflow ckpt  : ${GMFLOW_CKPT}"
  echo "  gmflow repo  : ${GMFLOW_REPO:-<PYTHONPATH/GMFLOW_REPO/third_party fallback>}"
fi
echo

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${MPLCONFIGDIR}"
fi

for METHOD in "${METHODS_ARR[@]}"; do
  CONFIG="$(config_for_method "${METHOD}")"

  if [[ ! -f "${CONFIG}" ]]; then
    if is_optional_config_method "${METHOD}"; then
      echo "SKIP: method '${METHOD}' requested but optional config is missing: ${CONFIG}" >&2
      continue
    fi
    echo "ERROR: config for method '${METHOD}' is missing: ${CONFIG}" >&2
    exit 3
  fi

  for ITEM in "${DATASETS[@]}"; do
    TASK_IDX=$((TASK_IDX + 1))
    read -r DATASET_NAME INPUT_DIR <<< "${ITEM}"
    OUT_DIR="${OUTPUT_ROOT}/${DATASET_NAME}/${METHOD}"
    LOG_FILE="${OUT_DIR}/run.log"
    OPW_LOG_FILE="${OUT_DIR}/opw.log"
    RUN_START_TS=$(date +%s)

    echo "================================================================"
    echo "[${TASK_IDX}/${TOTAL_TASKS}] ${METHOD} | ${DATASET_NAME}"
    echo "  method : ${METHOD}"
    echo "  dataset: ${DATASET_NAME}"
    echo "  config : ${CONFIG}"
    echo "  input  : ${INPUT_DIR}"
    echo "  output : ${OUT_DIR}"
    echo "  start  : $(date '+%F %T')"

    RUN_CMD=(
      "${PYTHON_CMD[@]}"
      run.py
      --config "${CONFIG}"
      --input "${INPUT_DIR}"
      --output "${OUT_DIR}"
      --save-pt
    )

    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "DRY_RUN: would create output directory:"
      echo "  mkdir -p ${OUT_DIR}"
      echo "DRY_RUN: would run:"
      print_command "${RUN_CMD[@]}"
      echo "DRY_RUN: would log to ${LOG_FILE}"
    else
      mkdir -p "${OUT_DIR}"
      {
        echo "================================================================"
        echo "[${TASK_IDX}/${TOTAL_TASKS}] ${METHOD} | ${DATASET_NAME}"
        echo "method : ${METHOD}"
        echo "dataset: ${DATASET_NAME}"
        echo "config : ${CONFIG}"
        echo "input  : ${INPUT_DIR}"
        echo "output : ${OUT_DIR}"
        echo "start  : $(date '+%F %T')"
        echo "================================================================"
      } | tee "${LOG_FILE}"

      "${RUN_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
    fi

    if [[ "${COMPUTE_OPW}" == "1" ]]; then
      OPW_CMD=(
        "${PYTHON_CMD[@]}"
        scripts/audit_opw_metric.py
        --gmflow-ckpt "${GMFLOW_CKPT}"
        --input-dir "${INPUT_DIR}"
        --pred-dir "${OUT_DIR}"
        --out-json "${OUT_DIR}/opw.json"
        --out-tsv "${OUT_DIR}/opw.tsv"
      )
      if [[ -n "${GMFLOW_REPO}" ]]; then
        OPW_CMD+=(--gmflow-repo "${GMFLOW_REPO}")
      fi

      if [[ "${DRY_RUN}" == "1" ]]; then
        echo "DRY_RUN: would compute OPW:"
        print_command "${OPW_CMD[@]}"
        echo "DRY_RUN: would log OPW to ${OPW_LOG_FILE}"
      else
        {
          echo "================================================================"
          echo "OPW audit | ${METHOD} | ${DATASET_NAME}"
          echo "gmflow_ckpt: ${GMFLOW_CKPT}"
          echo "gmflow_repo: ${GMFLOW_REPO:-<default resolution>}"
          echo "input      : ${INPUT_DIR}"
          echo "pred       : ${OUT_DIR}"
          echo "start      : $(date '+%F %T')"
          echo "================================================================"
        } | tee "${OPW_LOG_FILE}"

        "${OPW_CMD[@]}" 2>&1 | tee -a "${OPW_LOG_FILE}"
      fi
    fi

    RUN_END_TS=$(date +%s)
    RUN_ELAPSED=$((RUN_END_TS - RUN_START_TS))
    TOTAL_ELAPSED=$((RUN_END_TS - SCRIPT_START_TS))
    echo "[${TASK_IDX}/${TOTAL_TASKS}] finished ${METHOD} | ${DATASET_NAME} in ${RUN_ELAPSED}s"
    echo "elapsed total: ${TOTAL_ELAPSED}s"
    echo
  done
done

echo "All requested zero-shot VGGT-family baseline slots handled under ${OUTPUT_ROOT}"
