#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'
umask 022

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

log() { printf '[cct-run] %s\n' "$*"; }
fatal() { printf '[cct-run] ERROR: %s\n' "$*" >&2; exit 2; }
trap 'fatal "failed at line ${LINENO}: ${BASH_COMMAND}"' ERR

usage() {
  cat <<'USAGE'
Usage: bash run.sh

The script installs missing Ubuntu build/runtime dependencies when permitted, downloads and
prepares the governed real Track 1 sources, builds the native C++20 tools, runs CCT pretraining
and SFT, qualifies CCT/GRU/diagonal-SSM/dense-causal-attention on the same real-data contract,
runs the independent gates and full CTest suite, and writes all outputs below runs/<run-id>/.

Useful environment overrides:
  RUN_ID=...                         Stable output directory name.
  JOBS=2                             Native build parallelism.
  PRETRAIN_TOKEN_CAP=2000000        WikiText training token cap.
  SFT_EXAMPLES=8000                 SQuAD SFT training examples.
  SFT_EVAL_EXAMPLES=800             SQuAD SFT evaluation examples.
  PRETRAIN_STEPS=10000              Track 1 CCT pretraining steps.
  SFT_STEPS=2000                    Track 1 CCT supervised steps.
  CONTEXT_LENGTH=128                Track 1 context length.
  EMBEDDING_DIM=16                  Track 1 embedding width.
  HIDDEN_DIM=16                     Track 1 hidden width.
  ARCHITECTURE_STEPS=10000          Matched architecture qualification steps.
  ARCHITECTURE_BATCH=8              Matched architecture batch size.
  ARCHITECTURE_TRAIN_SEQUENCES=5000 Matched architecture train-sequence cap.
  ARCHITECTURE_EVAL_SEQUENCES=128   Matched architecture evaluation-sequence cap.
  SKIP_DATA_PREPARATION=1           Reuse PREP_DIR instead of downloading sources.
  PREP_DIR=/absolute/or/relative/path  Prepared Track 1 root.
  RUN_FULL_CTEST=0                  Skip the full 44-test repository suite.
  INSTALL_DEPENDENCIES=0            Do not attempt apt installation; fail if missing.
  SMOKE=1                           Use a bounded local validation configuration.

The default path is a real-data run. SMOKE=1 is only for checking the orchestration locally and
is not an architecture or language-quality result.
USAGE
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

command_exists() { command -v "$1" >/dev/null 2>&1; }

install_dependencies_if_needed() {
  local missing=()
  local command_name
  for command_name in cmake g++ pkg-config curl sha256sum zip unzip ctest; do
    command_exists "${command_name}" || missing+=("${command_name}")
  done
  if command_exists pkg-config && ! pkg-config --exists fftw3; then
    missing+=("fftw3")
  fi
  if ((${#missing[@]} == 0)); then
    log "all required host commands and FFTW3 are available"
    return
  fi
  if [[ "${INSTALL_DEPENDENCIES:-1}" != "1" ]]; then
    fatal "missing dependencies (${missing[*]}); set INSTALL_DEPENDENCIES=1 or install them manually"
  fi
  command_exists apt-get || fatal "missing dependencies (${missing[*]}) and apt-get is unavailable"
  local -a sudo_prefix=()
  if [[ "${EUID}" -ne 0 ]]; then
    command_exists sudo || fatal "missing dependencies (${missing[*]}); sudo is unavailable"
    sudo -n true >/dev/null 2>&1 || fatal "missing dependencies (${missing[*]}); passwordless sudo is required for automatic installation"
    sudo_prefix=(sudo)
  fi
  log "installing build and data dependencies: build-essential cmake pkg-config libfftw3-dev curl ca-certificates zip unzip"
  "${sudo_prefix[@]}" apt-get update
  DEBIAN_FRONTEND=noninteractive "${sudo_prefix[@]}" apt-get install -y build-essential cmake pkg-config libfftw3-dev curl ca-certificates zip unzip
  for command_name in cmake g++ pkg-config curl sha256sum zip unzip ctest; do
    command_exists "${command_name}" || fatal "dependency installation did not provide ${command_name}"
  done
  pkg-config --exists fftw3 || fatal "dependency installation did not provide FFTW3 through pkg-config"
}

number_or_default() {
  local name="$1" value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || fatal "${name} must be a non-negative integer, got '${value}'"
  printf '%s' "${value}"
}

install_dependencies_if_needed

JOBS="$(number_or_default JOBS "${JOBS:-2}")"
if [[ "${JOBS}" == "0" ]]; then JOBS=1; fi
RUN_ID="${RUN_ID:-track1-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/runs}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/${RUN_ID}}"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-seq}"
PREP_DIR="${PREP_DIR:-${RUN_DIR}/track1}"
TRAIN_DIR="${TRAIN_DIR:-${RUN_DIR}/training}"
ARCH_DIR="${ARCH_DIR:-${RUN_DIR}/architecture-qualification}"
GATE_DIR="${GATE_DIR:-${RUN_DIR}/track1-gate}"
mkdir -p "${RUN_DIR}" "${TRAIN_DIR}" "${ARCH_DIR}" "${GATE_DIR}"
ln -sfn "${RUN_DIR}" "${RUN_ROOT}/latest"

PRETRAIN_TOKEN_CAP="$(number_or_default PRETRAIN_TOKEN_CAP "${PRETRAIN_TOKEN_CAP:-2000000}")"
SFT_EXAMPLES="$(number_or_default SFT_EXAMPLES "${SFT_EXAMPLES:-8000}")"
SFT_EVAL_EXAMPLES="$(number_or_default SFT_EVAL_EXAMPLES "${SFT_EVAL_EXAMPLES:-800}")"
PRETRAIN_STEPS="$(number_or_default PRETRAIN_STEPS "${PRETRAIN_STEPS:-10000}")"
SFT_STEPS="$(number_or_default SFT_STEPS "${SFT_STEPS:-2000}")"
CONTEXT_LENGTH="$(number_or_default CONTEXT_LENGTH "${CONTEXT_LENGTH:-128}")"
EMBEDDING_DIM="$(number_or_default EMBEDDING_DIM "${EMBEDDING_DIM:-16}")"
HIDDEN_DIM="$(number_or_default HIDDEN_DIM "${HIDDEN_DIM:-16}")"
ARCHITECTURE_STEPS="$(number_or_default ARCHITECTURE_STEPS "${ARCHITECTURE_STEPS:-10000}")"
ARCHITECTURE_BATCH="$(number_or_default ARCHITECTURE_BATCH "${ARCHITECTURE_BATCH:-8}")"
ARCHITECTURE_TRAIN_SEQUENCES="$(number_or_default ARCHITECTURE_TRAIN_SEQUENCES "${ARCHITECTURE_TRAIN_SEQUENCES:-5000}")"
ARCHITECTURE_EVAL_SEQUENCES="$(number_or_default ARCHITECTURE_EVAL_SEQUENCES "${ARCHITECTURE_EVAL_SEQUENCES:-128}")"
SOURCE_ROW_LIMIT="$(number_or_default SOURCE_ROW_LIMIT "${SOURCE_ROW_LIMIT:-0}")"
SEED="$(number_or_default SEED "${SEED:-1701}")"
RUN_FULL_CTEST="${RUN_FULL_CTEST:-1}"
SKIP_DATA_PREPARATION="${SKIP_DATA_PREPARATION:-0}"
SMOKE="${SMOKE:-0}"

if [[ "${SMOKE}" == "1" ]]; then
  PRETRAIN_TOKEN_CAP=20000
  SFT_EXAMPLES=1000
  SFT_EVAL_EXAMPLES=100
  PRETRAIN_STEPS=4
  SFT_STEPS=4
  CONTEXT_LENGTH=16
  EMBEDDING_DIM=4
  HIDDEN_DIM=4
  ARCHITECTURE_STEPS=2
  ARCHITECTURE_BATCH=2
  ARCHITECTURE_TRAIN_SEQUENCES=32
  ARCHITECTURE_EVAL_SEQUENCES=16
  log "SMOKE=1: bounded orchestration validation only; this is not a quality result"
fi

TOKENIZER="${TOKENIZER:-${ROOT_DIR}/data/stage-10/tokenizer_snapshot.bin}"
[[ -s "${TOKENIZER}" ]] || fatal "tokenizer snapshot is missing: ${TOKENIZER}"
[[ -f cpp/CMakeLists.txt ]] || fatal "run.sh must be executed from a CCT repository checkout"

cat > "${RUN_DIR}/run_config.json" <<EOF
{
  "status":"RUNNING",
  "run_id":"${RUN_ID}",
  "repository":"${ROOT_DIR}",
  "build_dir":"${BUILD_DIR}",
  "prepared_data_dir":"${PREP_DIR}",
  "training_dir":"${TRAIN_DIR}",
  "architecture_dir":"${ARCH_DIR}",
  "seed":${SEED},
  "pretrain_token_cap":${PRETRAIN_TOKEN_CAP},
  "sft_examples":${SFT_EXAMPLES},
  "sft_eval_examples":${SFT_EVAL_EXAMPLES},
  "pretrain_steps":${PRETRAIN_STEPS},
  "sft_steps":${SFT_STEPS},
  "context_length":${CONTEXT_LENGTH},
  "embedding_dim":${EMBEDDING_DIM},
  "hidden_dim":${HIDDEN_DIM},
  "architecture_steps":${ARCHITECTURE_STEPS},
  "architecture_batch":${ARCHITECTURE_BATCH},
  "architecture_train_sequences":${ARCHITECTURE_TRAIN_SEQUENCES},
  "architecture_eval_sequences":${ARCHITECTURE_EVAL_SEQUENCES},
  "training_authorized":false
}
EOF

log "configuring native C++20 Release build in ${BUILD_DIR}"
cmake -S cpp -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DCCT_STRICT_WARNINGS=ON 2>&1 | tee "${RUN_DIR}/cmake_configure.log"
log "building native preparation, training, qualification, gate, and regression executables"
cmake --build "${BUILD_DIR}" --parallel "${JOBS}" \
  --target cct_track1_prepare cct_track1_train cct_track1_gate \
           cct_architecture_qualification cct_architecture_qualification_gate \
           cct_nlp_trainer_tests cct_track1_tests cct_documentation_consistency_tests \
  2>&1 | tee "${RUN_DIR}/build.log"

if [[ "${SKIP_DATA_PREPARATION}" == "1" ]]; then
  log "SKIP_DATA_PREPARATION=1: reusing ${PREP_DIR}"
else
  log "downloading and preparing pinned real Track 1 sources into ${PREP_DIR}"
  rm -rf "${PREP_DIR}"
  "${BUILD_DIR}/cct_track1_prepare" \
    --output "${PREP_DIR}" \
    --pretrain-token-cap "${PRETRAIN_TOKEN_CAP}" \
    --sft-examples "${SFT_EXAMPLES}" \
    --sft-eval-examples "${SFT_EVAL_EXAMPLES}" \
    --source-row-limit "${SOURCE_ROW_LIMIT}" \
    --seed "${SEED}" \
    2>&1 | tee "${RUN_DIR}/preparation.log"
fi
[[ -s "${PREP_DIR}/preparation_report.json" ]] || fatal "Track 1 preparation report is missing"
grep -q '"passed":true' "${PREP_DIR}/preparation_report.json" || fatal "Track 1 preparation did not pass"
for required_file in \
  "${PREP_DIR}/manifest.json" \
  "${PREP_DIR}/evaluation_contract.json" \
  "${PREP_DIR}/data/pretrain_train.txt" \
  "${PREP_DIR}/data/pretrain_validation.txt" \
  "${PREP_DIR}/data/pretrain_test.txt" \
  "${PREP_DIR}/data/squad_sft_train.jsonl" \
  "${PREP_DIR}/data/squad_sft_evaluation.jsonl" \
  "${PREP_DIR}/data/squad_final_test.jsonl"; do
  [[ -s "${required_file}" ]] || fatal "prepared Track 1 file is missing or empty: ${required_file}"
done

log "running native Track 1 CCT pretraining and supervised fine-tuning"
"${BUILD_DIR}/cct_track1_train" \
  --input "${PREP_DIR}" \
  --output "${TRAIN_DIR}" \
  --tokenizer "${TOKENIZER}" \
  --pretrain-steps "${PRETRAIN_STEPS}" \
  --sft-steps "${SFT_STEPS}" \
  --context "${CONTEXT_LENGTH}" \
  --embedding "${EMBEDDING_DIM}" \
  --hidden "${HIDDEN_DIM}" \
  --pretrain-selection-validation-limit 256 \
  --sft-selection-evaluation-limit 256 \
  --final-test-limit 0 \
  --sft-context-bytes 1024 \
  --seed "${SEED}" \
  2>&1 | tee "${RUN_DIR}/training.log"
[[ -s "${TRAIN_DIR}/training_report.json" ]] || fatal "Track 1 training report is missing"
grep -q '"status":"PASS"' "${TRAIN_DIR}/training_report.json" || fatal "Track 1 training did not pass"
[[ -s "${TRAIN_DIR}/pretrain_checkpoint.bin" && -s "${TRAIN_DIR}/sft_checkpoint.bin" ]] || fatal "Track 1 checkpoints are missing"

log "running the independent Track 1 governance and lineage gate"
"${BUILD_DIR}/cct_track1_gate" --output "${GATE_DIR}" 2>&1 | tee "${RUN_DIR}/track1_gate.log"
grep -q '"status":"PASS"' "${GATE_DIR}/release_record.json" || fatal "Track 1 independent gate did not pass"

log "running matched CCT/GRU/SSM/attention architecture qualification"
"${BUILD_DIR}/cct_architecture_qualification" \
  --train "${PREP_DIR}/data/pretrain_train.txt" \
  --validation "${PREP_DIR}/data/pretrain_validation.txt" \
  --test "${PREP_DIR}/data/pretrain_test.txt" \
  --tokenizer "${TOKENIZER}" \
  --output "${ARCH_DIR}/report.json" \
  --steps "${ARCHITECTURE_STEPS}" \
  --batch "${ARCHITECTURE_BATCH}" \
  --context "${CONTEXT_LENGTH}" \
  --embedding "${EMBEDDING_DIM}" \
  --hidden "${HIDDEN_DIM}" \
  --train-sequences "${ARCHITECTURE_TRAIN_SEQUENCES}" \
  --eval-sequences "${ARCHITECTURE_EVAL_SEQUENCES}" \
  --vocab-mode compact \
  --seed "${SEED}" \
  2>&1 | tee "${RUN_DIR}/architecture_qualification.log"
grep -q '"status":"COMPLETE"' "${ARCH_DIR}/report.json" || fatal "architecture qualification did not complete"

log "running the independent architecture qualification gate"
set +e
trap - ERR
"${BUILD_DIR}/cct_architecture_qualification_gate" \
  --report "${ARCH_DIR}/report.json" \
  --output "${ARCH_DIR}/gate" \
  2>&1 | tee "${RUN_DIR}/architecture_gate.log"
architecture_gate_exit=${PIPESTATUS[0]}
set -e
trap 'fatal "failed at line ${LINENO}: ${BASH_COMMAND}"' ERR
if [[ "${architecture_gate_exit}" -ne 0 || ! -s "${ARCH_DIR}/gate/checks.json" ]]; then
  if [[ "${SMOKE}" == "1" ]]; then
    log "SMOKE=1: architecture gate failure is expected for the deliberately undertrained bounded run"
    architecture_gate_status="EXPECTED_SMOKE_FAIL"
  else
    fatal "architecture qualification gate did not pass"
  fi
else
  architecture_gate_status="PASS"
fi

log "running focused native regression and documentation checks"
"${BUILD_DIR}/cct_nlp_trainer_tests" 2>&1 | tee "${RUN_DIR}/nlp_trainer_tests.log"
"${BUILD_DIR}/cct_track1_tests" 2>&1 | tee "${RUN_DIR}/track1_tests.log"
"${BUILD_DIR}/cct_documentation_consistency_tests" 2>&1 | tee "${RUN_DIR}/documentation_consistency.log"

grep -q 'SUMMARY 13/13 passed' "${RUN_DIR}/nlp_trainer_tests.log" || fatal "NLP trainer regression suite did not pass 13/13"
grep -q 'SUMMARY 1/1 passed' "${RUN_DIR}/documentation_consistency.log" || fatal "documentation consistency test did not pass 1/1"

if [[ "${RUN_FULL_CTEST}" == "1" ]]; then
  log "running the complete native CTest suite"
  ctest --test-dir "${BUILD_DIR}" --output-on-failure 2>&1 | tee "${RUN_DIR}/ctest.log"
else
  log "RUN_FULL_CTEST=${RUN_FULL_CTEST}: full CTest suite skipped"
fi

sha256sum \
  "${PREP_DIR}/manifest.json" \
  "${PREP_DIR}/preparation_report.json" \
  "${TRAIN_DIR}/training_report.json" \
  "${TRAIN_DIR}/pretrain_checkpoint.bin" \
  "${TRAIN_DIR}/sft_checkpoint.bin" \
  "${ARCH_DIR}/report.json" \
  "${ARCH_DIR}/gate/checks.json" \
  > "${RUN_DIR}/artifact_sha256.txt"

if [[ "${SMOKE}" == "1" ]]; then
  run_status="SMOKE_PASS"
else
  run_status="PASS"
fi
cat > "${RUN_DIR}/run_summary.json" <<EOF
{
  "status":"${run_status}",
  "run_id":"${RUN_ID}",
  "prepared_data":"${PREP_DIR}",
  "training_report":"${TRAIN_DIR}/training_report.json",
  "pretrain_checkpoint":"${TRAIN_DIR}/pretrain_checkpoint.bin",
  "sft_checkpoint":"${TRAIN_DIR}/sft_checkpoint.bin",
  "track1_gate":"${GATE_DIR}/release_record.json",
  "architecture_report":"${ARCH_DIR}/report.json",
  "architecture_gate":"${ARCH_DIR}/gate/checks.json",
  "architecture_gate_status":"${architecture_gate_status}",
  "artifact_hashes":"${RUN_DIR}/artifact_sha256.txt",
  "training_authorized":false,
  "claim_boundary":"bounded native C++20 real-data training and evaluation; not broad language competence or general intelligence"
}
EOF
sed -i 's/"status":"RUNNING"/"status":"PASS"/' "${RUN_DIR}/run_config.json"
log "${run_status}: requested training, evaluation, qualification, and artifact orchestration completed"
log "run summary: ${RUN_DIR}/run_summary.json"
