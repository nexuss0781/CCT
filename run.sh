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

The default mode installs missing Ubuntu build/runtime dependencies when permitted, downloads
pinned FineWeb-Edu and OpenAssistant ranges, builds the native C++20 tools, trains one competency
session from its parent checkpoint, writes a human mastery packet, and stops before advancement.
Set CURRICULUM_MODE=0 to run the legacy Track 1 pretraining/SFT and architecture qualification path.

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
  CURRICULUM_MODE=1                 Use competency-based continual sessions (default).
  CURRICULUM_MODULE1=1              Run the approved Module 1 submodule curriculum (default).
  CURRICULUM_ROOT=...               Durable curriculum state, data, sessions, and validation packets.
  HUMAN_VALIDATION_FILE=...         JSON result for the pending session; required before advancing.
  CURRICULUM_CHUNK_ROWS=100         FineWeb/OASST training rows per session.
  CURRICULUM_VALIDATION_ROWS=40     FineWeb/OASST validation rows per session.
  CURRICULUM_TEST_ROWS=40           FineWeb held-out test rows per session.
  CURRICULUM_PRETRAIN_STEPS=100     Native CCT steps per session pretraining phase.
  CURRICULUM_SFT_STEPS=50           Native CCT steps per session SFT phase.
  CURRICULUM_PAGE_DELAY_MS=5000      Delay between dataset API pages.
  CURRICULUM_RETRY_COUNT=12          Curl retries for transient/rate-limit failures.
  CURRICULUM_SFT_SCAN_MULTIPLIER=100 OpenAssistant rows scanned per accepted row.
  MODULE1_CONTEXT_LENGTH=128          Module 1 context length.
  MODULE1_EMBEDDING_DIM=32            Module 1 embedding width.
  MODULE1_HIDDEN_DIM=32               Module 1 hidden width.
  MODULE1_BATCH_SIZE=16               Module 1 token batch size.
  MODULE1_WORKERS=2                   Module 1 bounded batch-gradient workers.
  MODULE1_PAGE_DELAY_MS=1000          Module 1 dataset page pacing.

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
RUN_FULL_CTEST="${RUN_FULL_CTEST:-0}"
SKIP_DATA_PREPARATION="${SKIP_DATA_PREPARATION:-0}"
SMOKE="${SMOKE:-0}"
CURRICULUM_MODE="${CURRICULUM_MODE:-1}"
CURRICULUM_MODULE1="${CURRICULUM_MODULE1:-1}"
CURRICULUM_ROOT="${CURRICULUM_ROOT:-}"
if [[ -z "${CURRICULUM_ROOT}" ]]; then
  if [[ "${CURRICULUM_MODULE1}" == "1" ]]; then
    CURRICULUM_ROOT="${ROOT_DIR}/runs/curriculum-module1"
  else
    CURRICULUM_ROOT="${ROOT_DIR}/runs/curriculum-focused-english"
  fi
fi
CURRICULUM_CHUNK_ROWS="$(number_or_default CURRICULUM_CHUNK_ROWS "${CURRICULUM_CHUNK_ROWS:-100}")"
CURRICULUM_VALIDATION_ROWS="$(number_or_default CURRICULUM_VALIDATION_ROWS "${CURRICULUM_VALIDATION_ROWS:-40}")"
CURRICULUM_TEST_ROWS="$(number_or_default CURRICULUM_TEST_ROWS "${CURRICULUM_TEST_ROWS:-40}")"
CURRICULUM_PRETRAIN_STEPS="$(number_or_default CURRICULUM_PRETRAIN_STEPS "${CURRICULUM_PRETRAIN_STEPS:-100}")"
CURRICULUM_SFT_STEPS="$(number_or_default CURRICULUM_SFT_STEPS "${CURRICULUM_SFT_STEPS:-50}")"
CURRICULUM_MAX_LEVEL="$(number_or_default CURRICULUM_MAX_LEVEL "${CURRICULUM_MAX_LEVEL:-7}")"

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
  CURRICULUM_CHUNK_ROWS=4
  CURRICULUM_VALIDATION_ROWS=2
  CURRICULUM_TEST_ROWS=2
  CURRICULUM_PRETRAIN_STEPS=2
  CURRICULUM_SFT_STEPS=2
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
  "curriculum_module1":${CURRICULUM_MODULE1},
  "curriculum_root":"${CURRICULUM_ROOT}",
  "training_authorized":false
}
EOF

log "configuring native C++20 Release build in ${BUILD_DIR}"
cmake -S cpp -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DCCT_STRICT_WARNINGS=ON 2>&1 | tee "${RUN_DIR}/cmake_configure.log"
if [[ "${RUN_FULL_CTEST}" == "1" ]]; then
  log "building the complete native C++ target graph for full CTest"
  cmake --build "${BUILD_DIR}" --parallel "${JOBS}" 2>&1 | tee "${RUN_DIR}/build.log"
else
  log "building native preparation, training, qualification, gate, and regression executables"
  cmake --build "${BUILD_DIR}" --parallel "${JOBS}" \
    --target cct_track1_prepare cct_track1_train cct_track1_gate \
             cct_architecture_qualification cct_architecture_qualification_gate \
             cct_curriculum_prepare cct_curriculum_session cct_curriculum_inspect \
             cct_nlp_trainer_tests cct_track1_tests cct_documentation_consistency_tests \
    2>&1 | tee "${RUN_DIR}/build.log"
fi

if [[ "${CURRICULUM_MODE}" == "1" && "${CURRICULUM_MODULE1}" == "1" ]]; then
  MODULE1_STATE_FILE="${CURRICULUM_ROOT}/state.env"
  MODULE1_VALIDATION_ROOT="${CURRICULUM_ROOT}/validation"
  MODULE1_SESSIONS_ROOT="${CURRICULUM_ROOT}/sessions"
  MODULE1_DATA_ROOT="${CURRICULUM_ROOT}/data"
  MODULE1_CHUNK_STRIDE="$(number_or_default MODULE1_CHUNK_STRIDE "${MODULE1_CHUNK_STRIDE:-100000}")"
  MODULE1_RETRY_STRIDE="$(number_or_default MODULE1_RETRY_STRIDE "${MODULE1_RETRY_STRIDE:-50000}")"
  MODULE1_VALIDATION_GAP="$(number_or_default MODULE1_VALIDATION_GAP "${MODULE1_VALIDATION_GAP:-1000}")"
  MODULE1_TEST_GAP="$(number_or_default MODULE1_TEST_GAP "${MODULE1_TEST_GAP:-2000}")"
  MODULE1_SFT_VALIDATION_GAP="$(number_or_default MODULE1_SFT_VALIDATION_GAP "${MODULE1_SFT_VALIDATION_GAP:-1000}")"
  MODULE1_PAGE_DELAY_MS="$(number_or_default MODULE1_PAGE_DELAY_MS "${MODULE1_PAGE_DELAY_MS:-1000}")"
  MODULE1_RETRY_COUNT="$(number_or_default MODULE1_RETRY_COUNT "${MODULE1_RETRY_COUNT:-12}")"
  MODULE1_SFT_SCAN_MULTIPLIER="$(number_or_default MODULE1_SFT_SCAN_MULTIPLIER "${MODULE1_SFT_SCAN_MULTIPLIER:-100}")"
  MODULE1_FINEWEB_REVISION="${MODULE1_FINEWEB_REVISION:-87f09149ef4734204d70ed1d046ddc9ca3f2b8f9}"
  MODULE1_OASST_REVISION="${MODULE1_OASST_REVISION:-fdf72ae0827c1cda404aff25b6603abec9e3399b}"
  MODULE1_MINIMUM_EDUCATION_SCORE="${MODULE1_MINIMUM_EDUCATION_SCORE:-2.0}"
  MODULE1_CONTEXT_LENGTH="$(number_or_default MODULE1_CONTEXT_LENGTH "${MODULE1_CONTEXT_LENGTH:-128}")"
  MODULE1_EMBEDDING_DIM="$(number_or_default MODULE1_EMBEDDING_DIM "${MODULE1_EMBEDDING_DIM:-32}")"
  MODULE1_HIDDEN_DIM="$(number_or_default MODULE1_HIDDEN_DIM "${MODULE1_HIDDEN_DIM:-32}")"
  MODULE1_BATCH_SIZE="$(number_or_default MODULE1_BATCH_SIZE "${MODULE1_BATCH_SIZE:-16}")"
  MODULE1_WORKERS="$(number_or_default MODULE1_WORKERS "${MODULE1_WORKERS:-2}")"
  mkdir -p "${MODULE1_VALIDATION_ROOT}" "${MODULE1_SESSIONS_ROOT}" "${MODULE1_DATA_ROOT}"

  if [[ "${SMOKE}" == "1" ]]; then
    MODULE1_CHUNK_STRIDE=100
    MODULE1_RETRY_STRIDE=50
    MODULE1_VALIDATION_GAP=10
    MODULE1_TEST_GAP=20
    MODULE1_SFT_VALIDATION_GAP=10
    MODULE1_PAGE_DELAY_MS=0
    MODULE1_RETRY_COUNT=2
    MODULE1_SFT_SCAN_MULTIPLIER=100
    MODULE1_CONTEXT_LENGTH=16
    MODULE1_EMBEDDING_DIM=4
    MODULE1_HIDDEN_DIM=4
    MODULE1_BATCH_SIZE=2
    MODULE1_WORKERS=1
  fi

  log "running native Module 1 regression and documentation checks"
  "${BUILD_DIR}/cct_nlp_trainer_tests" 2>&1 | tee "${RUN_DIR}/module1_nlp_trainer_tests.log"
  "${BUILD_DIR}/cct_track1_tests" 2>&1 | tee "${RUN_DIR}/module1_track1_tests.log"
  "${BUILD_DIR}/cct_documentation_consistency_tests" 2>&1 | tee "${RUN_DIR}/module1_documentation_consistency.log"
  grep -q 'SUMMARY 14/14 passed' "${RUN_DIR}/module1_nlp_trainer_tests.log" || fatal "Module 1 NLP trainer regressions did not pass 14/14"
  grep -q 'SUMMARY 1/1 passed' "${RUN_DIR}/module1_documentation_consistency.log" || fatal "Module 1 documentation consistency did not pass 1/1"
  if [[ "${RUN_FULL_CTEST}" == "1" ]]; then
    log "running complete native CTest suite before Module 1 session"
    ctest --test-dir "${BUILD_DIR}" --output-on-failure 2>&1 | tee "${RUN_DIR}/module1_ctest.log"
  fi

  write_module1_state() {
    local status="$1" submodule_index="$2" failures="$3" pending_session="$4" pending_checkpoint="$5" pending_hash="$6" parent_checkpoint="$7"
    cat > "${MODULE1_STATE_FILE}" <<EOF
MODULE1_STATUS=${status}
MODULE1_SUBMODULE_INDEX=${submodule_index}
MODULE1_FAILURES=${failures}
MODULE1_PENDING_SESSION=${pending_session}
MODULE1_PENDING_CHECKPOINT=${pending_checkpoint}
MODULE1_PENDING_CHECKPOINT_HASH=${pending_hash}
MODULE1_PARENT_CHECKPOINT=${parent_checkpoint}
EOF
  }

  write_module1_summary() {
    local status="$1" message="$2"
    cat > "${RUN_DIR}/run_summary.json" <<EOF
{
  "status":"${status}",
  "curriculum":"module-1",
  "curriculum_status":"${status}",
  "message":"${message}",
  "curriculum_root":"${CURRICULUM_ROOT}",
  "state_file":"${MODULE1_STATE_FILE}",
  "training_authorized":false,
  "claim_boundary":"Module 1 submodule research workflow; human competency validation is mandatory"
}
EOF
    sed -i 's/"status":"RUNNING"/"status":"'"${status}"'/g' "${RUN_DIR}/run_config.json"
  }

  if [[ -f "${MODULE1_STATE_FILE}" ]]; then
    # This file is generated by this script and contains only scalar state values.
    # shellcheck disable=SC1090
    source "${MODULE1_STATE_FILE}"
  else
    MODULE1_STATUS=READY_TO_TRAIN
    MODULE1_SUBMODULE_INDEX=1
    MODULE1_FAILURES=0
    MODULE1_PENDING_SESSION=
    MODULE1_PENDING_CHECKPOINT=
    MODULE1_PENDING_CHECKPOINT_HASH=
    MODULE1_PARENT_CHECKPOINT=
    write_module1_state "${MODULE1_STATUS}" "${MODULE1_SUBMODULE_INDEX}" "${MODULE1_FAILURES}" "" "" "" ""
  fi

  if [[ "${MODULE1_STATUS}" == "AWAITING_HUMAN_VALIDATION" ]]; then
    HUMAN_VALIDATION_FILE="${HUMAN_VALIDATION_FILE:-${MODULE1_VALIDATION_ROOT}/${MODULE1_PENDING_SESSION}.json}"
    MASTERY_PACKET="${MODULE1_SESSIONS_ROOT}/${MODULE1_PENDING_SESSION}/mastery_prompt.md"
    if [[ ! -s "${HUMAN_VALIDATION_FILE}" ]]; then
      log "Module 1 session ${MODULE1_PENDING_SESSION} is awaiting human validation"
      log "review ${MASTERY_PACKET} and write PASS/FAIL JSON to ${HUMAN_VALIDATION_FILE}"
      write_module1_summary "AWAITING_HUMAN_VALIDATION" "one Module 1 submodule session completed; human mastery validation is required"
      exit 0
    fi
    grep -Eq '"session_id"[[:space:]]*:[[:space:]]*"'"${MODULE1_PENDING_SESSION}"'"' "${HUMAN_VALIDATION_FILE}" || fatal "Module 1 validation session_id does not match pending session"
    grep -Eq '"submodule"[[:space:]]*:[[:space:]]*"module-1-submodule-'"${MODULE1_SUBMODULE_INDEX}"'"' "${HUMAN_VALIDATION_FILE}" || fatal "Module 1 validation submodule does not match pending submodule"
    grep -Eq '"checkpoint_hash"[[:space:]]*:[[:space:]]*"'"${MODULE1_PENDING_CHECKPOINT_HASH}"'"' "${HUMAN_VALIDATION_FILE}" || fatal "Module 1 validation checkpoint hash does not match pending checkpoint"
    if grep -Eq '"result"[[:space:]]*:[[:space:]]*"PASS"' "${HUMAN_VALIDATION_FILE}"; then
      log "human mastery PASS for ${MODULE1_PENDING_SESSION}; advancing exactly one Module 1 submodule"
      PREVIOUS_MODULE1_CHECKPOINT="${MODULE1_PENDING_CHECKPOINT}"
      if ((MODULE1_SUBMODULE_INDEX >= 4)); then
        write_module1_state "MODULE1_COMPLETE" "5" "0" "" "" "" "${MODULE1_PENDING_CHECKPOINT}"
        write_module1_summary "MODULE1_COMPLETE" "all four Module 1 submodules have human PASS records"
        exit 0
      fi
      NEXT_SUBMODULE=$((MODULE1_SUBMODULE_INDEX + 1))
      write_module1_state "READY_TO_TRAIN" "${NEXT_SUBMODULE}" "0" "" "" "" "${MODULE1_PENDING_CHECKPOINT}"
      MODULE1_STATUS=READY_TO_TRAIN
      MODULE1_SUBMODULE_INDEX=${NEXT_SUBMODULE}
      MODULE1_FAILURES=0
      MODULE1_PENDING_SESSION=
      MODULE1_PENDING_CHECKPOINT=
      MODULE1_PENDING_CHECKPOINT_HASH=
      MODULE1_PARENT_CHECKPOINT="${PREVIOUS_MODULE1_CHECKPOINT}"
      MODULE1_PENDING_CHECKPOINT=""
      MODULE1_PENDING_CHECKPOINT_HASH=""
    elif grep -Eq '"result"[[:space:]]*:[[:space:]]*"FAIL"' "${HUMAN_VALIDATION_FILE}"; then
      if ((MODULE1_FAILURES == 0)); then
        write_module1_state "READY_TO_RETRY" "${MODULE1_SUBMODULE_INDEX}" "1" "" "" "" "${MODULE1_PARENT_CHECKPOINT}"
        cat > "${CURRICULUM_ROOT}/retry_required.md" <<EOF
# Module 1 Retry Required

Human validation failed for ${MODULE1_PENDING_SESSION}. The same Module 1 submodule will be retrained on a fresh disjoint source chunk using the original parent checkpoint and the same submodule contract.

The failed checkpoint remains immutable at ${MODULE1_PENDING_CHECKPOINT}.
EOF
        write_module1_summary "READY_TO_RETRY" "Module 1 submodule failed once; a fresh disjoint retry is required"
        exit 0
      fi
      write_module1_state "ARCHITECTURE_DIAGNOSIS_REQUIRED" "${MODULE1_SUBMODULE_INDEX}" "${MODULE1_FAILURES}" "" "${MODULE1_PENDING_CHECKPOINT}" "${MODULE1_PENDING_CHECKPOINT_HASH}" "${MODULE1_PARENT_CHECKPOINT}"
      cat > "${CURRICULUM_ROOT}/architecture_diagnosis_required.md" <<EOF
# Module 1 Diagnosis Required

The same Module 1 submodule failed twice under disjoint chunks with the same parent, model, tokenizer, and evaluation contract. Preserve both session directories and diagnose data coverage, optimization, decoding, or architecture before continuing.
EOF
      write_module1_summary "ARCHITECTURE_DIAGNOSIS_REQUIRED" "the same Module 1 submodule failed twice; diagnosis is required"
      exit 0
    else
      fatal "Module 1 validation result must be PASS or FAIL"
    fi
  elif [[ "${MODULE1_STATUS}" == "READY_TO_RETRY" ]]; then
    MODULE1_STATUS=READY_TO_TRAIN
  elif [[ "${MODULE1_STATUS}" == "ARCHITECTURE_DIAGNOSIS_REQUIRED" || "${MODULE1_STATUS}" == "MODULE1_COMPLETE" ]]; then
    write_module1_summary "${MODULE1_STATUS}" "Module 1 state is terminal until reviewed"
    exit 0
  fi

  case "${MODULE1_SUBMODULE_INDEX}" in
    1)
      MODULE1_SUBMODULE_ID=1.1
      MODULE1_SUBMODULE_NAME=character_and_symbol_awareness
      MODULE1_PRETRAIN_ROWS=500; MODULE1_VALIDATION_ROWS=100; MODULE1_TEST_ROWS=100; MODULE1_SFT_ROWS=250; MODULE1_SFT_VALIDATION_ROWS=50; MODULE1_PRETRAIN_STEPS=500; MODULE1_SFT_STEPS=250; MODULE1_PROMPT_COUNT=10
      MODULE1_OBJECTIVE="Reproduce letters, digits, punctuation, and common symbols without invalid or degenerate output."
      MODULE1_ACCEPTANCE="PASS requires at least 8 of 10 unseen prompts to produce inspectable non-degenerate output and both adversarial symbol cases to remain valid."
      ;;
    2)
      MODULE1_SUBMODULE_ID=1.2
      MODULE1_SUBMODULE_NAME=whitespace_and_word_boundaries
      MODULE1_PRETRAIN_ROWS=750; MODULE1_VALIDATION_ROWS=150; MODULE1_TEST_ROWS=150; MODULE1_SFT_ROWS=375; MODULE1_SFT_VALIDATION_ROWS=75; MODULE1_PRETRAIN_STEPS=750; MODULE1_SFT_STEPS=375; MODULE1_PROMPT_COUNT=10
      MODULE1_OBJECTIVE="Preserve spaces, word separation, punctuation spacing, contractions, and quoted boundaries."
      MODULE1_ACCEPTANCE="PASS requires at least 8 of 10 unseen prompts to preserve readable word separation and punctuation boundaries."
      ;;
    3)
      MODULE1_SUBMODULE_ID=1.3
      MODULE1_SUBMODULE_NAME=common_word_patterns
      MODULE1_PRETRAIN_ROWS=1000; MODULE1_VALIDATION_ROWS=200; MODULE1_TEST_ROWS=200; MODULE1_SFT_ROWS=500; MODULE1_SFT_VALIDATION_ROWS=100; MODULE1_PRETRAIN_STEPS=1000; MODULE1_SFT_STEPS=500; MODULE1_PROMPT_COUNT=10
      MODULE1_OBJECTIVE="Learn frequent English words, short and long word shapes, common affixes, repeated letters, and function-word patterns."
      MODULE1_ACCEPTANCE="PASS requires at least 8 of 10 unseen prompts to continue with recognizable English word patterns without unrelated or degenerate output."
      ;;
    4)
      MODULE1_SUBMODULE_ID=1.4
      MODULE1_SUBMODULE_NAME=stable_short_continuation
      MODULE1_PRETRAIN_ROWS=1250; MODULE1_VALIDATION_ROWS=250; MODULE1_TEST_ROWS=250; MODULE1_SFT_ROWS=625; MODULE1_SFT_VALIDATION_ROWS=125; MODULE1_PRETRAIN_STEPS=1250; MODULE1_SFT_STEPS=625; MODULE1_PROMPT_COUNT=12
      MODULE1_OBJECTIVE="Combine symbol, boundary, and word-pattern knowledge in stable short English continuations."
      MODULE1_ACCEPTANCE="PASS requires at least 10 of 12 unseen prompts to remain valid, readable, non-degenerate, and structurally consistent."
      ;;
    *) fatal "invalid Module 1 submodule index ${MODULE1_SUBMODULE_INDEX}" ;;
  esac
  if [[ "${SMOKE}" == "1" ]]; then
    MODULE1_PRETRAIN_ROWS=4; MODULE1_VALIDATION_ROWS=2; MODULE1_TEST_ROWS=2; MODULE1_SFT_ROWS=4; MODULE1_SFT_VALIDATION_ROWS=2; MODULE1_PRETRAIN_STEPS=2; MODULE1_SFT_STEPS=2; MODULE1_PROMPT_COUNT=2
  fi
  MODULE1_BASE_OFFSET=$(((MODULE1_SUBMODULE_INDEX - 1) * MODULE1_CHUNK_STRIDE + MODULE1_FAILURES * MODULE1_RETRY_STRIDE))
  MODULE1_SESSION_ID="module-1-submodule-${MODULE1_SUBMODULE_INDEX}-attempt-${MODULE1_FAILURES}"
  MODULE1_SESSION_DIR="${MODULE1_SESSIONS_ROOT}/${MODULE1_SESSION_ID}"
  MODULE1_DATA_DIR="${MODULE1_DATA_ROOT}/${MODULE1_SESSION_ID}"
  MODULE1_PROMPTS="${ROOT_DIR}/data/curriculum/module-1/prompts/${MODULE1_SUBMODULE_ID}.txt"
  mkdir -p "${MODULE1_SESSION_DIR}" "${MODULE1_DATA_DIR}"
  [[ -s "${MODULE1_PROMPTS}" ]] || fatal "Module 1 prompt packet is missing: ${MODULE1_PROMPTS}"
  log "preparing Module 1 submodule ${MODULE1_SUBMODULE_ID} at source offset ${MODULE1_BASE_OFFSET}"
  "${BUILD_DIR}/cct_curriculum_prepare" \
    --output "${MODULE1_DATA_DIR}" \
    --module module-1 \
    --submodule "${MODULE1_SUBMODULE_ID}" \
    --pretrain-offset "${MODULE1_BASE_OFFSET}" \
    --pretrain-rows "${MODULE1_PRETRAIN_ROWS}" \
    --validation-offset "$((MODULE1_BASE_OFFSET + MODULE1_PRETRAIN_ROWS + MODULE1_VALIDATION_GAP))" \
    --validation-rows "${MODULE1_VALIDATION_ROWS}" \
    --test-offset "$((MODULE1_BASE_OFFSET + MODULE1_PRETRAIN_ROWS + MODULE1_VALIDATION_GAP + MODULE1_VALIDATION_ROWS + MODULE1_TEST_GAP))" \
    --test-rows "${MODULE1_TEST_ROWS}" \
    --sft-offset "${MODULE1_BASE_OFFSET}" \
    --sft-rows "${MODULE1_SFT_ROWS}" \
    --sft-validation-offset "$((MODULE1_BASE_OFFSET + MODULE1_SFT_ROWS * MODULE1_SFT_SCAN_MULTIPLIER + MODULE1_SFT_VALIDATION_GAP))" \
    --sft-validation-rows "${MODULE1_SFT_VALIDATION_ROWS}" \
    --page-length 100 \
    --page-delay-ms "${MODULE1_PAGE_DELAY_MS}" \
    --retry-count "${MODULE1_RETRY_COUNT}" \
    --sft-scan-multiplier "${MODULE1_SFT_SCAN_MULTIPLIER}" \
    --minimum-education-score "${MODULE1_MINIMUM_EDUCATION_SCORE}" \
    --fineweb-revision "${MODULE1_FINEWEB_REVISION}" \
    --oasst-revision "${MODULE1_OASST_REVISION}" \
    2>&1 | tee "${RUN_DIR}/module1_prepare.log"
  [[ -s "${MODULE1_DATA_DIR}/manifest.json" && -s "${MODULE1_DATA_DIR}/pretrain_test.txt" ]] || fatal "Module 1 preparation did not publish governed data"
  PARENT_ARG=()
  if [[ -n "${MODULE1_PARENT_CHECKPOINT}" ]]; then PARENT_ARG=(--parent-checkpoint "${MODULE1_PARENT_CHECKPOINT}"); fi
  log "training exactly one Module 1 submodule from its immutable parent checkpoint"
  "${BUILD_DIR}/cct_curriculum_session" \
    --input "${MODULE1_DATA_DIR}" \
    --output "${MODULE1_SESSION_DIR}" \
    --tokenizer "${TOKENIZER}" \
    --module module-1 \
    --submodule "${MODULE1_SUBMODULE_ID}" \
    "${PARENT_ARG[@]}" \
    --session-id "${MODULE1_SESSION_ID}" \
    --level 1 \
    --pretrain-steps "${MODULE1_PRETRAIN_STEPS}" \
    --sft-steps "${MODULE1_SFT_STEPS}" \
    --context "${MODULE1_CONTEXT_LENGTH}" \
    --embedding "${MODULE1_EMBEDDING_DIM}" \
    --hidden "${MODULE1_HIDDEN_DIM}" \
    --batch "${MODULE1_BATCH_SIZE}" \
    --workers "${MODULE1_WORKERS}" \
    --seed "${SEED}" \
    2>&1 | tee "${RUN_DIR}/module1_session.log"
  [[ -s "${MODULE1_SESSION_DIR}/session_report.json" && -s "${MODULE1_SESSION_DIR}/checkpoint.bin" ]] || fatal "Module 1 session did not publish checkpoint and report"
  MODULE1_PENDING_HASH="$(grep -o '"checkpoint_hash":"[0-9a-f]*"' "${MODULE1_SESSION_DIR}/session_report.json" | head -1 | cut -d'"' -f4)"
  [[ "${#MODULE1_PENDING_HASH}" -eq 64 ]] || fatal "Module 1 report does not contain a valid checkpoint hash"
  log "running native deterministic Module 1 checkpoint inspection"
  "${BUILD_DIR}/cct_curriculum_inspect" \
    --checkpoint "${MODULE1_SESSION_DIR}/checkpoint.bin" \
    --tokenizer "${TOKENIZER}" \
    --prompts "${MODULE1_PROMPTS}" \
    --output "${MODULE1_SESSION_DIR}/inference.jsonl" \
    --max-new-tokens 64 \
    2>&1 | tee "${RUN_DIR}/module1_inspect.log"
  grep -q '"status":"PASS"' "${RUN_DIR}/module1_inspect.log" || fatal "Module 1 checkpoint inspection failed"
  [[ -s "${MODULE1_SESSION_DIR}/inference.jsonl" ]] || fatal "Module 1 inference output is missing"
  OBSERVATION_TEMPLATE=""
  for ((OBSERVATION_INDEX=1; OBSERVATION_INDEX<=MODULE1_PROMPT_COUNT; OBSERVATION_INDEX++)); do
    if [[ -n "${OBSERVATION_TEMPLATE}" ]]; then OBSERVATION_TEMPLATE+=", "; fi
    OBSERVATION_TEMPLATE+="\"prompt ${OBSERVATION_INDEX}: ...\""
  done
  cat > "${MODULE1_SESSION_DIR}/mastery_prompt.md" <<EOF
# Module 1 Human Competency Validation — ${MODULE1_SESSION_ID}

**Submodule:** ${MODULE1_SUBMODULE_ID} — ${MODULE1_SUBMODULE_NAME}

**Learning objective:** ${MODULE1_OBJECTIVE}

**Training and checkpoint report:** ${MODULE1_SESSION_DIR}/session_report.json

**Held-out source material:** ${MODULE1_DATA_DIR}/pretrain_test.txt

**Actual deterministic inference outputs:** ${MODULE1_SESSION_DIR}/inference.jsonl

Review the session report, the held-out material, and every generated continuation in inference.jsonl. The prompt file used by the native inspector was ${MODULE1_PROMPTS}. Use the generated outputs as evidence and add your own unseen prompts appropriate to this submodule. Record at least ${MODULE1_PROMPT_COUNT} prompt-by-prompt observations, including at least one adversarial or boundary case.

${MODULE1_ACCEPTANCE}

Do not mark PASS from loss, perplexity, or token accuracy alone. Reject invalid output, empty output, all-EOS output, systematic repetition, or output that fails the submodule objective. If the output packet is missing or the checkpoint cannot be run, record INSUFFICIENT_EVIDENCE and do not advance.

Write the result to ${MODULE1_VALIDATION_ROOT}/${MODULE1_SESSION_ID}.json using this exact shape:

{
  "module": "module-1",
  "submodule": "module-1-submodule-${MODULE1_SUBMODULE_INDEX}",
  "session_id": "${MODULE1_SESSION_ID}",
  "checkpoint_hash": "${MODULE1_PENDING_HASH}",
  "result": "PASS",
  "evaluator": "replace-with-your-identifier",
  "timestamp_utc": "replace-with-UTC-timestamp",
  "observations": [${OBSERVATION_TEMPLATE}]
}

The script will stop until this record exists. It will never self-approve the submodule.
EOF
  write_module1_state "AWAITING_HUMAN_VALIDATION" "${MODULE1_SUBMODULE_INDEX}" "${MODULE1_FAILURES}" "${MODULE1_SESSION_ID}" "${MODULE1_SESSION_DIR}/checkpoint.bin" "${MODULE1_PENDING_HASH}" "${MODULE1_PARENT_CHECKPOINT}"
  write_module1_summary "AWAITING_HUMAN_VALIDATION" "Module 1 submodule completed; human competency validation is required before advancement"
  log "Module 1 submodule ${MODULE1_SUBMODULE_ID} complete; review ${MODULE1_SESSION_DIR}/mastery_prompt.md"
  log "actual inference outputs: ${MODULE1_SESSION_DIR}/inference.jsonl"
  cat "${MODULE1_SESSION_DIR}/inference.jsonl"
  exit 0
fi

if [[ "${CURRICULUM_MODE}" == "1" ]]; then
  CURRICULUM_STATE_FILE="${CURRICULUM_ROOT}/state.env"
  CURRICULUM_VALIDATION_ROOT="${CURRICULUM_ROOT}/validation"
  CURRICULUM_SESSIONS_ROOT="${CURRICULUM_ROOT}/sessions"
  CURRICULUM_DATA_ROOT="${CURRICULUM_ROOT}/data"
  CURRICULUM_CHUNK_STRIDE="$(number_or_default CURRICULUM_CHUNK_STRIDE "${CURRICULUM_CHUNK_STRIDE:-10000}")"
  CURRICULUM_RETRY_STRIDE="$(number_or_default CURRICULUM_RETRY_STRIDE "${CURRICULUM_RETRY_STRIDE:-5000}")"
  CURRICULUM_VALIDATION_GAP="$(number_or_default CURRICULUM_VALIDATION_GAP "${CURRICULUM_VALIDATION_GAP:-1000}")"
  CURRICULUM_TEST_GAP="$(number_or_default CURRICULUM_TEST_GAP "${CURRICULUM_TEST_GAP:-2000}")"
  CURRICULUM_SFT_VALIDATION_GAP="$(number_or_default CURRICULUM_SFT_VALIDATION_GAP "${CURRICULUM_SFT_VALIDATION_GAP:-1000}")"
  CURRICULUM_PAGE_DELAY_MS="$(number_or_default CURRICULUM_PAGE_DELAY_MS "${CURRICULUM_PAGE_DELAY_MS:-5000}")"
  CURRICULUM_RETRY_COUNT="$(number_or_default CURRICULUM_RETRY_COUNT "${CURRICULUM_RETRY_COUNT:-12}")"
  CURRICULUM_SFT_SCAN_MULTIPLIER="$(number_or_default CURRICULUM_SFT_SCAN_MULTIPLIER "${CURRICULUM_SFT_SCAN_MULTIPLIER:-100}")"
  FINEWEB_REVISION="${FINEWEB_REVISION:-87f09149ef4734204d70ed1d046ddc9ca3f2b8f9}"
  OASST_REVISION="${OASST_REVISION:-fdf72ae0827c1cda404aff25b6603abec9e3399b}"
  MINIMUM_EDUCATION_SCORE="${MINIMUM_EDUCATION_SCORE:-2.0}"
  mkdir -p "${CURRICULUM_VALIDATION_ROOT}" "${CURRICULUM_SESSIONS_ROOT}" "${CURRICULUM_DATA_ROOT}"

  log "running native regression and documentation checks before curriculum training"
  "${BUILD_DIR}/cct_nlp_trainer_tests" 2>&1 | tee "${RUN_DIR}/curriculum_nlp_trainer_tests.log"
  "${BUILD_DIR}/cct_track1_tests" 2>&1 | tee "${RUN_DIR}/curriculum_track1_tests.log"
  "${BUILD_DIR}/cct_documentation_consistency_tests" 2>&1 | tee "${RUN_DIR}/curriculum_documentation_consistency.log"
  grep -q 'SUMMARY 14/14 passed' "${RUN_DIR}/curriculum_nlp_trainer_tests.log" || fatal "curriculum NLP trainer regressions did not pass 14/14"
  grep -q 'SUMMARY 1/1 passed' "${RUN_DIR}/curriculum_documentation_consistency.log" || fatal "curriculum documentation consistency did not pass 1/1"
  if [[ "${RUN_FULL_CTEST}" == "1" ]]; then
    log "running complete native CTest suite before curriculum session"
    ctest --test-dir "${BUILD_DIR}" --output-on-failure 2>&1 | tee "${RUN_DIR}/curriculum_ctest.log"
  fi

  write_curriculum_state() {
    local status="$1" level="$2" failures="$3" pending_session="$4" pending_checkpoint="$5" pending_hash="$6" parent_checkpoint="$7"
    cat > "${CURRICULUM_STATE_FILE}" <<EOF
CURRICULUM_STATUS=${status}
CURRICULUM_LEVEL=${level}
CURRICULUM_FAILURES=${failures}
CURRICULUM_PENDING_SESSION=${pending_session}
CURRICULUM_PENDING_CHECKPOINT=${pending_checkpoint}
CURRICULUM_PENDING_CHECKPOINT_HASH=${pending_hash}
CURRICULUM_PARENT_CHECKPOINT=${parent_checkpoint}
EOF
  }

  write_curriculum_summary() {
    local status="$1" message="$2"
    cat > "${RUN_DIR}/run_summary.json" <<EOF
{
  "status":"${status}",
  "curriculum_status":"${status}",
  "message":"${message}",
  "curriculum_root":"${CURRICULUM_ROOT}",
  "state_file":"${CURRICULUM_STATE_FILE}",
  "training_authorized":false,
  "claim_boundary":"competency-session research workflow; human mastery validation is mandatory and no broad language or intelligence claim is made"
}
EOF
    sed -i 's/"status":"RUNNING"/"status":"'"${status}"'"/' "${RUN_DIR}/run_config.json"
  }

  if [[ -f "${CURRICULUM_STATE_FILE}" ]]; then
    # This file is generated by this script and contains only scalar state values.
    # shellcheck disable=SC1090
    source "${CURRICULUM_STATE_FILE}"
  else
    CURRICULUM_STATUS=READY_TO_TRAIN
    CURRICULUM_LEVEL=0
    CURRICULUM_FAILURES=0
    CURRICULUM_PENDING_SESSION=
    CURRICULUM_PENDING_CHECKPOINT=
    CURRICULUM_PENDING_CHECKPOINT_HASH=
    CURRICULUM_PARENT_CHECKPOINT=
    write_curriculum_state "${CURRICULUM_STATUS}" "${CURRICULUM_LEVEL}" "${CURRICULUM_FAILURES}" "" "" "" ""
  fi

  if [[ "${CURRICULUM_STATUS}" == "AWAITING_HUMAN_VALIDATION" ]]; then
    HUMAN_VALIDATION_FILE="${HUMAN_VALIDATION_FILE:-${CURRICULUM_VALIDATION_ROOT}/${CURRICULUM_PENDING_SESSION}.json}"
    MASTERY_PACKET="${CURRICULUM_SESSIONS_ROOT}/${CURRICULUM_PENDING_SESSION}/mastery_prompt.md"
    if [[ ! -s "${HUMAN_VALIDATION_FILE}" ]]; then
      log "session ${CURRICULUM_PENDING_SESSION} is awaiting human validation"
      log "review ${MASTERY_PACKET} and write PASS/FAIL JSON to ${HUMAN_VALIDATION_FILE}"
      write_curriculum_summary "AWAITING_HUMAN_VALIDATION" "human mastery validation is required before the next session"
      exit 0
    fi
    grep -Eq '"session_id"[[:space:]]*:[[:space:]]*"'"${CURRICULUM_PENDING_SESSION}"'"' "${HUMAN_VALIDATION_FILE}" || fatal "human validation session_id does not match pending session"
    grep -Eq '"checkpoint_hash"[[:space:]]*:[[:space:]]*"'"${CURRICULUM_PENDING_CHECKPOINT_HASH}"'"' "${HUMAN_VALIDATION_FILE}" || fatal "human validation checkpoint hash does not match pending checkpoint"
    if grep -Eq '"result"[[:space:]]*:[[:space:]]*"PASS"' "${HUMAN_VALIDATION_FILE}"; then
      log "human mastery PASS for ${CURRICULUM_PENDING_SESSION}; advancing exactly one curriculum level"
      NEXT_LEVEL=$((CURRICULUM_LEVEL + 1))
      if ((NEXT_LEVEL > CURRICULUM_MAX_LEVEL)); then
        write_curriculum_state "CURRICULUM_COMPLETE" "${NEXT_LEVEL}" "0" "" "" "" "${CURRICULUM_PENDING_CHECKPOINT}"
        write_curriculum_summary "CURRICULUM_COMPLETE" "all declared curriculum levels have human PASS records"
        exit 0
      fi
      PREVIOUS_CHECKPOINT="${CURRICULUM_PENDING_CHECKPOINT}"
      write_curriculum_state "READY_TO_TRAIN" "${NEXT_LEVEL}" "0" "" "" "" "${PREVIOUS_CHECKPOINT}"
      CURRICULUM_STATUS=READY_TO_TRAIN
      CURRICULUM_LEVEL=${NEXT_LEVEL}
      CURRICULUM_FAILURES=0
      CURRICULUM_PENDING_SESSION=
      CURRICULUM_PENDING_CHECKPOINT=
      CURRICULUM_PENDING_CHECKPOINT_HASH=
      CURRICULUM_PARENT_CHECKPOINT="${PREVIOUS_CHECKPOINT}"
    elif grep -Eq '"result"[[:space:]]*:[[:space:]]*"FAIL"' "${HUMAN_VALIDATION_FILE}"; then
      if ((CURRICULUM_FAILURES == 0)); then
        NEXT_FAILURES=1
        write_curriculum_state "READY_TO_RETRY" "${CURRICULUM_LEVEL}" "${NEXT_FAILURES}" "" "" "" "${CURRICULUM_PENDING_CHECKPOINT}"
        cat > "${CURRICULUM_ROOT}/retry_required.md" <<EOF
# Retry Required

Human validation failed for session ${CURRICULUM_PENDING_SESSION}. The curriculum is stopped. The next invocation will select a fresh disjoint chunk at the same competency under the same model, tokenizer, optimizer, and validation contract.

The failed checkpoint remains immutable at ${CURRICULUM_PENDING_CHECKPOINT}. This is not yet an architectural conclusion; a second controlled failure is required.
EOF
        write_curriculum_summary "READY_TO_RETRY" "human mastery failed once; repeat the same competency on a fresh disjoint chunk"
        exit 0
      fi
      write_curriculum_state "ARCHITECTURE_DIAGNOSIS_REQUIRED" "${CURRICULUM_LEVEL}" "${CURRICULUM_FAILURES}" "" "${CURRICULUM_PENDING_CHECKPOINT}" "${CURRICULUM_PENDING_CHECKPOINT_HASH}" "${CURRICULUM_PARENT_CHECKPOINT}"
      cat > "${CURRICULUM_ROOT}/architecture_diagnosis_required.md" <<EOF
# Architecture Diagnosis Required

Human validation failed twice for curriculum level ${CURRICULUM_LEVEL} under disjoint data chunks with the same declared contract. Preserve both session directories and review their reports before changing architecture, data, or optimization.

The workflow is fail-closed and will not advance automatically.
EOF
      write_curriculum_summary "ARCHITECTURE_DIAGNOSIS_REQUIRED" "the same competency failed twice; architecture/data diagnosis is required"
      exit 0
    else
      fatal "human validation result must be PASS or FAIL"
    fi
  elif [[ "${CURRICULUM_STATUS}" == "READY_TO_RETRY" ]]; then
    CURRICULUM_STATUS=READY_TO_TRAIN
  elif [[ "${CURRICULUM_STATUS}" == "ARCHITECTURE_DIAGNOSIS_REQUIRED" || "${CURRICULUM_STATUS}" == "CURRICULUM_COMPLETE" ]]; then
    write_curriculum_summary "${CURRICULUM_STATUS}" "curriculum state is terminal until explicitly reviewed"
    exit 0
  fi

  BASE_OFFSET=$((CURRICULUM_LEVEL * CURRICULUM_CHUNK_STRIDE + CURRICULUM_FAILURES * CURRICULUM_RETRY_STRIDE))
  SESSION_ID="level-${CURRICULUM_LEVEL}-attempt-${CURRICULUM_FAILURES}"
  SESSION_DIR="${CURRICULUM_SESSIONS_ROOT}/${SESSION_ID}"
  DATA_DIR="${CURRICULUM_DATA_ROOT}/${SESSION_ID}"
  mkdir -p "${SESSION_DIR}" "${DATA_DIR}"
  log "preparing FineWeb-Edu and OpenAssistant session ${SESSION_ID} at source offset ${BASE_OFFSET}"
  "${BUILD_DIR}/cct_curriculum_prepare" \
    --output "${DATA_DIR}" \
    --pretrain-offset "${BASE_OFFSET}" \
    --pretrain-rows "${CURRICULUM_CHUNK_ROWS}" \
    --validation-offset "$((BASE_OFFSET + CURRICULUM_VALIDATION_GAP))" \
    --validation-rows "${CURRICULUM_VALIDATION_ROWS}" \
    --test-offset "$((BASE_OFFSET + CURRICULUM_TEST_GAP))" \
    --test-rows "${CURRICULUM_TEST_ROWS}" \
    --sft-offset "${BASE_OFFSET}" \
    --sft-rows "${CURRICULUM_CHUNK_ROWS}" \
    --sft-validation-offset "$((BASE_OFFSET + CURRICULUM_SFT_VALIDATION_GAP))" \
    --sft-validation-rows "${CURRICULUM_VALIDATION_ROWS}" \
    --page-length 100 \
    --page-delay-ms "${CURRICULUM_PAGE_DELAY_MS}" \
    --retry-count "${CURRICULUM_RETRY_COUNT}" \
    --sft-scan-multiplier "${CURRICULUM_SFT_SCAN_MULTIPLIER}" \
    --minimum-education-score "${MINIMUM_EDUCATION_SCORE}" \
    --fineweb-revision "${FINEWEB_REVISION}" \
    --oasst-revision "${OASST_REVISION}" \
    2>&1 | tee "${RUN_DIR}/curriculum_prepare.log"
  [[ -s "${DATA_DIR}/manifest.json" && -s "${DATA_DIR}/pretrain_test.txt" ]] || fatal "curriculum preparation did not publish its governed manifest and test split"
  PARENT_ARG=()
  if [[ -n "${CURRICULUM_PARENT_CHECKPOINT}" ]]; then PARENT_ARG=(--parent-checkpoint "${CURRICULUM_PARENT_CHECKPOINT}"); fi
  log "training exactly one competency session from its immutable parent checkpoint"
  "${BUILD_DIR}/cct_curriculum_session" \
    --input "${DATA_DIR}" \
    --output "${SESSION_DIR}" \
    --tokenizer "${TOKENIZER}" \
    "${PARENT_ARG[@]}" \
    --session-id "${SESSION_ID}" \
    --level "${CURRICULUM_LEVEL}" \
    --pretrain-steps "${CURRICULUM_PRETRAIN_STEPS}" \
    --sft-steps "${CURRICULUM_SFT_STEPS}" \
    --context "${CONTEXT_LENGTH}" \
    --embedding "${EMBEDDING_DIM}" \
    --hidden "${HIDDEN_DIM}" \
    --batch "${ARCHITECTURE_BATCH}" \
    --seed "${SEED}" \
    2>&1 | tee "${RUN_DIR}/curriculum_session.log"
  [[ -s "${SESSION_DIR}/session_report.json" && -s "${SESSION_DIR}/checkpoint.bin" ]] || fatal "curriculum session did not publish a checkpoint and report"
  PENDING_HASH="$(grep -o '"checkpoint_hash":"[0-9a-f]*"' "${SESSION_DIR}/session_report.json" | head -1 | cut -d'"' -f4)"
  [[ "${#PENDING_HASH}" -eq 64 ]] || fatal "curriculum session report does not contain a valid final checkpoint hash"
  case "${CURRICULUM_LEVEL}" in
    0) COMPETENCY_NAME="stable_symbols_and_word_boundaries"; PROMPT="Continue unseen English sentences exactly, preserving word boundaries, punctuation, and valid token output." ;;
    1) COMPETENCY_NAME="sentence_completion_and_local_grammar"; PROMPT="Complete unseen English sentences with grammatical agreement, tense, articles, and punctuation." ;;
    2) COMPETENCY_NAME="paragraph_coherence_and_topic_persistence"; PROMPT="Continue unseen educational paragraphs while preserving topic, referents, and sentence coherence." ;;
    3) COMPETENCY_NAME="reading_comprehension_and_answer_targeting"; PROMPT="Answer unseen English questions from supplied passages, abstaining when the passage does not support an answer." ;;
    4) COMPETENCY_NAME="instruction_following_and_response_structure"; PROMPT="Follow unseen English instructions with relevant, complete, correctly structured responses." ;;
    5) COMPETENCY_NAME="ambiguity_recognition_and_clarification"; PROMPT="Identify underspecified prompts and ask a useful clarification before answering." ;;
    6) COMPETENCY_NAME="conversational_continuity_and_repair"; PROMPT="Use prior turns, handle corrections, and repair contradictions without losing context." ;;
    *) COMPETENCY_NAME="bounded_transfer_and_generalization"; PROMPT="Handle new-domain English prompts while preserving the earlier declared competencies." ;;
  esac
  cat > "${SESSION_DIR}/mastery_prompt.md" <<EOF
# Human Mastery Validation — ${SESSION_ID}

**Competency:** ${COMPETENCY_NAME}

**Prompt:** ${PROMPT}

Review the session report, held-out metrics, checkpoint lineage, and the disjoint held-out test material at ${DATA_DIR}/pretrain_test.txt. Ask at least five unseen prompts appropriate to this competency, including one adversarial or boundary case. Record prompt-by-prompt observations. Do not mark PASS from loss alone. PASS requires the declared competency to be demonstrated consistently on unseen examples; otherwise mark FAIL.

Write the result to ${CURRICULUM_VALIDATION_ROOT}/${SESSION_ID}.json using this exact shape:

JSON shape:

{
  "session_id": "${SESSION_ID}",
  "checkpoint_hash": "${PENDING_HASH}",
  "competency": "${COMPETENCY_NAME}",
  "result": "PASS",
  "evaluator": "replace-with-your-identifier",
  "timestamp_utc": "replace-with-UTC-timestamp",
  "observations": ["prompt 1: ...", "prompt 2: ...", "prompt 3: ...", "prompt 4: ...", "prompt 5: ..."],
  "diagnosis": "optional"
}

The script will stop until this record exists. It will never self-approve the competency.
EOF
  write_curriculum_state "AWAITING_HUMAN_VALIDATION" "${CURRICULUM_LEVEL}" "${CURRICULUM_FAILURES}" "${SESSION_ID}" "${SESSION_DIR}/checkpoint.bin" "${PENDING_HASH}" "${CURRICULUM_PARENT_CHECKPOINT}"
  write_curriculum_summary "AWAITING_HUMAN_VALIDATION" "session completed; human mastery validation is required before advancement"
  log "session ${SESSION_ID} complete; review ${SESSION_DIR}/mastery_prompt.md and provide the validation JSON"
  exit 0
fi

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

grep -q 'SUMMARY 14/14 passed' "${RUN_DIR}/nlp_trainer_tests.log" || fatal "NLP trainer regression suite did not pass 14/14"
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
