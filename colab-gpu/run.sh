#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${ROOT_DIR}/data"
NATIVE_DIR="${ROOT_DIR}/native"
ARTIFACT_DIR="${ROOT_DIR}/artifacts/$(date -u +%Y%m%dT%H%M%SZ)"
CHECKPOINT_DIR="${ROOT_DIR}/checkpoints"
mkdir -p "${DATA_DIR}" "${ARTIFACT_DIR}" "${CHECKPOINT_DIR}"

WIKI_URL="${WIKI_URL:-https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream1.xml-p1p41242.bz2}"
OASST_URL="${OASST_URL:-https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/fdf72ae0827c1cda404aff25b6603abec9e3399b/2023-04-12_oasst_ready.messages.jsonl.gz}"
WIKI_ARCHIVE="${DATA_DIR}/enwiki-shard.bz2"
OASST_ARCHIVE="${DATA_DIR}/oasst1-ready.messages.jsonl.gz"
WIKI_PREFIX="${DATA_DIR}/wiki"
OASST_PREFIX="${DATA_DIR}/oasst"
PREPARE_BIN="${NATIVE_DIR}/prepare"
VALIDATE_BIN="${NATIVE_DIR}/validate"
TRAIN_BIN="${NATIVE_DIR}/cuda_train"
BASE_CHECKPOINT="${CHECKPOINT_DIR}/cct_base_cuda.bin"
SFT_CHECKPOINT="${CHECKPOINT_DIR}/cct_oasst_sft_cuda.bin"

MAX_WIKI_BYTES="${MAX_WIKI_BYTES:-0}"
MAX_OASST_BYTES="${MAX_OASST_BYTES:-0}"
MAX_TRAIN_TOKENS="${MAX_TRAIN_TOKENS:-16000000}"
MAX_VALIDATION_TOKENS="${MAX_VALIDATION_TOKENS:-2000000}"
MAX_TEST_TOKENS="${MAX_TEST_TOKENS:-2000000}"
PRETRAIN_STEPS="${PRETRAIN_STEPS:-1000}"
SFT_STEPS="${SFT_STEPS:-600}"
BATCH_SIZE="${BATCH_SIZE:-32}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-64}"
HIDDEN_DIM="${HIDDEN_DIM:-32}"
EMBEDDING_DIM="${EMBEDDING_DIM:-32}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"
SMOKE="${SMOKE:-0}"

log() { printf '[cct-colab] %s\n' "$*"; }
fatal() { printf '[cct-colab] ERROR: %s\n' "$*" >&2; exit 2; }
need() { command -v "$1" >/dev/null 2>&1 || fatal "required command not found: $1"; }

need curl
need sha256sum
need g++
need bzip2
need gzip
need nvcc
need nvidia-smi
nvidia-smi >/dev/null 2>&1 || fatal "no visible NVIDIA GPU; switch the Colab runtime to GPU before running"

if [[ "${MAX_WIKI_BYTES}" != "0" ]]; then
  log "MAX_WIKI_BYTES is set but the native downloader will not truncate compressed archives; unset it or use a complete shard"
fi
if [[ "${MAX_OASST_BYTES}" != "0" ]]; then
  log "MAX_OASST_BYTES is set but the native downloader will not truncate compressed archives; unset it or use a complete export"
fi

log "compiling native C++20 dataset preparer"
g++ -std=c++20 -O3 -Wall -Wextra -Wpedantic -Werror "${NATIVE_DIR}/prepare.cpp" -o "${PREPARE_BIN}"
g++ -std=c++20 -O3 -Wall -Wextra -Wpedantic -Werror "${NATIVE_DIR}/validate.cpp" -o "${VALIDATE_BIN}"
log "compiling native C++20/CUDA CCT trainer"
nvcc -std=c++20 -O3 --use_fast_math -lineinfo -Xcompiler=-Wall,-Wextra,-Wpedantic,-Werror "${NATIVE_DIR}/cuda_train.cu" -o "${TRAIN_BIN}"

fetch() {
  local url="$1" output="$2"
  if [[ -s "${output}" ]]; then
    log "using existing $(basename "${output}")"
    return
  fi
  log "downloading ${url}"
  curl --fail --location --retry 5 --retry-delay 3 --continue-at - --output "${output}.partial" "${url}"
  mv "${output}.partial" "${output}"
}

fetch "${WIKI_URL}" "${WIKI_ARCHIVE}"
fetch "${OASST_URL}" "${OASST_ARCHIVE}"
sha256sum "${WIKI_ARCHIVE}" | tee "${ARTIFACT_DIR}/wiki_archive.sha256"
sha256sum "${OASST_ARCHIVE}" | tee "${ARTIFACT_DIR}/oasst_archive.sha256"

if [[ ! -s "${WIKI_PREFIX}.train.bin" || ! -s "${WIKI_PREFIX}.validation.bin" || ! -s "${WIKI_PREFIX}.test.bin" ]]; then
  log "preparing Wikimedia byte-token streams; this may take substantial disk and time"
  rm -f "${WIKI_PREFIX}.train.bin" "${WIKI_PREFIX}.validation.bin" "${WIKI_PREFIX}.test.bin" "${WIKI_PREFIX}.manifest.json"
  bzip2 -dc "${WIKI_ARCHIVE}" | "${PREPARE_BIN}" wiki "${WIKI_PREFIX}"
fi
if [[ ! -s "${OASST_PREFIX}.train.bin" || ! -s "${OASST_PREFIX}.validation.bin" || ! -s "${OASST_PREFIX}.test.bin" ]]; then
  log "preparing OASST1 English assistant-message streams"
  rm -f "${OASST_PREFIX}.train.bin" "${OASST_PREFIX}.validation.bin" "${OASST_PREFIX}.test.bin" "${OASST_PREFIX}.manifest.json"
  gzip -dc "${OASST_ARCHIVE}" | "${PREPARE_BIN}" oasst "${OASST_PREFIX}"
fi
"${VALIDATE_BIN}" "${WIKI_PREFIX}.train.bin" "${WIKI_PREFIX}.validation.bin" "${WIKI_PREFIX}.test.bin" | tee "${ARTIFACT_DIR}/wiki_stream_validation.txt"
"${VALIDATE_BIN}" "${OASST_PREFIX}.train.bin" "${OASST_PREFIX}.validation.bin" "${OASST_PREFIX}.test.bin" | tee "${ARTIFACT_DIR}/oasst_stream_validation.txt"
cp "${ROOT_DIR}/data/sources.json" "${ARTIFACT_DIR}/sources.json"
cp "${WIKI_PREFIX}.manifest.json" "${ARTIFACT_DIR}/wiki_manifest.json"
cp "${OASST_PREFIX}.manifest.json" "${ARTIFACT_DIR}/oasst_manifest.json"

manifest_tokens() {
  local manifest="$1" split="$2"
  grep -o "\\\"${split}\\\":{[^}]*}" "${manifest}" | grep -o '\"tokens\":[0-9]*' | cut -d: -f2
}
if [[ "${SMOKE}" == "1" ]]; then
  MAX_TRAIN_TOKENS=1048576
  MAX_VALIDATION_TOKENS=131072
  MAX_TEST_TOKENS=131072
  WIKI_TRAIN_TOKENS="$(manifest_tokens "${WIKI_PREFIX}.manifest.json" train)"
  OASST_TRAIN_TOKENS="$(manifest_tokens "${OASST_PREFIX}.manifest.json" train)"
  PRETRAIN_STEPS=$(( ( (WIKI_TRAIN_TOKENS - 1) / CONTEXT_LENGTH + BATCH_SIZE - 1 ) / BATCH_SIZE ))
  SFT_STEPS=$(( ( (OASST_TRAIN_TOKENS - 1) / CONTEXT_LENGTH + BATCH_SIZE - 1 ) / BATCH_SIZE ))
  (( PRETRAIN_STEPS < 1 )) && PRETRAIN_STEPS=1
  (( SFT_STEPS < 1 )) && SFT_STEPS=1
  log "SMOKE=1: one training pass per prepared split; pretrain_steps=${PRETRAIN_STEPS}, sft_steps=${SFT_STEPS}"
fi

cat > "${ARTIFACT_DIR}/run_config.json" <<EOF
{
  "backend":"native-c++20-cuda",
  "gpu_required":true,
  "wiki_url":"${WIKI_URL}",
  "oasst_url":"${OASST_URL}",
  "max_train_tokens":${MAX_TRAIN_TOKENS},
  "max_validation_tokens":${MAX_VALIDATION_TOKENS},
  "max_test_tokens":${MAX_TEST_TOKENS},
  "pretrain_steps":${PRETRAIN_STEPS},
  "sft_steps":${SFT_STEPS},
  "batch_size":${BATCH_SIZE},
  "context_length":${CONTEXT_LENGTH},
  "hidden_dim":${HIDDEN_DIM},
  "embedding_dim":${EMBEDDING_DIM},
  "checkpoint_every":${CHECKPOINT_EVERY},
  "training_authorized":false
}
EOF

train_stage() {
  local stage="$1" train_path="$2" validation_path="$3" test_path="$4" checkpoint="$5" steps="$6" resume_flag="${7:-}"
  if [[ -s "${checkpoint}" ]]; then
    resume_flag="--resume ${checkpoint}"
    log "resuming ${stage} from an identity-checked checkpoint"
  fi
  log "starting ${stage} native CUDA training"
  local log_path="${ARTIFACT_DIR}/${stage}.log"
  "${TRAIN_BIN}" --train "${train_path}" --validation "${validation_path}" --test "${test_path}" \
    --checkpoint "${checkpoint}" --steps "${steps}" --batch "${BATCH_SIZE}" --context "${CONTEXT_LENGTH}" \
    --hidden "${HIDDEN_DIM}" --embedding "${EMBEDDING_DIM}" --max-train-tokens "${MAX_TRAIN_TOKENS}" \
    --max-validation-tokens "${MAX_VALIDATION_TOKENS}" --max-test-tokens "${MAX_TEST_TOKENS}" \
    ${resume_flag} 2>&1 | tee "${log_path}"
  tail -n 1 "${log_path}" > "${ARTIFACT_DIR}/${stage}_metrics.json"
  grep -q '"status":"PASS"' "${ARTIFACT_DIR}/${stage}_metrics.json" || fatal "${stage} did not produce a PASS metrics record"
}

train_stage "pretrain_wiki" "${WIKI_PREFIX}.train.bin" "${WIKI_PREFIX}.validation.bin" "${WIKI_PREFIX}.test.bin" "${BASE_CHECKPOINT}" "${PRETRAIN_STEPS}"
train_stage "sft_oasst" "${OASST_PREFIX}.train.bin" "${OASST_PREFIX}.validation.bin" "${OASST_PREFIX}.test.bin" "${SFT_CHECKPOINT}" "${SFT_STEPS}" "--resume-base ${BASE_CHECKPOINT}"

cat > "${ARTIFACT_DIR}/release_record.json" <<EOF
{
  "status":"PASS",
  "backend":"native-c++20-cuda",
  "pretraining_checkpoint":"${BASE_CHECKPOINT}",
  "sft_checkpoint":"${SFT_CHECKPOINT}",
  "pretraining_metrics":"${ARTIFACT_DIR}/pretrain_wiki_metrics.json",
  "sft_metrics":"${ARTIFACT_DIR}/sft_oasst_metrics.json",
  "training_authorized":false,
  "human_review_required":true,
  "claim_boundary":"bounded Colab GPU pilot; not broad language competence or production release"
}
EOF
log "completed; inspect ${ARTIFACT_DIR}/release_record.json and both metrics files"
