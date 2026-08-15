#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCHMARK="${ROOT_DIR}/build-seq/cct_optimization_benchmark"
OUTPUT_DIR="${ROOT_DIR}/artifacts/optimization/after_sweeps"
TRAIN="${ROOT_DIR}/artifacts/track1/real-release/data/pretrain_train.txt"
VALIDATION="${ROOT_DIR}/artifacts/track1/real-release/data/pretrain_validation.txt"
TEST="${ROOT_DIR}/artifacts/track1/real-release/data/pretrain_test.txt"
TOKENIZER="${ROOT_DIR}/data/stage-10/tokenizer_snapshot.bin"

[[ -x "${BENCHMARK}" ]] || { echo "missing native optimization benchmark: ${BENCHMARK}" >&2; exit 2; }
mkdir -p "${OUTPUT_DIR}"

for context in 32 64 128 256; do
  "${BENCHMARK}" --train "${TRAIN}" --validation "${VALIDATION}" --test "${TEST}" --tokenizer "${TOKENIZER}" \
    --output "${OUTPUT_DIR}/context-${context}.json" --context "${context}" --embedding 16 --hidden 16 \
    --steps 20 --batch 4 --workers 2 --train-sequences 64 --eval-sequences 32 --decode-tokens 64 --repeats 3
 done

for width in 8 16 32; do
  "${BENCHMARK}" --train "${TRAIN}" --validation "${VALIDATION}" --test "${TEST}" --tokenizer "${TOKENIZER}" \
    --output "${OUTPUT_DIR}/width-${width}.json" --context 128 --embedding "${width}" --hidden "${width}" \
    --steps 20 --batch 4 --workers 2 --train-sequences 64 --eval-sequences 32 --decode-tokens 64 --repeats 3
 done

printf 'status=COMPLETE\noutput_dir=%s\n' "${OUTPUT_DIR}"
