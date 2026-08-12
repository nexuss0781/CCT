# Track 1: Governed WikiText-2 and SQuAD 2.0 Preparation

Track 1 is the first compact, reproducible data pipeline for CCT-ASE. It prepares a capped WikiText-2 pretraining stream, a balanced SQuAD 2.0 supervised fine-tuning set, a held-out SFT evaluation set, and a frozen SQuAD final-test stream. The native C++20 preparer writes hashes, selected identifiers, source revisions, licenses, direct acquisition URLs, and evaluation requirements into its manifest.

| Component | Dataset and pinned source | Governed use |
|---|---|---|
| Pretraining | `Salesforce/wikitext`, `wikitext-2-raw-v1`, revision `b08601e04326c79dfdd32d625aee71d232d685c3`; direct raw archive pinned through `ggml-org/ci` revision `927b3642933080f1b0e811e2f916e14c292992f9` | WikiText-2 train, validation, and test streams; the train stream is capped at 2,000,000 byte tokens. |
| SFT and SFT evaluation | `GEM/squad_v2`, revision `67199807729e631955056c71c258b7acbee548a3`, `gem_data_split/train.json` | 8,000 examples balanced across answerable and unanswerable questions, deterministically partitioned into 7,200 SFT-train and 800 SFT-evaluation examples. |
| Final task test | `GEM/squad_v2`, revision `67199807729e631955056c71c258b7acbee548a3`, `gem_data_split/validation.json` | The entire 11,873-example SQuAD 2.0 dev/validation split, isolated from training and model-selection IDs. |

The preparer acquires WikiText through one pinned raw Zip archive and extracts its three fixed split members natively, avoiding the rate-limited rows endpoint. The GEM files are a pinned, direct-download Hugging Face representation of SQuAD 2.0. The manifest retains `rajpurkar/squad_v2` as the upstream benchmark identity, records immutable revisions and direct URLs, and hashes each acquired raw file. Answer spans are verified by decoding JSON UTF-16 surrogate pairs, converting source Unicode codepoint offsets to UTF-8 byte offsets, and requiring an exact context match. [1] [2] [3]

## Build and validate

Run the following from the repository root. All commands are native C++20; no Python data preparation is used.

```bash
make track1-test track1-gate
ctest --test-dir build-cpp --output-on-failure
```

The canonical cumulative target is:

```bash
make ci-track1
```

## Full production preparation

The preparer caches every non-empty downloaded file under `artifacts/track1/raw/` and reuses it on later invocations. Production acquisition uses one direct WikiText Zip archive plus one direct GEM JSON file per SQuAD split; it does not paginate the Hugging Face rows endpoint.

```bash
cmake -S cpp -B build-cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpp --parallel 2
./build-cpp/cct_track1_prepare \
  --output artifacts/track1 \
  --pretrain-token-cap 2000000 \
  --sft-examples 8000 \
  --sft-eval-examples 800 \
  --seed 1701
```

The completed output contains `manifest.json`, `preparation_report.json`, `evaluation_contract.json`, `data/pretrain_{train,validation,test}.txt`, `data/squad_sft_train.jsonl`, `data/squad_sft_evaluation.jsonl`, and `data/squad_final_test.jsonl`.

## Native CCT training handoff

After preparation, train the native CCT next-token model directly from the Track 1 artifacts:

```bash
make track1-train
```

The command creates `artifacts/track1/training/pretrain_checkpoint.bin`, `artifacts/track1/training/sft_checkpoint.bin`, and `artifacts/track1/training/training_report.json`. Pretraining uses WikiText-2 train data and selection validation data. SQuAD fine-tuning is formatted by the native Stage 13 formatter and applies the `target-span-only-v1` loss mask, so prompt/context tokens do not contribute to SFT loss. The frozen SQuAD final-test set is scored once only after SFT and is never used for updates or checkpoint selection.

The default command runs a bounded first-pass configuration with 200 pretraining steps and 120 SFT steps. It reports held-out next-token cross-entropy, perplexity, and token accuracy. It does **not** claim generative SQuAD exact-match or F1, because this runner does not yet include constrained answer decoding.

| Environment variable | Default | Purpose |
|---|---:|---|
| `TRACK1_PRETRAIN_STEPS` | 200 | WikiText-2 optimizer updates. |
| `TRACK1_SFT_STEPS` | 120 | Answer-target-only SQuAD optimizer updates. |
| `TRACK1_CONTEXT` | 32 | Native CCT token context length. |
| `TRACK1_EMBEDDING` / `TRACK1_HIDDEN` | 4 / 4 | Compact first-pass CCT dimensions. |
| `TRACK1_SFT_CONTEXT_BYTES` | 1024 | Maximum context bytes retained per SQuAD prompt. |
| `TRACK1_SEED` | 1701 | Reproducibility seed. |

For a short complete-corpus check before the first-pass run:

```bash
make track1-train \
  TRACK1_PRETRAIN_STEPS=2 \
  TRACK1_SFT_STEPS=2 \
  TRACK1_CONTEXT=16 \
  TRACK1_EMBEDDING=2 \
  TRACK1_HIDDEN=2 \
  TRACK1_SFT_CONTEXT_BYTES=256
```

## Bounded smoke preparation

This command is suitable for a quick, real-network exercise of the normal production acquisition path. It uses the same source revisions, manifest logic, answer-offset verifier, stable selection seed, and final-test isolation policy, but lower data budgets.

```bash
./build-cpp/cct_track1_prepare \
  --output /tmp/cct-track1-smoke \
  --pretrain-token-cap 2000 \
  --sft-examples 20 \
  --sft-eval-examples 10 \
  --source-row-limit 10000 \
  --seed 1701 \
  --fixture
```

The `--fixture` switch only permits the intentionally small SFT counts. It does not relax malformed-row, answer-span, split-overlap, manifest-digest, or missing-cache failures. For a cache-only replay, append `--no-download` after the initial acquisition has completed.

## Colab or local execution

Clone the repository, change to its root, then run the build, preparation, and `make track1-train` commands above. The Track 1 runner is a native C++20 CCT-library execution path and does not require CUDA. Do not use the frozen SQuAD final-test stream for checkpoint selection or parameter updates.

## Evidence from complete direct-file preparation

A complete production preparation exercised all three direct source routes: one WikiText Zip archive and two GEM SQuAD JSON files. It completed with `passed:true`, `source_pages:5`, `source_rows:173106`, `pretrain_tokens:2000000`, `sft_train_rows:7200`, `sft_evaluation_rows:800`, `final_test_rows:11873`, `overlap_ids:0`, `malformed_rows:0`, and 38,817 observed unanswerable training rows. The resulting manifest, report, and evaluation contract are tracked in `artifacts/track1/real-full-preparation/`.

A complete-corpus bounded-steps training verification then completed with `status:PASS`: it consumed the full prepared WikiText corpus, all 7,200 SFT examples, 64 held-out SFT-selection examples, and all 11,873 frozen final-test examples. It used target-span-only SQuAD loss masks and reported finite held-out metrics; the report is tracked at `artifacts/track1/real-training/training_report.json`.

> The completed preparation and bounded-steps native training run validate the acquisition, data-governance, optimization, checkpoint, and held-out next-token evaluation paths. They do not establish generative SQuAD exact-match or F1, broad language competence, or production readiness.

## References

[1]: https://huggingface.co/datasets/GEM/squad_v2/tree/67199807729e631955056c71c258b7acbee548a3 "GEM SQuAD v2 pinned repository"
[2]: https://huggingface.co/datasets/rajpurkar/squad_v2 "Official SQuAD 2.0 Hugging Face dataset"
[3]: https://huggingface.co/datasets/ggml-org/ci/tree/927b3642933080f1b0e811e2f916e14c292992f9 "Pinned WikiText raw archive mirror"
