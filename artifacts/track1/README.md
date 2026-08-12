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

Clone the repository, change to its root, then run the same build and preparation commands above. CUDA is not required for this preparation stage; the generated data artifacts can be consumed by either native CPU or CUDA training paths after their input adapter is selected. Do not use the frozen SQuAD final-test stream for checkpoint selection or parameter updates.

## Evidence from complete direct-file preparation

A complete production preparation exercised all three direct source routes: one WikiText Zip archive and two GEM SQuAD JSON files. It completed with `passed:true`, `source_pages:5`, `source_rows:173106`, `pretrain_tokens:2000000`, `sft_train_rows:7200`, `sft_evaluation_rows:800`, `final_test_rows:11873`, `overlap_ids:0`, `malformed_rows:0`, and 38,817 observed unanswerable training rows. The resulting manifest, report, and evaluation contract are tracked in `artifacts/track1/real-full-preparation/`.

> The completed preparation validates native acquisition parsing and data-governance controls. It does not substitute for a completed model-training and evaluation run.

## References

[1]: https://huggingface.co/datasets/GEM/squad_v2/tree/67199807729e631955056c71c258b7acbee548a3 "GEM SQuAD v2 pinned repository"
[2]: https://huggingface.co/datasets/rajpurkar/squad_v2 "Official SQuAD 2.0 Hugging Face dataset"
[3]: https://huggingface.co/datasets/ggml-org/ci/tree/927b3642933080f1b0e811e2f916e14c292992f9 "Pinned WikiText raw archive mirror"
