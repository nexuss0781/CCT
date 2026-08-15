# One-Command Native CCT Competency Training

The repository root `run.sh` is now the canonical focused-English continual-learning workflow. It is implemented in native C++20 and orchestrated by one shell command. On each invocation it verifies or installs Ubuntu build dependencies, builds the native preparation and session executables, selects one deterministic range from pinned real datasets, trains one competency session, publishes immutable checkpoints and lineage reports, writes a human mastery packet, and stops. It never automatically declares a competency mastered.

> **Run command:**
>
> ```bash
> cd CCT
> bash run.sh
> ```

With the approved Module 1 curriculum enabled by default, the first invocation creates the durable state directory `runs/curriculum-module1/`, prepares exactly one Module 1 submodule session, trains it, runs the native deterministic checkpoint inspector, prints the actual inference outputs, and stops with `AWAITING_HUMAN_VALIDATION`. Review the generated mastery packet and inference JSONL. Then write the required validation JSON at the path printed by the script and run `bash run.sh` again. A human `PASS` advances exactly one Module 1 submodule. A human `FAIL` causes one controlled retry of the same submodule on a fresh disjoint source range. Two failures stop the workflow in `ARCHITECTURE_DIAGNOSIS_REQUIRED`. Set `CURRICULUM_MODULE1=0` only when intentionally using the older level-based curriculum path.

The script requires Internet access for the pinned Hugging Face rows API. If a dependency is missing, it attempts non-interactive Ubuntu installation with `apt-get`; this requires root or passwordless `sudo`. No Python runtime is used by data preparation, training, checkpointing, or evaluation. The native requirements are C++20, CMake, `pkg-config`, FFTW3 development files, `curl`, `sha256sum`, `zip`, and `unzip`. The default `bash run.sh` path skips the full CTest suite so training can start directly; setting `RUN_FULL_CTEST=1` builds the complete CMake target graph before invoking all 44 tests.

## Dataset contract

The initial pretraining source is the English educational FineWeb-Edu `sample-10BT` configuration at the pinned revision recorded in the session manifest. FineWeb-Edu is a large educational web corpus with text, stable identifiers, language metadata, token counts, and educational quality scores.[1] The supervised source is English-filtered OpenAssistant/oasst1 at its pinned revision; the native preparer retains stable message IDs, language, role, review, deletion, and tree metadata.[2] The downloader requests explicit row ranges from the official datasets-server API, applies the declared filters, records selected IDs, and writes a source digest.

Each session contains disjoint pretraining train, validation, and test rows. The OpenAssistant training and validation rows are also selected from separate ranges. The actual source revisions, offsets, requested counts, selected IDs, text-file hashes, acquisition pacing/retry settings, and manifest hash are stored in `data/<session-id>/manifest.json` and `data/<session-id>/source_digest.txt`. Each API page is cached under the session data directory with its dataset revision, offset, and length in the filename. A rate-limit interruption can therefore be resumed by rerunning the same command; invalid or partial cache files are discarded. A future dataset change requires a new explicit revision and cannot silently reuse an old page cache.

| Source | Role | Default pinned configuration | Native output |
|---|---|---|---|
| FineWeb-Edu | Focused-English pretraining | `sample-10BT`, `train` | `pretrain_train.txt`, `pretrain_validation.txt`, `pretrain_test.txt` |
| OpenAssistant/oasst1 | English supervised response-format practice | `default`, `train` | `sft_train.txt`, `sft_validation.txt` |
| Human-provided prompts | Mastery validation only | User-controlled and unseen | `validation/<session-id>.json` |

## Competency ladder

The curriculum starts with stable English symbols and word boundaries, then progresses through local grammar, paragraph coherence, reading comprehension, instruction following, ambiguity recognition, conversational continuity, and bounded transfer. Levels 0–2 must be demonstrated before conversational levels are attempted. The exact prompt and validation requirements are copied into each session’s `mastery_prompt.md`.

The automatic report supplies finite before/after held-out metrics, test data identity, checkpoint hashes, parent lineage, optimizer steps, and token throughput. These are evidence for the human evaluator, not substitutes for human mastery judgment. The validation record must contain at least five unseen prompt observations, an evaluator identifier, a timestamp, the exact session ID, the exact checkpoint hash, and either `PASS` or `FAIL`.

## Required human validation record

The script prints the exact required path. The following is the required schema; preserve the session and checkpoint values generated by the script.

```json
{
  "session_id": "level-0-attempt-0",
  "checkpoint_hash": "replace-with-session-checkpoint-hash",
  "competency": "stable_symbols_and_word_boundaries",
  "result": "PASS",
  "evaluator": "replace-with-your-identifier",
  "timestamp_utc": "2026-08-15T00:00:00Z",
  "observations": [
    "prompt 1: ...",
    "prompt 2: ...",
    "prompt 3: ...",
    "prompt 4: ...",
    "prompt 5: ..."
  ],
  "diagnosis": "optional"
}
```

Do not mark `PASS` from training loss, perplexity, token accuracy, or a short automatic continuation alone. The mastery decision is intentionally external to the automatic training path. If a session fails, preserve its checkpoint and report; do not overwrite it or change architecture before the controlled retry has completed.

## Default curriculum configuration

| Variable | Default | Meaning |
|---|---:|---|
|   `CURRICULUM_MODE` | `1` | Enables the continual-learning state machine. Set `0` only for the legacy Track 1 workflow. |
| `CURRICULUM_MODULE1` | `1` | Runs the approved Module 1 submodule curriculum; one submodule session per invocation. |
| `CURRICULUM_ROOT` | `runs/curriculum-module1` | Durable Module 1 state, data, sessions, and validation root when Module 1 is enabled. |
| `CURRICULUM_ROOT` | `runs/curriculum-focused-english` when Module 1 is disabled | Durable state, data, validation, and session root for the legacy level-based curriculum. |
| `CURRICULUM_CHUNK_ROWS` | `100` | Requested accepted training rows per source per session. |
| `CURRICULUM_VALIDATION_ROWS` | `40` | Requested validation rows per source per session. |
| `CURRICULUM_TEST_ROWS` | `40` | Requested held-out FineWeb test rows per session. |
| `CURRICULUM_PRETRAIN_STEPS` | `100` | Native CCT pretraining steps per session. |
| `CURRICULUM_SFT_STEPS` | `50` | Native CCT SFT steps per session. |
| `CURRICULUM_PAGE_DELAY_MS` | `5000` | Delay between dataset API pages to reduce rate-limit pressure. |
| `CURRICULUM_RETRY_COUNT` | `12` | Curl retries for transient and HTTP 429 failures. |
| `CURRICULUM_SFT_SCAN_MULTIPLIER` | `100` | Bounded OpenAssistant rows scanned per accepted filtered row. |
| `CURRICULUM_MAX_LEVEL` | `7` | Last declared level in the curriculum. |
| `CONTEXT_LENGTH` | `128` | Native sequence context. |
| `EMBEDDING_DIM`, `HIDDEN_DIM` | `16`, `16` | Native compact CCT dimensions. |
| `ARCHITECTURE_BATCH` | `8` | Token-weighted mini-batch size. |
| `SEED` | `1701` | Deterministic initialization and session seed. |
| `FINEWEB_REVISION` | pinned SHA | FineWeb-Edu dataset revision. |
| `OASST_REVISION` | pinned SHA | OpenAssistant dataset revision. |
| `MINIMUM_EDUCATION_SCORE` | `2.0` | FineWeb-Edu quality threshold. |
| `RUN_FULL_CTEST` | `0` | Skips the full native repository suite by default; set to `1` for engineering validation. |
| `INSTALL_DEPENDENCIES` | `1` | Allows automatic package installation when needed. |

For a bounded orchestration check that does not count as language evidence, use an isolated state root:

```bash
SMOKE=1 \
MINIMUM_EDUCATION_SCORE=0 \
CURRICULUM_ROOT=/tmp/cct-curriculum-smoke \
RUN_ROOT=/tmp/cct-curriculum-runs \
RUN_FULL_CTEST=0 \
bash run.sh
```

## Checkpoint and lineage behavior

Every session publishes `pretrain_checkpoint.bin` and the final `checkpoint.bin`. The checkpoint format is V3 and retains the tokenizer hash, current dataset hash, optimizer state, global optimizer step, session ID, and parent checkpoint hash. The native trainer atomically publishes checkpoints, supports V2 loading for backward compatibility, verifies model configuration on continuation, resets the new chunk cursor, and retains optimizer moments and global step across sessions.

The state file is a small generated scalar record at `CURRICULUM_ROOT/state.env`. Its meaningful states are `READY_TO_TRAIN`, `AWAITING_HUMAN_VALIDATION`, `READY_TO_RETRY`, `ARCHITECTURE_DIAGNOSIS_REQUIRED`, and `CURRICULUM_COMPLETE`. Never edit it to bypass a pending validation record. The evidence of each session remains under `CURRICULUM_ROOT/sessions/<session-id>/`, and each retry uses a fresh source range.

## Module 1 academic submodules

Module 1 is executed in this order, with one submodule per session:

| Submodule | Objective | Qualification budget |
|---|---|---|
| `1.1` | Character and symbol awareness | 500 / 100 / 100 FineWeb train/validation/test rows; 250 / 50 SFT rows; 500 / 250 steps |
| `1.2` | Whitespace and word boundaries | 750 / 150 / 150 FineWeb rows; 375 / 75 SFT rows; 750 / 375 steps |
| `1.3` | Common word patterns | 1,000 / 200 / 200 FineWeb rows; 500 / 100 SFT rows; 1,000 / 500 steps |
| `1.4` | Stable short continuation | 1,250 / 250 / 250 FineWeb rows; 625 / 125 SFT rows; 1,250 / 625 steps |

For each submodule, the native workflow prepares the FineWeb-Edu and OpenAssistant chunks, trains pretraining followed by SFT, creates `inference.jsonl`, writes `mastery_prompt.md`, and stops. It never trains the next submodule in the same initial invocation.

## Main output files

| Path | Contents |
|---|---|
| `runs/<run-id>/run_config.json` | Exact invocation configuration and claim boundary. |
| `runs/<run-id>/run_summary.json` | Session completion or wait state. |
| `CURRICULUM_ROOT/state.env` | Durable curriculum pointer and lineage path. |
| `CURRICULUM_ROOT/data/<session-id>/manifest.json` | Dataset revisions, row ranges, filters, selected IDs, and output files. |
| `CURRICULUM_ROOT/data/<session-id>/source_digest.txt` | SHA-256 digests for source manifest and split text files. |
| `CURRICULUM_ROOT/sessions/<session-id>/session_report.json` | Before/after metrics, lineage, and checkpoint hashes. |
| `CURRICULUM_ROOT/sessions/<session-id>/pretrain_checkpoint.bin` | Intermediate native checkpoint after the pretraining phase. |
| `CURRICULUM_ROOT/sessions/<session-id>/checkpoint.bin` | Immutable final checkpoint for the session. |
| `CURRICULUM_ROOT/sessions/<session-id>/mastery_prompt.md` | Human competency test instructions and validation schema. |
| `CURRICULUM_ROOT/validation/<session-id>.json` | User-supplied mastery decision. |
| `CURRICULUM_ROOT/sessions/<module-1-session-id>/inference.jsonl` | Native deterministic continuations and symbol/boundary diagnostics for human review. |
| `data/curriculum/module-1/prompts/<submodule>.txt` | Versioned unseen prompt packet for the current Module 1 submodule. |
| `CURRICULUM_ROOT/retry_required.md` | First-failure controlled-retry explanation. |
| `CURRICULUM_ROOT/architecture_diagnosis_required.md` | Terminal two-failure diagnosis gate. |

## Claim boundary

This workflow is a disciplined training and evaluation protocol for a focused-English research engine. A human `PASS` means only that the declared competency was judged demonstrated on the supplied unseen prompts under the recorded session contract. It does not establish broad language competence, factual reliability, human-speaker equivalence, production readiness, safety approval, or general intelligence.

## References

[1]: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu "HuggingFaceFW/fineweb-edu dataset card"
[2]: https://huggingface.co/datasets/OpenAssistant/oasst1 "OpenAssistant/oasst1 dataset card"
[3]: https://huggingface.co/docs/dataset-viewer/en/rows "Hugging Face Dataset Viewer rows API"
