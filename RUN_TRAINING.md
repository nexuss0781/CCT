# One-Command Native CCT Training and Evaluation

The repository root `run.sh` is the canonical one-command workflow for the next Level 1 training run. It is a native C++20 path: it verifies or installs the Ubuntu build dependencies, downloads and prepares the pinned real Track 1 sources, builds the CCT executables, trains the Track 1 CCT recurrence through pretraining and supervised fine-tuning, evaluates the held-out and frozen splits, qualifies CCT against GRU, diagonal SSM, and dense causal attention under a matched contract, runs the independent gates, and writes a durable run bundle.

> **Run command:**
>
> ```bash
> cd CCT
> bash run.sh
> ```

The script requires Internet access for source acquisition. If a dependency is missing, it attempts non-interactive Ubuntu installation with `apt-get`; this requires root or passwordless `sudo`. No Python runtime is used by the training or evaluation path. The native requirements are C++20, CMake, `pkg-config`, FFTW3 development files, `curl`, `zip`, and `unzip`.

## Default training contract

| Setting | Default |
|---|---:|
| WikiText pretraining token cap | 2,000,000 |
| Track 1 pretraining steps | 10,000 |
| Track 1 SFT steps | 2,000 |
| Context length | 128 |
| Embedding and hidden width | 16 / 16 |
| Seed | 1701 |
| Architecture qualification steps | 10,000 |
| Architecture qualification batch | 8 |
| Architecture train/evaluation caps | 5,000 / 128 sequences |
| Architecture vocabulary mode | Compact |
| Full CTest suite | Enabled |

The default workflow is intentionally a real-data run rather than a smoke test. Training remains bounded and reproducible; it is not a claim of broad language competence, human-speaker equivalence, production readiness, or general intelligence.

## Configuration overrides

The workflow accepts environment variables so the same script can be used for a bounded validation or a larger run without editing source code.

| Variable | Purpose |
|---|---|
| `RUN_ID` | Names the output directory below `runs/`. |
| `JOBS` | Controls CMake build parallelism; default is `2`. |
| `PRETRAIN_TOKEN_CAP` | Changes the real WikiText preparation cap. |
| `SFT_EXAMPLES`, `SFT_EVAL_EXAMPLES` | Changes governed SQuAD training/evaluation selection sizes. |
| `PRETRAIN_STEPS`, `SFT_STEPS` | Changes Track 1 CCT training budgets. |
| `CONTEXT_LENGTH`, `EMBEDDING_DIM`, `HIDDEN_DIM` | Changes Track 1 and matched qualification model dimensions. |
| `ARCHITECTURE_STEPS`, `ARCHITECTURE_BATCH` | Changes the matched four-architecture qualification budget. |
| `ARCHITECTURE_TRAIN_SEQUENCES`, `ARCHITECTURE_EVAL_SEQUENCES` | Changes qualification sequence caps. |
| `SKIP_DATA_PREPARATION=1` and `PREP_DIR=...` | Reuses an already prepared Track 1 directory. |
| `RUN_FULL_CTEST=0` | Skips the complete 44-test suite when only focused training validation is desired. |
| `INSTALL_DEPENDENCIES=0` | Disables automatic package installation and fails if a dependency is missing. |
| `SMOKE=1` | Runs bounded orchestration validation; expected architecture coherence failure is recorded as `EXPECTED_SMOKE_FAIL` and is not a quality result. |

For example, a bounded local orchestration check using already prepared real data is:

```bash
SMOKE=1 \
SKIP_DATA_PREPARATION=1 \
PREP_DIR="$PWD/artifacts/track1/real-release" \
RUN_FULL_CTEST=0 \
bash run.sh
```

## Output bundle

Each run creates `runs/<run-id>/`, and `runs/latest` points to the most recent run. The most important files are shown below.

| File | Contents |
|---|---|
| `run_summary.json` | Final status and paths to all key outputs. |
| `run_config.json` | Exact configuration and claim boundary. |
| `track1/` | Prepared manifests, source attestations, evaluation contract, and data splits. |
| `training/training_report.json` | Track 1 pretraining/SFT metrics and checkpoint lineage. |
| `training/pretrain_checkpoint.bin` | Native CCT pretraining checkpoint. |
| `training/sft_checkpoint.bin` | Native CCT SFT checkpoint. |
| `track1-gate/` | Independent Track 1 provenance, isolation, replay, and training gate outputs. |
| `architecture-qualification/report.json` | Matched CCT/GRU/SSM/attention results and generation diagnostics. |
| `architecture-qualification/gate/checks.json` | Machine-readable 10-check architecture gate. |
| `ctest.log` | Complete native CTest output when enabled. |
| `artifact_sha256.txt` | SHA-256 identities for the principal run artifacts. |

The selected production decoding policy is deterministic no-repeat decoding. Greedy continuations remain recorded as a diagnostic baseline; the workflow does not relabel them as human-like fluency.
