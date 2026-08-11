# Stage 5 Expanded Gate Contract — Language and Code Scaling

**Project:** CCT-ASE  
**Stage:** 5 — Language and Code Scaling  
**Implementation:** Native C++20 only  
**Predecessor:** Stage 4 — Persistent Verifiable Memory  
**Transition:** Stage 6 only after explicit user approval

## Purpose and claim boundary

Stage 5 is a bounded, reproducible language-and-code scaling experiment. It uses a small public-domain language fixture and a repository-owned, MIT-licensed C++ fixture rather than claiming large-scale language-model training. The objective is an honest quality–compute–memory comparison across dense attention, GRU, diagonal SSM, CCT-ASE without memory, and CCT-ASE with frozen exact memory.

A passing gate demonstrates **small-scale next-token and code-token learning with reproducible data provenance, checkpoint replay, long-context diagnostics, sandboxed code checks, and memory attribution**. It does not demonstrate broad language competence, repository-level software engineering, or superintelligence.

## Data and provenance contract

The final manifest is `data/stage-5/manifests/stage5_manifest.txt`. Every fixture entry contains source ID, license boundary, source URL or repository commit, transformation, split, byte count, and SHA-256. The language fixture consists of official Project Gutenberg text endpoints for ebooks 1342 and 11, with the licensing boundary documented in `Stages/05_Data_Source_Findings.md`. The code fixture consists of native C++ files from the CCT repository at the release commit and is governed by the repository MIT license.

Evaluation canaries are generated into evaluator-only structures and are not included in training files. The gate checks manifest existence, unique hashes, immutable split names, and nonempty provenance fields. No corpus text is executed.

## Native model contract

The public API is in `cpp/include/cct/scaling.hpp` and implemented in `cpp/src/scaling.cpp`.

| Component | Contract |
|---|---|
| `Stage5Vocabulary` | Versioned byte-fallback vocabulary with deterministic encode/decode and unknown-byte behavior |
| `Stage5Dataset` | Manifest-addressed token streams with train/validation/test/canary splits and deterministic cursor order |
| `Stage5LanguageModel` | Native C++ sequence model with batched forward, recurrent step, SGD training, cross-entropy metrics, and checkpoint persistence |
| `Stage5Trainer` | Deterministic optimizer, clipping, schedule, epoch/cursor state, and resume-equivalence report |
| `Stage5MemoryAugmentor` | Frozen Stage 4 exact-memory lookup with retrieved IDs counted separately from parametric tokens |
| `Stage5CodeSandbox` | Static bounded code checks and explicit no-execution-on-host policy for generated text |

The reference scale is deliberately small. Mixed precision and distributed training are not claimed; those are later optimization paths. The trainer logs loss, token accuracy, tokens processed, wall time, parameter count, decode latency, state memory, and retrieval latency separately.

## Matched baseline protocol

The gate compares five configurations under the same token budget and deterministic split:

1. Dense causal attention baseline.
2. GRU baseline.
3. Diagonal state-space baseline.
4. CCT-ASE sequence core without memory.
5. CCT-ASE sequence core with frozen exact memory retrieval.

The comparison reports loss before and after training, validation loss, token accuracy, parameter count, state memory, forward/decode latency, and memory retrieval overhead. No universal superiority claim is made. A material advantage may be efficiency, long-context retention, or evidence attribution rather than lowest loss.

## Declared thresholds

Thresholds are frozen before the final release run and written to `artifacts/stage-5/cpp-gate/metrics.json`.

| Check | Pass condition |
|---|---|
| Data audit | Manifest entries are unique, provenance-complete, split-stable, and hash-addressed |
| Vocabulary | Encode/decode round-trip is exact for all fixture bytes, including unknown bytes |
| Loader replay | Same manifest and seed produce identical token order and cursor resume |
| Trainer correctness | Checkpoint resume reproduces next batch, loss, and parameter trajectory within `1e-12` on the micro-run |
| Language learning | Every trained comparator reduces validation cross-entropy by at least `5%` from its deterministic initial value |
| Code learning | CCT-ASE code-token accuracy exceeds its initial baseline and generated snippets satisfy static syntax/brace checks |
| Long context | Accuracy remains finite beyond training length and chunked/recurrent paths agree within declared tolerance |
| Memory value | Memory mode returns provenance IDs, improves canary evidence recall over no-memory, and reports retrieval separately |
| Code safety | No generated text is executed by the host; sandbox check records policy-safe non-execution and static diagnostics |
| Efficiency | Raw token throughput, decode latency, state memory, and quality/latency table are reported for all five configurations |
| Contamination | Evaluation canaries have no exact overlap with training token windows and manifest split identities remain immutable |
| Robustness | Duplicated passages, noisy metadata, and changed document order produce finite metrics and no policy/state corruption |
| Documentation | Model card records data, architecture, budgets, limitations, known failures, and inferior baseline cases |

The stage passes only when all mandatory checks pass and the release candidate is clean. The suite is intentionally a **small-scale native benchmark**, not a claim of state-of-the-art language or code modeling.

## Gate artifacts

The executable is `cct_stage5_gate --output artifacts/stage-5/cpp-gate`. It writes `gate.json`, `checks.json`, `metrics.json`, `manifest_audit.json`, `model_card.md`, `visible_eval.json`, and `evaluator_truth.json`. The evaluator-only file contains labels and canaries but no model-visible answer leakage.

## Transition package

The Stage 5 transition package contains the immutable data manifest and hashes, vocabulary contract, trainer/checkpoint implementation, all five comparator reports, memory attribution report, long-context and code-safety results, contamination audit, model card, known-failure list, and final commit SHA. Passing authorizes only **Stage 6 preparation after explicit user approval**.

## Non-goals

This stage does not implement unrestricted web data collection, autonomous tools, self-modification, distributed training, unrestricted code execution, open-ended agents, or superintelligence. The repository remains a research prototype with a declared small-scale evaluation boundary.
