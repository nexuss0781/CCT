# CCT Level 1 Todo

## 1. Purpose and execution rule

This file is the actionable execution companion to [`SPEC/Goal.md`](Goal.md). It contains no new capability claims. Every task must trace to a Level 1 objective, implementation task, gate-evidence requirement, failure boundary, artifact requirement, or transition condition in the goal specification.

A checkbox is not evidence by itself. A task may be marked complete only when the implementation exists in the repository, a realistic test or evaluator exercises it, the relevant failure path is covered, the result is recorded in an identity-linked artifact, and the stage gate is rerun from the resulting commit.

## 2. Status legend

| Status | Meaning |
|---|---|
| `[ ]` | Pending implementation, verification, evidence, or transition work. |
| `[x]` | Supported by recorded evidence for the declared contract. It still becomes `REVALIDATE` after a relevant change. |
| `REVALIDATE` | A historical implementation or gate record exists, but a fresh replay or artifact reconciliation is required before release use. |
| `[!]` | Failed or invalidated. Stop at the boundary, preserve the failure, remediate, and rerun. |
| `[?]` | Blocked by an explicitly optional dependency. Mandatory Level 1 gates remain closed. |

## 2.1 Completed remediation register

The former review backlog has been consolidated here. The remediation register is evidence-backed by the native C++20 implementation, realistic failure-path tests, identity-linked artifacts, and the pushed commit [`a3fc4d9`](https://github.com/nexuss0781/CCT/commit/a3fc4d9). It is not a claim that the later language-teacher stages are complete.

| Remediation area | Status | Evidence and contract boundary |
|---|---|---|
| Training safety and checkpoint identity | `[x]` | Transactional optimizer publication, validation cadence/timing, atomic checkpoints, and canonical training-contract digest. |
| Causal and field numerical behavior | `[x]` | Strict temporal policy, bounded QR ridge solver, conservative stability domain, explicit operator semantics, and finite-difference-checked rollout gradients. |
| Corpus governance | `[x]` | UTF-8-safe truncation, deterministic normalization, split-aware indexed contamination checks, bounded parsing, and heuristic-only policy candidates. |
| Public API and synchronization | `[x]` | Ownership, borrowed-reference lifetime, synchronization, and failure contracts documented in public headers. |
| Serialization hardening | `[x]` | Byte/count/dimension/string/parameter budgets and fail-closed version handling across supported serializers. |
| Retrieval baseline | `[x]` | Memory V3 external-vector identity, indexed lexical retrieval, linear correctness oracle, knowledge posting index, and provider identity. Semantic embedding quality is not claimed. |
| Track 1 evaluation scope | `[x]` | Durable artifacts distinguish target-token prediction from answer-quality evaluation; EM/F1 require a separate constrained evaluator. |
| Real application evidence | `[x]` | Durable Stage 16 checkpoint/tokenizer black-box release activation and real inference request. |
| Build and sanitizer validation | `[x]` | 39/39 strict Release tests, 39/39 expanded-warning tests, 20/20 ASan/UBSan unit tests, and 19/19 sanitizer gates with a bounded Stage 5 instrumentation workload. |

The next executable Level 1 move is **L1-6 fresh release-bound Track 1 training**. It must produce durable pretraining and supervised checkpoints from the release checkout before L1-7 instruction adaptation begins.

## 3. Global Level 1 controls

### 3.1 Program controls

- [x] Keep CCT Level 1 implementation native C++20, with native CUDA only behind an explicit accelerator gate.
- [x] Maintain one source of truth for model, parameter, state, tokenizer, data, checkpoint, evaluator, and artifact identities.
- [ ] Create a Level 1 configuration schema containing compiler, build type, model dimensions, tokenizer hash, dataset manifest hash, seed, optimizer, context, batch, evaluation slices, and resource limits.
- [ ] Create a Level 1 release-record schema containing commit, configuration, environment, data, checkpoint, evaluator, thresholds, known failures, and final status.
- [ ] Make every gate emit only `PASS`, `FAIL`, or `BLOCKED`.
- [ ] Reject missing mandatory evidence, unresolved discrepancies, non-finite metrics, contaminated splits, invalid checkpoints, and unsupported claims.
- [ ] Invalidate affected evidence after changes to source, configuration, data, tokenizer, environment, optimizer, checkpoint, or evaluation contract.
- [ ] Require a realistic normal-path test and a realistic failure-path test for every major capability.
- [ ] Require a baseline, ablation, or independent reference for each claimed CCT contribution.
- [ ] Preserve both machine-readable and human-readable reports.
- [ ] Keep side-effectful integration outside the Level 1 teacher interface unless a separate contract is approved.

### 3.2 Required evidence bundle for every Level 1 stage

- [ ] Stage implementation contract and frozen configuration.
- [ ] Native source and build record.
- [ ] Unit and integration test results.
- [ ] Formal gate result or deterministic evaluator result.
- [ ] Commit, seed, data/environment, software, hardware, and artifact manifest.
- [ ] Threshold report, limitations report, and known-failure record.
- [ ] Transition record naming the successor scope and required approval.
- [ ] Artifact hashes and publication classification.

### 3.3 Global completion check

- [ ] All mandatory tasks for L1-0 through L1-10 are complete.
- [ ] Every mandatory stage gate is `PASS`.
- [ ] Final model and tokenizer identities reproduce from the declared commit.
- [ ] Language, instruction, operation/API, teaching, safety, reliability, and resource evidence are complete.
- [ ] No unresolved mandatory failure remains.
- [ ] The final release report states exactly what was evaluated and what was not evaluated.

---

# Stage L1-0 — Contract and Reproducible Baseline

**Objective:** Establish the CCT Level 1 contract, clean build, deterministic execution, artifact protocol, and baseline measurement.

**Dependency:** None.

**Current status:** `[x] PASS` — clean-checkout replay completed from `dc590fadf51b16553f601df8c608c17d5dcf14c5`, with 39/39 native CTest tests, 39/39 expanded-warning tests, six L1-0 gate checks, a clean source/configuration tree, identity-linked artifacts, and an isolated missing-FFTW/PkgConfig configure failure.

## L1-0 implementation tasks

- [x] Define the supported compiler, CMake version, C++ standard, operating system, and FFTW dependency contract in CMake and the L1-0 environment record.
- [x] Define warning policy, warnings-as-errors policy, build types, and native CUDA policy.
- [x] Define native model, state, tokenizer, data, checkpoint, evaluator, and artifact identity fields across the configuration, manifest, checkpoint, and gate contracts.
- [x] Reproduce a clean native C++20 build from a fresh release-checkout directory at `dc590fadf51b16553f601df8c608c17d5dcf14c5`.
- [x] Implement deterministic fixed-seed baseline configuration and numerical replay.
- [x] Implement unit-test, integration-test, benchmark, and formal-gate entry points.
- [x] Define the standard L1-0 artifact tree for configuration, environment, tests, benchmark, manifest, gate, release record, and report.
- [x] Define PASS/FAIL gate status and nonzero exit behavior.
- [x] Add a clean-environment missing-FFTW/PkgConfig dependency failure replay; CMake exits nonzero with retained diagnostics.
- [x] Add malformed-configuration failure coverage.
- [x] Add non-finite-output failure coverage through the declared numerical solver validation matrix.
- [x] Add corrupted-artifact failure coverage through bounded fail-closed serializer tests.
- [x] Add deterministic-replay detection.

## L1-0 verification tasks

- [x] Build the final L1-0 commit from a fresh checkout with the declared native compiler.
- [x] Run baseline numerical replay twice with the same seed and configuration inside the L1-0 gate.
- [x] Run the fixed benchmark-schema workload and retain the configuration/commit/hardware/timestamp record.
- [x] Compare deterministic output identity within exact native CPU equality.
- [x] Confirm invalid configuration exits through a structured rejected check and retains diagnostics.
- [x] Confirm missing FFTW/CMake dependency exits nonzero in an isolated clean environment without fallback.
- [x] Confirm non-finite numerical behavior is rejected by the native numerical validation matrix.
- [x] Confirm final clean-checkout artifacts and gate envelope identify `dc590fadf51b16553f601df8c608c17d5dcf14c5` and configuration hash `8c1a11faf7fdc8d2827f333b79aa9d470fbdf97091feb303a0ebb6527e5e6fd3`.

## L1-0 artifacts

- [x] `config.json`
- [x] `environment.json`
- [x] `tests.json`
- [x] `benchmark_record.json`
- [x] `manifest.json`
- [x] `gate.json`
- [x] `release_record.json`
- [x] `report.md`
- [x] `gate_envelope.json`

## L1-0 gate and transition

- [x] Run the formal L1-0 gate from a fresh release-checkout commit.
- [x] Confirm all mandatory checks are `PASS` with a clean-tree envelope.
- [x] Record the isolated missing-dependency replay result; no mandatory dependency was treated as optional.
- [ ] Record transition approval for L1-1 after explicit user approval.

**Command template:**

```bash
cmake -S cpp -B build-cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpp --parallel 2
ctest --test-dir build-cpp --output-on-failure
```

**Transition:** `PASS` authorizes L1-1 numerical implementation.

---

# Stage L1-1 — Differentiable Numerical Substrate

**Objective:** Implement finite, stable, independently checked numerical operators required by CCT learning.

**Dependency:** L1-0 `PASS`.

**Current status:** `[x] PASS` for the declared native scalar-field numerical substrate. The fresh gate covers independent spectral/reference agreement, manufactured accuracy, convergence, energy stability, gradients, boundaries, deterministic serialization, explicit float64 policy, non-finite rejection, source causality, invalid-shape/overflow behavior, and fail-closed schema parsing. The gate artifact is regenerated from the milestone source commit and remains approval-gated for L1-2.

## L1-1 implementation tasks

- [x] Define native scalar-field storage, flattened shape/layout, float64 dtype, ownership, indexing, and lifetime contracts.
- [x] Implement spectral and independently discretized finite-difference forward operators, source propagation, boundaries, energy, and operator loss.
- [x] Implement stable loss normalization with non-empty targets and positive finite mask mass.
- [x] Implement bounded local potential and declared spectral primitives used by the field substrate.
- [x] Implement analytic one-step gradients and full temporal rollout gradients for the declared source and potential parameter groups.
- [x] Preserve explicit source/potential vector parameter identity and deterministic gradient accumulation across repeated native calls.
- [x] Implement deterministic initialize, step, rollout, serialization, and replay interfaces.
- [x] Add finite, range, stability-domain, and output checks.
- [x] Add overflow-safe shape arithmetic and finite loss/energy checks.
- [x] Add explicit NaN and infinity rejection for fields, sources, targets, masks, parameters, state, and configuration numbers.
- [x] Add invalid-shape and schema-version checks.
- [x] Add input-state mutation and source-causality checks.
- [x] Add corrupted-state and unsupported-precision rejection checks.

The declared L1-1 scope is the native scalar-field substrate; sequence-state batching, tokenizer representation, optimizer publication, and language-model adaptation are successor-stage contracts rather than unverified claims of this numerical gate.

## L1-1 verification tasks

- [x] Compare spectral forward results against the independently implemented finite-difference reference and manufactured solutions.
- [x] Compare analytic one-step and temporal rollout gradients with centered finite differences.
- [x] Run the complete declared scalar-field matrix across one- and two-dimensional shapes, integrators, ranges, and all supported boundaries.
- [x] Repeat the native matrix deterministically under fixed configuration and exact repeated-call checks.
- [x] Verify finite outputs, diagnostics, losses, gradients, and serialized precision identity.
- [x] Verify invalid shapes and unsupported schema/precision requests fail closed.
- [x] Verify input-state mutation and source-causality contracts.
- [x] Verify overflow-safe shape arithmetic, non-finite configuration/input/state rejection, and corrupted-state failure paths.

## L1-1 artifacts

- [x] Operator contract in `Stages/01_Numerical_Engine.md` and the public native field header.
- [x] Gradient-check report in `artifacts/stage-1/cpp-gate/checks.json` and `report.md`.
- [x] Numerical reference comparison in the same gate bundle.
- [x] Boundary and failure report, including precision, non-finite, causality, and schema checks.
- [x] Resource and dtype report with float64 identity and measured subquadratic scaling.
- [x] L1-1 gate record under `artifacts/stage-1/cpp-gate/`.
- [x] L1-1 transition record encoded in the gate/report as `PASS` with approval required for L1-2.

## L1-1 gate and transition

- [x] Freeze derivative tolerances before the final run.
- [x] Run the formal L1-1 gate from the strict native build.
- [x] Record every mandatory operator check; the final gate contains no failed or skipped check.
- [x] Keep claims bounded to the verified native scalar-field operator matrix.
- [ ] Record explicit user approval for transition to L1-2.

**Transition:** `PASS` authorizes L1-2 sequence-engine implementation.

---

# Stage L1-2 — Causal Sequence Engine

**Objective:** Implement efficient ordered sequence learning with recurrent state, causal boundaries, streaming equivalence, and bounded resources.

**Dependency:** L1-1 `PASS`.

**Current status:** `[x] PASS` for the declared native selective recurrent sequence contract. The fresh gate covers reference/scan/streaming equivalence, complex and segmented paths, selected and all-mode gradients, long-horizon stability, controlled algorithmic training and matched baselines, state-position/reset semantics, recurrent-state checkpoint resume, adversarial configured gate clamps, and realistic fail-closed inputs, updates, and checkpoint fixtures. Artifact publication remains approval-gated for L1-3.

## L1-2 implementation tasks

- [x] Define state initialization, active-position, and explicit reset-epoch semantics.
- [x] Define causal update order: only active events advance the supplied state; masked events preserve state and position.
- [x] Implement typed recurrent state with real/optional-imaginary hidden vectors, previous input, position, and reset epoch.
- [x] Implement state ownership, V3 serialization, versioning, bounded loading, and optional recurrent-state recovery.
- [x] Implement full-sequence reference execution.
- [x] Implement chunked execution with exact carried-state equivalence.
- [x] Implement one-event streaming execution with exact carried-state equivalence.
- [x] Implement incremental state reuse, suffix resume, and explicit reset with audited epochs.
- [x] Bound model dimensions, matrix arithmetic, checkpoint vector budgets, and constant decode-state storage.
- [x] Add invalid checkpoint-version, truncated, oversized, non-finite, mismatched-size, invalid-boolean, and trailing-data failures.
- [x] Add wrong-shape state, input, target, and mask failures.
- [x] Add non-binary masks and out-of-order reset-request failures.
- [x] Add unexpected reset position and reset-epoch-overflow failures.
- [x] Add non-finite recurrence input/state/parameter/gradient/update failures with atomic parameter publication.

## L1-2 verification tasks

- [x] Compare full-sequence and streaming outputs within the frozen `1e-12` tolerance.
- [x] Compare full-sequence and chunked outputs with carried-state equality.
- [x] Verify model reload, recurrent-state suffix resume, explicit reset, and deterministic replay.
- [x] Measure forward scaling, decode state memory, matched-baseline state memory, parameters, and timing.
- [x] Exercise 16,384-event long-horizon, segmented-mask, adversarial gate-clamp, and reset/resume fixtures.
- [x] Verify invalid state transitions, masks, configuration, checkpoint, parameter, and update paths fail closed.
- [x] Verify no future event is consumed: recurrence state advances only from the current active input and stored previous active input.

## L1-2 artifacts

- [x] State contract in `cpp/include/cct/sequence.hpp` and `Stages/02_Sequence_Core.md`.
- [x] Full-vs-streaming/scan/chunked equivalence report in `artifacts/stage-2/cpp-gate/checks.json`.
- [x] Resource and matched-baseline profile in the same native gate bundle.
- [x] State recovery report covering V3 recurrent-state suffix resume and normalization checkpoint recovery.
- [x] Failure-path report covering non-binary masks, state/reset ordering, non-finite parameters and updates, and corrupt checkpoints.
- [x] L1-2 gate record under `artifacts/stage-2/cpp-gate/`.
- [x] L1-2 transition record encoded as `PASS` with user approval required for L1-3.

## L1-2 gate and transition

- [x] Freeze equivalence, gradient, stability, resource, and algorithmic thresholds before the final run.
- [x] Run the formal native L1-2 gate.
- [x] Preserve the reference loop as the oracle for scan, chunked, streaming, masked, complex, normalized, and adversarial-gate paths.
- [ ] Record explicit user approval for transition to L1-3.

**Transition:** `PASS` authorizes L1-3 event and state discipline.

---

# Stage L1-3 — Event, Causality, and State Discipline

**Objective:** Process ordered events and internal state without leakage, ambiguity, or silent provenance loss.

**Dependency:** L1-2 `PASS`.

**Current status:** `[x] PASS` for the declared native event, graph, causality, leakage, intervention, counterfactual, robustness, and abstention contract. The fresh gate covers nine mandatory checks plus evaluator-truth separation, independent graph corruption, strict metadata/query failure closure, and transactional learner-fit preservation. Artifact publication remains approval-gated for L1-4.

## L1-3 implementation tasks

- [x] Define stable nonzero event identity with duplicate rejection and deterministic serialization.
- [x] Define event ordering, source/provenance metadata, timestamps, intervention modes, uncertainty, and causal relationships.
- [x] Define dependency structure, explicit unresolved-parent policy, temporal tie policy, and DAG cycle policy.
- [x] Implement future-information leakage checks in the encoder with strict or same-timestamp policy.
- [x] Implement evaluator-isolation checks with separate visible-input and evaluator-truth artifacts.
- [x] Implement observed, do-intervened, and counterfactual records with changed-input validation.
- [x] Implement paired-world fixtures sharing structural noise for counterfactual evaluation.
- [x] Implement omission/missing-parent, reorder, irrelevant-change, contradiction, corrupted-graph, and perturbation fixtures.
- [x] Preserve provenance, uncertainty, event identity, and intervention mode through snapshot and encoder paths.
- [x] Implement finite confidence/evidence records and abstention for incomplete or conflicting graphs.

## L1-3 verification tasks

- [x] Verify stable event identity, duplicate detection, deterministic export, and snapshot round trip.
- [x] Verify source/provenance fields, timestamp policy, insertion-order determinism, and dataset fingerprints.
- [x] Run the leakage audit with future-parent counts, masked/unmasked encodings, and graph-conditioned loop/scan equivalence.
- [x] Verify missing-parent, unresolved-parent, cycle, self-parent, enum, coordinate, uncertainty, and intervention validation.
- [x] Verify held-out intervention outcomes, effect direction, and intervention-aware versus observation-only error.
- [x] Verify paired counterfactual reproducibility with shared structural noise and irrelevant-change consistency.
- [x] Verify measurable degradation under corrupted graph hypotheses and finite behavior under robustness perturbation.
- [x] Verify incomplete and conflicting graph abstention with zero confidence.
- [x] Verify unsupported, malformed, non-finite, duplicate, and ambiguous event/learner inputs fail closed without corrupting a fitted model.

## L1-3 artifacts

- [x] Event schema and graph-store contract in `cpp/include/cct/causal.hpp` and `Stages/03_Causal_Event_Learning.md`.
- [x] Causal dependency and structural-recovery report in `artifacts/stage-3/cpp-gate/checks.json`.
- [x] Leakage audit and visible/evaluator-only separation in `visible_input.json` and `evaluator_truth.json`.
- [x] Intervention-effect report with observation-only control in the gate bundle.
- [x] Counterfactual consistency report with paired-world error and irrelevant-change metric.
- [x] Robustness report with corrupted-graph degradation and finite outputs.
- [x] Abstention report for incomplete and conflicting graph metadata.
- [x] L1-3 gate and transition record under `artifacts/stage-3/cpp-gate/`.

## L1-3 gate and transition

- [x] Freeze event, provenance, uncertainty, intervention, temporal-policy, and visible/evaluator-only schemas.
- [x] Run the formal native L1-3 gate with nine mandatory checks.
- [x] Record the declared limitations: evidence is confined to the tested synthetic structural-equation distributions and does not establish general causal understanding or real-world causal discovery.
- [ ] Record explicit user approval for transition to L1-4.

**Transition:** `PASS` authorizes L1-4 representation locking.

---

# Stage L1-4 — Tokenizer and Representation Lock

**Objective:** Freeze one deterministic representation contract for Level 1 data, training, checkpoints, and evaluation.

**Dependency:** L1-3 `PASS`.

**Current status:** `[x] PASS` for the declared native tokenizer and representation-lock contract. The fresh gate measures byte, subword, and hybrid candidates on the same governed fixture and passes twelve mandatory checks, including strict snapshot-schema and packed/padded metadata failure closure. The selected immutable hybrid snapshot is identity-linked for downstream data, trainer checkpoints, and inference loading. Artifact publication remains approval-gated for L1-5.

## L1-4 implementation tasks

- [x] Implement byte tokenizer candidate with exhaustive byte fallback.
- [x] Implement deterministic frequency-ranked subword tokenizer candidate.
- [x] Implement deterministic longest-first hybrid tokenizer candidate with byte fallback.
- [x] Compare all candidates under fixed governed data, record order, normalization, and batch settings.
- [x] Freeze selected vocabulary, ordering, canonical snapshot, and SHA-256 hash.
- [x] Implement byte-safe source-to-token half-open offsets and record provenance.
- [x] Implement explicit `preserve-bytes-v1` malformed-input policy.
- [x] Implement byte fallback for all `256` byte values, including NUL and malformed UTF-8 fragments.
- [x] Implement packed causal batches with independent sequence boundaries.
- [x] Implement padded causal batches with inactive pad positions.
- [x] Implement padding, boundary, control-category, tokenizer-version, source-span, and record masks.
- [x] Measure token efficiency, source-byte ratio, estimated/resident memory, throughput, and exact round-trip behavior.

## L1-4 verification tasks

- [x] Verify snapshot identity, exact canonical loading, SHA-256 matching, duplicate-singleton/trailing-data rejection, and immutable release record.
- [x] Verify Unicode offsets, malformed UTF-8, NUL, separators, and byte-preserving normalization.
- [x] Verify universal byte fallback coverage and exact decode.
- [x] Verify packed and padded masks, zero cross-boundary loss, exact loss checksums, and tampered metadata rejection.
- [x] Verify deterministic tokenization, source spans, literal-control separation, record ordering, and byte-exact round trips.
- [x] Compare candidate token efficiency and resource behavior under the same declared fixture.
- [x] Verify tokenizer version and snapshot hash are carried in batch data and checked by native training, checkpoint, and inference consumers.

## L1-4 artifacts

- [x] Candidate comparison report in `artifacts/stage-10/cpp-gate/candidate_comparison.json`.
- [x] Frozen tokenizer snapshot in `tokenizer_snapshot.bin` and `tokenizer_snapshot.json`.
- [x] Vocabulary hash in the snapshot manifest and release record.
- [x] Offset and fallback test report in `checks.json`.
- [x] Batch, loss-mask, and control/boundary metadata report in `batch_report.json` and `checks.json`.
- [x] Efficiency report in `metrics.json` and candidate comparison.
- [x] L1-4 gate and approval-gated transition record under `artifacts/stage-10/cpp-gate/`.

## L1-4 gate and transition

- [x] Freeze tokenizer and vocabulary before downstream data preparation.
- [x] Run the formal native L1-4 gate with twelve mandatory checks.
- [x] Record that tokenizer/version/snapshot changes invalidate downstream batch, training, checkpoint, and inference identities unless a new migration is formally verified.
- [ ] Record explicit user approval for transition to L1-5.

**Transition:** `PASS` authorizes L1-5 native language training.

---

# Stage L1-5 — Native Trainable Language Core

**Objective:** Train CCT through a native next-token objective with analytic gradients, optimizer safety, checkpoint recovery, and held-out evaluation.

**Dependency:** L1-4 `PASS`.

**Current status:** `[x] PASS` for the declared native trainable language-core contract. The current native gate contains thirteen mandatory checks covering the categorical objective, analytic gradients, optimizer and schedule, stability/failure closure, three-seed held-out improvement, no-training capability control, repeated-corpus overfit, matched controls, checkpoint resume, cursor/context/budget discipline, contamination, corruption/identity rejection, reproducibility, and artifact integrity. Artifact publication remains approval-gated for L1-6.

## L1-5 implementation tasks

- [x] Define next-token targets, context windows, final-position masks, and causal loss masks.
- [x] Define mean categorical cross-entropy, token counts, finite-objective rules, and stable log-sum-exp evaluation.
- [x] Implement analytic CCT selective-recurrence backpropagation through time.
- [x] Implement independent centered finite-difference gradient checks over embedding, recurrent, gate, and output parameters.
- [x] Implement optimizer, global clipping, warmup, linear decay, AdamW-equivalent moments, deterministic initialization, and atomic finite updates.
- [x] Implement model parameter and optimizer-state serialization with bounded dimensions and explicit model kind.
- [x] Implement atomic checkpoint save and strict checkpoint load.
- [x] Implement exact cursor, optimizer-step, scheduler, moment, and model resume recovery.
- [x] Implement tokenizer, data, configuration, context, split, and model identity validation.
- [x] Implement corruption, truncation, trailing-data, non-finite, wrong-identity, and incompatible-checkpoint rejection.
- [x] Implement matched dense-attention, GRU, and diagonal-SSM native reference baselines under equal declared budgets.
- [x] Measure cross-entropy, perplexity, token accuracy, throughput, state memory, parameter count, gradient norm, and checkpoint hashes.

## L1-5 verification tasks

- [x] Verify loss, gradients, parameters, optimizer moments, logits, and metrics remain finite.
- [x] Verify analytic-gradient agreement at the declared `1e-4` relative threshold.
- [x] Run three fixed-seed CCT configurations and an exact repeated same-seed reproducibility run.
- [x] Compare initial and final held-out validation behavior and require every seed to improve.
- [x] Require a separate no-training capability control and retain failure if improvement is absent.
- [x] Verify exact checkpoint load and resume equivalence at cursors `0`, `1`, and `3`.
- [x] Verify wrong tokenizer, dataset, context, optimizer budget, model kind, and model allocation identities fail closed.
- [x] Verify corrupted, truncated, malformed, trailing-data, non-finite, and incompatible checkpoints fail closed.
- [x] Compare dense attention, GRU, and diagonal SSM controls under equal declared data, context, seed, and optimizer budgets.

## L1-5 artifacts

- [x] Objective and binary-mask contract in `Stages/11_Trainable_Native_NLP_Core.md` and the native trainer.
- [x] Gradient report in `artifacts/stage-11/cpp-gate/gradient_report.json`.
- [x] Optimizer configuration and schedule evidence in `checks.json`, `seed_comparison.json`, and `metrics.json`.
- [x] Baseline comparison in `baseline_comparison.json`.
- [x] Training and held-out validation report in `seed_comparison.json`, `metrics.json`, and `report.md`.
- [x] Checkpoint manifest and hashes in `checkpoint_report.json`, `release_record.json`, and `selected_checkpoint.bin`.
- [x] Resume, corruption, wrong-identity, and cursor report in `checkpoint_report.json` and `checks.json`.
- [x] L1-5 thirteen-check gate and approval-gated transition record under `artifacts/stage-11/cpp-gate/`.

## L1-5 gate and transition

- [x] Freeze optimizer, context, batch, seed, split, and baseline configurations in the gate and training contract hash.
- [x] Run the formal native L1-5 gate with thirteen mandatory checks.
- [x] Record finite metrics and fail-closed diagnostics; the final gate contains no failed configuration.
- [ ] Record explicit user approval for transition to L1-6.

**Transition:** `PASS` authorizes L1-6 governed language acquisition.

---

# Stage L1-6 — Track 1 Real Language Acquisition

**Objective:** Train CCT on a small governed real-data bundle and create reproducible language-learning evidence.

**Dependency:** L1-5 `PASS` and frozen tokenizer identity.

**Current status:** `REVALIDATE` — Track 1 preparation and bounded training are recorded as PASS for their declared contracts; a fresh replay is required after changes or in a new environment.

## L1-6 implementation tasks

### L1-6 source and preparation tasks

- [x] Pin WikiText-2 source revision, direct archive URL, license metadata, split members, and digest.
- [x] Pin SQuAD 2.0 source identity, GEM direct-file revision, upstream provenance, license metadata, and direct URLs.
- [x] Implement native resumable downloads, cache reuse, pacing, retry, and atomic extraction.
- [x] Implement fail-closed missing-cache and malformed-source behavior.
- [x] Parse WikiText raw archive members natively.
- [x] Parse GEM flat SQuAD files natively.
- [x] Decode JSON Unicode surrogate pairs correctly.
- [x] Convert SQuAD codepoint offsets to UTF-8 byte offsets.
- [x] Validate answer text against context.
- [x] Select deterministic balanced answerable and unanswerable records.
- [x] Isolate supervised training, selection evaluation, and frozen final-test IDs.
- [x] Generate source manifests, data manifests, digests, preparation reports, evaluation contracts, and release records.
- [x] Preserve fixture mode for local tests without weakening production validation.

### L1-6 training tasks

- [x] Implement the native CCT Track 1 training runner.
- [x] Consume prepared WikiText pretraining data.
- [x] Consume prepared SQuAD supervised data.
- [x] Apply `target-span-only-v1` masks.
- [x] Filter zero-loss prompt-only chunks.
- [x] Save a pretraining checkpoint.
- [x] Continue from the pretraining checkpoint for supervised tuning.
- [x] Save the supervised checkpoint with identity lineage.
- [x] Evaluate selection data and frozen final-test target-token metrics.
- [x] Record that answer-target next-token metrics are measured and exact-answer EM/F1 are not claimed.

## L1-6 verification and gate tasks

- [x] Run native Track 1 unit tests.
- [x] Run the formal Track 1 gate.
- [x] Run the full native suite when the declared toolchain is available.
- [x] Run a bounded real-source preparation.
- [x] Run complete governed preparation when the declared source cache and toolchain are available.
- [x] Verify no malformed rows and no train/evaluation/final overlap.
- [x] Verify deterministic manifests under the same seed and source.
- [x] Verify checkpoint reload under declared identity.
- [ ] Re-run the complete Track 1 preparation and training command from a fresh Level 1 release checkout.
- [ ] Preserve fresh non-temporary checkpoint paths in the release artifact bundle.

## L1-6 artifacts

- [x] `artifacts/track1/cpp-gate/checks.json`
- [x] `artifacts/track1/cpp-gate/source_manifest.json`
- [x] `artifacts/track1/cpp-gate/evaluation_contract.json`
- [x] `artifacts/track1/real-gem-smoke/preparation_report.json`
- [x] `artifacts/track1/real-training/training_report.json`
- [ ] Fresh release-bound preparation report.
- [ ] Fresh release-bound training report with durable checkpoint references.

## L1-6 gate and transition

- [x] Record Track 1 preparation `PASS` for the declared contract.
- [x] Record bounded native training `PASS` for the declared target-token contract.
- [ ] Run the fresh release-candidate Track 1 gate after any training or data change.
- [ ] Record transition approval for L1-7 instruction adaptation.

**Command sequence:**

```bash
cmake -S cpp -B build-cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpp --parallel 2
make track1-test track1-gate
./build-cpp/cct_track1_prepare \
  --output artifacts/track1 \
  --pretrain-token-cap 2000000 \
  --sft-examples 8000 \
  --sft-eval-examples 800 \
  --seed 1701
make track1-train
cat artifacts/track1/preparation_report.json
cat artifacts/track1/training/training_report.json
```

**Transition:** `PASS` authorizes L1-7 supervised instruction adaptation.

---

# Stage L1-7 — Supervised Instruction Adaptation

**Objective:** Teach CCT canonical instruction-to-target behavior with target-only supervision and held-out evaluation.

**Dependency:** L1-6 `PASS` with fresh release-bound evidence.

**Current status:** `[ ] PENDING` — the current repository has Track 1 supervised target training, but the full Level 1 instruction-adaptation contract is not yet complete.

## L1-7 implementation tasks

- [ ] Define instruction, context, target, metadata, eligibility, policy, and evaluator schemas.
- [ ] Define canonical instruction serialization and version it.
- [ ] Implement `target-span-only-v1` offsets and masks for all supported task forms.
- [ ] Reject malformed targets and zero-loss examples.
- [ ] Implement full-parameter supervised continuation from an identity-checked base checkpoint.
- [ ] Preserve base checkpoint immutability.
- [ ] Implement structured output validation.
- [ ] Implement citation and evidence fields where the task schema requires them.
- [ ] Implement missing-evidence abstention.
- [ ] Implement safe refusal and unsupported-request behavior.
- [ ] Implement deletion and lineage records for adapted data and checkpoints.

## L1-7 verification tasks

- [ ] Verify prompt tokens have zero supervised target contribution where required.
- [ ] Verify target offsets and masks across Unicode, empty-target, and malformed examples.
- [ ] Compare held-out instruction behavior against the untouched base.
- [ ] Verify structured outputs parse deterministically.
- [ ] Verify missing evidence causes bounded abstention.
- [ ] Verify unsafe and unsupported requests remain denied.
- [ ] Verify base immutability, adapter or checkpoint lineage, and deletion.
- [ ] Verify results across declared task slices and seeds.

## L1-7 artifacts

- [ ] Instruction schema and formatter version.
- [ ] Task manifest and split manifest.
- [ ] Mask and offset report.
- [ ] Base and adapted checkpoint manifests.
- [ ] Held-out task report.
- [ ] Structure, citation, abstention, and safety report.
- [ ] Deletion and lineage report.
- [ ] L1-7 gate and transition record.

## L1-7 gate and transition

- [ ] Freeze the instruction schema, formatter, mask policy, base identity, data split, and evaluator.
- [ ] Run the formal L1-7 gate.
- [ ] Confirm every failure class fails closed.
- [ ] Record transition approval for L1-8.

**Transition:** `PASS` authorizes L1-8 operation and API adaptation.

---

# Stage L1-8 — Operation and API Teacher Adaptation

**Objective:** Teach CCT declared internal operation and API contracts, including valid serialization, explanation, validation, correction, and safe rejection.

**Dependency:** L1-7 `PASS`.

**Current status:** `[ ] PENDING`.

## L1-8 implementation tasks

- [ ] Define versioned operation schemas.
- [ ] Define required and optional fields, types, bounds, defaults, and error classes.
- [ ] Define authorization class for every operation.
- [ ] Create governed demonstrations for valid calls.
- [ ] Create governed demonstrations for invalid calls, retries, missing evidence, refusal, and ambiguity.
- [ ] Fine-tune serialization and explanation behavior.
- [ ] Implement schema validation before operation acceptance.
- [ ] Implement unknown-operation and malformed-argument rejection.
- [ ] Implement unauthorized-operation rejection.
- [ ] Preserve operation-schema identity in training, checkpoint, and evaluation manifests.
- [ ] Keep external side effects disabled in the Level 1 teacher interface.

## L1-8 verification tasks

- [ ] Verify valid calls serialize and validate.
- [ ] Verify invalid calls fail with the correct error class.
- [ ] Verify unknown operations fail closed.
- [ ] Verify malformed arguments fail closed.
- [ ] Verify unauthorized calls fail closed.
- [ ] Verify ambiguous requests abstain or request clarification within the contract.
- [ ] Verify CCT explains required fields and identifies malformed calls.
- [ ] Verify every demonstration maps to a governed source record.
- [ ] Verify operation schema changes invalidate incompatible checkpoints.
- [ ] Verify no test can trigger an unapproved external side effect.

## L1-8 artifacts

- [ ] Operation schema registry.
- [ ] Operation demonstration manifest.
- [ ] Formatter and validator report.
- [ ] Error-class report.
- [ ] Authorization report.
- [ ] API checkpoint identity report.
- [ ] Side-effect isolation report.
- [ ] L1-8 gate and transition record.

## L1-8 gate and transition

- [ ] Freeze operation schemas and authorization classes.
- [ ] Run the formal L1-8 gate.
- [ ] Verify invalid and unauthorized operation coverage.
- [ ] Record transition approval for L1-9.

**Transition:** `PASS` authorizes L1-9 bounded teaching evaluation.

---

# Stage L1-9 — Bounded Teaching Behavior

**Objective:** Demonstrate that CCT can communicate, demonstrate, evaluate, correct, and abstain through a defined teacher interface.

**Dependency:** L1-8 `PASS`.

**Current status:** `[ ] PENDING`.

## L1-9 implementation tasks

- [ ] Define teaching episode identity and schema.
- [ ] Define task, demonstration, expected result, evidence, critique, correction, outcome, and evaluator fields.
- [ ] Implement a bounded teacher interaction harness.
- [ ] Ensure the harness cannot perform unapproved external side effects.
- [ ] Implement task explanation.
- [ ] Implement step decomposition.
- [ ] Implement demonstration generation within declared schemas.
- [ ] Implement result checking.
- [ ] Implement error identification.
- [ ] Implement evidence-linked corrective response.
- [ ] Implement abstention for missing evidence, unsupported schemas, and ambiguity.
- [ ] Log model version, instruction, output, evidence, evaluator, and decision for every episode.

## L1-9 verification tasks

- [ ] Replay teaching episodes deterministically.
- [ ] Verify demonstrations are valid for declared tasks.
- [ ] Inject errors and verify independent detection.
- [ ] Verify corrections cite relevant evidence or abstain.
- [ ] Verify unsupported teaching requests are rejected.
- [ ] Verify evaluator disagreement is recorded.
- [ ] Verify lesson identity and output lineage.
- [ ] Verify side-effect isolation.
- [ ] Separate language quality, schema correctness, evaluator quality, and unsupported claims in the report.

## L1-9 artifacts

- [ ] Teaching episode schema.
- [ ] Teacher harness configuration.
- [ ] Demonstration report.
- [ ] Error-injection report.
- [ ] Evidence and correction report.
- [ ] Abstention report.
- [ ] Episode replay log.
- [ ] L1-9 gate and transition record.

## L1-9 gate and transition

- [ ] Freeze teaching schemas, evaluator, task slices, and episode budgets.
- [ ] Run the formal L1-9 gate.
- [ ] Verify all unsupported and unsafe teaching requests fail closed.
- [ ] Record transition approval for L1-10.

**Transition:** `PASS` authorizes L1-10 local release review.

---

# Stage L1-10 — Local Teacher-Engine Release

**Objective:** Freeze and release a measured CCT Level 1 teacher engine for its declared local interface.

**Dependency:** L1-0 through L1-9 `PASS` with complete identity-linked evidence.

**Current status:** `[ ] PENDING`.

## L1-10 implementation tasks

- [ ] Freeze model, tokenizer, data, training, operation schemas, policy, evaluator, runtime, and release configuration.
- [ ] Save pretraining and adapted checkpoints with complete lineage.
- [ ] Verify checkpoint reload and exact resume.
- [ ] Verify corruption and configuration-mismatch rejection.
- [ ] Measure local memory, throughput, latency, checkpoint size, and load time.
- [ ] Measure resource exhaustion and bounded failure behavior.
- [ ] Produce model card and evaluation report.
- [ ] Produce limitations and support-boundary report.
- [ ] Produce release record and artifact manifest.
- [ ] Define rollback, retirement, deletion, and revalidation procedures.

## L1-10 verification tasks

- [ ] Verify all predecessor gates are `PASS`.
- [ ] Verify language, instruction, operation/API, teaching, safety, reliability, and resource reports are complete.
- [ ] Verify final checkpoint and tokenizer hashes.
- [ ] Verify reproducibility from the release commit.
- [ ] Verify no mandatory failure remains unresolved.
- [ ] Verify every claim is supported by an evaluation result.
- [ ] Verify release rollback and retirement behavior.
- [ ] Verify deletion and revalidation procedures.

## L1-10 artifacts

- [ ] Frozen release configuration.
- [ ] Final model checkpoint and hash.
- [ ] Final tokenizer snapshot and hash.
- [ ] Data and operation-schema manifests.
- [ ] Model card.
- [ ] Language evaluation report.
- [ ] Instruction evaluation report.
- [ ] Operation/API evaluation report.
- [ ] Teaching evaluation report.
- [ ] Safety and reliability report.
- [ ] Resource profile.
- [ ] Release record.
- [ ] Final L1-10 gate record.

## L1-10 gate and transition

- [ ] Run the final Level 1 release gate from the frozen release candidate.
- [ ] Confirm all mandatory checks are `PASS`.
- [ ] Record all known limitations and unmeasured capabilities.
- [ ] Confirm the local teacher interface is bounded and side-effect-safe.
- [ ] Record the Level 1 completion decision.
- [ ] Authorize only the separately specified downstream integration stage.

**Transition:** `PASS` completes CCT Level 1.

---

## 4. Level 1 metric checklist

- [ ] Numerical: finite outputs, gradient agreement, parameter finiteness, update stability, and deterministic replay.
- [ ] Language: cross-entropy, perplexity, token accuracy, held-out loss, continuation slices, and robustness slices.
- [ ] Instruction: target-token loss, task success, structure validity, abstention, and regression against base.
- [ ] Operation/API: schema-valid rate, invalid-call rejection, unknown-operation rejection, correction accuracy, and trace completeness.
- [ ] Teaching: demonstration validity, independent error detection, evidence-linked correction, abstention, and episode replay.
- [ ] Systems: memory, throughput, latency, checkpoint size, load time, recovery, and resource-bound failure.
- [ ] Reproducibility: manifest equality, tokenizer and data identity, checkpoint hash, seed replay, and environment record.
- [ ] Attach split, seed, configuration, data identity, reference, threshold, and artifact path to every metric.
- [ ] Keep answer-target next-token metrics separate from exact-answer EM/F1.
- [ ] Do not claim a metric for an unimplemented evaluator.

## 5. Failure and safety checklist

- [ ] Missing or changed source data fails closed.
- [ ] Malformed rows fail closed.
- [ ] Invalid Unicode answer offsets fail closed.
- [ ] Train/evaluation/final contamination fails closed.
- [ ] Unknown tokenizer or data identity fails closed.
- [ ] Incompatible checkpoint fails closed.
- [ ] Corrupted state fails closed.
- [ ] Non-finite values fail closed.
- [ ] Unsupported operation fails closed.
- [ ] Malformed operation arguments fail closed.
- [ ] Unauthorized operation fails closed.
- [ ] Missing teaching evidence causes abstention.
- [ ] Ambiguous teaching request causes abstention or bounded clarification.
- [ ] Resource exhaustion triggers a bounded failure.
- [ ] External side effects remain disabled unless a separate contract authorizes them.

## 6. Release and revalidation checklist

- [ ] Start the release run from a clean checkout.
- [ ] Record `git rev-parse HEAD`.
- [ ] Record compiler, CMake, CUDA, operating system, CPU/GPU, and dependency versions.
- [ ] Recompute configuration and environment hashes.
- [ ] Recompute source, tokenizer, data, operation-schema, checkpoint, and artifact hashes.
- [ ] Run all predecessor gates in order.
- [ ] Rerun affected failure-path tests after any change.
- [ ] Rerun baseline and ablation comparisons after model changes.
- [ ] Rerun resource profiles after context, batch, model, optimizer, or backend changes.
- [ ] Preserve old evidence and write a new release record rather than overwriting history.
- [ ] Review the final report for unsupported claims.
- [ ] Push the release commit after all required gates pass.

## 7. Current command references

### Native build and repository tests

```bash
cmake -S cpp -B build-cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpp --parallel 2
ctest --test-dir build-cpp --output-on-failure
```

### Track 1 preparation

```bash
./build-cpp/cct_track1_prepare \
  --output artifacts/track1 \
  --pretrain-token-cap 2000000 \
  --sft-examples 8000 \
  --sft-eval-examples 800 \
  --seed 1701
```

### Track 1 gates and training

```bash
make track1-test track1-gate
make track1-train
cat artifacts/track1/preparation_report.json
cat artifacts/track1/training/training_report.json
```

These commands are valid only when the declared native toolchain is installed and the current checkout contains the required source and configuration. A successful build or smoke run does not by itself complete any Level 1 transition.

## 8. Completion rule

`SPEC/Todo.md` is complete only when every required task from L1-0 through L1-10 is checked, every mandatory gate is `PASS`, every artifact is identity-linked and reviewable, all failure boundaries are exercised, every transition is recorded, and the final report states the exact measured Level 1 capability and limitation boundary.

## References

- [CCT Level 1 Goal Specification](Goal.md)
- [CCT architecture](../Architecture.md)
- [CCT internal goal map](../Goal.md)
- [CCT actionable todo](../Todo.md)
- [Track 1 operational guide](../artifacts/track1/README.md)
