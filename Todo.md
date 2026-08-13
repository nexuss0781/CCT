# CCT Todo

## Operating rule

This file is the actionable execution companion to [`Goal.md`](Goal.md). It does not introduce a second roadmap. Every task below exists to satisfy a CCT Goal objective, implementation breakdown, gate, evidence requirement, or transition condition.

A task is not complete because code exists. It is complete only when the implementation is exercised by a realistic test or harness, the failure path is covered, the evidence artifact is written, and the relevant gate is rerun from the resulting commit.

## Status legend

| Marker | Meaning |
|---|---|
| `[ ]` | Not complete. It remains an actionable task. |
| `[x]` | Complete in the repository and supported by recorded evidence. |
| `[!]` | Failed or invalidated. Stop at the boundary, remediate, and rerun. |
| `[?]` | Blocked by an explicitly declared optional dependency. The core gate remains closed. |
| `REVALIDATE` | Previously passed, but must be rerun if code, configuration, data, environment, or artifact identity changes. |

## Global execution checklist

- [x] Keep the implementation native C++20 unless a stage specification explicitly declares another native backend.
- [x] Define one canonical build, test, gate, benchmark, artifact, and release-record contract.
- [x] Pin repository commit, configuration, seeds, data or environment manifests, software versions, hardware context, and artifact hashes for every release-quality run.
- [x] Treat `PASS`, `FAIL`, and `BLOCKED` as the only valid gate statuses.
- [x] Stop at any failed correctness, reproducibility, provenance, safety, recovery, or operational criterion.
- [x] Require a realistic failure path for each major feature rather than only a happy-path smoke test.
- [x] Preserve a human-readable report together with machine-readable checks, manifests, logs, and benchmark records.
- [x] Rerun an invalidated gate whenever implementation, configuration, source, environment, or checkpoint identity changes.
- [x] Require a baseline or ablation for every claimed contribution of recurrence, spectral operation, causal metadata, workspace, memory, deliberation, or optimization.
- [x] Do not claim a capability beyond its declared evaluation contract.

## Current execution position

| Work item | Status | Evidence or next action |
|---|---|---|
| Stages 0–17 | `[x]` COMPLETE | Existing stage gates and release artifacts. Revalidate after any change. |
| Track 1 source preparation | `[x]` COMPLETE | `artifacts/track1/` reports, manifests, unit tests, and formal gate. |
| Track 1 native training handoff | `[x]` COMPLETE for the declared bounded contract | Native runner, checkpoints, and training report. Exact-answer EM/F1 remains outside the current contract. |
| Ongoing maintenance | `REVALIDATE` | Pull the release commit, rerun the declared gate chain, and preserve new evidence when changing CCT. |

---

# Stage 0 — CCT Todo: Reproducible Baseline

**Goal:** Create a clean native C++20 baseline that builds, tests, benchmarks, and replays from a known repository state.

**Status:** `[x] COMPLETE — REVALIDATE after changes.`

## Implementation tasks

- [x] Define supported compiler, CMake version, C++ standard, operating system, warning policy, and dependency contract.
- [x] Configure native C++20 targets with strict warnings and warnings-as-errors.
- [x] Implement deterministic unit and integration test entry points with explicit seeds and configuration.
- [x] Implement a stable benchmark schema containing workload, seed, elapsed time, throughput, memory, and environment.
- [x] Define canonical artifact directories, report schemas, status values, and release-record fields.
- [x] Add missing-dependency, malformed-configuration, non-finite-output, and nondeterministic-replay failures.

## Gate and evidence tasks

- [x] Run a clean checkout build.
- [x] Run all baseline tests and verify exit codes.
- [x] Run the benchmark twice under the same declared environment.
- [x] Compare repeated test and benchmark identities within declared tolerances.
- [x] Verify every artifact identifies the exact commit and configuration.
- [x] Write the Stage 0 gate record and human-readable report.

## Transition

- [x] Record `PASS` for Stage 0.
- [x] Record authorization for Stage 1 numerical implementation.
- [x] Confirm the record does not claim learning, language capability, or deployment.

**Evidence locations:** `artifacts/stage-0/`, Stage 0 gate executable, CTest records, and release report.

---

# Stage 1 — CCT Todo: Differentiable Numerical Engine

**Goal:** Implement a correct and numerically stable field/operator substrate with independently verified analytic gradients.

**Status:** `[x] COMPLETE — REVALIDATE after changes.`

## Implementation tasks

- [x] Define tensor or field storage, shape, layout, dtype, ownership, indexing, and lifetime contracts.
- [x] Implement forward numerical operators, stable reductions, normalization, activation, mixing, and required spectral or adaptive primitives.
- [x] Implement analytic gradients for every trainable operator used by the later sequence core.
- [x] Implement gradient accumulation, parameter identity, and deterministic update interfaces.
- [x] Add finite, range, overflow, underflow, aliasing, mutation, invalid-shape, and boundary checks.
- [x] Ensure corrupted or non-finite numerical state fails closed.

## Gate and evidence tasks

- [x] Compare forward results with an independent reference.
- [x] Compare analytic gradients with finite differences or an independent derivative implementation.
- [x] Run the full numerical test matrix with fixed seeds.
- [x] Verify deterministic replay and finite outputs.
- [x] Exercise invalid-shape, aliasing, overflow, and corrupted-state failures.
- [x] Record tolerances, environment, operator coverage, gradient results, and artifact hashes.
- [x] Write the Stage 1 gate record and transition report.

## Transition

- [x] Record `PASS` for Stage 1.
- [x] Authorize Stage 2 sequence-core work.
- [x] Preserve the numerical contract as the correctness reference for later optimized paths.

**Evidence locations:** `artifacts/stage-1/`, numerical engine tests, gradient reports, and gate record.

---

# Stage 2 — CCT Todo: Efficient Sequence Core

**Goal:** Build a stable selective recurrent sequence mechanism with causal semantics and full-vs-streaming equivalence.

**Status:** `[x] COMPLETE — REVALIDATE after changes.`

## Implementation tasks

- [x] Define sequence state, update order, reset semantics, causal boundaries, and state ownership.
- [x] Implement recurrent state updates using the Stage 1 numerical substrate.
- [x] Implement typed state initialization, serialization, versioning, and reset.
- [x] Implement chunked and streaming execution with incremental state reuse and flush behavior.
- [x] Bound state memory and allocations.
- [x] Add invalid-state-version, wrong-shape, out-of-order-event, unexpected-reset, and non-finite-recurrence failures.

## Gate and evidence tasks

- [x] Compare full-sequence and streaming outputs.
- [x] Verify deterministic state initialization, update, serialization, reload, and reset.
- [x] Measure memory, throughput, latency, and long-sequence behavior.
- [x] Exercise invalid state transitions and ensure they fail closed.
- [x] Record state identity, resource thresholds, seeds, and comparison tolerances.
- [x] Write the Stage 2 gate record and transition report.

## Transition

- [x] Record `PASS` for Stage 2.
- [x] Authorize Stage 3 causal-event work.
- [x] Retain the simple sequence path as a correctness oracle for later optimization.

**Evidence locations:** `artifacts/stage-2/`, sequence-core tests, streaming-equivalence report, and gate record.

---

# Stage 3 — CCT Todo: Causal Event Learning

**Goal:** Add event identity, causal ordering, dependency structure, interventions, counterfactuals, robustness, and abstention.

**Status:** `[x] COMPLETE — REVALIDATE after changes.`

## Implementation tasks

- [x] Represent stable event IDs, ordering or timestamps, source metadata, and causal relationships.
- [x] Build and validate causal dependency structures with a declared cycle policy.
- [x] Add topological execution and future-information leakage checks.
- [x] Implement interventions with changed-input records, outcome comparison, and temporary-state rollback.
- [x] Implement paired counterfactual worlds with shared seed control and changed-variable manifests.
- [x] Add reorder, omission, irrelevant-event, contradiction, perturbation, and adversarial-event tests.
- [x] Implement confidence, evidence thresholds, uncertainty records, and safe abstention.

## Gate and evidence tasks

- [x] Verify event identity, duplicate detection, source digests, and ordering.
- [x] Pass the causal DAG and leakage audit.
- [x] Verify intervention and counterfactual consistency.
- [x] Verify robustness under declared perturbations.
- [x] Verify unsupported and ambiguous requests abstain safely.
- [x] Record the expanded gate contract and approval boundary.
- [x] Write the Stage 3 gate record and transition report.

## Transition

- [x] Record `PASS` for Stage 3.
- [x] Preserve explicit approval before Stage 4 transition.
- [x] Authorize Stage 4 only within the documented scope.

**Evidence locations:** `artifacts/stage-3/`, causal-event fixtures, expanded gate report, and approval record.

---

# Stage 4 — CCT Todo: Persistent Verifiable Memory

**Goal:** Implement durable, checksummed, provenance-linked, retrievable, deletable, conflict-aware, retention-governed, and recoverable memory.

**Status:** `[x] COMPLETE — REVALIDATE after changes.`

## Implementation tasks

- [x] Implement checksummed records, versions, atomic commits, indexes, and durable metadata.
- [x] Implement exact lookup, deterministic ranking, provenance, citation, and missing-result behavior.
- [x] Implement retention, expiry, deletion tombstones, rebuild, conflict records, and ownership.
- [x] Reject malformed, duplicated, poisoned, and unauthorized records.
- [x] Implement crash, partial-write, corrupted-checksum, and committed-state recovery.
- [x] Preserve append or record-log identity and audit history.

## Gate and evidence tasks

- [x] Verify exact retrieval and provenance integrity.
- [x] Verify deletion, retention, expiry, conflict resolution, and deterministic rebuild.
- [x] Verify poisoning isolation and malformed-record rejection.
- [x] Simulate interruption and confirm recovery of the last committed state.
- [x] Verify corrupt checksums fail closed.
- [x] Record memory manifest, recovery report, deletion report, and gate decision.
- [x] Write the Stage 4 transition report.

## Transition

- [x] Record `PASS` for Stage 4.
- [x] Preserve explicit approval before Stage 5 transition.
- [x] Authorize language and code scaling within the approved data and resource scope.

**Evidence locations:** `artifacts/stage-4/`, memory logs, recovery fixtures, deletion records, and gate report.

---

# Stage 5 — CCT Todo: Language and Code Scaling

**Goal:** Demonstrate native language and code micro-training with matched baselines, memory attribution, long-context checks, replay, and code-safety controls.

**Status:** `[x] COMPLETE — REVALIDATE after changes.`

## Implementation tasks

- [x] Pin governed language and code micro-corpora with source, license, split, and contamination records.
- [x] Implement fixed-budget native training, finite-objective checks, checkpoint, and resume.
- [x] Implement matched recurrent or dense reference baselines using equal data, token budget, and evaluation splits.
- [x] Build context, memory, throughput, and parameter-budget scaling matrix.
- [x] Attribute resource cost to recurrence, spectral operation, memory, and data processing.
- [x] Add syntax-aware code fixtures, malformed-code behavior, execution denial, and sandbox boundaries.

## Gate and evidence tasks

- [x] Verify finite training and stable checkpoints.
- [x] Compare CCT to the declared matched controls.
- [x] Verify checkpoint replay and deterministic restoration.
- [x] Produce memory and runtime attribution.
- [x] Produce long-context evidence.
- [x] Exercise unsafe-code and malformed-code failures.
- [x] Record the scaling report, baseline report, attribution report, and gate decision.

## Transition

- [x] Record `PASS` for Stage 5.
- [x] Preserve explicit approval before Stage 6 transition.
- [x] Authorize bounded deliberation and verification.

**Evidence locations:** `artifacts/stage-5/`, scaling matrix, matched-baseline report, memory profile, and code-safety gate.

---

# Stage 6 — CCT Todo: Deliberation and Verification

**Goal:** Add bounded planning, independent verification, evidence-aware abstention, deny-by-default tools, replay, interruption recovery, and incident logging.

**Status:** `[x] COMPLETE — REVALIDATE after changes.`

## Implementation tasks

- [x] Represent typed plans with steps, dependencies, preconditions, expected results, budget, timeout, and cancellation.
- [x] Implement an independent verifier that cannot silently self-certify the same result.
- [x] Attach evidence, disagreement, approval, and abstention status to each material claim.
- [x] Register offline tools with deny-by-default authorization and input/output validation.
- [x] Implement idempotent retry, interruption, cancellation, partial-completion, and incident reconstruction.
- [x] Add unsupported-plan, ambiguous-result, missing-evidence, and unsafe-action failures.

## Gate and evidence tasks

- [x] Inject planning errors and verify independent detection.
- [x] Verify plans remain within budget and timeout limits.
- [x] Verify tools remain denied without authorization.
- [x] Verify evidence and abstention outcomes.
- [x] Replay interrupted plans deterministically.
- [x] Confirm incidents are reconstructable from logs.
- [x] Record the Stage 6 gate and transition package.

## Transition

- [x] Record `PASS` for Stage 6.
- [x] Authorize controlled multimodal and open-ended research in Stage 7.

**Evidence locations:** `artifacts/stage-6/`, plan fixtures, verifier report, tool-policy report, replay logs, and incident records.

---

# Stage 7 — CCT Todo: Multimodal and Open-Ended Research

**Goal:** Extend CCT with typed multimodal events and validated adapters while preserving alignment, masks, typed memory, deterministic simulation, transfer checks, auditability, and safety.

**Status:** `[x] COMPLETE — REVALIDATE after changes.`

## Implementation tasks

- [x] Define typed modality interfaces, adapter lifecycle, source, timestamp, confidence, shape, and validity fields.
- [x] Implement the seven declared adapters with versioning, capability checks, and malformed-input failures.
- [x] Implement mask-aware fusion with missing-modality, temporal-order, conflict, and future-leakage handling.
- [x] Store typed multimodal memory with source identity, retention, retrieval, and deletion behavior.
- [x] Implement deterministic simulation, transfer boundaries, audit logging, and unsafe-input controls.

## Gate and evidence tasks

- [x] Run contract tests for every declared adapter.
- [x] Verify alignment, masking, missing-modality, and temporal-order behavior.
- [x] Verify deterministic simulation and transfer evaluation.
- [x] Verify typed-memory provenance and deletion.
- [x] Verify audit and safety controls.
- [x] Record the terminal research gate and controlled-continuation boundary.

## Transition

- [x] Record `PASS` for Stage 7.
- [x] Authorize the production NLP governance segment.
- [x] Keep unrestricted deployment closed.

**Evidence locations:** `artifacts/stage-7/`, adapter reports, multimodal fixtures, transfer report, safety report, and gate record.

---

# Stage 8 — CCT Todo: Production NLP Governance Foundation

**Goal:** Establish the governance registry, policy boundaries, realistic application fixtures, artifact protocol, adversarial controls, and readiness evidence.

**Status:** `[x] COMPLETE — REVALIDATE after changes.`

## Implementation tasks

- [x] Register models, data, tools, policies, evaluators, and releases with immutable identities.
- [x] Define allowed, disallowed, review-required, and evaluator-only operations.
- [x] Build realistic bounded application fixtures with user-visible reports and evaluator ownership.
- [x] Produce manifest, config, environment, tests, benchmarks, gate, logs, and report artifacts.
- [x] Exercise prompt, data, tool, identity, leakage, policy-bypass, and rollback challenges.

## Gate and evidence tasks

- [x] Verify registry and policy integrity.
- [x] Run realistic application fixtures, including failure paths.
- [x] Verify artifact completeness and identity linkage.
- [x] Verify adversarial controls and unsafe-operation denial.
- [x] Document the readiness boundary.
- [x] Record that the gate authorizes governed data work only, not unrestricted deployment.

## Transition

- [x] Record `PASS` for Stage 8.
- [x] Authorize governed real-source corpus work in Stage 9.

**Evidence locations:** `artifacts/stage-8/`, governance registry, policy records, application fixtures, adversarial report, and gate record.

---

# Stage 9 — CCT Todo: Governed Data and Corpus

**Goal:** Acquire and prepare real data through native, reproducible, rights-aware, contamination-resistant corpus processing.

**Status:** `[x] COMPLETE — REVALIDATE after changes.`

## Implementation tasks

- [x] Pin every source, revision, split, license record, and acquisition URL.
- [x] Quarantine rights, privacy, malformed, unsafe, and disallowed records.
- [x] Implement exact and near-duplicate detection.
- [x] Enforce cross-split contamination barriers.
- [x] Build deterministic shards with stable ordering, seed, counts, digests, and replay.
- [x] Implement deletion, re-preparation, audit, stale-artifact detection, and missing-source failure.

## Gate and evidence tasks

- [x] Verify source identity and rights metadata.
- [x] Verify quarantine and safety decisions.
- [x] Verify exact and near deduplication.
- [x] Verify zero unexplained train, validation, and test overlaps.
- [x] Replay shard preparation from the same seed and source.
- [x] Verify deletion and stale-artifact detection.
- [x] Record the data gate and source audit.

## Transition

- [x] Record `PASS` for Stage 9.
- [x] Authorize tokenizer and representation work in Stage 10.

**Evidence locations:** `artifacts/stage-9/`, source manifests, quarantine records, deduplication reports, deterministic shards, deletion report, and gate record.

---

# Stage 10 — CCT Todo: Tokenizer and Representation

**Goal:** Create an immutable native tokenizer and representation contract with offsets, byte fallback, provenance, causal batches, and efficiency measurement.

**Status:** `[x] COMPLETE — REVALIDATE after changes.`

## Implementation tasks

- [x] Implement byte, subword, and hybrid tokenizer candidates.
- [x] Compare candidates under a declared data and evaluation contract.
- [x] Freeze the selected vocabulary and tokenizer snapshot with a stable hash.
- [x] Implement Unicode-safe source-to-token offsets and malformed-input policy.
- [x] Implement byte fallback for every supported byte.
- [x] Build deterministic packed and padded causal batches with padding, boundary, and record masks.
- [x] Measure token efficiency, source bytes, memory, throughput, and round-trip behavior.

## Gate and evidence tasks

- [x] Verify tokenizer snapshot identity and immutable loading.
- [x] Verify source offsets, Unicode behavior, and byte fallback.
- [x] Verify deterministic round trips.
- [x] Verify packed and padded batch masks and boundaries.
- [x] Compare efficiency and representation results.
- [x] Record that tokenizer changes invalidate downstream data and checkpoint identity.
- [x] Write the Stage 10 gate and transition record.

## Transition

- [x] Record `PASS` for Stage 10.
- [x] Authorize the trainable native NLP core in Stage 11.

**Evidence locations:** `artifacts/stage-10/`, tokenizer snapshots, offset fixtures, batch reports, efficiency comparison, and gate record.

---

# Stage 11 — CCT Todo: Trainable Native NLP Core

**Goal:** Make CCT trainable through a native categorical next-token objective with analytic recurrence gradients, optimizer safety, checkpoint recovery, real-source evidence, and matched controls.

**Status:** `[x] COMPLETE — REVALIDATE after changes.`

## Implementation tasks

- [x] Define the next-token objective, causal masks, sequence contract, target validation, and token accounting.
- [x] Implement analytic recurrence gradients and independently verify them numerically.
- [x] Implement stable optimization, clipping, scheduling, finite checks, and deterministic initialization.
- [x] Save and reload checkpoints containing tokenizer, dataset, configuration, optimizer, step, cursor, and model identity.
- [x] Implement wrong-identity, corruption, and incompatible-checkpoint rejection.
- [x] Run a real-source pilot and matched controls.
- [x] Measure validation loss, perplexity, token accuracy, throughput, memory, and parameters.

## Gate and evidence tasks

- [x] Verify finite objectives and gradients.
- [x] Verify analytic-gradient agreement.
- [x] Verify deterministic multi-seed behavior.
- [x] Verify held-out behavior or explain the result without weakening the criterion.
- [x] Verify checkpoint save, exact load, wrong-identity rejection, corruption failure, and resume equivalence.
- [x] Verify real-source provenance and matched controls.
- [x] Record complete machine-readable and human-readable artifacts.

## Transition

- [x] Record `PASS` for Stage 11.
- [x] Authorize scaling and accelerator systems in Stage 12.

**Evidence locations:** `artifacts/stage-11/`, NLP trainer tests, gradient report, pilot report, checkpoint artifacts, and gate record.

---

# Stage 12 — CCT Todo: Scaling and Accelerator Systems

**Goal:** Scale the native path through reference and optimized implementations with parity, resource profiling, ordered workers, atomic recovery, and a backend decision.

**Status:** `[x] COMPLETE — REVALIDATE after changes.`

## Implementation tasks

- [x] Preserve the simple CPU reference path as correctness oracle.
- [x] Implement optimized or fused path with explicit fallback.
- [x] Build the context, batch, model, and data scaling matrix.
- [x] Profile memory, throughput, latency, allocations, and failure behavior.
- [x] Implement ordered worker behavior and seed partitioning.
- [x] Implement atomic checkpoint, partial-write simulation, corrupt-checkpoint rejection, and committed-state replay.
- [x] Record backend selection and unsupported-platform capabilities honestly.

## Gate and evidence tasks

- [x] Verify reference and optimized numerical parity.
- [x] Run the complete resource-accounted scaling matrix.
- [x] Verify worker equivalence and interruption behavior.
- [x] Verify atomic recovery and corruption failure.
- [x] Record the backend decision and absent capabilities.
- [x] Write the Stage 12 gate and transition package.

## Transition

- [x] Record `PASS` for Stage 12.
- [x] Authorize supervised fine-tuning and adapters in Stage 13.

**Evidence locations:** `artifacts/stage-12/`, parity report, scaling matrix, resource profile, recovery report, architecture decision, and gate record.

---

# Stage 13 — CCT Todo: Supervised Fine-Tuning and Adapters

**Goal:** Turn the trainable CCT core into a bounded task-adaptation system with explicit formatting, target-only masks, full and low-rank adaptation, structured outputs, citations, safety retention, authorization, and deletion lineage.

**Status:** `[x] COMPLETE — REVALIDATE after changes.`

## Implementation tasks

- [x] Define the six task schemas, example identity, target provenance, eligibility, policy class, and evaluator ownership.
- [x] Implement canonical instruction formatting and `target-span-only-v1` loss masks.
- [x] Implement and compare full-parameter and parameter-efficient adapter training.
- [x] Validate structured outputs, grounded citations, missing-evidence behavior, and safe refusal.
- [x] Enforce adapter authorization, base immutability, merge equivalence, deletion, and lineage.
- [x] Evaluate representative held-out tasks against the base model.

## Gate and evidence tasks

- [x] Verify representative task improvement.
- [x] Verify target-only mask correctness.
- [x] Verify structured-output validation.
- [x] Verify grounded citations and missing-evidence handling.
- [x] Verify unsafe-request denial and safety retention.
- [x] Verify adapter permissions, base immutability, merge/runtime agreement, deletion, and identity linkage.
- [x] Record the Stage 13 gate and transition package.

## Transition

- [x] Record `PASS` for Stage 13.
- [x] Authorize preference tuning and alignment in Stage 14.

**Evidence locations:** `artifacts/stage-13/`, SFT tests, adapter reports, structured-output fixtures, safety report, deletion lineage, and gate record.

---

# Stage 14 — CCT Todo: Preference Tuning and Alignment

**Goal:** Improve controllability, helpfulness, refusal quality, citations, style, calibration, and task quality without unacceptable truthfulness, safety, or regression damage.

**Status:** `[x] COMPLETE — REVALIDATE after changes.`

## Implementation tasks

- [x] Govern preference pairs or rankings with annotator, task, provenance, conflict, and split identity.
- [x] Compare preference optimization candidates under the same evaluation contract.
- [x] Implement verifier-weighted reranking or the declared equivalent control.
- [x] Measure helpfulness, truthfulness, calibration, refusal quality, citation integrity, and task quality.
- [x] Run adversarial and blind evaluations against the Stage 13 baseline.
- [x] Preserve deletion, replay, checkpoint, approval, and decision records.

## Gate and evidence tasks

- [x] Verify preference data governance and quarantine.
- [x] Compare candidate methods without presuming universal superiority.
- [x] Verify independent quality checks and disagreement handling.
- [x] Verify blind, adversarial, calibration, safety, truthfulness, and regression results.
- [x] Verify checkpoint lineage, deletion, and approvals.
- [x] Record the selected method and all known limitations.
- [x] Write the Stage 14 gate and transition package.

## Transition

- [x] Record `PASS` for Stage 14.
- [x] Authorize verified retrieval and knowledge in Stage 15.

**Evidence locations:** `artifacts/stage-14/`, preference manifest, alignment report, blind review, adversarial report, calibration report, and gate record.

---

# Stage 15 — CCT Todo: Verified Retrieval and Knowledge

**Goal:** Provide typed retrieval, freshness, citations, conflict handling, deletion, poisoning isolation, auditability, and verified grounding.

**Status:** `[x] COMPLETE — REVALIDATE after changes.`

## Implementation tasks

- [x] Define typed knowledge records with source, authority, freshness, version, digest, and retention.
- [x] Implement deterministic retrieval with query, ranking, evidence bundle, and citation identity.
- [x] Distinguish fresh, stale, conflicting, missing, and deleted knowledge.
- [x] Quarantine poisoned and unauthorized records.
- [x] Require grounded output or bounded abstention when evidence is insufficient.
- [x] Audit retrieval, use, deletion, refresh, and conflict resolution.

## Gate and evidence tasks

- [x] Verify citation correctness and provenance.
- [x] Verify freshness, stale boundaries, and conflicting-source behavior.
- [x] Verify deletion and rebuild.
- [x] Verify poisoning isolation and unauthorized-source rejection.
- [x] Verify deterministic retrieval and complete audit trace.
- [x] Verify grounded or abstaining outputs.
- [x] Write the Stage 15 gate and transition package.

## Transition

- [x] Record `PASS` for Stage 15.
- [x] Authorize production inference and operations in Stage 16.

**Evidence locations:** `artifacts/stage-15/`, knowledge manifest, retrieval report, freshness/conflict fixtures, poisoning report, audit traces, and gate record.

---

# Stage 16 — CCT Todo: Production Inference and Operations

**Goal:** Provide a production-like native inference service with versioned APIs, batching, state/cache isolation, observability, SLOs, fault controls, canaries, and rollback.

**Status:** `[x] COMPLETE — REVALIDATE after changes.`

## Implementation tasks

- [x] Define versioned request, response, error, compatibility, and authentication-boundary contracts.
- [x] Implement scheduling, batching, streaming, recurrent-state lifecycle, and retrieval integration.
- [x] Isolate user, request, model, adapter, retrieval, and cache state.
- [x] Measure latency, throughput, resource use, saturation, and SLOs.
- [x] Enforce quotas, policy checks, fault handling, timeout, retry, and circuit controls.
- [x] Implement version registry, canary, approval, rollback, deletion, retirement, and incident workflow.

## Gate and evidence tasks

- [x] Verify API correctness and compatibility.
- [x] Verify state and cache isolation.
- [x] Verify metrics, traces, logs, privacy controls, and SLO evidence.
- [x] Inject faults and verify timeout, retry, circuit, quota, and recovery behavior.
- [x] Verify canary and rollback.
- [x] Verify deletion and retirement behavior.
- [x] Write the Stage 16 gate and transition package.

## Transition

- [x] Record `PASS` for Stage 16.
- [x] Authorize only the controlled pilot and bounded release in Stage 17.

**Evidence locations:** `artifacts/stage-16/`, API contract, runtime report, SLO profile, fault report, canary records, rollback records, and gate record.

---

# Stage 17 — CCT Todo: Controlled Pilot and Production Release

**Goal:** Move a narrowly defined CCT-NLP use case through shadow evaluation, bounded pilot, oversight, incident response, rollback, deletion, drift review, and release approval.

**Status:** `[x] COMPLETE — TERMINAL BOUNDED RELEASE; REVALIDATE after changes.`

## Implementation tasks

- [x] Freeze the named scope, model, data, policy, evaluator, runtime, and release configuration.
- [x] Run shadow evaluation without external side effects.
- [x] Record disagreement, error taxonomy, evaluator decisions, and approval.
- [x] Operate the bounded pilot with named users, quotas, permissions, and human review.
- [x] Exercise incident response, rollback, deletion, drift detection, and release review.
- [x] Preserve visible limitations, evaluator records, release note, scope, expiry, and terminal artifacts.
- [x] Keep training authorization, external action, and scope expansion closed unless separately specified and approved.

## Gate and evidence tasks

- [x] Verify shadow evidence and reference comparison.
- [x] Verify bounded pilot controls and activity logs.
- [x] Verify policy, refusal, escalation, incident, rollback, and deletion behavior.
- [x] Verify data, quality, calibration, latency, and behavior drift signals.
- [x] Verify approvals, known limitations, scope, expiry, and final release record.
- [x] Record terminal `PASS` for the named bounded scope.

## Transition

- [x] Confirm Stage 17 has no automatic successor.
- [x] Require a new CCT specification and new gate for any expanded capability, different domain, changed model, external action, or broader deployment.

**Evidence locations:** `artifacts/stage-17/`, shadow report, pilot report, incident records, rollback evidence, drift report, approval record, release record, and terminal gate.

---

# Current CCT Training Milestone — Track 1 Todo

**Goal:** Prepare and train CCT on a governed real-data bundle using WikiText-2 pretraining and SQuAD 2.0 supervised target learning with frozen evaluation.

**Status:** `[x] COMPLETE for the declared Track 1 contract.`

## Preparation tasks

- [x] Pin WikiText-2 source identity, raw archive URL, revision, license metadata, and split members.
- [x] Pin SQuAD 2.0 source identity, GEM direct-file revision, upstream provenance, license metadata, and direct URLs.
- [x] Implement native resumable downloads, cache reuse, atomic extraction, retry, pacing, and fail-closed missing-cache behavior.
- [x] Parse WikiText raw members natively without the rate-limited rows endpoint in production mode.
- [x] Parse GEM flat SQuAD files natively without paginated row acquisition.
- [x] Decode JSON Unicode surrogate pairs correctly.
- [x] Convert SQuAD codepoint answer offsets to UTF-8 byte offsets.
- [x] Validate answer text against the source context.
- [x] Select deterministic balanced answerable and unanswerable SFT examples.
- [x] Keep SFT evaluation and frozen final-test identities isolated from training.
- [x] Generate source manifests, dataset manifests, digests, preparation reports, and evaluation contracts.
- [x] Support a bounded local fixture mode without weakening production validation.

## Preparation gate tasks

- [x] Run Track 1 unit tests.
- [x] Run the Track 1 formal gate.
- [x] Run the full native CTest suite.
- [x] Run cumulative Track 1 CI.
- [x] Run a real bounded network preparation.
- [x] Run complete production-scale preparation.
- [x] Verify `passed: true`, `malformed_rows: 0`, and `overlap_ids: 0`.
- [x] Preserve preparation report, source manifest, dataset manifest, evaluation contract, and release record.

## Native training tasks

- [x] Implement the native Track 1 training runner in C++20.
- [x] Consume prepared WikiText pretraining records.
- [x] Consume prepared SQuAD SFT records.
- [x] Use `target-span-only-v1` supervision so prompt tokens do not contribute to SFT target loss.
- [x] Filter zero-loss prompt-only chunks.
- [x] Train the pretraining phase and save a checkpoint.
- [x] Continue from the pretraining checkpoint for SFT.
- [x] Save the SFT checkpoint with dataset, tokenizer, configuration, optimizer, and step identity.
- [x] Evaluate validation and the frozen final-test target-token metrics.
- [x] Write a training report that states the exact evaluation scope and limitations.
- [x] Verify checkpoint reload and declared identity.
- [x] Keep the frozen final-test split out of updates and checkpoint selection.

## Track 1 gate and boundaries

- [x] Record the Track 1 preparation gate as `PASS`.
- [x] Record the native training handoff as `PASS` for its declared bounded contract.
- [x] Keep exact-answer SQuAD EM/F1 unclaimed until constrained answer decoding is implemented and evaluated.
- [x] Keep broader capability claims outside the report.
- [ ] Re-run the full Track 1 preparation and training command from a fresh Colab checkout after any training-runner change.
- [ ] Add constrained answer decoding and exact-match/F1 only as a separately specified CCT task, with new tests and a new gate.

**Primary evidence:** `artifacts/track1/`, `artifacts/track1/cpp-gate/`, `artifacts/track1/real-full-preparation/`, `artifacts/track1/real-training/`, `cpp/tools/track1_train.cpp`, `cpp/tools/track1_gate.cpp`, and `artifacts/track1/README.md`.

## Reproducible Track 1 commands

```bash
cd CCT
git pull origin main
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

---

# Maintenance and Revalidation Todo

These tasks remain active after a stage is marked complete because a later code, data, configuration, or environment change invalidates prior evidence.

- [ ] Before each milestone, start from a clean checkout and record `git rev-parse HEAD`.
- [ ] Rebuild with the declared strict native flags.
- [ ] Run the complete prior-stage chain before declaring a later gate valid.
- [ ] Recompute configuration, environment, data, checkpoint, and artifact hashes after changes.
- [ ] Rerun all affected failure-path tests.
- [ ] Rerun baseline and ablation comparisons after changing a model component.
- [ ] Rerun resource profiles after changing context, batch, model, optimizer, or backend.
- [ ] Preserve old evidence and write a new release record rather than overwriting historical results.
- [ ] Review reports for claims beyond their metrics and remove unsupported language.
- [ ] Push the release commit after the gate passes and verify the remote branch.

# Completion Rule

The CCT Todo is complete for a declared scope only when every required task for that scope is checked, all mandatory gates are `PASS`, artifacts are identity-linked and reviewable, failure conditions remain covered, transitions are recorded, and no implementation change has invalidated the evidence. A task must be reopened whenever its supporting code, data, configuration, environment, checkpoint, or gate artifact changes.

## References

- [CCT Goal](Goal.md)
- [CCT stage specifications](Stages/README.md)
- [CCT architecture](Architecture.md)
- [Track 1 operational guide](artifacts/track1/README.md)
