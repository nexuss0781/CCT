# CCT Goal

## Internal Engineering Objective

This document defines the repository-internal goals for the **CCT-ASE** program. Every section is labelled as a **CCT Goal** and is limited to architecture, implementation, verification, data governance, training, evaluation, operations, and controlled release.

The document intentionally does not describe external product narratives, broader non-CCT concepts, unvalidated intelligence claims, or capabilities that are outside the repository’s measured evidence. A stage is complete only when its implementation, tests, formal gate, artifacts, and transition record are all complete.

> **Primary CCT objective:** build a native C++20 adaptive learning and inference engine whose numerical behavior, causal state, memory, training, evaluation, serving, and release controls are independently testable, reproducible, and fail-closed.

## Global CCT Gate Rules

Every stage follows the same lifecycle: define the contract, implement the smallest complete native path, create realistic failure tests, run the formal gate, preserve artifacts, and record the transition decision. A passing capability metric cannot override a failed correctness, reproducibility, provenance, safety, or operational criterion.

| Gate status | Meaning | Allowed transition |
|---|---|---|
| `PASS` | All mandatory implementation, evaluation, artifact, and failure-path criteria passed. | Begin the documented successor within scope. |
| `FAIL` | A mandatory criterion failed. | Stop, add or repair regression coverage, remediate, and rerun. |
| `BLOCKED` | An explicitly optional dependency is unavailable. | Continue only where the stage specification permits it; the core gate remains closed. |

Every release-quality gate records the repository commit, configuration, source or environment manifest, software and hardware context, random seeds, test results, benchmark results, thresholds, known failures, artifact hashes, and final decision. Any implementation change after a gate invalidates that gate until rerun.

The required evidence classes are correctness, deterministic replay, realistic failure handling, resource accounting, baseline or ablation comparison, provenance, checkpoint or state recovery, and human-readable reporting. Smoke tests may support development but cannot substitute for the formal gate.

## CCT Stage Map

| Stage | CCT goal | Primary transition result |
|---:|---|---|
| 0 | Reproducible baseline | Authorizes numerical implementation. |
| 1 | Differentiable numerical engine | Authorizes sequence-core implementation. |
| 2 | Efficient sequence core | Authorizes causal-event learning. |
| 3 | Causal event learning | Authorizes persistent verifiable memory after its approval boundary. |
| 4 | Persistent verifiable memory | Authorizes language and code scaling after its approval boundary. |
| 5 | Language and code scaling | Authorizes deliberation and verification after its approval boundary. |
| 6 | Deliberation and verification | Authorizes multimodal and open-ended research under controlled continuation. |
| 7 | Multimodal and open-ended research | Authorizes the production NLP governance segment. |
| 8 | Production NLP governance foundation | Authorizes governed real-source data work. |
| 9 | Governed data and corpus | Authorizes tokenizer and representation work. |
| 10 | Tokenizer and representation | Authorizes the trainable native NLP core. |
| 11 | Trainable native NLP core | Authorizes scaling and accelerator systems. |
| 12 | Scaling and accelerator systems | Authorizes supervised fine-tuning and adapters. |
| 13 | Supervised fine-tuning and adapters | Authorizes preference tuning and alignment. |
| 14 | Preference tuning and alignment | Authorizes verified retrieval and knowledge. |
| 15 | Verified retrieval and knowledge | Authorizes production inference and operations. |
| 16 | Production inference and operations | Authorizes a controlled pilot and bounded release. |
| 17 | Controlled pilot and production release | Terminal bounded release for the named scope; expansion requires a new specification. |

---

# Stage 0 — CCT Goal: Reproducible Baseline

## Goal

Create a clean native C++20 baseline that can be built, tested, benchmarked, and replayed from a known repository state. Establish the project’s first trustworthy measurement and artifact protocol before adding model complexity.

## End-to-end objectives

1. Define the supported compiler, CMake, warning, language-standard, operating-system, and dependency contract.
2. Implement a clean build path with strict warnings and warnings-as-errors for native C++ sources.
3. Create deterministic unit, integration, and benchmark harnesses with explicit seeds and configuration.
4. Define canonical artifact directories, report schemas, status values, and release-record fields.
5. Verify that a clean checkout produces the same test and benchmark identity under the declared environment.

## Implementation breakdown

| Workstream | Required implementation |
|---|---|
| Build | CMake project, native C++20 targets, strict compiler flags, dependency checks, and reproducible configuration. |
| Test | Deterministic test runner, failure reporting, exit-code contract, and repeatable invocation. |
| Benchmark | Stable benchmark schema with workload, seed, elapsed time, throughput, memory, and environment fields. |
| Artifacts | Versioned reports, manifest hashes, release records, logs, and configuration snapshots. |
| Failure handling | Missing dependency, malformed configuration, non-finite output, and nondeterministic replay must fail closed. |

## Gate

The Stage 0 gate passes only when the clean build succeeds, all baseline tests pass, repeated runs agree within declared tolerances, benchmark output is schema-valid, and artifacts identify the exact commit and configuration.

## Transition

A Stage 0 `PASS` authorizes numerical field and operator work in Stage 1. It does not authorize claims about learning, language capability, or deployment.

---

# Stage 1 — CCT Goal: Differentiable Numerical Engine

## Goal

Implement a correct and numerically stable field/operator substrate that supports the later CCT recurrence and provides analytic gradients with independently verified derivatives.

## End-to-end objectives

1. Define tensor, field, shape, layout, dtype, and ownership contracts.
2. Implement the required numerical operators with deterministic behavior and explicit boundary handling.
3. Implement analytic gradients for every trainable operator used by the sequence core.
4. Verify analytic gradients against finite differences or an independent reference implementation.
5. Detect overflow, underflow, NaN, infinity, invalid shape, aliasing, and mutation errors.

## Implementation breakdown

| Workstream | Required implementation |
|---|---|
| Representation | Native tensor or field storage, shape validation, indexing, ownership, and lifetime rules. |
| Operators | Forward operators, stable reductions, normalization, activation, mixing, and spectral or adaptive primitives used by CCT. |
| Gradients | Analytic backward path, gradient accumulation, parameter identity, and deterministic update interface. |
| Numerical safety | Finite checks, range checks, stable special cases, and fail-closed invalid-input behavior. |
| Verification | Finite-difference checks, reference comparisons, seed replay, and boundary-value tests. |

## Gate

The gate requires correct forward results, gradient agreement within declared tolerances, deterministic replay, finite outputs across the test matrix, and realistic failures for invalid shapes, aliasing, overflow, and corrupted state.

## Transition

A Stage 1 `PASS` authorizes the efficient sequence core in Stage 2.

---

# Stage 2 — CCT Goal: Efficient Sequence Core

## Goal

Build the stable selective recurrent sequence mechanism that carries information through ordered events while preserving causal semantics and equivalence between full-sequence and streaming execution.

## End-to-end objectives

1. Define the sequence state, update order, reset semantics, and causal boundary contract.
2. Implement recurrent state updates using the Stage 1 numerical substrate.
3. Support chunked and streaming execution without changing valid outputs.
4. Bound state memory and expose explicit state ownership and reset behavior.
5. Measure throughput, latency, memory, and numerical stability against a declared baseline.

## Implementation breakdown

| Workstream | Required implementation |
|---|---|
| State | Typed recurrent state, initialization, update, reset, serialization, and versioning. |
| Causality | Strict no-future-read contract, ordered events, masks, and sequence boundaries. |
| Streaming | Chunked execution, incremental state reuse, flush behavior, and full-vs-stream equivalence. |
| Efficiency | Memory accounting, throughput measurement, bounded allocation, and long-sequence tests. |
| Failure handling | Invalid state version, wrong shape, out-of-order event, unexpected reset, and non-finite recurrence. |

## Gate

The gate passes only when full and streaming execution agree, recurrent state is deterministic and recoverable, memory remains within declared bounds, and invalid state transitions fail closed.

## Transition

A Stage 2 `PASS` authorizes causal-event identity and intervention work in Stage 3.

---

# Stage 3 — CCT Goal: Causal Event Learning

## Goal

Give CCT explicit event identity, causal ordering, dependency structure, intervention handling, counterfactual evaluation, robustness tests, and safe abstention behavior.

## End-to-end objectives

1. Represent events with stable identity, timestamps or order, source metadata, and causal relationships.
2. Build and validate causal dependency structures without leakage from future or evaluator-only information.
3. Implement intervention and counterfactual pathways with explicit provenance.
4. Test robustness under reordering, missing events, irrelevant events, contradictory evidence, and perturbation.
5. Implement confidence and abstention rules for unsupported or ambiguous conclusions.

## Implementation breakdown

| Workstream | Required implementation |
|---|---|
| Event identity | Stable IDs, source digest, ordering, provenance, and duplicate detection. |
| Causal graph | Dependency representation, acyclicity or declared cycle policy, topological execution, and validation. |
| Intervention | Do-operations, changed-input records, outcome comparison, and rollback of temporary state. |
| Counterfactual | Paired worlds, shared seed control, changed-variable manifest, and comparison metrics. |
| Robustness | Leakage audit, reorder tests, omission tests, conflict tests, and adversarial event inputs. |
| Abstention | Evidence thresholds, unsupported-query response, uncertainty record, and safe default. |

## Gate

The gate must demonstrate event identity correctness, no leakage, causal ordering, intervention and counterfactual consistency, robustness under declared perturbations, and bounded abstention. The expanded gate must pass before Stage 4 is treated as complete.

## Transition

A Stage 3 `PASS` authorizes Stage 4 only within its documented scope and after the required approval boundary.

---

# Stage 4 — CCT Goal: Persistent Verifiable Memory

## Goal

Implement a durable memory substrate whose records are checksummed, provenance-linked, retrievable, deletable, conflict-aware, retention-governed, and recoverable after interruption or corruption.

## End-to-end objectives

1. Persist CCT state and memory records with checksums, versions, and atomic commit semantics.
2. Retrieve exact records with provenance and citation metadata.
3. Support deletion, retention, expiry, conflict resolution, and rebuild without stale data leakage.
4. Isolate or reject poisoned, malformed, duplicated, and unauthorized records.
5. Recover the last committed state after partial writes, corruption, or process interruption.

## Implementation breakdown

| Workstream | Required implementation |
|---|---|
| Storage | Checksummed append or record log, index, version, atomic commit, and durable metadata. |
| Retrieval | Exact lookup, deterministic ranking, provenance, citation, and missing-result behavior. |
| Governance | Retention policy, deletion tombstones, rebuild, conflict records, and ownership. |
| Safety | Poisoning boundary, malformed record rejection, access controls, and audit trail. |
| Recovery | Crash simulation, partial-write recovery, corrupt-checksum failure, and replay. |

## Gate

The gate requires exact retrieval, provenance integrity, correct deletion, conflict and retention behavior, poisoning isolation, deterministic rebuild, and committed-state recovery. The expanded Stage 4 contract must pass before Stage 5 transition.

## Transition

A Stage 4 `PASS` authorizes language and code scaling within the approved data and resource scope.

---

# Stage 5 — CCT Goal: Language and Code Scaling

## Goal

Demonstrate that the native engine can perform a small, governed language and code training run with matched baselines, memory attribution, long-context checks, checkpoint replay, and code-safety controls.

## End-to-end objectives

1. Train CCT on declared language and code micro-corpora with immutable provenance.
2. Compare CCT against matched reference baselines using equal data, token budget, and evaluation splits.
3. Attribute memory and runtime costs to the recurrent, spectral, memory, and data components.
4. Measure behavior as context length and sequence size increase.
5. Verify checkpoint replay, deterministic restoration, code parsing, and unsafe-code boundaries.

## Implementation breakdown

| Workstream | Required implementation |
|---|---|
| Data | Governed fixtures, licensing or source records, deterministic splits, contamination controls, and manifests. |
| Training | Native optimizer, checkpoint, resume, finite-objective checks, and fixed-budget micro-runs. |
| Baselines | Matched simple recurrent, dense, or other declared references with the same evaluation contract. |
| Scaling | Context, memory, throughput, and parameter-budget matrix with resource accounting. |
| Code safety | Syntax-aware fixtures, malformed-code behavior, execution denial by default, and sandbox boundary. |

## Gate

The gate requires finite training, matched baseline results, reproducible checkpoint replay, memory attribution, long-context evidence, and code-safety failures that close correctly. A small improvement alone cannot pass the gate if provenance or safety fails.

## Transition

A Stage 5 `PASS` authorizes bounded deliberation and verification in Stage 6.

---

# Stage 6 — CCT Goal: Deliberation and Verification

## Goal

Add bounded planning, independent verification, evidence-aware abstention, deny-by-default tools, deterministic replay, interruption recovery, and incident logging.

## End-to-end objectives

1. Represent bounded plans with explicit steps, dependencies, preconditions, and expected results.
2. Execute independent verification rather than allowing the same reasoning path to self-certify.
3. Require evidence for claims and abstain when evidence is incomplete or contradictory.
4. Keep offline tools deny-by-default with explicit authorization and audit records.
5. Preserve replayability across interruption, retry, cancellation, and partial completion.

## Implementation breakdown

| Workstream | Required implementation |
|---|---|
| Planning | Typed plan, step identity, dependency checks, budget, timeout, and cancellation. |
| Verification | Independent verifier, evidence attachment, disagreement handling, and approval status. |
| Tool boundary | Capability registry, deny-by-default policy, input/output validation, and audit logging. |
| Replay | Deterministic plan replay, idempotent retries, interruption, and incident reconstruction. |
| Abstention | Unsupported plan, ambiguous result, missing evidence, and unsafe-action response. |

## Gate

The gate passes only when plans remain within budgets, independent verification detects injected errors, tools fail closed, evidence and abstention are correct, replay survives interruption, and incidents are reconstructable.

## Transition

A Stage 6 `PASS` authorizes controlled multimodal and open-ended research in Stage 7.

---

# Stage 7 — CCT Goal: Multimodal and Open-Ended Research

## Goal

Extend CCT with typed multimodal events and validated adapters while preserving alignment, masks, typed memory, deterministic simulation, transfer checks, auditability, and safety controls.

## End-to-end objectives

1. Define typed interfaces for the supported modalities and adapter lifecycle.
2. Implement the declared adapter set with explicit provenance, timestamps, masks, and failure behavior.
3. Fuse modalities without allowing invalid, missing, or future information to leak.
4. Store typed multimodal memory with source identity and deletion behavior.
5. Demonstrate deterministic simulation, transfer boundaries, audit records, and unsafe-input controls.

## Implementation breakdown

| Workstream | Required implementation |
|---|---|
| Typed events | Modality, shape, timestamp, source, confidence, and validity contract. |
| Adapters | Seven declared adapters, versioning, capability checks, and malformed-input failures. |
| Fusion | Mask-aware alignment, missing-modality behavior, temporal ordering, and conflict handling. |
| Memory | Typed storage, retrieval, provenance, retention, and deletion. |
| Research controls | Deterministic simulation, transfer evaluation, audit log, and safety review. |

## Gate

The terminal research gate requires all declared adapters to pass contract tests, alignment and masking to be correct, transfer and simulation results to be reproducible, and safety controls to remain fail-closed. Passing authorizes controlled continuation only; it does not authorize unrestricted deployment.

## Transition

A Stage 7 `PASS` authorizes the production NLP governance foundation in Stage 8.

---

# Stage 8 — CCT Goal: Production NLP Governance Foundation

## Goal

Create the governance registry, policy boundary, realistic application fixtures, artifact protocol, adversarial controls, and readiness evidence required before real production-oriented NLP work.

## End-to-end objectives

1. Register models, datasets, tools, policies, evaluators, and releases with immutable identities.
2. Define allowed, disallowed, review-required, and evaluator-only operations.
3. Test realistic application fixtures rather than toy-only interactions.
4. Produce reviewable artifacts that preserve provenance, configuration, environment, and decision history.
5. Exercise adversarial input, policy bypass, data leakage, and unsafe-operation controls.

## Implementation breakdown

| Workstream | Required implementation |
|---|---|
| Registry | Versioned identity for model, data, code, policy, evaluator, and artifact. |
| Policy | Capability boundary, approval path, deny-by-default actions, and escalation. |
| Application fixtures | Realistic bounded tasks, error paths, user-visible reports, and evaluator ownership. |
| Artifacts | Manifest, config, environment, tests, benchmarks, gate, logs, and report. |
| Adversarial controls | Prompt, data, tool, identity, leakage, and rollback challenges. |

## Gate

The gate requires policy and registry integrity, realistic application evidence, artifact completeness, adversarial resistance, and a documented readiness boundary. It authorizes governed data work, not external action or unrestricted deployment.

## Transition

A Stage 8 `PASS` authorizes governed real-source corpus work in Stage 9.

---

# Stage 9 — CCT Goal: Governed Data and Corpus

## Goal

Acquire and prepare real data through a native, reproducible, rights-aware, contamination-resistant corpus pipeline.

## End-to-end objectives

1. Pin every source, revision, split, license record, and acquisition URL.
2. Quarantine rights, privacy, malformed, unsafe, or disallowed records before training.
3. Detect exact and near duplicates and prevent contamination across train, validation, and test.
4. Build deterministic shards and replayable manifests without hidden source changes.
5. Support deletion, re-preparation, audit, and fail-closed missing-source behavior.

## Implementation breakdown

| Workstream | Required implementation |
|---|---|
| Acquisition | Native downloader, immutable revisions, retry and resume policy, and cache validation. |
| Governance | License, privacy, rights, safety, quarantine, and evaluator-only metadata. |
| Deduplication | Exact digest, near-duplicate policy, cross-split contamination barrier, and report. |
| Sharding | Stable ordering, seed, shard identity, counts, digests, and replay. |
| Deletion | Source removal, rebuild, stale-artifact detection, and audit record. |

## Gate

The gate passes only when source identity, rights and quarantine rules, contamination barriers, shard replay, deletion, and audit evidence all pass with zero unexplained rows or overlaps.

## Transition

A Stage 9 `PASS` authorizes tokenizer and representation work in Stage 10.

---

# Stage 10 — CCT Goal: Tokenizer and Representation

## Goal

Create an immutable native tokenizer and representation contract that preserves offsets, byte fallback, provenance, packed and padded causal batches, and measurable efficiency.

## End-to-end objectives

1. Implement and compare the declared byte, subword, and hybrid tokenizer candidates.
2. Freeze one selected vocabulary and tokenizer snapshot with a stable hash.
3. Preserve exact source-to-token offsets, including Unicode and malformed-input behavior.
4. Guarantee byte fallback coverage for all supported input bytes.
5. Build packed and padded causal batches with deterministic masks and boundary metadata.
6. Measure token efficiency, memory, throughput, and decode or round-trip behavior.

## Implementation breakdown

| Workstream | Required implementation |
|---|---|
| Candidates | Native candidate builders, training records, vocabulary identity, and comparison report. |
| Snapshot | Immutable serialization, hash, version, vocabulary ordering, and loader validation. |
| Offsets | Unicode-safe source spans, byte fallback, round-trip tests, and invalid-UTF-8 policy. |
| Batching | Packed and padded causal sequences, padding masks, boundary masks, and record IDs. |
| Efficiency | Token count, source bytes, memory, throughput, and quality comparison. |

## Gate

The gate requires tokenizer snapshot identity, offset correctness, byte coverage, deterministic round trips, valid batches, and an explicit efficiency comparison. A tokenizer change invalidates downstream dataset and checkpoint identities.

## Transition

A Stage 10 `PASS` authorizes the trainable native NLP core in Stage 11.

---

# Stage 11 — CCT Goal: Trainable Native NLP Core

## Goal

Make the CCT language engine trainable through a native categorical next-token objective with analytic recurrence gradients, optimizer safety, checkpoint recovery, real-source pilot evidence, and matched controls.

## End-to-end objectives

1. Define the next-token objective, causal masks, sequence contract, and loss accounting.
2. Implement analytic gradients for the CCT recurrence and compare them against an independent numerical check.
3. Implement stable optimization, clipping, scheduling, finite checks, and deterministic initialization.
4. Save and reload checkpoints with tokenizer, dataset, configuration, optimizer, step, and cursor identity.
5. Run a real-source pilot and compare against declared matched controls.
6. Measure validation loss, perplexity, token accuracy, throughput, memory, and parameter count.

## Implementation breakdown

| Workstream | Required implementation |
|---|---|
| Objective | Masked categorical loss, target validation, token accounting, and finite behavior. |
| Model | CCT recurrence, parameter layout, forward path, analytic backward path, and model identity. |
| Optimizer | Learning-rate schedule, moments, clipping, weight decay, state serialization, and replay. |
| Dataset | Tokenizer-encoded documents, fixed context, eligibility flags, hashes, and split isolation. |
| Evaluation | Initial versus final validation, held-out test, seed study, baseline comparison, and report. |
| Recovery | Checkpoint save, exact load, wrong-identity rejection, corruption failure, and resume equivalence. |

## Gate

The gate requires finite objectives, analytic-gradient agreement, deterministic multi-seed behavior, held-out improvement or an explicitly explained result, checkpoint identity and recovery, real-source evidence, matched controls, and complete artifacts.

## Transition

A Stage 11 `PASS` authorizes scaling and accelerator systems in Stage 12.

---

# Stage 12 — CCT Goal: Scaling and Accelerator Systems

## Goal

Scale the native training path through reference and fused CPU implementations, resource profiling, parity checks, ordered worker behavior, atomic checkpoint recovery, and an evidence-based backend decision.

## End-to-end objectives

1. Preserve a simple CPU reference implementation as the correctness oracle.
2. Implement the optimized or fused path without changing numerical results beyond declared tolerances.
3. Measure the complete scaling matrix across context, batch, model, and data budgets.
4. Profile memory, throughput, latency, allocations, and failure behavior.
5. Verify ordered worker equivalence, atomic recovery, and corrupt-checkpoint rejection.
6. Record absent accelerator capabilities honestly rather than fabricating backend support.

## Implementation breakdown

| Workstream | Required implementation |
|---|---|
| Reference | Clear deterministic CPU path with baseline metrics and assertions. |
| Fused path | Optimized kernels or operators, parity harness, and fallback behavior. |
| Scaling matrix | Declared configurations, resource limits, timing, memory, and failure thresholds. |
| Parallelism | Ordered work, seed partitioning, worker-equivalence tests, and interruption handling. |
| Recovery | Atomic checkpoint, partial-write simulation, corruption rejection, and committed-state replay. |
| Decision | Backend selection, unsupported-platform record, and architecture decision log. |

## Gate

The gate requires reference/fused parity, finite and resource-accounted scaling results, worker equivalence, atomic recovery, corruption failure, and an honest backend decision.

## Transition

A Stage 12 `PASS` authorizes supervised fine-tuning and adapter work in Stage 13.

---

# Stage 13 — CCT Goal: Supervised Fine-Tuning and Adapters

## Goal

Turn the trainable CCT core into a bounded task-adaptation system with explicit instruction formatting, target-only loss masks, full and low-rank adaptation, structured outputs, citation behavior, safety retention, permissions, and deletion lineage.

## End-to-end objectives

1. Define task schemas, example identity, input and target provenance, eligibility, and evaluator ownership.
2. Implement deterministic instruction formatting and explicit target-span-only loss masks.
3. Compare full-parameter fine-tuning against parameter-efficient adapters.
4. Validate structured outputs, grounded citations, missing-evidence behavior, and safe refusals.
5. Enforce adapter authorization, base immutability, merge equivalence, deletion, and lineage.
6. Evaluate representative tasks with held-out data and regression against the base model.

## Implementation breakdown

| Workstream | Required implementation |
|---|---|
| Schema | Six declared task types, output contract, labels or structure, bounds, and policy class. |
| Formatter | Canonical instruction serialization, token offsets, target-only masks, and malformed-target rejection. |
| Training | Full SFT, adapter gradients, optimizer, checkpoint, merge, and runtime loading. |
| Validation | Structure, citations, abstention, safety, calibration, task quality, and base-retention checks. |
| Governance | Permissions, base hash, training-manifest hash, deletion lineage, and authorization registry. |

## Gate

The gate requires representative task improvement, target-only mask correctness, structured validation, grounded citation behavior, unsafe-request denial, adapter authorization, base immutability, merge/runtime agreement, deletion, and identity-linked artifacts.

## Transition

A Stage 13 `PASS` authorizes preference tuning and alignment in Stage 14.

---

# Stage 14 — CCT Goal: Preference Tuning and Alignment

## Goal

Improve controllability, helpfulness, refusal quality, citation behavior, output style, calibration, and task quality using governed preference evidence without unacceptable truthfulness, safety, or regression damage.

## End-to-end objectives

1. Acquire and govern preference records with annotator, task, provenance, and split identity.
2. Compare preference optimization candidates rather than assuming one method is universally reliable.
3. Use verifier-weighted reranking or equivalent controls where declared.
4. Measure helpfulness, truthfulness, calibration, refusal quality, citation integrity, and task performance.
5. Run adversarial and blind evaluations with regression against the Stage 13 baseline.
6. Preserve deletion, replay, checkpoint, and approval records.

## Implementation breakdown

| Workstream | Required implementation |
|---|---|
| Preference data | Pair or ranking schema, provenance, annotator metadata, conflict handling, and quarantine. |
| Optimization | Candidate methods, hyperparameters, identity, rollback, and reproducible training. |
| Verification | Independent quality checks, reranking, disagreement, and evidence requirements. |
| Evaluation | Blind review, adversarial tests, calibration, safety, truthfulness, and task regression. |
| Operations | Checkpoint lineage, deletion, approvals, and decision record. |

## Gate

The gate passes only when the selected method improves declared behavior without unacceptable regressions in truthfulness, safety, calibration, task quality, or operations.

## Transition

A Stage 14 `PASS` authorizes verified retrieval and knowledge-plane work in Stage 15.

---

# Stage 15 — CCT Goal: Verified Retrieval and Knowledge

## Goal

Provide typed retrieval, freshness tracking, citations, conflict handling, deletion, poisoning isolation, auditability, and verified grounding for knowledge-dependent outputs.

## End-to-end objectives

1. Define typed knowledge records, provenance, source authority, freshness, and version identity.
2. Retrieve evidence deterministically with query, ranking, and citation records.
3. Distinguish fresh, stale, conflicting, missing, and deleted knowledge.
4. Prevent poisoned or unauthorized records from influencing trusted outputs.
5. Require grounded answers or bounded abstention when evidence is insufficient.
6. Audit retrieval, use, deletion, refresh, and conflict resolution.

## Implementation breakdown

| Workstream | Required implementation |
|---|---|
| Knowledge records | Typed schema, source, version, timestamp, authority, digest, and retention. |
| Retrieval | Deterministic index, typed query, ranking, evidence bundle, and citation identity. |
| Freshness | Refresh policy, stale boundary, conflicting-source behavior, and update replay. |
| Safety | Poisoning quarantine, unauthorized-source rejection, and unsupported-answer abstention. |
| Audit | Retrieval trace, citation trace, deletion trace, and reviewable report. |

## Gate

The gate requires correct citations, freshness and conflict behavior, deletion, poisoning isolation, deterministic retrieval, audit completeness, and grounded or abstaining outputs.

## Transition

A Stage 15 `PASS` authorizes production inference and operations in Stage 16.

---

# Stage 16 — CCT Goal: Production Inference and Operations

## Goal

Turn an approved model and knowledge candidate into a production-like native inference service with versioned APIs, batching, state and cache isolation, observability, SLOs, fault controls, canaries, and rollback.

## End-to-end objectives

1. Expose a versioned native API with request, response, error, and compatibility contracts.
2. Implement scheduling, batching, streaming, recurrent-state lifecycle, and retrieval integration.
3. Isolate user, request, model, adapter, retrieval, and cache state.
4. Measure latency, throughput, resource use, saturation, and declared SLOs.
5. Detect faults, enforce quotas and policy checks, and preserve incident records.
6. Support canary release, rollback, checkpoint pinning, and deletion or retirement.

## Implementation breakdown

| Workstream | Required implementation |
|---|---|
| API | Versioned request/response schema, error contract, compatibility, and authentication boundary. |
| Runtime | Batching, streaming, state lifecycle, cache isolation, retrieval, and policy enforcement. |
| Observability | Metrics, traces, logs, SLOs, saturation, drift signals, and privacy controls. |
| Reliability | Fault injection, timeout, retry, circuit control, quota, rollback, and recovery. |
| Release | Version registry, canary, approval, rollback, deletion, and incident workflow. |

## Gate

The gate requires API correctness, state/cache isolation, observability, SLO evidence, fault controls, canary behavior, rollback, and deletion or retirement behavior.

## Transition

A Stage 16 `PASS` authorizes only the controlled pilot and bounded release in Stage 17.

---

# Stage 17 — CCT Goal: Controlled Pilot and Production Release

## Goal

Move a narrowly defined CCT-NLP use case through shadow evaluation, bounded pilot, human oversight, incident response, rollback, deletion, drift review, and release approval.

## End-to-end objectives

1. Freeze the named scope, model, data, policy, evaluator, runtime, and release configuration.
2. Run shadow evaluation without external side effects and record disagreement or failure.
3. Operate a bounded pilot with explicit users, quotas, permissions, and human oversight.
4. Exercise incident response, rollback, deletion, drift detection, and release review.
5. Preserve user-visible limitations, evaluator records, and terminal decision artifacts.
6. Keep training authorization, external agency, and scope expansion closed unless separately approved.

## Implementation breakdown

| Workstream | Required implementation |
|---|---|
| Shadow | Offline or shadow traffic, reference comparison, error taxonomy, and evaluator sign-off. |
| Pilot | Named use case, users, quotas, human review, support boundary, and activity logs. |
| Safety | Policy checks, refusal, escalation, incident response, rollback, and deletion. |
| Drift | Data, quality, calibration, latency, and behavior drift signals with thresholds. |
| Release | Approval record, release note, known limitations, scope, expiry, and terminal gate. |

## Gate

The terminal gate passes only when shadow evidence, bounded pilot operation, human oversight, incident and rollback exercises, deletion, drift review, approvals, and artifacts are complete. A `PASS` authorizes only the named bounded release. It does not authorize general autonomy, unrestricted deployment, or automatic expansion.

## Transition

Stage 17 has no automatic successor. Any expanded capability, different domain, materially changed model, new external action, or broader deployment requires a new CCT specification and a new gate.

---

# Current CCT Training Milestone — Track 1

Track 1 is an internal training milestone supporting the Stage 11–13 training path. It is not a replacement for any numbered stage and does not change the stage transition rules.

## Goal

Prepare and train CCT on a small, governed real-data bundle using WikiText-2 for pretraining and SQuAD 2.0 for supervised target learning and frozen evaluation.

## Objectives

1. Acquire pinned real sources through native C++20 paths with resumable caching and fail-closed validation.
2. Create deterministic pretraining, supervised-training, supervised-selection, and frozen-final-test artifacts.
3. Preserve answerability, Unicode-safe answer spans, split isolation, source manifests, licenses, and digests.
4. Run native CCT pretraining and `target-span-only-v1` SQuAD supervised continuation tuning.
5. Save pretraining and supervised checkpoints with tokenizer and dataset identity.
6. Evaluate validation and frozen final-test target-token metrics without claiming answer exact-match or F1 until constrained answer decoding exists.

## Required transition evidence

Track 1 is complete only when the preparation report passes, unit tests and formal gate pass, the complete native test suite remains green, the real corpus training runner completes, checkpoint files reload under the declared identity, the frozen test set is not used for updates or selection, and the training report clearly states its evaluation scope.

## Current implementation command

```bash
make track1-train
```

The command produces native CCT pretraining and SFT checkpoints and a JSON training report under `artifacts/track1/training/`. The default first pass is intentionally bounded and must be expanded only with measured resource and quality evidence.

---

# CCT Completion Standard

The CCT program is complete for a declared scope only when the corresponding stage chain is green, the release artifacts are reviewable, the data and model identities are reproducible, known limitations are explicit, and no capability is claimed beyond its evaluation contract.

The engine must be improved by evidence: stronger real-data runs, matched baselines, ablations, scaling measurements, exact task metrics, robustness, recovery, and operational behavior. When a result fails, the correct response is to preserve the failure, add the missing test or implementation, remediate, and rerun the gate—not to weaken the criterion or relabel the result.

## Internal references

- [CCT stage specifications](Stages/README.md)
- [CCT architecture](Architecture.md)
- [CCT evolution proposal](CCT_EVOLUTION_PROPOSAL.md)
- [Track 1 operational guide](artifacts/track1/README.md)
- [Track 1 training report](artifacts/track1/real-training/training_report.json)
