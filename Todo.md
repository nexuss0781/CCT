# CCT Todo

## Purpose and compliance

This file is the executable task plan derived from [`Goal.md`](Goal.md). `Goal.md` is the canonical specification of what CCT must accomplish. `Todo.md` converts each goal into implementation, verification, artifact, and transition tasks.

A task is complete only when its code, realistic tests, formal gate, artifacts, and transition evidence are complete. Smoke tests support development but do not close a gate. No task may weaken a criterion, hide a failure, or claim behavior outside its measured evaluation contract.

## Global CCT tasks

- [x] Keep implementation native C++20 with strict compilation and reproducible build configuration.
- [x] Keep implementation, regression tests, formal gates, artifacts, and transition records separate.
- [x] Record commit, configuration, source or environment manifest, seeds, hardware, software, tests, benchmarks, thresholds, known failures, artifact hashes, and final status.
- [x] Use only `PASS`, `FAIL`, and explicitly permitted `BLOCKED` statuses.
- [x] Preserve deterministic replay and fail closed on missing, malformed, corrupt, incompatible, or non-finite inputs.
- [x] Require baseline or ablation evidence before claiming a component advantage.
- [x] Invalidate and rerun a gate after an affecting implementation change.
- [x] Keep artifacts reviewable and exclude secrets, private source material, hidden test data, and evaluator-only information.

## Stage 0 — Reproducible Baseline

**Status:** [x] Implemented and gated.

### Implementation

- [x] Freeze compiler, CMake, language-standard, warning, dependency, operating-system, and build contracts.
- [x] Maintain a clean native C++20 build with strict warnings and warnings-as-errors.
- [x] Provide deterministic unit, integration, and benchmark harnesses with explicit seeds.
- [x] Define artifact directories, report schemas, status values, and release-record fields.
- [x] Add missing-dependency, malformed-configuration, non-finite-output, and replay failure paths.

### Gate and transition

- [x] Verify clean checkout build, repeated test identity, benchmark schema, and commit/configuration identity.
- [x] Record Stage 0 `PASS`.
- [x] Authorize Stage 1 numerical implementation only after all mandatory checks pass.

## Stage 1 — Differentiable Numerical Engine

**Status:** [x] Implemented and gated.

### Implementation

- [x] Define tensor, field, shape, layout, dtype, ownership, and lifetime contracts.
- [x] Implement required numerical operators and deterministic boundary behavior.
- [x] Implement analytic gradients for every operator used by the sequence core.
- [x] Add finite-difference or independent-reference gradient verification.
- [x] Add overflow, underflow, NaN, infinity, invalid-shape, aliasing, and mutation checks.

### Gate and transition

- [x] Verify forward correctness, gradient tolerance, deterministic replay, finite outputs, and invalid-input failures.
- [x] Record Stage 1 `PASS`.
- [x] Authorize Stage 2 sequence-core implementation.

## Stage 2 — Efficient Sequence Core

**Status:** [x] Implemented and gated.

### Implementation

- [x] Define sequence state, update order, reset semantics, and causal boundaries.
- [x] Implement recurrent state updates on the Stage 1 substrate.
- [x] Implement chunked and streaming execution with bounded state memory.
- [x] Expose state ownership, reset, serialization, and versioning.
- [x] Measure throughput, latency, memory, and numerical stability.
- [x] Add invalid-state, wrong-shape, out-of-order, reset, and non-finite failures.

### Gate and transition

- [x] Verify full-sequence/streaming equivalence, deterministic recovery, resource bounds, and fail-closed transitions.
- [x] Record Stage 2 `PASS`.
- [x] Authorize Stage 3 causal-event learning.

## Stage 3 — Causal Event Learning

**Status:** [x] Implemented and gated.

### Implementation

- [x] Add stable event identity, order, source metadata, and causal relationships.
- [x] Validate causal dependency structures without future or evaluator leakage.
- [x] Implement intervention and counterfactual pathways with provenance.
- [x] Add reorder, omission, irrelevant-event, contradiction, and perturbation tests.
- [x] Implement confidence and abstention for unsupported or ambiguous conclusions.

### Gate and transition

- [x] Verify event identity, causal ordering, leakage audit, intervention, counterfactuals, robustness, and abstention.
- [x] Verify the expanded gate contract.
- [x] Record Stage 3 `PASS` and approval boundary.
- [x] Authorize Stage 4 only within its documented scope.

## Stage 4 — Persistent Verifiable Memory

**Status:** [x] Implemented and gated.

### Implementation

- [x] Implement checksummed records, versions, atomic commits, indexes, and durable metadata.
- [x] Implement exact retrieval with provenance and citation metadata.
- [x] Implement deletion, retention, expiry, conflict resolution, and deterministic rebuild.
- [x] Isolate or reject poisoned, malformed, duplicated, and unauthorized records.
- [x] Test partial writes, corruption, interruption, and last-committed-state recovery.

### Gate and transition

- [x] Verify exact retrieval, provenance, deletion, retention, conflicts, poisoning isolation, rebuild, and recovery.
- [x] Verify the expanded gate contract.
- [x] Record Stage 4 `PASS` and approval boundary.
- [x] Authorize Stage 5 language and code scaling.

## Stage 5 — Language and Code Scaling

**Status:** [x] Implemented and gated.

### Implementation

- [x] Run governed native language and code micro-corpora.
- [x] Establish matched baselines with equal data, token budgets, and evaluation splits.
- [x] Attribute memory and runtime to recurrent, spectral, memory, and data components.
- [x] Measure context-length and sequence-size behavior.
- [x] Verify checkpoint replay and deterministic restoration.
- [x] Add syntax-aware code fixtures and deny-by-default execution boundaries.

### Gate and transition

- [x] Verify finite training, provenance, matched baselines, memory attribution, long-context behavior, checkpoint replay, and code-safety failures.
- [x] Verify the expanded gate contract.
- [x] Record Stage 5 `PASS` and approval boundary.
- [x] Authorize Stage 6 deliberation and verification.

## Stage 6 — Deliberation and Verification

**Status:** [x] Implemented and gated.

### Implementation

- [x] Define typed plans, dependencies, preconditions, expected results, budgets, and timeouts.
- [x] Implement independent verification and disagreement handling.
- [x] Require evidence and bounded abstention for unsupported conclusions.
- [x] Keep tools deny-by-default with explicit authorization and audit records.
- [x] Implement deterministic replay, idempotent retries, cancellation, and incident reconstruction.

### Gate and transition

- [x] Verify plan budgets, injected-error detection, evidence, abstention, tool denial, replay, interruption, and incident logging.
- [x] Verify the expanded gate contract.
- [x] Record Stage 6 `PASS` and approval boundary.
- [x] Authorize Stage 7 controlled multimodal research.

## Stage 7 — Multimodal and Open-Ended Research

**Status:** [x] Implemented and gated.

### Implementation

- [x] Define typed interfaces for all declared modalities and adapter lifecycle.
- [x] Implement the declared adapter set with provenance, timestamps, masks, and failure behavior.
- [x] Implement mask-aware fusion with missing and invalid modality behavior.
- [x] Store typed multimodal memory with source identity and deletion behavior.
- [x] Add deterministic simulation, transfer checks, audit records, and unsafe-input controls.

### Gate and transition

- [x] Verify every adapter, alignment, masks, temporal ordering, conflicts, typed memory, deletion, simulation, transfer, audit, and safety.
- [x] Verify the expanded terminal research gate.
- [x] Record controlled-continuation `PASS`.
- [x] Authorize Stage 8 governance foundation only within controlled scope.

## Stage 8 — Production NLP Governance Foundation

**Status:** [x] Implemented and gated.

### Implementation

- [x] Register models, datasets, tools, policies, evaluators, and releases with immutable identities.
- [x] Define allowed, disallowed, review-required, and evaluator-only operations.
- [x] Create realistic bounded application fixtures and user-visible failure reports.
- [x] Implement complete reviewable artifact protocol.
- [x] Test adversarial inputs, policy bypass, leakage, and unsafe-operation controls.

### Gate and transition

- [x] Verify registry and policy integrity, realistic evidence, artifact completeness, adversarial controls, and readiness boundary.
- [x] Record governance-only `PASS`.
- [x] Authorize Stage 9 governed real-source data work.

## Stage 9 — Governed Data and Corpus

**Status:** [x] Implemented and gated.

### Implementation

- [x] Pin sources, revisions, splits, license records, and acquisition URLs.
- [x] Quarantine rights, privacy, malformed, unsafe, and disallowed records.
- [x] Detect exact and near duplicates and prevent cross-split contamination.
- [x] Build deterministic shards with stable ordering, seeds, counts, and digests.
- [x] Implement deletion, re-preparation, stale-artifact detection, and audit.

### Gate and transition

- [x] Verify source identity, rights/quarantine rules, contamination barriers, deterministic replay, deletion, and audit.
- [x] Verify zero unexplained rows and overlaps.
- [x] Record Stage 9 `PASS`.
- [x] Authorize Stage 10 tokenizer and representation work.

## Stage 10 — Tokenizer and Representation

**Status:** [x] Implemented and gated.

### Implementation

- [x] Implement and compare declared byte, subword, and hybrid tokenizer candidates.
- [x] Freeze the selected vocabulary and tokenizer snapshot with stable hash.
- [x] Preserve Unicode-safe source-to-token offsets and invalid-input policy.
- [x] Guarantee byte fallback for the supported byte range.
- [x] Build packed and padded causal batches with masks and boundary metadata.
- [x] Measure token efficiency, memory, throughput, and round-trip behavior.

### Gate and transition

- [x] Verify snapshot identity, offsets, byte coverage, round trips, batches, and efficiency comparison.
- [x] Record Stage 10 `PASS` and tokenizer identity.
- [x] Authorize Stage 11 trainable native NLP core.

## Stage 11 — Trainable Native NLP Core

**Status:** [x] Implemented and gated.

### Implementation

- [x] Define categorical next-token objective, causal masks, and token accounting.
- [x] Implement analytic recurrence gradients and independent numerical verification.
- [x] Implement optimizer scheduling, clipping, finite checks, and deterministic initialization.
- [x] Implement checkpoint identity for tokenizer, dataset, model, optimizer, step, and cursor.
- [x] Run real-source pilots and matched controls.
- [x] Measure validation loss, perplexity, token accuracy, throughput, memory, and parameters.

### Gate and transition

- [x] Verify finite objective and analytic-gradient agreement.
- [x] Verify deterministic multi-seed behavior and held-out results.
- [x] Verify matched controls and exact checkpoint save/load/resume.
- [x] Verify wrong-identity rejection, corruption failure, and complete artifacts.
- [x] Record Stage 11 `PASS`.
- [x] Authorize Stage 12 scaling and accelerator systems.

## Stage 12 — Scaling and Accelerator Systems

**Status:** [x] Implemented and gated.

### Implementation

- [x] Preserve a deterministic CPU reference correctness oracle.
- [x] Implement optimized or fused paths with numerical parity checks.
- [x] Run the declared scaling matrix across context, batch, model, and data budgets.
- [x] Profile memory, throughput, latency, allocations, and failure behavior.
- [x] Verify ordered worker equivalence, interruption handling, atomic recovery, and corruption rejection.
- [x] Record an evidence-based backend decision without fabricating unsupported backends.

### Gate and transition

- [x] Verify reference/fused parity, finite resource-accounted scaling, worker equivalence, recovery, and corruption failure.
- [x] Record Stage 12 `PASS`.
- [x] Authorize Stage 13 supervised fine-tuning and adapters.

## Stage 13 — Supervised Fine-Tuning and Adapters

**Status:** [x] Implemented and gated.

### Implementation

- [x] Define task schemas, example identity, provenance, eligibility, and evaluator ownership.
- [x] Implement deterministic formatting and `target-span-only-v1` masks.
- [x] Implement and compare full-parameter and low-rank adaptation.
- [x] Validate structured outputs, grounded citations, missing-evidence behavior, and safe refusals.
- [x] Enforce adapter authorization, base immutability, merge equivalence, deletion, and lineage.
- [x] Evaluate representative tasks against the base model.

### Gate and transition

- [x] Verify task improvement, target-only masks, structured validation, citations, abstention, and safety retention.
- [x] Verify permissions, base immutability, merge/runtime agreement, deletion, and identity-linked artifacts.
- [x] Record Stage 13 `PASS`.
- [x] Authorize Stage 14 preference tuning and alignment.

## Stage 14 — Preference Tuning and Alignment

**Status:** [x] Implemented and gated.

### Implementation

- [x] Govern preference records with task, provenance, annotator, and split identity.
- [x] Compare preference optimization candidates.
- [x] Implement verifier-weighted reranking or the declared alternative.
- [x] Measure helpfulness, truthfulness, calibration, refusal quality, citation integrity, and task quality.
- [x] Run adversarial and blind evaluations against the prior baseline.
- [x] Preserve deletion, replay, checkpoint, and approval records.

### Gate and transition

- [x] Verify declared improvement without unacceptable regression.
- [x] Verify truthfulness, safety, calibration, task, operations, adversarial, and blind evidence.
- [x] Verify checkpoint and approval lineage.
- [x] Record Stage 14 `PASS`.
- [x] Authorize Stage 15 verified retrieval and knowledge.

## Stage 15 — Verified Retrieval and Knowledge

**Status:** [x] Implemented and gated.

### Implementation

- [x] Define typed knowledge records, provenance, source authority, freshness, and version identity.
- [x] Implement deterministic retrieval with query, ranking, evidence, and citation records.
- [x] Distinguish fresh, stale, conflicting, missing, and deleted knowledge.
- [x] Isolate poisoned or unauthorized records.
- [x] Require grounded responses or bounded abstention when evidence is insufficient.
- [x] Audit retrieval, use, deletion, refresh, and conflicts.

### Gate and transition

- [x] Verify citations, freshness, conflicts, deterministic retrieval, deletion, poisoning isolation, grounding, and abstention.
- [x] Verify audit completeness.
- [x] Record Stage 15 `PASS`.
- [x] Authorize Stage 16 production inference and operations.

## Stage 16 — Production Inference and Operations

**Status:** [x] Implemented and gated.

### Implementation

- [x] Expose versioned native request, response, error, and compatibility APIs.
- [x] Implement batching, streaming, state lifecycle, retrieval integration, and policy enforcement.
- [x] Isolate request, user, model, adapter, retrieval, and cache state.
- [x] Measure latency, throughput, resource use, saturation, and SLOs.
- [x] Implement fault detection, quotas, incident records, canary release, rollback, and deletion.

### Gate and transition

- [x] Verify API and state/cache isolation.
- [x] Verify observability, SLO evidence, saturation, fault controls, canary, rollback, and deletion.
- [x] Record Stage 16 `PASS`.
- [x] Authorize Stage 17 controlled pilot and bounded release.

## Stage 17 — Controlled Pilot and Production Release

**Status:** [x] Implemented and gated for the named bounded scope.

### Implementation

- [x] Freeze named scope, model, data, policy, evaluator, runtime, and release configuration.
- [x] Run shadow evaluation without external side effects.
- [x] Operate a bounded pilot with quotas, permissions, human oversight, and activity logs.
- [x] Exercise incident response, rollback, deletion, drift review, and release approval.
- [x] Preserve limitations, evaluator records, and terminal decision artifacts.
- [x] Keep scope expansion and new external actions closed without a new specification.

### Gate and transition

- [x] Verify shadow evidence and disagreement records.
- [x] Verify bounded pilot controls and human oversight.
- [x] Verify incident, rollback, deletion, and drift exercises.
- [x] Verify approvals, known limitations, and complete artifacts.
- [x] Record terminal bounded-release `PASS`.
- [x] Close the automatic stage chain.
- [x] Require a new CCT specification for expanded capability, different domain, materially changed model, new external action, or broader deployment.

## Current CCT Training Milestone — Track 1

**Status:** [x] Preparation, native gates, and bounded complete-corpus training handoff implemented and verified.

### Data preparation

- [x] Pin sources, revisions, splits, licenses, and acquisition URLs.
- [x] Acquire WikiText-2 through the pinned direct archive path.
- [x] Acquire SQuAD 2.0 through pinned direct GEM JSON files with upstream provenance.
- [x] Implement resumable caching, retries, source-window controls, and fail-closed cache validation.
- [x] Validate Unicode-safe answer offsets, including non-BMP surrogate pairs.
- [x] Build deterministic pretraining, SFT, selection, and frozen-final-test artifacts.
- [x] Verify balanced answerability selection and zero final-test overlap.

### Native training

- [x] Add the native C++20 Track 1 training runner.
- [x] Train CCT on WikiText-2 preparation artifacts.
- [x] Apply `target-span-only-v1` SQuAD supervision.
- [x] Keep frozen final-test data out of updates and checkpoint selection.
- [x] Save pretraining and SFT checkpoints with tokenizer and dataset identity.
- [x] Emit finite validation, held-out, and frozen-test target-token metrics.

### Track 1 gate

- [x] Pass Track 1 unit tests.
- [x] Pass the formal Track 1 gate.
- [x] Pass the full native CTest suite.
- [x] Pass cumulative `make ci-track1`.
- [x] Run complete prepared-corpus bounded training verification.
- [ ] Implement constrained answer decoding and exact-match/F1 evaluation before claiming those metrics.
- [ ] Run longer-budget training with recorded resource and quality evidence before treating it as a larger competence result.

## CCT maintenance and completion

- [x] Keep every stage specification and this task plan consistent with `Goal.md`.
- [x] Preserve source, model, tokenizer, dataset, checkpoint, and release identities.
- [x] Keep explicit known limitations in every report.
- [ ] Add or update matched baselines and component ablations whenever architecture claims change.
- [ ] Rerun affected gates after any gated implementation change.
- [ ] Expand training budgets only after resource, stability, and held-out quality measurements are recorded.
- [ ] Create a new specification before any post-Stage-17 capability or scope expansion.

## Final completion rule

The Todo is complete for a declared scope only when every mandatory task in the corresponding `Goal.md` section is checked, the associated gate is `PASS`, artifacts are reviewable, and the transition record is valid. Open tasks remain open when the implementation does not yet provide the required evidence; they must not be silently marked complete.
