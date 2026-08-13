# CCT Level 1 Goal Specification

## Document status

**Specification:** CCT Level 1 — New Architecture and Teacher Engine

**Repository:** CCT-ASE

**Implementation language:** Native C++20, with native CUDA only where an explicitly gated accelerator path is available.

**Status:** Active Level 1 specification. A stage is not complete because source code exists; it is complete only after its implementation contract, realistic tests, failure paths, artifacts, and gate record pass from a declared commit.

## 1. Level 1 mission

The CCT Level 1 mission is to create a new, efficient, trainable AI architecture that can operate locally, learn language from governed data, accept supervised instruction, learn declared internal operations and API schemas, and function as a reliable teacher engine for a later downstream learning system.

CCT is not specified here as a complete downstream system, operating system, or unrestricted intelligence. Level 1 is the **engine stage**: it establishes the model architecture, training behavior, language capability, instruction interface, tool or API representation, reproducibility, and release discipline required before downstream integration is permitted.

> **CCT Level 1 end state:** a native, locally runnable, reproducible language-and-teaching engine whose measured behavior is sufficient for a downstream system to receive instructions, demonstrations, corrections, structured operation knowledge, and bounded evaluation feedback.

## 2. Delegated CCT role

CCT is delegated the **Level 1 teacher-engine role**. It must first become a capable language model through controlled training, then learn the declared instruction, operation, and API contracts through supervised adaptation. Its role is to communicate, demonstrate, explain, validate, and correct within the interfaces that Level 1 actually measures.

The delegated role has five boundaries:

| Boundary | CCT Level 1 requirement |
|---|---|
| Language | CCT must learn and evaluate language behavior from governed data. |
| Instruction | CCT must accept a canonical instruction format and produce a valid target response. |
| Operations | CCT must represent declared operation and API schemas accurately and reject unsupported forms. |
| Teaching | CCT must provide bounded demonstrations, explanations, corrections, and evidence-linked judgments. |
| Reliability | CCT must be reproducible, checkpointable, auditable, and fail closed outside its declared contract. |

The delegated role is deliberately narrower than a general autonomous system. Any capability beyond the measured language, instruction, operation, teaching, safety, and reliability contracts requires a new specification or an approved successor specification.

## 3. End-to-end Level 1 objective

> **Build, train, evaluate, and release CCT as an efficient native language-teacher engine: a distinct architecture that learns language, follows instruction, represents declared operations and APIs, produces bounded demonstrations and corrections, and remains reproducible and recoverable on local hardware.**

The complete Level 1 execution path is:

```text
Architecture contract
    → numerical correctness
    → causal sequence engine
    → state and event discipline
    → tokenizer and representation lock
    → native language training
    → real Track 1 pretraining
    → supervised instruction adaptation
    → operation and API adaptation
    → teaching-behavior evaluation
    → checkpointed local release
```

## 4. Level 1 goals

| Goal ID | CCT goal | Completion meaning |
|---|---|---|
| G1 | Establish a distinct architecture | CCT has a written model contract, native implementation, parameter identity, state identity, and independently testable operators. |
| G2 | Achieve numerical correctness | Forward and backward behavior is finite, deterministic, and verified against independent references or numerical checks. |
| G3 | Provide efficient sequence learning | Ordered and streaming execution preserve causal behavior while respecting declared memory and latency limits. |
| G4 | Preserve state and event discipline | CCT tracks state, event order, provenance, and boundaries without future leakage or silent mutation. |
| G5 | Freeze representation identity | Tokenizer, vocabulary, offsets, masks, and batch formats are immutable and identity-linked to data and checkpoints. |
| G6 | Demonstrate real language learning | CCT trains on governed language data and improves or explains its held-out result under a declared budget. |
| G7 | Demonstrate supervised instruction learning | CCT learns canonical instruction-to-target behavior with explicit target-only loss and held-out evaluation. |
| G8 | Learn declared operations and APIs | CCT can serialize, explain, validate, and reject declared operation calls according to fixed schemas. |
| G9 | Demonstrate bounded teaching behavior | CCT can provide a demonstration, explain a result, identify an error, and issue an evidence-linked correction within scope. |
| G10 | Release a reliable local engine | CCT checkpoints, resumes, replays, reports, and fails closed under identity, corruption, safety, and resource failures. |

## 5. Global Level 1 acceptance rules

Every CCT Level 1 stage must produce the following evidence before transition:

1. A stage-specific implementation contract and configuration.
2. Native source code with strict compilation under the declared toolchain.
3. Unit and integration tests for the normal path and realistic failure paths.
4. A formal gate executable or equivalent deterministic evaluator.
5. A manifest containing commit, configuration, seed, data or environment identity, software versions, hardware context, and artifact hashes.
6. A human-readable report explaining results, thresholds, limitations, and known failures.
7. A transition record that names the successor scope and any required approval.

The only allowed gate outcomes are `PASS`, `FAIL`, and `BLOCKED`. `BLOCKED` is valid only for an explicitly optional dependency. A missing mandatory test, missing artifact, unresolved discrepancy, non-finite metric, contaminated split, invalid checkpoint, or unsupported claim is a gate failure.

A smoke run can guide implementation but cannot authorize transition. Any change to source, configuration, data, tokenizer, environment, optimizer, checkpoint, or evaluation contract invalidates affected prior evidence until rerun.

## 6. Level 1 stage plan

### Stage L1-0 — Contract and reproducible baseline

**Objective:** Establish the CCT Level 1 contract, build system, deterministic execution, artifact protocol, and baseline measurement.

**Implementation tasks:**

- Define supported compiler, CMake, C++ standard, operating system, dependencies, and warning policy.
- Define model, parameter, state, tokenizer, dataset, checkpoint, evaluation, and artifact identities.
- Implement clean native builds with strict warnings and warnings-as-errors.
- Implement deterministic unit, integration, benchmark, and gate entry points.
- Define report schemas, status values, threshold fields, and release records.
- Add failures for missing dependencies, malformed configuration, non-finite output, corrupted artifacts, and nondeterministic replay.

**Gate evidence:**

- Clean checkout build passes.
- Baseline tests pass with fixed seeds.
- Repeated benchmark and test identities agree within declared tolerances.
- Artifact manifests identify exact commit and configuration.
- Failure paths close without silent fallback.

**Transition:** `PASS` authorizes L1-1 numerical implementation.

### Stage L1-1 — Differentiable numerical substrate

**Objective:** Implement the finite, stable, independently checked numerical operators required by CCT learning.

**Implementation tasks:**

- Define tensor or field storage, shape, layout, dtype, ownership, and lifetime rules.
- Implement forward operators, reductions, normalization, activation, mixing, and declared spectral or adaptive primitives.
- Implement analytic gradients, accumulation, parameter identity, and deterministic update interfaces.
- Add overflow, underflow, NaN, infinity, aliasing, mutation, invalid-shape, and range checks.
- Compare gradients against finite differences or an independent reference.

**Gate evidence:**

- Forward results match independent references.
- Analytic gradients meet declared tolerances.
- The complete numerical matrix remains finite.
- Invalid shapes, aliasing, overflow, and corrupted state fail closed.
- Replay from the same seed and configuration is deterministic.

**Transition:** `PASS` authorizes L1-2 sequence-engine implementation.

### Stage L1-2 — Causal sequence engine

**Objective:** Implement efficient ordered sequence learning with recurrent state, causal boundaries, streaming equivalence, and bounded resource behavior.

**Implementation tasks:**

- Define state initialization, update order, reset semantics, and causal boundaries.
- Implement typed recurrent state with serialization, versioning, and ownership.
- Implement full, chunked, and streaming execution.
- Verify state memory, allocation, latency, and throughput bounds.
- Add invalid-version, wrong-shape, out-of-order, reset, and non-finite recurrence failures.

**Gate evidence:**

- Full-sequence and streaming results agree within declared tolerances.
- State reload, reset, and replay are deterministic.
- Resource measurements remain within declared limits.
- Invalid state transitions fail closed.

**Transition:** `PASS` authorizes L1-3 event and state-discipline work.

### Stage L1-3 — Event, causality, and state discipline

**Objective:** Ensure CCT can process ordered events and internal state without leakage, ambiguity, or silent provenance loss.

**Implementation tasks:**

- Assign stable event identity, ordering, source metadata, and causal relationships.
- Validate dependency structures and declared cycle policy.
- Enforce no-future-read and evaluator-isolation rules.
- Implement intervention, counterfactual, omission, reordering, contradiction, and perturbation fixtures.
- Add confidence, evidence, uncertainty, and safe abstention records.

**Gate evidence:**

- Event identity and duplicate detection pass.
- Leakage audit passes.
- Intervention and counterfactual comparisons are reproducible.
- Robustness tests pass under declared perturbations.
- Unsupported and ambiguous inputs abstain safely.

**Transition:** `PASS` authorizes L1-4 representation locking.

### Stage L1-4 — Tokenizer and representation lock

**Objective:** Freeze one deterministic representation contract for all Level 1 data, training, checkpoints, and evaluation.

**Implementation tasks:**

- Implement and compare byte, subword, and hybrid candidates under a fixed contract.
- Freeze the selected vocabulary, tokenizer snapshot, ordering, and hash.
- Implement Unicode-safe offsets, malformed-input policy, and byte fallback.
- Implement packed and padded causal batches with padding, boundary, and record masks.
- Measure token efficiency, source-byte ratio, memory, throughput, and round-trip behavior.

**Gate evidence:**

- Snapshot identity is immutable and loadable.
- Source-to-token offsets are correct, including Unicode and fallback cases.
- Every supported byte has a valid representation.
- Packed and padded batches are deterministic and mask-correct.
- Efficiency comparison is recorded.

**Transition:** `PASS` authorizes L1-5 native language training.

### Stage L1-5 — Native trainable language core

**Objective:** Make CCT trainable through a native next-token objective with analytic gradients, optimizer safety, checkpoint recovery, and held-out evaluation.

**Implementation tasks:**

- Define next-token targets, causal masks, loss accounting, and finite-objective rules.
- Implement analytic CCT recurrence gradients and independent gradient checks.
- Implement optimizer, clipping, schedule, finite checks, initialization, and deterministic updates.
- Implement checkpoint save, load, resume, identity validation, and corruption rejection.
- Compare against a declared matched reference under equal data and budget.

**Gate evidence:**

- Loss and gradients remain finite.
- Gradient agreement passes.
- Multiple fixed-seed runs are deterministic within tolerance.
- Held-out behavior improves or is explicitly explained.
- Checkpoint reload and resume are equivalent.
- Real-source evidence and matched controls are recorded.

**Transition:** `PASS` authorizes L1-6 governed language acquisition.

### Stage L1-6 — Track 1 real language acquisition

**Objective:** Train CCT on a small, governed real-data bundle and create the first reproducible language-learning evidence.

**Approved data contract:** WikiText-2 for pretraining and SQuAD 2.0 for supervised target learning, selection evaluation, and frozen final evaluation.

**Implementation tasks:**

- Pin source revisions, direct acquisition URLs, source licenses, upstream identities, and split identities.
- Acquire WikiText through the pinned direct archive route and SQuAD through the pinned direct flat-file route.
- Implement resumable caching, atomic extraction, pacing, retry, cache validation, and fail-closed missing-source behavior.
- Validate Unicode source offsets, answer text, answerability, malformed rows, and split isolation.
- Select deterministic balanced supervised examples.
- Generate manifests, digests, preparation reports, evaluation contracts, and release records.
- Train the native CCT pretraining phase and save a verified checkpoint.
- Evaluate validation and held-out target-token metrics.

**Gate evidence:**

- Unit tests and formal Track 1 gate pass.
- Real bounded acquisition passes with no malformed rows or overlap.
- Complete governed preparation reports `passed: true`.
- Train, selection, and frozen final identities are isolated.
- The training runner produces finite metrics and reloadable checkpoints.
- The report states that answer-target next-token metrics are measured; exact-answer EM/F1 are not claimed until constrained decoding exists.

**Transition:** `PASS` authorizes L1-7 supervised instruction adaptation.

### Stage L1-7 — Supervised instruction adaptation

**Objective:** Teach CCT to accept a canonical instruction format and produce supervised target behavior without training the prompt itself as the target.

**Implementation tasks:**

- Define instruction, context, target, metadata, eligibility, policy, and evaluator schemas.
- Implement canonical formatting and `target-span-only-v1` loss masks.
- Filter zero-loss or malformed examples.
- Train full-parameter supervised continuation and preserve base checkpoint identity.
- Compare held-out task behavior against the untouched base.
- Validate structure, citations, missing-evidence behavior, and safe refusal.

**Gate evidence:**

- Target-only mask tests pass.
- Held-out instruction behavior improves or is explained.
- Structured outputs validate deterministically.
- Missing evidence produces bounded abstention.
- Unsafe or unsupported requests remain denied.
- Base immutability, checkpoint lineage, and deletion behavior pass.

**Transition:** `PASS` authorizes L1-8 operation and API adaptation.

### Stage L1-8 — Operation and API teacher adaptation

**Objective:** Teach CCT the declared internal operation and API contracts that a downstream system is allowed to expose.

**Implementation tasks:**

- Define versioned operation schemas, required fields, optional fields, types, bounds, and error classes.
- Create governed demonstrations for valid calls, invalid calls, retries, missing evidence, and refusal.
- Fine-tune CCT on serialization, explanation, validation, and correction of declared operation calls.
- Implement schema validation before any operation is accepted.
- Test unknown operations, malformed arguments, unauthorized operations, and ambiguous requests.
- Preserve operation schema identity in checkpoints and reports.

**Gate evidence:**

- Valid operation calls serialize and validate.
- Invalid, unknown, unauthorized, or ambiguous calls fail closed.
- CCT can explain required fields and identify malformed calls.
- Demonstrations and corrections are traceable to governed examples.
- No external side effect is permitted by the Level 1 teacher interface.

**Transition:** `PASS` authorizes L1-9 bounded teaching evaluation.

### Stage L1-9 — Bounded teaching behavior

**Objective:** Demonstrate that CCT can teach within a defined interface through language, demonstrations, evaluation, correction, and evidence.

**Implementation tasks:**

- Define teaching episode identity, task, demonstration, expected result, evidence, critique, correction, and outcome schema.
- Implement a bounded teacher interaction harness with no unapproved external side effects.
- Test task explanation, step decomposition, demonstration, result checking, error identification, and corrective response.
- Use an independent evaluator or verifier for material judgments.
- Require abstention when evidence or schema support is missing.
- Log every episode, model version, prompt or instruction, output, evidence, and evaluator decision.

**Gate evidence:**

- Teaching episodes replay deterministically.
- CCT provides valid demonstrations for declared tasks.
- Injected errors are detected by the independent evaluator.
- Corrections cite the relevant evidence or abstain.
- Unsupported teaching requests are rejected.
- The report separates language quality, schema correctness, evaluation quality, and unsupported claims.

**Transition:** `PASS` authorizes L1-10 local teacher release review.

### Stage L1-10 — Local teacher-engine release

**Objective:** Freeze and release a measured CCT Level 1 teacher engine for the declared downstream interface.

**Implementation tasks:**

- Freeze model, tokenizer, data, training, operation schemas, policy, evaluator, runtime, and release configuration.
- Save pretraining and adapted checkpoints with complete identity lineage.
- Verify checkpoint reload, resume, corruption rejection, and configuration mismatch rejection.
- Measure local memory, throughput, latency, checkpoint size, and resource behavior.
- Produce model card, evaluation report, limitations, support boundary, and release record.
- Define rollback, retirement, deletion, and revalidation procedures.

**Gate evidence:**

- All Level 1 predecessors are `PASS` or explicitly recorded as approved evidence.
- Complete language, instruction, operation, teaching, reliability, and resource reports are present.
- No unresolved mandatory failure remains.
- Release artifacts are hash-linked to the final commit.
- Claims are limited to the declared evaluation scope.

**Transition:** `PASS` completes Level 1 and authorizes only the separately specified downstream integration stage.

## 7. CCT Level 1 metrics

The following metrics are mandatory where applicable:

| Metric class | Required measurements |
|---|---|
| Numerical | Finite outputs, gradient agreement, parameter finiteness, update stability, and deterministic replay. |
| Language | Cross-entropy, perplexity, token accuracy, held-out loss, continuation slices, and robustness slices. |
| Instruction | Target-token loss, task success, structure validity, abstention, and regression against the base. |
| Operation/API | Schema-valid call rate, invalid-call rejection, unknown-operation rejection, correction accuracy, and trace completeness. |
| Teaching | Demonstration validity, independent error detection, evidence-linked correction, abstention, and episode replay. |
| Systems | Memory, throughput, latency, checkpoint size, load time, recovery, and resource-bound failures. |
| Reproducibility | Manifest equality, tokenizer and data identity, checkpoint hash, seed replay, and environment record. |

No metric may be reported without its split, seed, configuration, data identity, baseline or reference, threshold, and artifact path.

## 8. Failure and safety boundaries

CCT Level 1 must fail closed for missing or changed source data, malformed rows, invalid Unicode offsets, contaminated splits, unknown tokenizer identity, incompatible checkpoints, corrupted state, non-finite values, unsupported operations, malformed arguments, unauthorized operations, missing evidence, ambiguous teaching requests, and resource exhaustion.

The Level 1 teacher interface may return text, structured operation representations, explanations, evaluations, corrections, or abstentions. It must not silently perform unapproved external side effects. Side-effectful integration is outside this specification and requires a separate contract.

## 9. Level 1 completion rule

Level 1 is complete only when Stages L1-0 through L1-10 have complete implementation and evidence records, all mandatory gates are `PASS`, the final checkpoint and tokenizer identities are reproducible, the local teacher interface is tested on realistic tasks and failures, and the release report states the exact capabilities and limitations demonstrated.

A Track 1 preparation or bounded training result is necessary evidence for Level 1 but is not by itself completion of the Level 1 teacher-engine goal. Track 1 proves the initial governed data and training handoff; later Level 1 stages must prove instruction, operation/API, bounded teaching, and release behavior.

## 10. References

- [CCT architecture](../Architecture.md)
- [CCT internal goals](../Goal.md)
- [CCT actionable todo](../Todo.md)
- [CCT stage specifications](../Stages/README.md)
- [Track 1 operational guide](../artifacts/track1/README.md)
