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

**Current status:** `PASS — fresh native replay at a3fc4d9` for the declared reproducible baseline contract; a clean external release-checkout replay remains required before release use.

## L1-0 implementation tasks

- [ ] Define the supported compiler, CMake version, C++ standard, operating system, and dependency versions.
- [ ] Define warning policy, warnings-as-errors policy, build types, and native CUDA policy.
- [ ] Define model, parameter, state, tokenizer, data, checkpoint, evaluator, and artifact identity fields.
- [ ] Implement a clean native C++20 build from a clean checkout.
- [ ] Implement deterministic initialization and fixed-seed configuration.
- [ ] Implement unit-test, integration-test, benchmark, and formal-gate entry points.
- [ ] Define the standard artifact tree for configuration, environment, tests, benchmarks, manifests, logs, checkpoints, gate, and report.
- [ ] Define status values and exit-code behavior.
- [ ] Add missing-dependency failure coverage.
- [ ] Add malformed-configuration failure coverage.
- [ ] Add non-finite-output failure coverage.
- [ ] Add corrupted-artifact failure coverage.
- [ ] Add nondeterministic-replay detection.

## L1-0 verification tasks

- [ ] Build the clean checkout with the declared native compiler.
- [ ] Run baseline tests twice with the same seed.
- [ ] Run benchmark workloads twice with the same configuration.
- [ ] Compare test identity and benchmark identity within declared tolerances.
- [ ] Confirm invalid configuration exits nonzero and produces a useful failure record.
- [ ] Confirm missing dependency exits nonzero without silently falling back.
- [ ] Confirm non-finite output is rejected.
- [ ] Confirm artifact hashes identify the exact commit and configuration.

## L1-0 artifacts

- [ ] `config.json`
- [ ] `environment.json`
- [ ] `tests.json`
- [ ] `benchmarks.json`
- [ ] `manifest.json`
- [ ] `gate.json`
- [ ] `release_record.json`
- [ ] `report.md`

## L1-0 gate and transition

- [ ] Run the formal L1-0 gate from the release candidate commit.
- [ ] Confirm all mandatory checks are `PASS`.
- [ ] Record known limitations and any optional blocked dependency.
- [ ] Record transition approval for L1-1.

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

**Current status:** `PASS — fresh native replay at a3fc4d9` for the declared numerical and field contracts; a clean external release-checkout replay remains required before release use.

## L1-1 implementation tasks

- [ ] Define tensor or field storage, shape, layout, dtype, ownership, indexing, and lifetime contracts.
- [ ] Implement required forward operators.
- [ ] Implement stable reductions and normalization.
- [ ] Implement activation, mixing, and declared spectral/adaptive primitives.
- [ ] Implement analytic gradients for every trainable operator used by CCT.
- [ ] Implement gradient accumulation and parameter identity.
- [ ] Implement deterministic update interfaces.
- [ ] Add finite and range checks.
- [ ] Add overflow and underflow checks.
- [ ] Add NaN and infinity checks.
- [ ] Add invalid-shape checks.
- [ ] Add aliasing and mutation checks.
- [ ] Add corrupted-state checks.

## L1-1 verification tasks

- [ ] Compare forward results against an independent reference.
- [ ] Compare analytic gradients with finite differences or an independent derivative implementation.
- [ ] Run the complete numerical test matrix across declared shapes, dtypes, ranges, and boundary values.
- [ ] Repeat the numerical matrix with fixed seeds.
- [ ] Verify finite outputs, gradients, and parameters.
- [ ] Verify invalid shapes fail closed.
- [ ] Verify aliasing and mutation failures are detected.
- [ ] Verify overflow, underflow, and corrupted state fail closed.

## L1-1 artifacts

- [ ] Operator contract.
- [ ] Gradient-check report.
- [ ] Numerical reference comparison.
- [ ] Boundary and failure report.
- [ ] Resource and dtype report.
- [ ] L1-1 gate record.
- [ ] L1-1 transition record.

## L1-1 gate and transition

- [ ] Freeze derivative tolerances before the final run.
- [ ] Run the formal L1-1 gate.
- [ ] Record every failed or skipped operator.
- [ ] Confirm no capability claim exceeds the verified operator matrix.
- [ ] Record transition approval for L1-2.

**Transition:** `PASS` authorizes L1-2 sequence-engine implementation.

---

# Stage L1-2 — Causal Sequence Engine

**Objective:** Implement efficient ordered sequence learning with recurrent state, causal boundaries, streaming equivalence, and bounded resources.

**Dependency:** L1-1 `PASS`.

**Current status:** `PASS — fresh native replay at a3fc4d9` for the declared causal sequence contract; a clean external release-checkout replay remains required before release use.

## L1-2 implementation tasks

- [ ] Define state initialization and reset semantics.
- [ ] Define update order and causal boundaries.
- [ ] Implement typed recurrent state.
- [ ] Implement state ownership, serialization, versioning, and loading.
- [ ] Implement full-sequence execution.
- [ ] Implement chunked execution.
- [ ] Implement streaming execution.
- [ ] Implement incremental state reuse and flush behavior.
- [ ] Bound state memory and allocations.
- [ ] Add invalid state-version failure.
- [ ] Add wrong-shape state failure.
- [ ] Add out-of-order event failure.
- [ ] Add unexpected reset failure.
- [ ] Add non-finite recurrence failure.

## L1-2 verification tasks

- [ ] Compare full-sequence and streaming outputs within declared tolerances.
- [ ] Compare full-sequence and chunked outputs.
- [ ] Verify state reload, reset, and replay.
- [ ] Measure memory, throughput, latency, and allocation behavior.
- [ ] Exercise long-sequence and repeated-reset fixtures.
- [ ] Verify invalid state transitions fail closed.
- [ ] Verify no future event is consumed.

## L1-2 artifacts

- [ ] State contract.
- [ ] Full-vs-streaming equivalence report.
- [ ] Resource profile.
- [ ] State recovery report.
- [ ] Failure-path report.
- [ ] L1-2 gate record.
- [ ] L1-2 transition record.

## L1-2 gate and transition

- [ ] Freeze equivalence and resource thresholds.
- [ ] Run the formal L1-2 gate.
- [ ] Preserve the simple sequence path as a correctness oracle.
- [ ] Record transition approval for L1-3.

**Transition:** `PASS` authorizes L1-3 event and state discipline.

---

# Stage L1-3 — Event, Causality, and State Discipline

**Objective:** Process ordered events and internal state without leakage, ambiguity, or silent provenance loss.

**Dependency:** L1-2 `PASS`.

**Current status:** `PASS — fresh native replay at a3fc4d9` for the declared event, causality, and state-discipline contract; a clean external release-checkout replay remains required before release use.

## L1-3 implementation tasks

- [ ] Define stable event identity.
- [ ] Define event ordering, source metadata, and causal relationships.
- [ ] Define dependency structure and cycle policy.
- [ ] Implement future-information leakage checks.
- [ ] Implement evaluator-isolation checks.
- [ ] Implement interventions with changed-input records.
- [ ] Implement counterfactual paired-world fixtures.
- [ ] Implement omission, reorder, irrelevant-event, contradiction, and perturbation fixtures.
- [ ] Preserve provenance for every intervention and counterfactual.
- [ ] Implement confidence, evidence, uncertainty, and abstention records.

## L1-3 verification tasks

- [ ] Verify stable event identity and duplicate detection.
- [ ] Verify source digest and ordering.
- [ ] Run the leakage audit.
- [ ] Verify causal dependency validation.
- [ ] Verify intervention outcome comparisons.
- [ ] Verify counterfactual reproducibility with shared seed control.
- [ ] Verify robustness under reordering and omission.
- [ ] Verify contradiction handling.
- [ ] Verify unsupported and ambiguous input abstention.

## L1-3 artifacts

- [ ] Event schema.
- [ ] Causal dependency report.
- [ ] Leakage audit.
- [ ] Intervention report.
- [ ] Counterfactual report.
- [ ] Robustness report.
- [ ] Abstention report.
- [ ] L1-3 gate and transition record.

## L1-3 gate and transition

- [ ] Freeze event and provenance schema.
- [ ] Run the formal L1-3 gate.
- [ ] Record unresolved causal or state limitations.
- [ ] Record transition approval for L1-4.

**Transition:** `PASS` authorizes L1-4 representation locking.

---

# Stage L1-4 — Tokenizer and Representation Lock

**Objective:** Freeze one deterministic representation contract for Level 1 data, training, checkpoints, and evaluation.

**Dependency:** L1-3 `PASS`.

**Current status:** `PASS — fresh native replay at a3fc4d9` for the declared tokenizer and representation contract; a clean external release-checkout replay remains required before release use.

## L1-4 implementation tasks

- [ ] Implement byte tokenizer candidate.
- [ ] Implement subword tokenizer candidate.
- [ ] Implement hybrid tokenizer candidate.
- [ ] Compare candidates under a fixed data and evaluation contract.
- [ ] Freeze selected vocabulary, ordering, snapshot, and hash.
- [ ] Implement Unicode-safe source-to-token offsets.
- [ ] Implement malformed-input policy.
- [ ] Implement byte fallback for every supported byte.
- [ ] Implement packed causal batches.
- [ ] Implement padded causal batches.
- [ ] Implement padding, boundary, and record masks.
- [ ] Measure token efficiency, source-byte ratio, memory, throughput, and round-trip behavior.

## L1-4 verification tasks

- [ ] Verify snapshot identity and immutable loading.
- [ ] Verify Unicode offsets and malformed-input behavior.
- [ ] Verify byte fallback coverage.
- [ ] Verify packed and padded masks.
- [ ] Verify deterministic tokenization and round trips.
- [ ] Compare token efficiency and resource behavior.
- [ ] Verify tokenizer identity is embedded into data and checkpoint manifests.

## L1-4 artifacts

- [ ] Candidate comparison report.
- [ ] Frozen tokenizer snapshot.
- [ ] Vocabulary hash.
- [ ] Offset and fallback test report.
- [ ] Batch and mask report.
- [ ] Efficiency report.
- [ ] L1-4 gate and transition record.

## L1-4 gate and transition

- [ ] Freeze tokenizer and vocabulary before downstream data preparation.
- [ ] Run the formal L1-4 gate.
- [ ] Record that tokenizer changes invalidate downstream identities.
- [ ] Record transition approval for L1-5.

**Transition:** `PASS` authorizes L1-5 native language training.

---

# Stage L1-5 — Native Trainable Language Core

**Objective:** Train CCT through a native next-token objective with analytic gradients, optimizer safety, checkpoint recovery, and held-out evaluation.

**Dependency:** L1-4 `PASS`.

**Current status:** `PASS — fresh native replay at a3fc4d9` for the declared native trainable language-core contract; a clean external release-checkout replay remains required before release use.

## L1-5 implementation tasks

- [ ] Define next-token targets and causal masks.
- [ ] Define loss accounting and finite-objective rules.
- [ ] Implement analytic CCT recurrence gradients.
- [ ] Implement independent gradient checks.
- [ ] Implement optimizer, clipping, schedule, initialization, and deterministic updates.
- [ ] Implement parameter and optimizer state serialization.
- [ ] Implement checkpoint save and load.
- [ ] Implement exact resume and cursor recovery.
- [ ] Implement tokenizer, data, configuration, and model identity validation.
- [ ] Implement corruption and incompatible-checkpoint rejection.
- [ ] Implement a matched reference baseline.
- [ ] Measure cross-entropy, perplexity, token accuracy, throughput, memory, and parameter count.

## L1-5 verification tasks

- [ ] Verify loss, gradients, parameters, and optimizer state remain finite.
- [ ] Verify analytic-gradient agreement.
- [ ] Run multiple fixed-seed configurations.
- [ ] Compare initial and final validation behavior.
- [ ] Explain any non-improvement without weakening the gate.
- [ ] Verify exact checkpoint load and resume equivalence.
- [ ] Verify wrong tokenizer, data, configuration, and model identities fail closed.
- [ ] Verify corrupted checkpoints fail closed.
- [ ] Compare against the matched reference under equal data and budget.

## L1-5 artifacts

- [ ] Objective and mask contract.
- [ ] Gradient report.
- [ ] Optimizer configuration.
- [ ] Baseline comparison.
- [ ] Training and validation report.
- [ ] Checkpoint manifest and hashes.
- [ ] Resume and corruption report.
- [ ] L1-5 gate and transition record.

## L1-5 gate and transition

- [ ] Freeze optimizer, context, batch, seed, and baseline configurations.
- [ ] Run the formal L1-5 gate.
- [ ] Record finite metrics and all failed configurations.
- [ ] Record transition approval for L1-6.

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
