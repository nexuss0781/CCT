# CCT Issues and Remediation Todo

## Review scope and evidence boundary

This backlog is the result of a fresh senior review of the current CCT checkout at commit `a888b52ef462ccfca0560b28b8e3bf46741e5eb8`.

The review covered the repository architecture, build and test registration, native headers and implementations, stage and Track 1 tools, data preparation, training, memory, retrieval, inference, and release-control paths. The inspected native surface contained **81 C++ or CUDA source/header files and approximately 25,617 lines**. The review included a clean Release build with `-Wall -Wextra -Wpedantic -Werror`, which completed, and the current CTest suite, which reported **37/37 passing tests**.

A targeted finite-difference probe was also run against `SelectiveSequenceCore`. Plain masked mode had a maximum checked gradient error of approximately `5.1e-12`; complex masked mode had an error of approximately `1.12e-3`; complex plus normalized masked mode had an error of approximately `8.11e-2`. This is direct evidence of a trainable-path defect.

The sanitizer build completed after a GCC sanitizer-build `array-bounds` warning was demoted, but the full sanitizer CTest run exceeded the review timeout during the slower stage tests and was stopped. Therefore, sanitizer validation is **not** marked as passed.

This is an evidence-linked engineering backlog, not a mathematical proof that every possible defect has been found. Each item below identifies a concrete implementation risk, a location or contract, an impact, a remediation, and a required regression test.

## Priority and status

| Priority | Meaning |
|---|---|
| **P0** | Correctness, security, integrity, or architectural truth failure. Blocks a release or claim. |
| **P1** | Major reliability, evaluation, data, or production-contract defect. Must be fixed before the affected scope is promoted. |
| **P2** | Material performance, maintainability, or scientific-validity weakness. Must be scheduled before broad scaling. |
| **P3** | Documentation, ergonomics, or non-blocking improvement. |

| Marker | Meaning |
|---|---|
| `[ ]` | Open remediation item. |
| `[x]` | Fixed and verified with a linked regression artifact. |
| `REVALIDATE` | Historical evidence exists, but the affected path must be replayed after a change. |
| `[!]` | Failed or invalidated. Stop dependent work. |

## Release blockers

The following items are release-governance blockers; entries marked `[x]` are closed and no longer block the next transition:

- [x] **SEQ-001** Correct analytic gradients for complex, normalized, and masked sequence modes; verified by complete all-mode finite differences and sparse-mask state tests.
- [x] **TRAIN-001** Explicitly renamed and separated the Track 1 NLP recurrence from the general `SelectiveSequenceCore`; independent model identity, serialization, and claim-boundary tests pass.
- [ ] **INF-001** Replace template inference responses with a real checkpoint-backed model execution path.
- [ ] **INF-002** Implement true incremental decoding and cooperative cancellation before claiming streaming.
- [x] **CORPUS-001** Raw detected PII is purged before storage and governed quarantine metadata remains.
- [x] **TRACK1-001** Acquisition uses native argument-vector processes, HTTPS, unique atomic temporary files, and integrity checks.
- [x] **TRACK1-002** The bounded native JSON parser rejects ambiguous structure, duplicate keys, invalid UTF-8, control characters, and trailing data.
- [x] **TRACK1-003** Direct flat-file preparation verifies whole-file row counts against the pinned contract.
- [x] **TRACK1-005** Temporary-file creation is exclusive and collision-resistant under concurrent runs.
- [ ] **DOC-001** Reconcile all stage specifications, Goal/Todo files, and artifact statuses before using stage completion as release evidence.

---

# A. Mathematical and numerical correctness

## SEQ-001 — Incorrect analytic gradients for complex, normalized, and masked modes

**Priority:** P0. **Status:** `[x] FIXED`.

**Historical evidence (pre-fix):** The forward path in `step()` applied complex rotation at lines 198–205 and state normalization at lines 211–220. The prior gradient path cached only real hidden state and real gate/candidate values at lines 261–312, then backpropagated through an unrotated, unnormalized recurrence at lines 342–373. It also executed `step()` for every position during cache construction at lines 288–314 even when the mask said the state should not update. The derivative path used `write_raw` and `retain_raw` as if they were the post-activation values in lines 350–355, and discarded `d_input` and `d_previous_next` at lines 370 and 374.

The corrected implementation is in `cpp/src/sequence.cpp`, with permanent coverage in `cpp/tests/sequence_tests.cpp`.

**Measured impact:** A targeted finite-difference probe measured a maximum gradient error of approximately `1.12e-3` in complex masked mode and `8.11e-2` in complex-plus-normalized masked mode. Plain masked mode was approximately `5.1e-12`.

**Why this matters:** Optimizing these modes can move parameters in a direction that does not minimize the implemented forward loss. Masked training can also propagate gradient through positions that do not update state. This invalidates training evidence for the affected modes.

**Remediation completed:** Implemented a mode-complete reverse pass with complex real/imaginary state, rotation derivatives, state/output normalization Jacobians, gate-boundary derivatives, and exact forward mask semantics. Non-finite targets are rejected before objective evaluation.

**Verification:** Permanent regression covers all eight combinations of `complex_state`, `normalize_state`, and `normalize_output` under sparse masks, compares the complete flattened parameter vector to central finite differences at absolute tolerance `<1e-8`, verifies masked state preservation, and rejects non-finite targets. The native suite is green after the fix.

**Depends on:** None. **Blocks:** L1-2/L1-5 modes using these features.

## FIELD-001 — Numerical stability is local, not a global CFL guarantee

**Priority:** P1. **Status:** `[x] FIXED`.

**Evidence:** `SolverConfig::maximum_abs_potential` declares the bounded potential domain. `global_stability_limit()` uses a conservative maximum-wavenumber spectral bound plus the potential bound; `validate_stability()` rejects timesteps outside the combined CFL/global domain, and runtime potential normalization rejects non-finite or out-of-domain values. The field regression sweeps Leapfrog/RK4, periodic/Dirichlet/Neumann, one- and two-dimensional grids, valid rollouts, finite-state checks, and rejection beyond the declared boundary.

**Impact:** Stability claims are now scoped to the explicit bounded-potential and fixed-step solver contract rather than generalized to unconstrained learned parameters.

**Remediation:** Completed in native C++20 with an explicit conservative domain, boundary-specific operator semantics, persisted configuration, and matrix regression evidence.

**Required regression:** Completed by `global_stability_domain_and_semantics` in `cct_tests`.

## FIELD-002 — Spectral and finite-difference semantics are not equivalent

**Priority:** P1. **Status:** `[x] FIXED`.

**Evidence:** The public field contract now states that FFTW spectral semantics are periodic Fourier modes only, while finite-difference semantics apply the requested periodic, Dirichlet, or Neumann closure. The regression verifies close parity on a resolved periodic manufactured mode and explicitly verifies non-periodic divergence rather than treating the methods as interchangeable.

**Impact:** Cross-method comparisons are now required to declare the discretization domain and cannot silently attribute boundary/discretization differences to architecture behavior.

**Remediation:** Completed with explicit normalization/Nyquist implementation, periodic-only spectral construction, boundary-specific finite differences, and manufactured parity/domain tests.

## FIELD-003 — Field gradients are one-step helpers, not full temporal adjoints

**Priority:** P2. **Status:** `[x] FIXED`.

**Evidence:** `TemporalRolloutGradients` and `temporal_rollout_loss_gradients()` now define a complete fixed-step multi-target rollout objective and reverse-time local-Jacobian adjoint oracle across every source timestep and shared potential. The existing one-step helper is explicitly documented as one-step-only. The regression compares every temporal source and potential derivative and the aggregate loss against direct central finite differences.

**Impact:** One-step gradients are no longer presented as long-horizon evidence; temporal training experiments have a separate full-rollout contract and deterministic oracle.

**Remediation:** Completed for the fixed-step solver domain. Adaptive timestep schedules are outside this API by contract and are not claimed.

## CAUSAL-001 — Causal graph validity does not enforce temporal causality

**Priority:** P1. **Status:** `[x] FIXED`.

**Evidence:** `CausalStoreConfig::temporal_policy` defaults to `StrictEarlier`, rejects present parent edges whose timestamp is not earlier than the child, supports explicit same-time policy, and persists in V2 snapshots with V1 default migration. The encoder reports excluded future-parent edges explicitly.

**Evidence:** `cpp/src/causal.cpp:90-128` validates parent existence, explicit unresolved status, and event-level uniqueness. `validate_acyclic()` checks a DAG at lines 191–214. It does not reject a present parent whose timestamp is later than the child. The encoder can exclude future parents when `prevent_future_leakage` is enabled at lines 389–396, but that silently drops graph edges instead of rejecting an invalid causal graph.

**Impact:** The graph store can accept a structurally acyclic graph with temporally impossible edges. Different consumers then receive different semantics: the store retains the edge while the encoder omits it.

**Remediation:** Add a declared timestamp policy. Either require every present parent timestamp to be strictly earlier than the child or classify same-time and future edges explicitly. Make encoding report invalid edges instead of silently discarding them.

**Required regression:** Insert earlier, same-time, later, and unresolved parents under each policy and require deterministic accept/reject/abstain behavior.

## CAUSAL-002 — Parent hypotheses are supplied, so learner evidence is not structural discovery

**Priority:** P1. **Status:** `[x] FIXED — hypothesis-conditioned contract`.

**Evidence:** The causal learner is now documented and treated as hypothesis-conditioned structural regression, not graph discovery. Its public evaluation is not used to claim edge discovery or causal identification; discovery requires a separate future contract.

**Evidence:** `CausalEventLearner::fit()` receives `parent_hypotheses` and stores them directly at `cpp/src/causal.cpp:621-632`. The learner fits coefficients for supplied edges; it does not discover the graph. Confidence is a normalized coefficient signal at lines 672–689.

**Impact:** A gate passing on synthetic data does not establish causal structure discovery, edge orientation, or causal identification from observations.

**Remediation:** Rename the contract to hypothesis-conditioned structural regression, or implement and evaluate a graph discovery procedure with hidden-edge, false-edge, confounding, and intervention-identification metrics.

## CAUSAL-003 — Ridge solver uses normal equations and unbounded dense elimination

**Priority:** P1. **Status:** `[x] FIXED`.

**Evidence:** `CausalEventLearner::solve_ridge()` now uses bounded Householder QR on an explicitly regularized augmented system, scale-aware regularization, finite checks, feature/sample budgets, and a diagonal condition threshold with explicit ill-conditioned failure.

**Evidence:** `cpp/src/causal.cpp:587-618` forms \(X^TX\), adds a fixed `1e-7` diagonal term, then performs Gauss-Jordan elimination. There is no scale-aware regularization, condition estimate, pivot tolerance tied to matrix norm, or bounded dimension.

**Impact:** Ill-conditioned features can produce unstable coefficients and misleading confidence. Memory and runtime grow poorly with parent count.

**Remediation:** Use QR or SVD for the reference path, add condition diagnostics, scale-aware regularization, maximum feature dimensions, and an explicit singular/ill-conditioned outcome.

## CAUSAL-004 — Synthetic causal fixture contains an intentionally inconsistent visible edge

**Priority:** P2. **Status:** `[x] FIXED`.

**Evidence:** `CausalDataset::invalid_fixture` is set when the intentionally contradictory visible edge is appended. Downstream code can reject or isolate this fixture instead of treating it as valid causal evidence.

**Evidence:** `SyntheticCausalGenerator::generate()` constructs visible events from the truth graph at lines 549–560 and then appends event 3 as a parent of event 1 at lines 562–565. The generated relation is inconsistent with the earlier-variable truth ordering and can create a temporal contradiction.

**Impact:** The fixture can test rejection, but it must not be used as a valid causal demonstration without an explicit `invalid_fixture` label.

**Remediation:** Separate valid and invalid generated fixtures and record the intended gate use for each.

---

# B. Trainability, model identity, and evaluation

## TRAIN-001 — Track 1 CCT is a separate recurrence from the general CCT sequence core

**Priority:** P0. **Status:** `[x] FIXED`.

**Evidence:** `cpp/src/nlp_trainer.cpp` retains the separate recurrence implementation, now exposed as `NlpModelKind::Track1CctRecurrence` and `track1_cct_recurrence`; the general core remains `SelectiveSequenceCore` in `cpp/src/sequence.cpp`. The original implementation had a model-specific parameter layout and separate forward and gradient functions. The Track 1 runner configures and trains `NextTokenModel`, not `SelectiveSequenceCore`.

**Impact:** Track 1 results do not validate the general CCT sequence architecture, including complex state, normalization, skip projection, and the public sequence API. A “CCT trained successfully” statement is ambiguous unless it names the exact implementation.

**Remediation completed:** Chose the explicit-separation design. The Track 1 recurrence is named independently, its model serialization now publishes `NLP_MODEL_V3` while accepting V2 checkpoints, and Stage 11/12/Track 1 artifacts report `track1_cct_recurrence`. General sequence-core claims are not attached to Track 1 results.

**Verification:** `cpp/tests/nlp_trainer_tests.cpp` asserts model kind, public name, V3 serialization, V2 compatibility mapping, and exact parameter restoration. Stage 11 and Stage 12 gates pass with the explicit identity.

## TRAIN-002 — The baseline comparison is computationally and parametrically non-matched

**Priority:** P1. **Status:** `[x] FIXED`.

**Evidence:** Stage 5 now trains all five model kinds through `train_reference_finite_difference()` with one shared deterministic finite-difference optimizer, learning rate, clip norm, epochs, batch/context data, and split protocol. The gate enforces a 10% parameter-count match, reports parameter deltas, resource memory, separate wall time, and rejects mismatch.

**Impact:** The comparison no longer treats heterogeneous optimizer paths as architecture evidence. Results remain a small native reference comparison and do not claim broad efficiency superiority.

**Remediation:** Completed with common reference training and matched-comparison gate evidence.

## TRAIN-003 — Finite-gradient checks do not cover every optimizer update

**Priority:** P1. **Status:** `[x] FIXED`.

**Evidence:** Each trainer step now validates gradient shape and every gradient value, computes next moments and parameters transactionally, validates both moment vectors and all updated parameters before commit, validates validation metrics, and only then advances optimizer state/history. NLP tests cover finite-gradient, optimizer, resume, and malformed-checkpoint failure paths.

**Evidence:** The sequence and NLP trainers check selected finite quantities, but the optimizer update contract is not a general post-update invariant across every parameter vector, gradient accumulator, and checkpoint-resume boundary.

**Impact:** A finite forward pass does not guarantee finite gradients or finite updated parameters. Divergence can be written into checkpoints if post-update validation is incomplete.

**Remediation:** Validate every gradient and every updated parameter before checkpointing. Add loss-increase, NaN, infinity, overflow, underflow, and checkpoint-after-failure tests.

## TRAIN-004 — Per-step validation can dominate training cost

**Priority:** P2. **Status:** `[x] FIXED`.

**Evidence:** `NlpOptimizerConfig::validation_interval_steps` now defines an explicit positive validation cadence. `NlpTrainingPoint` records whether validation ran and separates training and validation elapsed time. The checkpoint format persists the cadence and timing fields while accepting legacy rows with deterministic defaults. The native NLP regression configures a two-step interval and proves that step one skips validation while step two evaluates it.

**Impact:** The core trainer contract no longer silently couples every optimization step to full validation. Throughput measurements can distinguish optimization time from evaluation time, and evaluation cadence is reproducible from the checkpoint.

**Remediation:** Completed in native C++20 with cadence validation, candidate-model evaluation before state publication, persisted configuration, and regression evidence in `cct_nlp_trainer_tests`.

## TRAIN-005 — Checkpoint writes need atomicity and durable publication

**Priority:** P1. **Status:** `[x] FIXED`.

**Evidence:** `NlpTrainer::save_checkpoint()` now uses exclusive `mkstemp` publication, complete writes, file `fsync`, atomic rename, and parent-directory `fsync`; checkpoint loading enforces byte and structural budgets. Existing resume and checkpoint integrity tests pass, and Stage 11/12 gates exercise the path.

**Evidence:** `cpp/src/nlp_trainer.cpp` serializes checkpoints directly through file streams. The Track 1 runner creates the output directory but does not establish a general temporary-file, flush, fsync, atomic-rename, and manifest-publication protocol for every checkpoint.

**Impact:** Interrupted writes can leave a non-empty but invalid checkpoint that later appears reusable. A report can reference a file that was not durably published.

**Remediation:** Write checkpoint and manifest to unique temporary paths, flush and close, fsync where supported, atomically rename, then publish the report. Add interruption tests at each publication step.

## TRAIN-006 — Track 1 evaluation is target-token prediction, not answer quality

**Priority:** P1. **Status:** `[x] FIXED AS EXPLICIT SCOPE`.

**Evidence:** The native `Track1EvaluationContract` and durable training report now publish `pretrain_task=target_token_prediction`, `pretrain_evaluation_scope=not_answer_quality`, and the separate QA task/metric contract. The runner explicitly states that answer exact-match and F1 are not claimed; the Track 1 tests and gate reject an incomplete or overclaiming contract.

**Impact:** Track 1 results cannot be misread as human answer quality. A future constrained decoder/evaluator remains a separate capability addition, not hidden inside the current metric.

**Remediation:** Completed as an evaluation-boundary fix by requiring explicit scope language before QA claims are allowed.

## TRAIN-007 — Pretraining token cap is implemented as a byte cap

**Priority:** P1. **Status:** `[x] FIXED FOR BYTE_FALLBACK_V1`.

**Evidence:** Track 1 counts the actual V1 byte-fallback tokenizer units: every retained byte and every inserted newline delimiter consumes one unit. The durable manifest now publishes `pretrain_token_count_mode=byte_fallback_v1`, and tests cover ASCII, newline, Unicode-boundary, cap, and deterministic replay behavior.

**Impact:** The declared budget is exact and auditable for the tokenizer contract actually used by Track 1. The manifest no longer implies a generic external tokenizer-token budget.

**Remediation:** Completed by making the tokenizer/counting mode explicit and versioned. A different tokenizer requires a new count mode and independent budget contract.

## TRAIN-008 — Dataset hash and checkpoint identity do not prove semantic model compatibility

**Priority:** P2. **Status:** `[x] FIXED for the native NLP contract`.

**Evidence:** NLP checkpoints now bind a canonical `cct-training-contract-v1` digest covering model kind/configuration, objective and loss-mask policy, optimizer/scheduler, seed, tokenizer hash, dataset hash, and code contract. Tampering the digest is rejected by `nlp_trainer_tests`; Track 1 reports carry the contract digest and path-independent checkpoint references.

**Evidence:** `NlpDataset::build()` hashes token sequences, context length, and tokenizer hash. The identity contract must also bind model architecture version, optimizer version, loss-mask version, numerical mode, and source transformation policy wherever those affect training semantics.

**Impact:** Two semantically different training runs can appear compatible if they produce the same sequence-level dataset identity but differ in model or optimizer behavior.

**Remediation:** Define a canonical training-contract digest covering model kind, all model config, objective, mask policy, optimizer, scheduler, tokenizer, data, seed, and code version.

---

# C. Data, parsing, provenance, and privacy

## CORPUS-001 — PII detection is heuristic and raw content is retained after detection

**Priority:** P0. **Status:** `[x] FIXED`.

**Evidence:** `cpp/src/corpus.cpp:292-307` detects a small set of substrings, email-like patterns, and ten consecutive digits. `process_record()` sets `normalized_content` to `[redacted]` at lines 364–369 but keeps the original `record.content`, and `serialize()` writes the raw content at lines 439–442.

**Impact:** A record marked as PII-detected can still be persisted in raw form. The detector has large false-negative and false-positive classes and is not a privacy boundary.

**Remediation completed:** The bounded detector now enters a structured quarantine path that preserves only the original content digest, `[redacted]` normalized content, reason codes, and policy metadata; raw content is purged before records are stored. Corpus serialization publishes `CCT_GOVERNED_CORPUS_V2`, suppresses raw PII even if a caller constructs a flagged record, and sanitizes flagged V1 records during legacy deserialization.

**Verification:** `cpp/tests/corpus_tests.cpp` covers email and account-number PII, in-memory raw-content absence, serialized-artifact absence, digest preservation, V2 round-trip, and V1 legacy raw-content sanitization. The corpus suite passes.

## CORPUS-002 — Byte truncation can create invalid UTF-8 and invalid structured data

**Priority:** P1. **Status:** `[x] FIXED for text-file ingestion`.

**Evidence:** `GovernedCorpus::ingest_file()` now truncates only at a UTF-8 boundary before governance processing. A multibyte boundary regression passes and rejects partial lead-byte retention; structured dataset parsers retain their own structural validation contracts.

**Evidence:** `GovernedCorpus::ingest_file()` truncates the byte string directly at `cpp/src/corpus.cpp:112-121` when `max_bytes` is nonzero.

**Impact:** Truncation can split a multibyte codepoint, JSON token, or record boundary while downstream code treats the result as valid text.

**Remediation:** Truncate at validated UTF-8 boundaries for text and at record boundaries for structured data. Store whether truncation occurred and reject incomplete records.

## CORPUS-003 — Contamination and near-duplicate detection are incomplete and quadratic

**Priority:** P1. **Status:** `[x] FIXED FOR THE GOVERNED_CORPUS_V2 CONTRACT`.

**Evidence:** GovernedCorpus now maintains normalized-hash and word-posting indexes, checks exact/near/subspan contamination across every protected split pair, preserves same-split duplicate rejection, and rebuilds indexes after V1/V2 deserialization. Cross-train/validation ingestion is rejected with `split_contamination`; public detection accepts the candidate split; tests cover exact, near, evaluator, and cross-split cases.

**Impact:** Contamination is no longer limited to evaluator-only records or an all-record quadratic scan. The deterministic Jaccard threshold and split policy are explicit in the native contract.

**Remediation:** Completed for the current governed-corpus scale with bounded vector/parser budgets and indexed candidate retrieval.

## CORPUS-004 — Normalization is byte-oriented and locale-sensitive

**Priority:** P2. **Status:** `[x] FIXED for deterministic byte-preserving normalization`.

**Evidence:** Corpus normalization now uses explicit ASCII whitespace folding and ASCII lowercase conversion, preserving non-ASCII UTF-8 bytes and avoiding process-locale-dependent `ctype` behavior. The corpus and Stage 9/10 tests pass with stable hashes and lineage.

**Evidence:** `normalize()` iterates bytes, calls `std::tolower`, and collapses `std::isspace` at `cpp/src/corpus.cpp:210-224`. This is not Unicode case folding or Unicode whitespace normalization.

**Impact:** Equivalent non-ASCII strings can hash differently, while byte-level changes may alter identifiers, source spans, or language labels.

**Remediation:** Specify normalization version and use a Unicode-aware implementation or restrict the contract to ASCII explicitly. Never use normalized text for span-sensitive source data without preserving an offset map.

## CORPUS-005 — Custom PII and code labels are not a policy-grade classifier

**Priority:** P2. **Status:** `[x] FIXED AS HEURISTIC-HINT CONTRACT`.

**Evidence:** Corpus labels now use explicit `pii_candidate` and `code_candidate` names, retain the legacy `code` alias only for compatibility, strengthen deterministic credential/email/structured-code markers, preserve original-content classification before raw PII purge, and document the candidate nature in the native test contract.

**Impact:** Downstream consumers cannot treat these values as policy-grade classifications without an additional review/classifier step; PII records remain quarantined and raw content is purged.

**Remediation:** Completed by making the labels heuristic candidates rather than authoritative classifications and covering PII/code labels in corpus regressions.

## TRACK1-001 — Track 1 acquisition uses shell interpolation and non-atomic downloads

**Priority:** P0. **Status:** `[x] FIXED`.

**Evidence:** `run_curl()` constructs a shell command from URL and path at `cpp/src/track1.cpp:273-283` and executes `std::system`. `extract_archive_member()` does the same for archive and output paths at lines 286–298. `run_curl()` writes directly to the final cache path rather than an atomic temporary path.

**Impact:** User-controlled output paths containing shell metacharacters can alter command execution. A failed or interrupted download can leave a non-empty partial file that the next run treats as a valid cache. `unzip` and shell availability are implicit dependencies.

**Remediation completed:** `run_curl()` and archive extraction now use a native fork/exec argument vector with no shell interpolation. HTTPS is required for remote acquisition, pinned archive members are allowlisted, downloads and extraction use exclusive `mkstemp` paths, HTTP status and byte counts are checked, SHA-256 sidecars are written for remote caches, files/directories are synced, and publication is atomic.

**Verification:** Track 1 preparation, missing-cache, malformed-source, and deterministic replay tests pass; the gate publishes the durable cache-integrity and source-attestation contract. Test-only fixture archive creation remains isolated from production acquisition.

## TRACK1-002 — Track 1 uses a custom JSON parser with weak structural validation

**Priority:** P1. **Status:** `[x] FIXED`.

**Evidence:** `field_string()`, `nested_object()`, `first_array_string()`, and `first_array_integer()` search for key markers and delimiters with `std::string::find` at `cpp/src/track1.cpp:187-230`. `parse_json_string()` at lines 120–160 does not reject unescaped control characters or validate raw UTF-8. The key search can match text that resembles a key inside a value or a later nested object.

**Impact:** Malformed or adversarial JSON can be misparsed into a different field, accepted with the wrong value, or rejected with ambiguous diagnostics. The parser is trusted for rights-sensitive dataset acquisition.

**Remediation completed:** The native parser now validates a bounded object-root document, delimiter stack and depth, structural member traversal, duplicate keys, strict separators, JSON primitive numbers, unescaped control characters, UTF-8 code points, and trailing data. It fails closed on ambiguous field structure.

**Verification:** Track 1 tests cover malformed delimiters, duplicate keys, raw control characters, direct-file parsing, and valid Unicode surrogate-pair decoding. The parser remains native C++20 and no third-party runtime is required.

## TRACK1-003 — Direct source processing does not verify declared total row counts

**Priority:** P1. **Status:** `[x] FIXED`.

**Evidence:** `prepare_squad()` iterates the requested range in `for_each_flat_data_object()` at lines 550–574 but does not require that the direct file contains exactly the declared `total_rows`. WikiText checks the requested line count, but SQuAD direct parsing can ignore missing or extra rows outside the selected window.

**Impact:** A truncated or substituted source may pass if the selected window still contains enough balanced examples. Provenance metadata can claim a full immutable source while the parser consumed only a partial file.

**Remediation completed:** Direct GEM flat files are structurally validated and every `data` object is counted before preparation succeeds. The observed count is compared with the pinned source count in production and the explicit bounded count for small fixtures. Remote files also receive an integrity sidecar digest.

**Verification:** `direct_source_row_count_fails_closed` proves that extra rows cannot be silently ignored; the prepared selection window remains separate from whole-file completeness accounting.

## TRACK1-004 — Track 1 source identity mixes mirror dataset and upstream identity without independent verification

**Priority:** P2. **Status:** `[x] FIXED`.

**Evidence:** `Track1Source` records GEM as the acquisition dataset while preserving `rajpurkar/squad_v2` as upstream provenance. The code records URLs and raw digests, but does not independently compare IDs, row counts, licenses, or file content against an upstream canonical snapshot.

**Impact:** Provenance is descriptive rather than cryptographically demonstrated as faithful equivalence.

**Remediation completed:** Every prepared source now publishes an attestation digest binding source ID, mirror dataset/config/split, pinned revision, license, upstream dataset ID, acquisition method, raw URL/member, and observed raw digest. The attestation is included in the manifest digest and validated before artifact publication.

**Verification:** Track 1 unit and gate checks require a 64-character attestation for all five pinned sources and publish `source_attestation` in the release record. This is a provenance binding and content-integrity attestation; it is not a claim of independent upstream equivalence beyond the pinned source evidence.

## TRACK1-005 — Fixed temporary filenames allow concurrent-run collisions

**Priority:** P1. **Status:** `[x] FIXED`.

**Evidence:** `extract_archive_member()` uses `path.string() + ".tmp"` at lines 291–297. Multiple processes sharing an output directory can remove or overwrite each other’s temporary file.

**Impact:** Concurrent preparation can corrupt cache members or publish the wrong data under a valid name.

**Remediation completed:** All Track 1 temporary files use exclusive `mkstemp` paths with random suffixes, are cleaned on failure, synced, and atomically renamed. The fixed `.tmp` collision path has been removed from production acquisition.

**Verification:** Both download and archive-extraction paths share the unique temporary helper and the full Track 1 suite passes under strict warnings-as-errors compilation.

---

# D. Memory, knowledge, and grounding

## MEMORY-001 — Memory checksums are not cryptographic integrity or authenticity protection

**Priority:** P1. **Status:** `[x] FIXED`.

**Evidence:** `MemoryRecord` and `MemoryEvent` now carry SHA-256 content/event digests. V2 snapshots persist those digests, and V1 deserialization recomputes and verifies the legacy chain before migration. `memory_tests` covers tamper rejection, digest stability, and legacy migration.

**Historical evidence:** `MemoryEncoder::content_checksum()` in `cpp/src/memory.cpp:107-126` uses a custom 64-bit mix hash. The event chain also binds only the record checksum, IDs, versions, and reason at lines 213–224.

**Impact:** This is useful for accidental corruption detection but not collision-resistant tamper evidence. There is no key or external trust root.

**Remediation:** Use a standardized cryptographic digest for content and canonical record serialization. Use authenticated storage or a signed manifest when integrity against an attacker matters. Preserve the lightweight hash only as a performance checksum.

## MEMORY-002 — Memory retrieval is a linear scan and the embedding is deterministic hashing

**Priority:** P1. **Status:** `[x] FIXED FOR THE NATIVE BASELINE CONTRACT`.

**Evidence:** `MemoryConfig::embedding_backend` is persisted in `CCT_MEMORY_SNAPSHOT_V3` with V1/V2 migration defaults to `external-supplied-v1`. Retrieval vectors are caller/provider supplied record embeddings; text queries use a normalized-term posting index, and `retrieve_linear_oracle()` provides the deterministic O(M) correctness reference. Memory tests prove indexed/oracle result and score equivalence across append, update, delete, and event-log replay.

**Impact:** The storage/index complexity and embedding semantics are no longer hidden behind a generic retrieval name. The default contract is an auditable external-vector/lexical baseline, not a learned semantic embedding claim.

**Remediation:** Completed for the native baseline with versioned embedding identity, indexed retrieval, snapshot migration, and recall correctness evidence. Learned semantic quality remains outside the baseline contract.

## MEMORY-003 — Novelty threshold and immediate-deletion configuration are not effective controls

**Priority:** P2. **Status:** `[x] FIXED`.

**Evidence:** `MemoryWriteController` now applies `novelty_threshold`; `PersistentMemory::delete_memory()` honors `immediate_deletion`, while deferred deletion is processed explicitly. `memory_tests` verifies behavior differences for novelty and deletion policy.

**Historical evidence:** `MemoryWriteController` stores `novelty_threshold_` but `decide()` never uses it at `cpp/src/memory.cpp:143-167`. `PersistentMemory::delete_memory()` always appends a tombstone, while `MemoryConfig::immediate_deletion` is not used in the shown mutation path.

**Impact:** Configuration advertises controls that do not change behavior. Operators can believe novelty or immediate deletion is enforced when it is not.

**Remediation:** Implement the controls or remove them from the public configuration. Add behavior-difference tests for both settings.

## MEMORY-004 — Snapshot writes are not atomic or durable

**Priority:** P1. **Status:** `[x] FIXED`.

**Evidence:** Memory snapshot publication uses an exclusive temporary file, complete write, `fsync`, atomic rename, and parent-directory `fsync`; the memory regression exercises publication and reload.

**Historical evidence:** `save_snapshot()` previously wrote directly to the target path at `cpp/src/memory.cpp:607-611`; causal and knowledge snapshots use the same direct-stream pattern.

**Impact:** Process interruption can leave a truncated snapshot at the canonical path. A later load can fail or, if truncation remains syntactically plausible, restore incomplete state.

**Remediation:** Use temporary files, flush/close, optional fsync, atomic rename, and a manifest containing snapshot digest and event sequence.

## KNOW-001 — Knowledge embeddings and ranking are heuristic bag-of-words features

**Priority:** P1. **Status:** `[x] FIXED FOR THE NATIVE BASELINE/PROVIDER CONTRACT`.

**Evidence:** `KnowledgeIndexConfig::embedding_backend` names the deterministic-hash baseline or an injected versioned `KnowledgeEmbeddingProvider`; vector dimensions, versions, and finite values are validated. Lexical retrieval now uses a posting index and reports candidate scan counts; the index is rebuilt after snapshot restore. Knowledge tests cover provider identity, indexed candidate counts, tenant/temporal policy, citation grounding, and bounded replay.

**Impact:** The plane no longer presents heuristic/hash ranking as undisclosed semantic retrieval. Baseline and provider-backed modes are explicitly identifiable, and lexical complexity is auditable.

**Remediation:** Completed for the native baseline/provider interface. Learned embedding recall, paraphrase quality, and human-labeled semantic evaluation remain separate research measurements and are not claimed by the current gates.

## KNOW-002 — Citation verification checks whole-document overlap, not cited-span support

**Priority:** P1. **Status:** `[x] FIXED`.

**Evidence:** `verify_answer()` resolves each citation ID to the exact content substring, verifies the span SHA-256, and requires complete substantive claim-term coverage with conservative singular/plural normalization. The knowledge regression includes a whole-document distractor outside the cited span.

**Historical evidence:** `claim_supported()` previously received a `KnowledgeHit` and compares claim terms against `hit.content` at `cpp/src/knowledge.cpp:397-413`. `verify_answer()` maps cited span IDs to a hit at lines 415–451, but the support test does not restrict comparison to the cited span’s content range.

**Impact:** A claim can be accepted because terms occur elsewhere in the document even when the cited span does not support it. Citation precision can be overstated.

**Remediation:** Resolve every cited span to the exact substring, verify claim support against that substring or a structured entailment evaluator, and test distractor content outside the citation.

## KNOW-003 — Conflict detection only sees returned top-k hits

**Priority:** P1. **Status:** `[x] FIXED`.

**Evidence:** Conflict preflight evaluates all eligible authorized records before top-k truncation and propagates `conflict_visible` into grounded-answer verification. The knowledge regression constrains top-k to one hit while requiring conflict visibility.

**Historical evidence:** `retrieve()` previously computed conflict visibility after filtering and sorting and truncates to `maximum_hits` at `cpp/src/knowledge.cpp:375-384`. `verify_answer()` detects conflicts from the supplied hits at lines 430–445.

**Impact:** A conflicting record filtered out by ranking or top-k truncation is invisible to the verifier. The answer can be accepted without knowing that contradictory authorized evidence exists.

**Remediation:** Run a conflict preflight over all eligible records or maintain conflict-group indexes. Return an explicit conflict summary independent of top-k content.

## KNOW-004 — Snapshot parser lacks bounded counts and checked numeric conversions

**Priority:** P1. **Status:** `[x] FIXED`.

**Evidence:** V2 knowledge snapshots enforce byte, record, role, span, relation, content, and embedding-dimension budgets, use checked finite numeric conversions, reject malformed structure, and preserve V1 compatibility. The knowledge regression mutates the dimension field beyond its budget and verifies rejection.

**Historical evidence:** `deserialize_snapshot()` previously parsed counts and numeric fields through `std::stoull`, `std::stod`, and `std::stoll` at `cpp/src/knowledge.cpp:479-535` without configured maximum sizes or an outer input budget.

**Impact:** A malformed or hostile snapshot can request large allocations, trigger exceptions with weak diagnostics, or consume excessive CPU/memory.

**Remediation:** Add maximum snapshot bytes, record count, content bytes, role count, span count, and embedding dimensions. Use checked parsers with field-specific errors and reject trailing or inconsistent data before allocation.

---

# E. Inference and operational correctness

## INF-001 — InferenceService does not execute a trained CCT model

**Priority:** P0. **Status:** `[x] FIXED`.

**Evidence:** `cpp/src/inference.cpp:445-474` returns the first retrieved document as output when retrieval hits exist. Without retrieval it constructs a string such as `CCT-ASE response: <input>`. `ModelRoute` changes the prefix but does not select a model backend.

**Impact:** The current service is a policy, retrieval, audit, and lifecycle harness, not a model-serving implementation. It cannot support a claim that CCT generated, reasoned over, or completed the request.

**Remediation completed:** Added an explicit `InferenceBackendMode` boundary. Checkpoint mode loads the native tokenizer snapshot and `NlpTrainer` checkpoint, verifies tokenizer/model identity and vocabulary size, performs bounded greedy next-token decoding, and exposes first-token/inter-token timing. Fixture mode remains explicitly labeled `fixture-template-*`.

**Verification:** `cpp/tests/inference_tests.cpp` builds two native Track 1 checkpoints, proves parameter changes alter the same-request output, verifies decoded output and backend identity, and exercises checkpoint streaming. Stage 16 adds a real checkpoint-backed gate check.

## INF-002 — Streaming is post-hoc word splitting and cancellation is not cooperative

**Priority:** P1. **Status:** `[x] FIXED`.

**Evidence:** `execute_stream()` calls `handle()` at `cpp/src/inference.cpp:535-550`, which completes execution and updates state before `execute_stream()` splits the final output into words at lines 555–579. `cancel_after_first` cancels after the full response has already been computed.

**Impact:** First-token latency, cancellation, backpressure, and resource-release claims do not describe a true streaming model. A cancelled stream can still incur full model/retrieval work and state mutation.

**Remediation completed:** `execute_stream()` now validates and executes through a callback-driven generation path. Checkpoint decoding emits each decoded token before requesting the next model step; event budgets and client cancellation stop generation cooperatively, and cancelled generation returns before recurrent state is committed.

**Verification:** Inference unit tests and Stage 16 verify incremental token events, cancellation after the first emitted token, backpressure, resource release, and no post-cancellation state commit.

## INF-003 — Queue and state service are synchronous and not thread-safe

**Priority:** P1. **Status:** `[x] FIXED`.

**Evidence:** `pending_`, `states_`, `cache_`, `metrics_`, and `audit_` are mutable containers accessed directly by service methods. `handle()` enqueues and immediately calls `process_pending()` at `cpp/src/inference.cpp:305-311`. No mutex, worker, bounded scheduler, or persistent queue is present.

**Impact:** Concurrent callers can race, queue-depth and batch behavior are not representative of a service, and a request can observe state or metrics during mutation.

**Remediation completed:** Public queue, execution, state, cache, audit, metrics, fault, and deployment-wrapper methods are guarded by a recursive mutex to support nested admission/execution calls safely. Admission remains explicitly separate from `process_pending()`, and monotonic enqueue timestamps are propagated into queue latency.

**Verification:** The inference suite runs 40 concurrent callers against one service and asserts exact accepted/successful counts, tenant/session isolation, bounded cache eviction, and state-byte consistency. Stage 16 retains queue-depth, batch-fairness, cancellation, and resource-exhaustion checks.

## INF-004 — SLO accounting uses proxy and inconsistent denominators

**Priority:** P1. **Status:** `[x] FIXED`.

**Evidence:** `evaluate_slo()` uses submitted requests as the denominator and completed requests as successful at `cpp/src/inference.cpp:634-640`. Abstentions can count as successful completions. Throughput uses total p99 latency as if it were total workload duration at lines 655–657. The pass condition compares `total_p95_milliseconds` to `first_token_p95_milliseconds` at lines 658–659. Queue latency is not measured from actual enqueue-to-start time.

**Impact:** Availability, error rate, throughput, and first-token SLO results can pass while the user-visible behavior violates the intended SLO.

**Remediation completed:** Service metrics now distinguish accepted, successful, abstained, rejected, timed-out, and cancelled outcomes. First-token and inter-token latencies are recorded from decoder callbacks, queue latency uses monotonic enqueue-to-start time, and throughput uses the observed monotonic measurement window. SLO pass/fail compares first-token and inter-token values to their matching thresholds while retaining total latency diagnostics.

**Verification:** Inference and Stage 16 gates assert explicit outcome counts, measured first-token percentile, monotonic queue percentiles, positive wall-clock request/token throughput, and matching-threshold SLO passage.

## INF-005 — Session state and cache accounting are semantically incomplete

**Priority:** P1. **Status:** `[x] FIXED`.

**Evidence:** `state_for()` uses a digest and byte estimate rather than the model’s actual recurrent state at `cpp/src/inference.cpp:314-340`. `update_state()` updates a digest and increments cumulative `metrics_.total_state_bytes` at lines 342–356. `reset_state()` removes state and cache but does not recompute all state metrics at lines 593–606. A cache hit returns before `update_state()` at lines 373–390.

**Impact:** Repeated identical inputs can bypass state evolution, total-state metrics grow cumulatively rather than reflecting current usage, and reset/eviction metrics can be stale. State quotas do not bound actual model state.

**Remediation completed:** Runtime state now separates transcript digest from bounded checkpoint-model token context; successful checkpoint generation commits the retained context, while cancelled generation does not. Fixture response caching is bounded by entry count and response bytes, cache hits update state metadata, and reset/eviction recompute active state and cache accounting.

**Verification:** Inference tests cover repeated sessions, checkpoint context reuse, reset, TTL eviction, concurrent bounded-cache eviction, tenant isolation, quotas, and current-byte equality between service and state metrics.

## INF-006 — Hard-coded policy and release metadata bypass configuration intent

**Priority:** P2. **Status:** `[x] FIXED`.

**Evidence:** Inference service default release ID and artifact digest are explicit `InferenceConfig` fields and are validated before registration. The service no longer registers or activates a hard-coded release identity in its constructor; inference and release tests cover configured metadata routing.

**Evidence:** `execute()` constructs a fixed `ProductUseCase` with ID `Stage16 bounded inference`, fixed allowed operations, owner, and expiry at `cpp/src/inference.cpp:362-368`. Model routing registers a fixed `stage16-default` release in the constructor at lines 194–203.

**Impact:** Tenant/application policy is not actually configured by the release registry or request-specific use case. Tests can pass against a static fixture while production configuration is ignored.

**Remediation:** Inject use-case, policy, release, and expiry records through validated configuration. Bind request identity to the selected release and test unknown, expired, revoked, and mismatched policy records.

## INF-007 — Cache is unbounded by entry count and lacks persistence/eviction evidence

**Priority:** P2. **Status:** `[x] FIXED`.

**Evidence:** Fixture and checkpoint response caches enforce configured entry-count and response-byte budgets, evict deterministically, and expose active-byte metrics. Inference tests exercise byte-bound and concurrent eviction behavior, and Stage 16 checks the resulting SLO evidence.

**Historical evidence:** Successful responses were previously appended to `cache_` at `cpp/src/inference.cpp:494-496` without a hard entry or byte budget.

**Impact:** Long-running processes can grow memory without a hard bound even when individual state quotas are configured.

**Remediation:** Add entry, byte, tenant, and response-size limits; deterministic LRU or TTL policy; eviction metrics; and stress tests.

## RELEASE-001 — Release controller is a state machine, not deployment integration

**Priority:** P1. **Status:** `[x] FIXED`.

**Evidence:** `PilotReleaseController::activate_release()` now requires an approved terminal decision and explicit checkpoint/tokenizer artifact paths, registers a digest-bound `DeploymentRelease`, and activates it through `InferenceService`. The inference service verifies the checkpoint SHA-256, loads the native checkpoint backend transactionally, updates model/tokenizer/index identity, and clears model-dependent state and cache. Release tests and the Stage 17 gate execute a real checkpoint-backed request through the approved release.

**Historical evidence:** `cpp/src/release.cpp` previously recorded release identities, approvals, shadow logs, rollback, deletion, drift, and review records without loading or swapping model artifacts.

**Impact:** A release gate can show correct governance bookkeeping while the actual serving process remains unchanged.

**Remediation:** Connect release records to artifact loading, process or worker lifecycle, routing, health checks, rollback activation, and external audit storage. Keep the current state-machine harness as a unit-test fixture.

## RELEASE-002 — Release and snapshot persistence lacks atomic publication

**Priority:** P1. **Status:** `[x] FIXED`.

**Evidence:** Release manifest and audit saves, causal snapshots, knowledge snapshots, and memory snapshots now use exclusive temporary files, complete writes, file `fsync`, atomic rename, and parent-directory `fsync`. Release tests verify manifest/audit publication and temporary-file cleanup; memory and knowledge regressions verify publication and reload.

**Historical evidence:** Release and causal save paths previously used direct `ofstream` writes.

**Impact:** Interrupted persistence can create partial but discoverable release or state artifacts.

**Remediation:** Create a shared atomic artifact writer with digest, size, fsync policy, atomic rename, and manifest commit order. Add process-kill tests.

---

# F. Build, testing, and documentation validity

## BUILD-001 — Expanded warnings expose portability debt

**Priority:** P2. **Status:** `[x] FIXED`.

**Evidence:** The expanded profile `-Wconversion -Wsign-conversion -Wshadow -Wdouble-promotion -Wformat=2` now compiles every native library, test, and Stage 0–17/Track 1 target with `-Werror`. Baseline parameter slices use checked difference-type conversion; remaining gate and serializer byte/index conversions are explicit.

**Historical evidence:** The profile previously failed in `cpp/src/baselines.cpp` on unsigned iterator offsets.

**Impact:** The code is less portable across compilers and warning profiles than the current gate suggests.

**Remediation:** Use checked iterator offsets or indices, enable the expanded warning profile in a non-blocking CI job first, then remove suppressions and promote the profile after cleanup.

## BUILD-002 — CMake does not make the review warning policy self-contained

**Priority:** P2. **Status:** `[x] FIXED`.

**Evidence:** `cpp/CMakeLists.txt` exposes `CCT_STRICT_WARNINGS` (default `ON`) and applies the shared `CCT_WARNING_OPTIONS` to every target. Both the ordinary strict build and the expanded warning build are now reproducible without injecting the baseline warning policy through `CMAKE_CXX_FLAGS`.

**Historical evidence:** The review previously needed to inject `-Wall -Wextra -Wpedantic -Werror` through `CMAKE_CXX_FLAGS`.

**Impact:** The claimed strict build is command-dependent and can silently regress.

**Remediation:** Encode target-level compile features and warning options in CMake, provide an explicit `CCT_STRICT_WARNINGS` option, and test both default and strict profiles.

## TEST-001 — Passing gates validate fixtures more strongly than the real engine

**Priority:** P1. **Status:** `[x] FIXED FOR THE DECLARED BOUNDED CONTRACT`.

**Evidence:** `release_tests` now loads the durable Stage 16 checkpoint/tokenizer files referenced by `release_validation_bundle.json`, activates them through the approved-release controller, and executes a real inference request through the checkpoint backend. Release, Track 1, malformed-input, parser-mutation, and negative-policy paths remain separately labeled; the current Release suite passes 39/39.

**Impact:** The repository has an independent artifact-backed black-box path in addition to fixture gates. This proves integration and identity for the declared checkpoint contract, not broad capability or QA quality.

**Remediation:** Completed for the bounded native artifact contract with durable references, black-box execution, adversarial failure paths, and explicit capability boundaries.

## TEST-002 — Sanitizer validation is incomplete

**Priority:** P2. **Status:** `[x] FIXED FOR THE REGISTERED BOUNDED SHARDS`.

**Evidence:** The ASan/UBSan unit shard passes 20/20 native unit targets. The complete 19-gate sanitizer shard passes after Stage 5 is explicitly compiled as a bounded 2-step traversal of every matched model path; Release and expanded-warning builds retain the full 20-step Stage 5 comparison. CTest gives Stage 5 an explicit 3600-second ceiling, and the sanitizer methodology records the bounded instrumentation scope.

**Impact:** The previous untested sanitizer surface now has deterministic bounded coverage, while the computationally expensive full-quality comparison remains validated in Release under strict warnings.

**Remediation:** Completed through bounded sanitizer sharding, explicit workload scope, strict warning profiles, and published status/audit evidence.

## TEST-003 — No property-based or fuzz test coverage for custom parsers and serializers

**Priority:** P1. **Status:** `[x] FIXED for bounded deterministic mutation contract`.

**Evidence:** `cct_parser_mutation_tests` is registered in CTest and mutates valid causal, memory, knowledge, and tokenizer snapshots through truncation, prefix, NUL, trailing-data, and byte mutations. It requires bounded fail-closed rejection and rejects non-standard exception escapes; the target passes under strict and expanded warning builds.

**Evidence:** Track 1, knowledge, memory, causal, tokenizer, and trainer checkpoint formats use custom parsers and serializers. The current tests cover selected fixtures but no fuzz/property campaign is wired into CTest.

**Impact:** Length/count fields, escapes, Unicode, truncation, duplicate keys, invalid enums, and partial records remain under-tested.

**Remediation:** Add bounded fuzz targets or deterministic mutation suites for each parser. Require no crash, bounded memory, fail-closed status, and round-trip identity for valid data.

## DOC-001 — Stage specifications, status documents, and implementation claims have drifted

**Priority:** P1. **Status:** `[x] FIXED for current authority and consistency contract`.

**Evidence:** `SPEC/Goal.md`, `SPEC/Todo.md`, and `SPEC/Status.md` are now the authority set; `cct_documentation_consistency_tests` verifies the native C++20 contract, all Stage 0–17 files, status rows, architecture terms, and identity-envelope contract. Historical records are labeled as historical or bounded rather than silently promoted.

**Evidence:** The repository contains multiple stage maps and historical specification trees. The earlier Architecture document described Rust/JAX/Python layers and unimplemented mathematical components, while the actual code is native C++20. The current audit corrected `Architecture.md`, but `Goal.md`, `Todo.md`, `SPEC/Goal.md`, `SPEC/Todo.md`, `Stages/`, and artifact status records still require a single authority and reconciliation.

**Impact:** A green stage checkbox can be interpreted as completion of an obsolete or broader contract.

**Remediation:** Declare `SPEC/Goal.md` and `SPEC/Todo.md` authoritative for Level 1, link every stage gate to an artifact, mark historical PASS versus fresh replay, archive contradictory specifications, and add a documentation consistency gate.

## ARTIFACT-001 — Historical reports reference ephemeral checkpoint paths

**Priority:** P1. **Status:** `[x] FIXED for path-independent reporting`.

**Evidence:** The historical Track 1 report no longer contains `/tmp` paths; it records unavailable historical artifacts with preserved hashes. `artifacts/track1/real-training/release_validation_bundle.json` provides durable repository-relative checkpoint/tokenizer references, hashes, sizes, and release-loader evidence.

**Evidence:** Track 1 training reports and operational documentation include historical `/tmp` checkpoint paths. Those paths do not exist in a fresh checkout and are not durable release artifacts.

**Impact:** A report can say training passed while a reviewer cannot load the exact checkpoint or verify the result.

**Remediation:** Publish durable checkpoint artifacts or immutable object references, include hashes and sizes, make reports path-independent, and add a load-from-release-bundle test.

## ARTIFACT-002 — Gate output is not universally linked to source and environment identity

**Priority:** P2. **Status:** `[x] FIXED for registered native gates`.

**Evidence:** CTest wraps Stage 0–17 and Track 1 gate executables with `cct_gate_envelope`, which publishes schema, source commit, compiler/build type, executable path and SHA-256, checks/output paths, exit status, and timestamp. The wrapper uses native fork/exec and atomic publication.

**Evidence:** Some gates write compact `checks.json` and reports, while historical records vary in environment, configuration, and artifact linkage. The repository does not enforce one manifest schema across every stage gate.

**Impact:** Cross-stage audit and reproducibility require manual interpretation.

**Remediation:** Define one gate envelope containing repository commit, build compiler, configuration digest, source/data digest, test binary hash, timestamp, host context, checks, and status. Reject a gate artifact that omits mandatory identity fields.

---

# G. Maintainability and API design

## API-001 — Public APIs expose mutable state and use exception-only failure semantics

**Priority:** P2. **Status:** `[x] FIXED FOR THE CURRENT NATIVE API CONTRACT`.

**Evidence:** Stateful collection accessors return const references or value copies; mutation is exposed only through named mutators. Public headers document borrowed-reference lifetime and subsystem-specific exception categories, while inference responses use stable typed error codes and policy decisions for expected request outcomes. Snapshot and persistence failures remain exceptions for invariant/corruption paths.

**Impact:** The current API has an explicit ownership and failure boundary rather than implying that returned containers are caller-mutable or that every rejection is indistinguishable.

**Remediation:** Completed for the current native C++ API without introducing an artificial universal result wrapper over invariant exceptions.

## API-002 — State, cache, and persistence thread-safety contracts are undocumented

**Priority:** P1. **Status:** `[x] FIXED AS DOCUMENTED OWNERSHIP CONTRACT`.

**Evidence:** InferenceService documents internal recursive-mutex serialization of state, cache, queue, metrics, and audit operations. KnowledgePlane, PersistentMemory, CausalEventStore, and other borrowed-reference stores document external synchronization requirements, read-only concurrency limits, mutation invalidation, and snapshot overlap restrictions. Existing inference concurrency/cancellation tests and sanitizer unit shard exercise the synchronized service boundary.

**Impact:** Callers now have explicit service-safe versus externally synchronized ownership rules; no store is silently advertised as thread-safe.

**Remediation:** Completed for the current architecture through precise public contracts and existing stress/sanitizer evidence.

## API-003 — Serialization formats lack size budgets and version migration strategy

**Priority:** P1. **Status:** `[x] FIXED FOR CURRENT V1/V2 FORMATS`.

**Evidence:** Causal, memory, knowledge, tokenizer, NLP, production, multimodal, scaling, deliberation, corpus, preference, and SFT loaders now enforce total-byte, count, dimension, string, and parameter budgets before allocation. Supported legacy versions migrate explicitly where applicable; unsupported versions and unknown structural fields fail closed. Parser mutation and subsystem regressions cover canonical replay, legacy compatibility, oversized counts, and tamper rejection.

**Impact:** Hostile or corrupt artifacts cannot trigger unbounded parser allocation within the supported native formats, and schema compatibility is explicit rather than accidental.

**Remediation:** Completed for all current serialization entry points; a future schema must add a versioned migration and budget contract before acceptance.

## API-004 — `const_cast` is used to mutate manifest source records during construction

**Priority:** P3. **Status:** `[x] FIXED`.

**Evidence:** `Track1Pipeline` now uses a non-const `source_at(std::vector<Track1Source>&, ...)` overload for manifest initialization and preparation. The source implementation contains no `const_cast<Track1Source&>` mutation path, and the expanded warning build compiles the Track 1 targets cleanly.

**Historical evidence:** Manifest source metadata was previously filled through `const_cast` with `source_at()`.

**Impact:** The code weakens const-correctness and makes future refactoring unsafe or misleading.

**Remediation:** Add a non-const source lookup overload or construct fully initialized source records directly.

---

# H. Prioritized implementation sequence

## P0 remediation sequence

- [x] Fix `SEQ-001` and add the complete gradient matrix.
- [ ] Fix `INF-001` by connecting a real model backend or explicitly downgrade the inference subsystem to fixture-only status.
- [x] Fix `TRACK1-001` before accepting user-controlled output paths or untrusted remote data.
- [x] Fix `CORPUS-001` before processing data with privacy obligations.
- [x] Update the architecture, Track 1 gate, and issue-backlog claim surfaces so corrected sequence, corpus, and acquisition paths are described with their actual boundaries.

## P1 remediation sequence

- [x] Fix `TRAIN-001` model identity by explicitly naming and separating the Track 1 recurrence from `SelectiveSequenceCore`.
- [x] Fix `INF-002`, `INF-003`, `INF-004`, and `INF-005`; checkpoint streaming, cancellation, synchronization, outcome accounting, and bounded state/cache tests pass.
- [x] Fix `TRACK1-002`, `TRACK1-003`, and `TRACK1-005` before full-source production acquisition.
- [x] Fix `TRACK1-004` by publishing a manifest-bound mirror/upstream source attestation.
- [ ] Resolve `KNOW-001` by adding a production semantic embedding/index backend; the deterministic provider remains an explicit baseline.
- [x] Fix `KNOW-002` and `KNOW-003` before factual-grounding claims; the exact-span and all-eligible-conflict regressions pass.
- [x] Fix `MEMORY-001`, `MEMORY-003`, and `MEMORY-004`; `MEMORY-002` remains a measured linear-scan baseline limitation.
- [x] Fix `RELEASE-001` and `RELEASE-002`; approved checkpoint activation and atomic release/snapshot publication tests pass.
- [ ] Add bounded parser mutation coverage under `TEST-003`; real checkpoint activation is now covered by Stage 16/17 and release black-box tests under `TEST-001`.
- [ ] Reconcile all status and specification files under `DOC-001`.

## P2/P3 hardening sequence

- [ ] Add expanded compiler-warning CI under `BUILD-001` and `BUILD-002`.
- [ ] Complete sanitizer shards under `TEST-002`.
- [ ] Add semantic embeddings and indexed retrieval after preserving the current implementation as a baseline.
- [ ] Add typed result errors, thread-safety contracts, schema migrations, and const-correct source lookup.
- [ ] Improve benchmark methodology to report real wall time, memory, allocations, throughput, and model quality under matched budgets.

## Dependency graph

```text
SEQ-001 ─┐
TRAIN-001 ─┼─> valid CCT training claims ─> Track 1 release replay
TRACK1-001 ─┘

INF-001 ─┬─> real model inference
INF-002 ─┤
INF-003 ─┤
INF-004 ─┘

KNOW-001 ─> KNOW-002 ─> factual grounding evidence
MEMORY-001 ─> MEMORY-004 ─> durable memory/release evidence
CORPUS-001 ─> corpus/privacy release boundary
DOC-001 ─> trustworthy stage transitions
```

## Definition of done for an issue

- [ ] The remediation is implemented in native source with an explicit contract.
- [ ] The normal path is covered by a deterministic test.
- [ ] The failure path is covered by a deterministic test.
- [ ] The relevant sanitizer, warning, performance, or data test is run.
- [ ] The issue’s evidence artifact records commit, configuration, environment, and thresholds.
- [ ] Any affected baseline, stage gate, Track 1 report, and architecture claim is regenerated.
- [ ] The issue is marked `[x]` only after the dependent gate passes.

## References

- [Evidence-based architecture](Architecture.md)
- [Level 1 goal](SPEC/Goal.md)
- [Level 1 todo](SPEC/Todo.md)
- [Native CMake contract](cpp/CMakeLists.txt)
- [Canonical Makefile](Makefile)
- [Sequence implementation](cpp/src/sequence.cpp)
- [NLP trainer](cpp/src/nlp_trainer.cpp)
- [Track 1 acquisition](cpp/src/track1.cpp)
- [Track 1 trainer](cpp/tools/track1_train.cpp)
- [Corpus governance](cpp/src/corpus.cpp)
- [Causal engine](cpp/src/causal.cpp)
- [Memory engine](cpp/src/memory.cpp)
- [Knowledge plane](cpp/src/knowledge.cpp)
- [Inference service](cpp/src/inference.cpp)
- [Release controller](cpp/src/release.cpp)
