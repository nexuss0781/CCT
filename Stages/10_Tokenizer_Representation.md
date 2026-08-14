## Versioned NLP Inputs, Packed Batches, and Compatibility

**Predecessor:** Stage 9 — Governed Data and Corpus Pipeline  
**Successor:** Stage 11 — Trainable Native NLP Core  
**Status:** Implemented and gated `PASS` on declared real-source and application-shaped fixtures. The current native gate has twelve mandatory checks, including strict snapshot-schema and batch-metadata failure closure; Stage 11 requires explicit approval.
**Implementation:** Native C++20 tokenizer, candidate comparison, snapshot format, causal batch packer, regression suite, and artifact-producing gate

## Purpose

Stage 10 defines the model’s input contract. It converts governed corpus records into versioned token streams and deterministic training batches while preserving byte fallback, Unicode coverage, code identifiers, structured control tokens, offsets, provenance, and split boundaries. A tokenizer change is a model-interface change and must be released as a versioned artifact.

## Scope and non-goals

The stage includes byte-level, subword, and hybrid tokenizer candidates; vocabulary construction; normalization; reserved/control tokens; offset mapping; token-to-source spans; packed and padded causal batches; attention/loss masks; sequence boundaries; tokenizer serialization; migration diagnostics; and throughput measurement. It does not train the language model or select a production architecture.

The implementation is located in `cpp/include/cct/tokenizer.hpp` and `cpp/src/tokenizer.cpp`. Native regression coverage is in `cpp/tests/tokenizer_tests.cpp`; the machine-readable gate is `cpp/tools/stage10_gate.cpp`; the expanded threshold and failure contract is `Stages/10_Expanded_Gate_Contract.md`.

## Candidate tokenization strategies

| Candidate | Strength | Required risk test | Stage 10 result |
|---|---|---|---|
| Byte fallback | Universal coverage and deterministic recovery | Sequence expansion and generation efficiency | Measured baseline; exact byte round-trip |
| Subword | Token compression and throughput | Rare identifiers, Unicode, code, and contamination | Measured on identical fixtures; byte fallback retained |
| Hybrid byte/subword | Coverage plus compression | Boundary semantics and tokenizer complexity | **Selected** for the Stage 11 input pilot |

The selected hybrid candidate is a deterministic longest-first byte-piece candidate with universal byte fallback. Vocabulary construction consumes only records explicitly marked training-allowed and non-evaluator. Frequency ties are resolved deterministically; validation and evaluator records are not read by the builder.

The Stage 10 comparison uses the fixed real-source and application-shaped fixture set from the expanded contract. The observed selected-candidate compression ratio was `1.85448×` relative to byte tokenization, with `100%` offset coverage and positive throughput/memory measurements. These are representation metrics, not language-model quality metrics.

## Required implementation

| Component | Implementation | Contract | Result |
|---|---|---|---|
| Normalizer | `preserve-bytes-v1` | Same input/version gives same normalized bytes | PASS; malformed bytes and security-sensitive separators preserved |
| Vocabulary | Reserved IDs, byte IDs, learned pieces, frequency report | IDs are immutable after release | PASS; reserved IDs `0–8`, byte IDs `256–511`, learned IDs from `512` |
| Encoder | Byte, subword, and hybrid candidates with offsets | Round-trip and source-span mapping are tested | PASS |
| Decoder | Deterministic ID-to-byte decoding | Valid IDs decode; invalid IDs fail closed | PASS |
| Special tokens | BOS/EOS/PAD/UNK/TASK/SCHEMA/CITATION/document and sequence boundaries | Control tokens cannot collide with content | PASS |
| Batch packer | Packed and padded causal sequences | Boundaries, control categories, and loss masks are exact | PASS; packed/padded loss checksums agree exactly and tampered metadata rejects |
| Provenance | Record ID and half-open source spans | Every content token is traceable | PASS; control tokens carry explicit categories |
| Serialization | Canonical tokenizer/config/vocabulary snapshot | Snapshot hash is part of model identity | PASS; incompatible, duplicate-field, trailing-data, and malformed snapshots reject |
| Throughput | Fixed-fixture bytes/sec, tokens/sec, and memory measurements | Reported under fixed settings | PASS; all candidates exceeded the `10,000` bytes/sec gate threshold |

## Token and batch contract

Every batch item exposes token IDs, sequence boundaries, record IDs, source offsets, loss mask, padding mask, boundary mask, control labels, and tokenizer version. The trainer can distinguish a real end-of-document token from padding and packed-sequence boundaries. No loss is computed across unrelated packed records.

The tokenizer supports an explicit byte-fallback path for any input that cannot be represented by the primary vocabulary. Fallback behavior does not silently drop bytes or normalize away security-sensitive distinctions.

For content tokens, source offsets are half-open byte intervals `[start,end)`. Control tokens have empty spans and explicit control categories. The default `preserve-bytes-v1` policy makes normalized offsets identical to source offsets for all valid and invalid byte sequences.

For a document represented as `[BOS, content..., EOS]`, the target at position `i` is the next token within that document. The final EOS position has no target and has `loss_mask=false`. Padding positions contain `<PAD>`, have `loss_mask=false`, and have an inactive padding mask. Packed sequence boundaries never target the first token of the next document.

## Evaluation harness

The native harness includes exhaustive byte round-trips for all `256` byte values; valid Unicode and malformed-byte fixtures; code identifiers, indentation, literals, and comments; structured JSON and delimiters; control-token collision tests; exact encode/decode tests; source-offset and record-provenance tests; packed boundary and loss-mask tests; padded-batch equivalence; snapshot serialization and hash tests; incompatible-version, invalid-ID, duplicate-singleton, and trailing-data rejection; externally supplied document provenance validation; tampered packed/padded metadata rejection; throughput and memory reports; fixed-data comparison of all three candidates; evaluator-only construction rejection; and reproducibility under reordered input records.

The current native tokenizer regression suite contains **8/8** tests, and `cct_stage10_gate` has **12/12** mandatory checks. The full repository CTest matrix is recorded with the released gate evidence.

## Mandatory gate checks

| Check | Pass condition | Result |
|---|---|---|
| Byte fallback | All byte values round-trip without loss | PASS |
| Unicode | Declared Unicode and invalid-input suite passes without silent corruption | PASS |
| Code coverage | Identifiers, indentation, literals, and comments preserve required distinctions | PASS |
| Special tokens | Reserved IDs are stable and cannot be confused with content | PASS |
| Round-trip | Encode/decode passes declared exact or normalized equivalence rules | PASS |
| Offsets | Every trainable token maps to a source span or explicit control category | PASS; `100%` |
| Packed loss | No loss is charged across document or sequence boundaries | PASS; zero cross-boundary loss |
| Padding | Packed and padded evaluation agree within tolerance | PASS; exact checksum equality |
| Strict metadata closure | Externally supplied document provenance and packed/padded control and boundary metadata reject if tampered | PASS |
| Versioning | Snapshot/config/hash round-trip exactly; incompatible versions, duplicate singleton fields, and trailing data fail closed | PASS |
| Efficiency | Candidate metrics are measured at fixed data, hardware, and batch settings | PASS; all candidates measured |
| Contamination | Tokenizer construction cannot read evaluator-only records | PASS; builder rejects them |
| Reproducibility | Same corpus/config/seed produces identical vocabulary and batches | PASS |

## Selected release artifact

The selected Stage 10 tokenizer is the hybrid candidate with tokenizer version `cct-ase-tokenizer-v1`. The gate records the immutable snapshot hash in `artifacts/stage-10/cpp-gate/tokenizer_snapshot.json`, `tokenizer_snapshot.bin`, and `release_record.json`. The release record retains `training_authorized: false` and `approval_required: true`; Stage 11 may consume the snapshot only after explicit approval.

The gate writes `checks.json`, `metrics.json` (including mandatory-check count), `candidate_comparison.json`, `tokenizer_snapshot.json`, `tokenizer_snapshot.bin`, `batch_report.json`, `reproducibility.json`, `incident_log.json`, `release_record.json`, and `report.md` beneath `artifacts/stage-10/cpp-gate/`.

## Pass/fail transition

Stage 10 passes only when one tokenizer candidate is selected for the next training pilot, all alternatives have comparison reports, and the selected snapshot has an immutable hash. The clean release gate satisfies these conditions and authorizes Stage 11 preparation only after explicit approval. It does not authorize large-scale training or claim that token compression predicts model quality.

A `FAIL` requires correction of round-trip, masking, provenance, contamination, compatibility, or efficiency defects. A `BLOCKED` result must name unresolved Unicode, licensing, or compatibility constraints.

## Deliverables

Stage 10 delivers the native public API and implementation in `cpp/include/cct/tokenizer.hpp` and `cpp/src/tokenizer.cpp`, the byte/subword/hybrid candidate comparison, versioned vocabulary builder, immutable canonical snapshot format, universal byte fallback, source-offset and record-provenance mapping, packed and padded causal batch packer, throughput and memory measurement, native regression suite in `cpp/tests/tokenizer_tests.cpp`, artifact-producing gate in `cpp/tools/stage10_gate.cpp`, expanded gate contract, tokenizer model card, machine-readable gate artifacts, and the `stage10-test`, `stage10-gate`, and `ci-stage10` build targets.

## Explicit limitations

Tokenizer metrics are not language-model quality metrics. The selected hybrid candidate was validated on a small declared fixture set and does not establish production-scale throughput, multilingual completeness, vocabulary optimality, safety behavior, or model quality. The normalization implementation currently exposes the byte-preserving `preserve-bytes-v1` policy; future normalization changes require new versioned snapshots and migration tests. Any production tokenizer change after Stage 10 requires a new model checkpoint or a formally tested migration.
