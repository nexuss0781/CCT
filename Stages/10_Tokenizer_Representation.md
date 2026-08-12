# Stage 10 — Tokenizer and Representation Engine
## Versioned NLP Inputs, Packed Batches, and Compatibility

**Predecessor:** Stage 9 — Governed Data and Corpus Pipeline  
**Successor:** Stage 11 — Trainable Native NLP Core  
**Status:** Specification; implementation not started  
**Implementation:** Native C++20 tokenizer, batcher, tests, and gate

## Purpose

Stage 10 defines the model’s input contract. It converts governed corpus records into versioned token streams and deterministic training batches while preserving byte fallback, Unicode coverage, code identifiers, structured control tokens, offsets, provenance, and split boundaries. A tokenizer change is a model-interface change and must be released as a versioned artifact.

## Scope and non-goals

The stage includes byte-level, subword, and hybrid tokenizer candidates; vocabulary construction; normalization; reserved/control tokens; offset mapping; token-to-source spans; packed and padded causal batches; attention/loss masks; sequence boundaries; tokenizer serialization; migration diagnostics; and throughput measurement. It does not train the language model or select a production architecture.

## Candidate tokenization strategies

| Candidate | Strength | Required risk test |
|---|---|---|
| Byte fallback | Universal coverage and deterministic recovery | Sequence expansion and generation efficiency |
| Subword | Token compression and throughput | Rare identifiers, Unicode, code, and contamination |
| Hybrid byte/subword | Coverage plus compression | Boundary semantics and tokenizer complexity |

The selected tokenizer must be evaluated on general text, code, multilingual Unicode, structured data, malformed input, adversarial separators, and long-context documents. No candidate is accepted solely by average compression.

## Required implementation

| Component | Implementation | Contract |
|---|---|---|
| Normalizer | Versioned Unicode and whitespace policy | Same input/version gives same normalized bytes |
| Vocabulary | Reserved IDs, token strings, byte fallback, frequency report | IDs are immutable after release |
| Encoder | Text/code to token IDs with offsets | Round-trip and source-span mapping are tested |
| Decoder | IDs to text/bytes | Valid IDs decode deterministically; invalid IDs fail closed |
| Special tokens | BOS/EOS/PAD/UNK/task/schema/citation markers | Control tokens cannot collide with content tokens |
| Batch packer | Packed and padded causal sequences | Boundaries and loss masks are exact |
| Provenance | Record and span identity | Tokens remain traceable to source records |
| Serialization | Tokenizer/config/vocabulary snapshot | Snapshot hash is part of model identity |
| Throughput | Tokens/sec and memory measurement | Reported at fixed hardware and batch settings |

## Token and batch contract

Every batch item must expose token IDs, sequence boundaries, record IDs, source offsets, loss mask, padding mask, task/control labels, and tokenizer version. The trainer must be able to distinguish a real end-of-document token from padding and packed-sequence boundaries. No loss may be computed across unrelated packed records.

The tokenizer must support an explicit byte-fallback path for any input that cannot be represented by the primary vocabulary. Fallback behavior must not silently drop bytes or normalize away security-sensitive distinctions.

## Evaluation harness

The harness must include:

1. exhaustive byte round-trip tests for all 256 byte values;
2. Unicode normalization and invalid-byte tests;
3. code identifier, indentation, string, and comment fixtures;
4. structured JSON and delimiter fixtures;
5. control-token collision tests;
6. encode/decode round-trip tests;
7. source-offset and citation-span tests;
8. packed-batch boundary and loss-mask tests;
9. padded-batch equivalence tests;
10. tokenizer snapshot serialization and hash tests;
11. tokenizer version incompatibility rejection;
12. throughput and memory reports over increasing sequence lengths.

The harness must compare all candidates on identical data and report token count, compression ratio, unknown/fallback rate, batch efficiency, and source-offset coverage.

## Mandatory gate checks

| Check | Pass condition |
|---|---|
| Byte fallback | All byte values round-trip without loss |
| Unicode | Declared Unicode and invalid-input suite passes without silent corruption |
| Code coverage | Identifiers, indentation, literals, and comments preserve required distinctions |
| Special tokens | Reserved IDs are stable and cannot be confused with content |
| Round-trip | Encode/decode passes declared exact or normalized equivalence rules |
| Offsets | Every trainable token maps to a source span or explicit control category |
| Packed loss | No loss is charged across document or sequence boundaries |
| Padding | Packed and padded evaluation agree within tolerance |
| Versioning | Snapshot/config/hash round-trip exactly; incompatible versions fail closed |
| Efficiency | Candidate metrics are measured at fixed data, hardware, and batch settings |
| Contamination | Tokenizer construction cannot read evaluator-only records |
| Reproducibility | Same corpus/config/seed produces identical vocabulary and batches |

## Pass/fail transition

Stage 10 passes only when one tokenizer candidate is selected for the next training pilot, all alternatives have comparison reports, and the selected snapshot has an immutable hash. A `PASS` authorizes Stage 11 to consume the tokenizer. It does not authorize large-scale training or claim that token compression predicts model quality.

A `FAIL` requires correction of round-trip, masking, provenance, or efficiency defects. A `BLOCKED` result must name unresolved Unicode, licensing, or compatibility constraints.

## Deliverables

The stage must deliver tokenizer implementations, vocabulary builder, snapshot format, batch packer, candidate comparison report, source-offset mapping, native regression suite, machine-readable gate artifacts, tokenizer model card, and CI command.

## Explicit limitations

Tokenizer metrics are not language-model quality metrics. A tokenizer with fewer tokens may still produce worse learning or safety behavior. Any production tokenizer change after Stage 10 requires a new model checkpoint or a formally tested migration.
