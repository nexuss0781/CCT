# Stage 10 Expanded Gate Contract

## Tokenizer and Representation Engine

**Predecessor:** Stage 9 — Governed Data and Corpus Pipeline  
**Successor:** Stage 11 — Trainable Native NLP Core  
**Implementation boundary:** Native C++20 only; no Python tokenizer, vocabulary builder, or batch implementation is accepted.  
**Gate status:** Native C++20 implementation complete; the current formal gate is PASS with twelve mandatory checks and explicit approval still required for the successor training stage.

## 1. Gate purpose

Stage 10 establishes the immutable input contract consumed by the future native training pilot. It must convert governed training records into deterministic token streams and causal batches while preserving bytes, source provenance, control-token semantics, sequence boundaries, and version compatibility. The gate evaluates candidate tokenizers on the same declared training fixture and does not treat token compression as evidence of language-model quality.

A passing gate authorizes Stage 11 to consume the selected tokenizer snapshot. It does not authorize large-scale training, claim production NLP quality, or imply that the selected candidate is optimal outside the declared fixtures.

## 2. Declared application-shaped fixtures

The gate uses a fixed corpus manifest derived from Stage 9 and additional in-memory adversarial records. Vocabulary construction may consume only records explicitly marked `Train` and `training_allowed=true`. Validation, test, and evaluator-only records are available only to evaluation checks and must never affect vocabulary IDs, frequencies, snapshots, or batch ordering.

| Fixture family | Required content | Split/use | Failure risk exercised |
|---|---|---|---|
| Real reference text | Project Gutenberg `pg1342.txt` and `pg11.txt` as governed Stage 9 sources | `pg1342`: train; `pg11`: validation only | Real-source provenance, long documents, punctuation, Unicode-adjacent text |
| Real native code | CCT `production.cpp` and `corpus.cpp` from the governed Stage 9 manifest | Train | Namespaces, types, comments, string literals, braces, operators, indentation |
| Code boundary fixture | `auto user_id = read_value(); // preserve user_id` plus tabs, spaces, escaped literals, and adjacent identifiers | Train regression fixture | Identifier and whitespace distinctions |
| Structured data | JSON containing nested objects, escaped quotes, colons, commas, arrays, and duplicate-looking keys | Train regression fixture | Delimiter preservation and exact round-trip |
| Multilingual Unicode | Valid UTF-8 containing accented Latin, Greek, Cyrillic, CJK, combining marks, and emoji | Train regression fixture | Unicode coverage and byte offsets |
| Invalid-byte fixture | Every byte value, including malformed UTF-8 sequences and embedded zero bytes | Regression only | Universal byte fallback and no silent corruption |
| Control collision fixture | Literal text containing the strings `<PAD>`, `<BOS>`, `<EOS>`, `<UNK>`, `<TASK>`, `<SCHEMA>`, `<CITATION>`, and `<DOC_BOUNDARY>` | Train regression fixture | Content/control namespace separation |
| Separator fixture | Repeated newlines, CRLF, CR, tabs, spaces, NUL, record delimiters, and packed-document boundaries | Train regression fixture | Boundary semantics and normalization policy |
| Evaluator canary | A unique held-out sentence that must be detectable as evaluator content | Evaluator-only | Contamination prevention |

The real files are read from the repository paths and checked against their Stage 9 manifest hashes before candidate construction. The gate fails closed if a declared real file is missing, changed, or ingested under the wrong split.

## 3. Candidate contract

Three deterministic candidates must be implemented and measured on identical documents, configuration, and batch settings.

| Candidate | Construction rule | Required fallback |
|---|---|---|
| Byte | One content token per source byte | None required because all 256 byte values are represented |
| Subword | Deterministic frequency-ranked lexical pieces learned from train documents only | Byte tokens for every unmatched span |
| Hybrid | Subword pieces plus deterministic frequent byte n-grams matched by longest-first scan | Byte tokens for every unmatched span |

All candidates must use the same normalization version, special-token IDs, record order, batch limits, and measurement corpus. Vocabulary ties are resolved by lexicographic byte order after frequency, so construction does not depend on hash-map iteration order or thread scheduling. Evaluator-only and validation records must be rejected by the builder rather than silently ignored.

The candidate comparison must report token count, bytes per token, compression ratio versus bytes, fallback/unknown rate, source-offset coverage, packed utilization, padded utilization, throughput, and measured resident-memory samples. A candidate cannot be selected solely because it has the lowest token count.

## 4. Immutable vocabulary and snapshot contract

The reserved control namespace is fixed for Stage 10 and must remain stable:

| ID | Control token | Category |
|---:|---|---|
| 0 | `<PAD>` | Padding; never a content byte |
| 1 | `<BOS>` | Beginning of document/sequence |
| 2 | `<EOS>` | End of document |
| 3 | `<UNK>` | Invalid vocabulary ID or explicit unknown diagnostic only |
| 4 | `<TASK>` | Task control |
| 5 | `<SCHEMA>` | Schema control |
| 6 | `<CITATION>` | Citation control |
| 7 | `<DOC_BOUNDARY>` | Packed document boundary |
| 8 | `<SEQ_BOUNDARY>` | Packed sequence boundary |

Byte content IDs occupy a separate immutable range beginning at `256`; candidate-specific learned content IDs begin at `512`. A content token is never assigned a reserved ID, and a literal control-token string in source text is encoded as content bytes/pieces rather than as a control ID. Duplicate IDs, duplicate content entries, missing reserved IDs, invalid ranges, and control/content collisions are fatal snapshot errors.

Every snapshot must include the format version, tokenizer version, candidate kind, normalization version, seed, reserved-ID table, vocabulary entries, frequency report, and construction provenance. Serialization must be canonical: identical configuration, train corpus bytes, and seed produce identical snapshot bytes and SHA-256 hash. Deserialization must reject unknown format versions, incompatible tokenizer major versions, malformed records, duplicate IDs, truncated data, and hash mismatches.

## 5. Token and provenance contract

Each encoded item exposes the token ID, token kind, candidate name, source record ID, source byte start, source byte end, and either a source span or an explicit control category. Content spans use half-open byte intervals `[start,end)` and must be non-empty. Control tokens use an explicit control category and an empty source span; they are not counted as missing provenance. The source record ID must be retained through batching.

For normalized inputs, the tokenizer must retain a deterministic normalized-to-source mapping. The default Stage 10 policy is `preserve-bytes-v1`, which leaves all bytes—including malformed UTF-8 and security-sensitive separators—unchanged. Optional canonical-newline behavior, if exposed, must provide source-span mappings and must never be silently applied under the preserve-bytes version.

The hard offset threshold is **100% coverage**: every trainable content token maps to a source span, every control token has an explicit control category, and no token has a negative, reversed, or out-of-range span.

## 6. Causal batch contract

The packer must expose both packed and padded representations containing input IDs, target IDs, source record IDs, source offsets, loss masks, padding masks, boundary masks, control categories, tokenizer version, and sequence/document boundaries.

For a document represented as `[BOS, content..., EOS]`, the target at position `i` is the next token within that document. The final EOS position has no target and must have `loss_mask=false`. A packed document’s final position must never target the first position of the next document. Padding positions must have `padding_mask=false`, `loss_mask=false`, and `target_id=<PAD>`. BOS and document-boundary controls are never charged as accidental cross-document losses.

The gate uses a deterministic token-local checksum evaluator—not a language-model quality metric—to compare packed and padded semantics. For every document, the sum and count of target IDs under the loss mask must match exactly between packed and padded forms. The permitted discrepancy is **zero**; any cross-boundary or padding loss fails the gate.

## 7. Mandatory checks and hard thresholds

| Check | Hard pass condition |
|---|---|
| Byte fallback | All 256 byte values, including NUL and malformed UTF-8 bytes, encode/decode byte-exactly |
| Unicode | Declared valid Unicode round-trips under the active normalization version; invalid bytes are preserved without dropping or substituting bytes |
| Code coverage | Identifiers, indentation, literals, comments, operators, and adjacent identifier boundaries remain distinguishable and source-mappable |
| Structured data | JSON delimiters, escapes, whitespace, and key/value boundaries round-trip exactly |
| Special tokens | All nine reserved IDs and strings are stable; no content token uses a reserved ID or becomes a control token |
| Round-trip | Byte and malformed-input suites are exact; normalized text follows only the declared normalization equivalence |
| Offsets | 100% of content tokens have valid source spans; 100% of controls have explicit categories |
| Packed loss | Zero loss is charged across document or sequence boundaries; all padding loss masks are false |
| Padding equivalence | Packed and padded token-local loss checksum agrees exactly per document |
| Versioning | Snapshot bytes and SHA-256 round-trip exactly; incompatible versions, malformed snapshots, duplicate singleton fields, and trailing data fail closed |
| Strict metadata closure | Externally supplied document provenance and packed/padded control and boundary metadata reject if tampered |
| Efficiency measurement | Every candidate reports non-zero tokens/sec, bytes/sec, resident-memory samples, identical fixture/config/batch metadata, and no candidate is omitted |
| Candidate selection | Selected candidate passes every mandatory check, has 100% offset coverage and zero unknown/fallback corruption, and its selection rationale is recorded |
| Contamination | Builder rejects evaluator-only or non-training documents; evaluator canary does not change vocabulary or snapshot |
| Reproducibility | Same corpus/config/seed produces byte-identical vocabulary, snapshot, encoded streams, and packed/padded batches |

Additional quantitative thresholds are fixed for the gate environment: all three candidates must process the fixed comparison fixture at **at least 10,000 source bytes per second**, each candidate must report a positive resident-memory sample, and the selected candidate must achieve a compression ratio of **at least 1.05× relative to byte tokenization** on the comparison fixture. If the fixture cannot demonstrate that threshold without sacrificing exactness, the gate must report `BLOCKED` rather than selecting a candidate by assertion.

## 8. Adversarial and failure-path requirements

The harness must attempt and verify fail-closed behavior for an evaluator-only document passed to the vocabulary builder, a missing real-source file, a changed real-source hash, a duplicate reserved ID, a content/control collision, an invalid token ID at decode time, a truncated snapshot, an incompatible snapshot version, duplicate singleton snapshot fields, trailing snapshot data, malformed UTF-8, NUL-containing content, mismatched externally supplied token provenance, cross-document packed boundaries, tampered control/boundary metadata, and padding positions. Each rejected input must produce a deterministic failure or diagnostic; silent repair is not accepted unless explicitly governed by the normalization version.

The harness must also run the same construction twice with different container insertion orders and confirm identical output. It must verify that adding validation/evaluator content after construction cannot mutate the released vocabulary or snapshot.

## 9. Required machine-readable artifacts

The Stage 10 gate writes all artifacts beneath `artifacts/stage-10/cpp-gate/`:

| Artifact | Required contents |
|---|---|
| `checks.json` | One record per mandatory check with status, duration, and measured details |
| `metrics.json` | Mandatory-check count, candidate counts, token totals, compression, fallback, offsets, batch utilization, throughput, and memory |
| `candidate_comparison.json` | Identical-fixture comparison for byte, subword, and hybrid candidates |
| `tokenizer_snapshot.json` | Selected candidate metadata, snapshot hash, version, and provenance |
| `tokenizer_snapshot.bin` | Canonical serialized snapshot bytes |
| `batch_report.json` | Packed/padded shapes, boundaries, loss-mask checksums, and utilization |
| `reproducibility.json` | Repeated-build and repeated-batch equality evidence |
| `incident_log.json` | Rights/split/contamination/version/offset/boundary incident flags |
| `release_record.json` | Stage status, selected candidate, immutable snapshot hash, training authorization boundary, and approval requirement |
| `report.md` | Human-readable evidence, metrics, limitations, and claim boundary |

## 10. Transition decision

The gate returns `PASS` only when every mandatory check is `PASS`, all three candidates have comparison metrics, exactly one selected snapshot is immutable and reproducible, and the report records the fact that Stage 10 evaluates representation mechanics rather than model quality. A `FAIL` requires correction before transition. A `BLOCKED` result must identify the unresolved Unicode, provenance, rights, compatibility, or efficiency constraint and must not authorize Stage 11.

Stage 10’s final claim is limited to a deterministic tokenizer and batch interface validated on the declared fixtures. It does not establish multilingual completeness, production-scale throughput, tokenizer optimality, training quality, safety behavior, or general intelligence.
