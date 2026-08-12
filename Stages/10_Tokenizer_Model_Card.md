# CCT-ASE Stage 10 Tokenizer Model Card

## Model identity

This card describes the **CCT-ASE tokenizer and representation snapshot** released by Stage 10. The selected candidate is `hybrid`, with tokenizer version `cct-ase-tokenizer-v1`, snapshot format version `1`, and normalization policy `preserve-bytes-v1`. The immutable snapshot hash is recorded by the final gate in `artifacts/stage-10/cpp-gate/tokenizer_snapshot.json` and `release_record.json`.

The artifact is an input-interface component for a future native C++20 next-token training pilot. It is **not a language model**, does not generate text, and has not been evaluated for language-model quality.

## Intended use

The tokenizer is intended to convert governed text and code records into deterministic token streams for later controlled training experiments. It preserves raw source bytes, supports universal byte fallback, records source offsets and record identity, distinguishes content from control tokens, and produces packed or padded causal batches with explicit loss and padding masks.

The implementation supports three candidates for controlled comparison: byte, subword, and hybrid. The hybrid candidate was selected for Stage 11 preparation because it preserved exact byte round-trips and achieved a measured compression ratio above the Stage 10 threshold on the declared comparison fixture. This selection is an engineering decision for the next pilot, not a claim of global tokenizer optimality.

## Out-of-scope use

The snapshot must not be treated as evidence of language understanding, factuality, reasoning, safety, multilingual completeness, production serving readiness, or general intelligence. It must not be used as a production tokenizer without a new release review, compatibility check, and model-checkpoint migration plan. Stage 10 retains `training_authorized: false` until the next stage is explicitly approved.

## Data and provenance

The gate used the governed Stage 9 manifest and fixed application-shaped fixtures. The real-source inputs include two declared Project Gutenberg text fixtures and two CCT repository C++ fixtures. Their local paths, source URIs, declared licenses, hashes, split assignments, transformation lineage, and permissions are recorded in `data/stage-9/manifests/stage9_manifest.txt` and verified before tokenizer construction. The source declarations do not replace jurisdiction-specific legal review.

Vocabulary construction reads only records explicitly marked training-allowed and non-evaluator. Validation and evaluator-only records are used only for evaluation or contamination checks. The gate rejects evaluator-only records passed to the builder and verifies that adding them cannot mutate the baseline vocabulary or snapshot.

## Representation contract

Reserved control IDs are immutable: `<PAD>=0`, `<BOS>=1`, `<EOS>=2`, `<UNK>=3`, `<TASK>=4`, `<SCHEMA>=5`, `<CITATION>=6`, `<DOC_BOUNDARY>=7`, and `<SEQ_BOUNDARY>=8`. Byte fallback IDs occupy `256–511`; learned candidate pieces begin at `512`.

Content tokens map to non-empty half-open source byte spans and retain the originating record ID. Control tokens have empty source spans and explicit control categories. The `preserve-bytes-v1` normalizer leaves valid, malformed, and NUL-containing byte sequences unchanged. Packed and padded batches do not charge loss at document boundaries or on padding positions.

## Evaluation evidence

The final clean Stage 10 CI run passed **21/21 CTest targets**. The Stage 10 native regression executable passed **7/7 tests**, and the Stage 10 gate passed all mandatory checks, including exhaustive byte round-trip, malformed-input preservation, Unicode/code/structured-data fixtures, control collision separation, offset coverage, packed/padded loss equivalence, snapshot compatibility, evaluator isolation, candidate comparison, throughput, memory measurement, and reproducibility.

The selected hybrid candidate’s final comparison artifact reports a `1.85448×` source-byte-to-content-token compression ratio and `100%` source-offset coverage on the declared fixture. The three candidates were measured with the same fixture set, configuration, and repetition count. These measurements characterize representation mechanics only.

| Evidence item | Result |
|---|---:|
| All byte values round-tripped | PASS |
| Malformed and NUL-containing bytes preserved | PASS |
| Content-token offset coverage | 100% |
| Packed/padded loss checksum discrepancy | 0 |
| Candidate count measured | 3/3 |
| Selected candidate | Hybrid |
| Selected compression ratio on gate fixture | 1.85448× |
| Evaluator-only builder records | 0 accepted |
| Snapshot/config/hash round-trip | PASS |
| Incompatible snapshot rejection | PASS |

## Risks and limitations

The vocabulary and frequency statistics are derived from a small declared fixture set and are not representative of a production language distribution. Compression can improve sequence length while harming downstream learning, identifier behavior, or safety properties; no such downstream conclusion is made here. The tokenizer does not perform broad Unicode canonicalization because its released normalization policy is byte-preserving. Future normalization, vocabulary, special-token, or boundary changes require a new versioned snapshot and a compatibility/migration gate.

Throughput and resident-memory measurements are environment-specific and should not be generalized to other hardware, workloads, or deployment settings. The gate’s real-source rights fields are provenance controls, not legal opinions. Privacy detection remains a governance control and does not establish that all sensitive information has been removed.

## Reproduction

From the repository root, run:

```bash
make clean && make ci-stage10
```

The final Stage 10 evidence is written beneath `artifacts/stage-10/cpp-gate/`. The release is valid only when the implementation, configuration, fixture manifest, and commit are unchanged from the gated release candidate.

## References

[1]: https://www.gutenberg.org/cache/epub/1342/pg1342.txt "Project Gutenberg Pride and Prejudice fixture"

[2]: https://www.gutenberg.org/cache/epub/11/pg11.txt "Project Gutenberg Alice's Adventures in Wonderland fixture"

[3]: https://github.com/nexuss0781/CCT "CCT repository"
