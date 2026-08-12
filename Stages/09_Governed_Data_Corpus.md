# Stage 9 — Governed Data and Corpus Pipeline
## Licensed, Private, Deduplicated, and Reproducible Training Data

**Predecessor:** Stage 8 — Production Foundation and Governance  
**Successor:** Stage 10 — Tokenizer and Representation Engine  
**Status:** Implemented and gated on declared real-source fixtures; Stage 10 requires explicit approval
**Implementation:** Native C++20 pipeline, regression suite, and gate

## Purpose

Stage 9 turns approved data sources into an immutable, auditable corpus suitable for language-model pretraining, supervised fine-tuning, preference tuning, and evaluation. The stage treats data governance as an engineering dependency: no training run may consume an unmanifested, unlicensed, privacy-unreviewed, or contamination-prone source.

## Scope and non-goals

The stage includes source registration, license and consent metadata, privacy classification, collection and transformation logs, exact and near-duplicate removal, quality filtering, language/domain classification, PII detection, opt-out deletion, contamination canaries, split assignment, shard creation, integrity checks, and reproducible corpus packaging. It does not decide final model quality, implement the language trainer, or authorize production deployment.

## Required data record

Every source item must resolve to a record with at least:

```text
record_id
source_id
source_uri
license_or_consent
jurisdiction
collection_method
collection_timestamp
privacy_classification
content_hash
normalized_hash
transformation_chain
language_and_domain_labels
quality_labels
split_assignment
retention_policy
delete_after
opt_out_status
```

The pipeline must preserve the raw-source reference separately from transformed training content. Transformations are append-only and versioned. A changed transformation version creates a new artifact identity.

## Required implementation

| Component | Implementation | Contract |
|---|---|---|
| Source registry | Native manifest parser and validator | Unknown sources fail closed |
| License review | Allowlist/denylist and human-review state | Unresolved rights are quarantined |
| Privacy review | PII detector, redaction policy, sensitive-class labels | High-risk records cannot enter training by default |
| Ingestion | Resumable, checksummed source acquisition | Interrupted ingestion resumes without duplication |
| Normalization | Unicode, encoding, markup, and document normalization | Original and transformed hashes are recorded |
| Deduplication | Exact hash and near-duplicate fingerprinting | Duplicate policy is deterministic and reported |
| Quality filtering | Language, length, corruption, spam, boilerplate, and domain filters | Every removal has a reason code |
| Contamination control | Evaluator canaries, benchmark overlap, split barriers | Evaluation-only content is inaccessible to training |
| Sharding | Size-bounded deterministic shards | Shard order and offsets are reproducible |
| Deletion | Source/item tombstones and rebuild plan | Deletion propagates to derived artifacts |
| Audit | Append-only data lineage log | Every record is reconstructable |

## Data classes

The initial corpus must separate general text, high-quality reference text, code, instruction data, preference data, grounded enterprise data, safety data, and evaluation-only data. Each class receives independent license, privacy, quality, and retention thresholds. A mixture report must state token counts and percentages after filtering rather than only raw source sizes.

## Evaluation harness

The Stage 9 harness must create a fixed synthetic manifest fixture plus a reviewable sample of every real source class. It must test:

1. valid and invalid license states;
2. unresolved-license quarantine;
3. PII detection and redaction behavior;
4. exact and near-duplicate removal;
5. deterministic normalization hashes;
6. language/domain labels;
7. split isolation and contamination canaries;
8. interrupted ingestion and resume equivalence;
9. deletion tombstone propagation;
10. malformed manifest rejection;
11. shard replay and item-count equality;
12. audit-log completeness.

The harness must include evaluator-only records that the training reader cannot access. It must test a deliberate canary collision and require a contamination failure rather than silently proceeding.

## Mandatory gate checks

| Check | Pass condition |
|---|---|
| Manifest schema | 100% of accepted records validate; malformed records fail closed |
| Rights | Accepted training records have resolved license/consent state; unresolved records are quarantined |
| Privacy | High-risk PII records are removed or approved by policy; redaction is deterministic |
| Hash integrity | Raw, normalized, and shard hashes reproduce exactly |
| Deduplication | Exact duplicates are removed deterministically; near-duplicate rate is reported |
| Quality | Filter decisions include reason codes and do not silently discard data |
| Split isolation | No record crosses train/validation/test/evaluator-only barriers |
| Contamination | Injected evaluator canary is detected and blocks the affected corpus |
| Resume | Interrupted ingestion produces byte- and manifest-equivalent output |
| Deletion | Tombstoned source is absent from all rebuilt derived shards |
| Auditability | Every accepted, removed, quarantined, and deleted item has a trace |
| Reproducibility | Same manifest/config/seed produces identical corpus artifacts |

## Pass/fail transition

Stage 9 passes only if all mandatory checks are green and the corpus release record names the exact real sources, declared rights, privacy policy, transformation versions, hashes, shard counts, token estimates, and evaluation split ownership. The native Stage 9 gate passes 8/8 mandatory checks on the declared real-source fixtures. A `PASS` authorizes Stage 10 tokenizer work against the released corpus contract. It does not authorize training on unreviewed data.

A `FAIL` requires remediation. A `BLOCKED` result must identify the unresolved source or policy boundary. Blocked data must remain unavailable to downstream training targets.

## Deliverables

The stage delivers a native corpus tool, source and item manifest schemas, license review report, privacy report, deduplication report, contamination report, shard index, deletion test artifact, immutable corpus release record, native regression tests, gate executable, and CI command. The implementation uses the real Project Gutenberg and repository-code fixtures recorded in `data/stage-9/manifests/stage9_manifest.txt`.

## Explicit limitations

Automated license and privacy classifiers are not legal advice and cannot replace human review for high-risk data. Near-duplicate detection is approximate and requires reported thresholds. A clean manifest does not prove that every fact in the corpus is accurate. Data quality and rights remain separate release decisions.
