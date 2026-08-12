# Stage 9 Expanded Gate Contract — Governed Data and Corpus Pipeline

**Stage:** 9  
**Implementation:** Native C++20  
**Status:** Implemented and gated on declared real-source fixtures  
**Transition:** Stage 10 requires explicit approval

## Gate intent

This gate validates governed corpus mechanics against actual public-domain source files and repository MIT code files, not synthetic-only content. It verifies that source rights, privacy, provenance, deduplication, contamination controls, deterministic sharding, replay, deletion, and audit lineage behave as declared.

The public-domain labels in this contract are source declarations and do not replace jurisdiction-specific legal review. Automated privacy detection is a quarantine control, not legal advice. The gate does not claim the corpus is representative of production language or that a model trained on it would be production-ready.

## Real-source manifest

| Source | Split | Class | Declared rights | Evidence |
|---|---|---|---|---|
| Project Gutenberg fixture 1342 | Train | Reference text | Declared US public-domain status | Official URL, local bytes, SHA-256 |
| Project Gutenberg fixture 11 | Validation | Reference text | Declared US public-domain status | Official URL, local bytes, SHA-256 |
| CCT production C++ source | Train | Code | Repository MIT license | Immutable repository reference and local bytes |
| CCT corpus C++ source | Train | Code | Repository MIT license | Local repository-under-test reference and local bytes |
| Stage 9 evaluator canary | Evaluator-only | Evaluator data | Restricted | Held-out truth, never trainable |

Each source record includes URI, local path, license/consent label, jurisdiction, collection method, timestamp, transformation chain, SHA-256, split, training/evaluation permissions, and privacy class.

## Mandatory checks

| Check | Required threshold |
|---|---|
| Real source manifest | 5 entries, 4 real files hashed, 1 evaluator-only entry |
| Hash integrity | Every declared real-file hash matches the native SHA-256 implementation |
| Rights | Resolved or explicitly evaluator-only sources are accepted; unresolved rights quarantine |
| Real ingestion | 4/4 declared real files accepted with content and normalized hashes |
| Privacy | PII fixture is detected, redacted, and quarantined |
| Quality | Short/corrupt fixture is rejected with a reason code |
| Exact deduplication | Case/whitespace-equivalent duplicate is rejected by normalized hash |
| Near deduplication | High-overlap duplicate is rejected with an explicit near-duplicate reason |
| Split isolation | Evaluator-only records in training: `0` |
| Contamination | Injected evaluator canary collision detected and affected corpus blocked |
| Shards | Deterministic shard IDs, record order, byte counts, and content hashes |
| Replay | Serialize/deserialize is byte-equivalent and shard-equivalent |
| Deletion | Tombstoned record absent from rebuilt shards and remains deleted after replay |
| Audit | At least one lineage event per record plus deletion event |
| Reproducibility | Same fixture/config produces identical corpus snapshot |

## Automatic failure conditions

The gate fails for any hash mismatch, accepted unresolved rights, PII record entering training, evaluator-only record entering training, missed contamination canary, silent filter decision, malformed source acceptance, shard/replay divergence, deleted record surviving rebuild, missing audit event, or claim that the Stage 9 corpus proves model quality.

## Required artifacts

```text
artifacts/stage-9/cpp-gate/
├── gate.json
├── checks.json
├── metrics.json
├── manifest.json
├── source_hashes.json
├── privacy_report.json
├── deduplication_report.json
├── contamination_report.json
├── shards.json
├── audit.json
├── incident_log.json
├── release_record.json
└── report.md
```

The source manifest is also retained at `data/stage-9/manifests/stage9_manifest.txt`. Evaluator truth is not published as training data and remains separated from accepted corpus shards.

## Transition decision

`PASS` authorizes Stage 10 tokenizer work against the released corpus contract only. It does not authorize unreviewed training, broader data acquisition, production deployment, or claims about language-model quality. `FAIL` requires remediation and rerun. `BLOCKED` is valid for unresolved rights, privacy, or data ownership, but blocked items remain unavailable downstream.
