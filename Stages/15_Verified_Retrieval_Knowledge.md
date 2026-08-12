# Stage 15 — Verified Retrieval and Knowledge Plane
## Grounded Generation, Provenance, Freshness, and Tenant-Safe Memory

**Predecessor:** Stage 14 — Preference Tuning and Alignment  
**Successor:** Stage 16 — Production Inference and Operations  
**Status:** Specification; implementation not started  
**Implementation:** Native C++20 retrieval, memory, verification, and gate integration

## Purpose

Stage 15 turns the persistent memory substrate into a production knowledge plane for current, private, and source-grounded NLP. It connects retrieval, causal/event identity, citations, validity, conflicts, deletion, access control, and independent answer verification while preserving the distinction between evidence and executable instructions.

## Scope and non-goals

The stage includes lexical/vector/hybrid retrieval, tenant and document permissions, temporal validity, document versioning, citation spans, evidence hashes, conflict sets, deletion propagation, stale-data handling, prompt-injection isolation, memory poisoning detection, retrieval monitoring, and verified generation. It does not authorize arbitrary external tools, autonomous action, or treating retrieved content as policy.

## Knowledge record contract

Each knowledge item must expose:

```text
knowledge_id
tenant_id
document_id
document_version
source_uri_or_reference
content_hash
embedding_version
lexical_index_version
created_at
valid_from
valid_until
access_policy
provenance
citation_spans
quality_and_confidence
supersedes_or_conflicts
retention_and_deletion_state
```

Retrieval must return typed evidence, not only text. Every evidence hit must carry source identity, document version, access decision, score, temporal validity, citation span, and transformation version.

## Required implementation

| Component | Implementation | Contract |
|---|---|---|
| Ingestion adapter | Stage 9 manifests to knowledge records | Rights/privacy state required |
| Lexical retrieval | Exact and ranked term search | Deterministic scores and filters |
| Vector retrieval | Versioned embedding/index path | Index hash and embedding version recorded |
| Hybrid ranking | Combined score with explanation | Ranking weights are versioned |
| Access control | Tenant/document/role filters | Unauthorized hits are impossible by default |
| Freshness | Validity intervals and version selection | Stale evidence is explicit |
| Citations | Span IDs and content hashes | Answer claims bind to evidence |
| Conflict manager | Contradictory versions and sources | Conflicts are surfaced, not flattened |
| Deletion | Immediate logical delete plus rebuild | Deleted evidence cannot be returned |
| Poisoning defense | Instruction/evidence separation and source risk | Retrieved data cannot change policy |
| Answer verifier | Evidence entailment/schema/uncertainty checks | Unsupported answers abstain or are marked |
| Audit | Query, hits, filters, answer claims, decisions | Full path is reconstructable |

## Retrieval and generation modes

The evaluation must compare:

1. no retrieval;
2. retrieval without verification;
3. retrieval with citation verification;
4. retrieval with citation and independent answer verification;
5. retrieval with stale/conflicting/poisoned evidence.

The mode and policy must appear in every response trace. The system must never silently change from verified to unverified generation because a verifier times out.

## Evaluation harness

The harness must include:

1. exact and semantic retrieval fixtures;
2. tenant-isolation and unauthorized-query tests;
3. current versus historical document versions;
4. stale and expired evidence;
5. contradictory sources and confidence ranking;
6. citation span precision and recall;
7. supported and unsupported answer claims;
8. missing evidence and abstention;
9. retrieval prompt injection and instruction-like documents;
10. poisoned memory records;
11. deletion and restart persistence;
12. embedding/index version mismatch;
13. latency and memory under increasing corpus sizes;
14. no-retrieval, retrieval, and verified-retrieval ablations;
15. human review of grounded answers.

## Mandatory gate checks

| Check | Pass condition |
|---|---|
| Retrieval quality | Declared precision/recall or nDCG targets pass on held-out queries |
| Citation integrity | Claims cite correct source spans and hashes |
| Freshness | Current valid evidence outranks expired/superseded evidence |
| Conflict handling | Conflicts are exposed; unsupported certainty is blocked |
| Access control | Cross-tenant and unauthorized document retrieval rate is zero |
| Deletion | Deleted records disappear immediately and after rebuild/restart |
| Poisoning | Instruction-like evidence cannot alter policy or execute tools |
| Grounded generation | Verified mode reduces unsupported-claim rate against retrieval baseline |
| Abstention | Missing/contradictory evidence yields calibrated uncertainty or abstention |
| Versioning | Index/embedding/document/model versions are recorded and checked |
| Efficiency | Query latency, memory, and rebuild cost are within declared budgets |
| Auditability | Query-to-evidence-to-answer path is fully reconstructable |
| Regression | Prior Stage 0–14 gates remain green |

## Pass/fail transition

Stage 15 passes only when verified generation improves grounded correctness, citation integrity, freshness, access control, deletion, and poisoning resistance on held-out fixtures. A `PASS` authorizes Stage 16 serving integration.

A `FAIL` requires retrieval, access, verification, or deletion remediation. A `BLOCKED` result is valid for any source class whose rights, privacy, or access policy is unresolved.

## Deliverables

The stage must deliver knowledge-record schema, ingestion/index adapters, lexical/vector/hybrid retrieval, access-control integration, citation and validity system, conflict/deletion handling, answer verifier, poisoning suite, human-grounding report, native tests, gate executable, and CI command.

## Explicit limitations

Retrieval can improve grounding without making the underlying model generally truthful. Vector similarity is not proof of entailment. Citation presence is not citation correctness unless claims are verified. Enterprise data requires separate privacy, access, retention, and legal review.
