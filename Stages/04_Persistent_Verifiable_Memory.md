# Stage 4 — Persistent Verifiable Memory

**Project:** CCT-ASE  
**Stage ID:** 4  
**Predecessor:** Stage 3 — Causal Event Learning  
**Successor:** Stage 5 — Language and Code Scaling  
**Status:** Implemented in native C++20; Stage 4 gate validation in progress; Stage 5 approval required

## Purpose

Stage 4 adds persistent episodic and semantic memory that can write, retrieve, cite, update, expire, and delete information under an explicit policy. The key requirement is not merely storing embeddings; it is maintaining an auditable relationship between generated outputs and the evidence that influenced them.

The memory system must be evaluated as an independent subsystem and as a model-augmented subsystem. A language-model loss improvement without retrieval precision, provenance integrity, stale-memory handling, and deletion correctness is not sufficient for a pass.

## Scope and non-goals

The stage includes a checksummed append-only event log, versioned records, deterministic exact embedding and metadata indexes, auditable write/read controllers, provenance-aware retrieval, citation generation, retention/deletion policies, conflict handling, poisoning-boundary tests, causal-event integration, recovery tests, and long-context evaluation. Approximate indexing and distributed storage remain outside this correctness stage. It does not scale the full language model, add autonomous tools, or claim that topological summaries alone constitute memory.

The implemented local baseline uses an in-process exact search implementation. Approximate nearest-neighbor indexing and distributed storage are optimization paths, not prerequisites for correctness or claimed by this stage.

## Memory contract

A memory record must contain content, identity, temporal information, causal relationships, provenance, confidence, access history, retention state, and schema version:

```text
MemoryRecord {
    memory_id: MemoryId,
    version: u64,
    content: Payload,
    embedding: Vector,
    event_ids: Vec<EventId>,
    causal_parents: Vec<EventId>,
    created_at: LogicalTime,
    valid_from: LogicalTime,
    valid_until: Optional[LogicalTime],
    source: SourceRef,
    confidence: float,
    status: Active | Superseded | Deleted | Quarantined,
    retention_class: RetentionClass,
    checksum: Digest
}
```

The system must distinguish:

| State | Meaning | Output behavior |
|---|---|---|
| Active | Eligible for retrieval under policy | May be returned with provenance |
| Superseded | Replaced by a newer version | May be returned only when historical context is requested |
| Deleted | Removed under policy or user request | Must not be retrieved or cited |
| Quarantined | Suspected corruption, conflict, or policy violation | Must not influence normal generation |
| Uncertain | Content exists but confidence is below threshold | Must be labeled and may trigger abstention |

Deletion must be tested as a semantic property. Removing a record from the index is insufficient if stale copies remain in caches, checkpoints, summaries, or training artifacts.

## Required implementation

| Component | Required implementation | Testable contract |
|---|---|---|
| Event log | Append-only, versioned record store with checksums | Replay reconstructs identical active state |
| Exact index | Deterministic metadata and ID index | Filters are correct before approximate search is added |
| Vector index | Exact baseline first; approximate index behind an interface | Recall is measured against exact search |
| Memory encoder | Encode content plus event, time, provenance, and uncertainty features | Encoding is versioned and deterministic |
| Write controller | Decide write, update, ignore, or quarantine | Decisions and reasons are logged |
| Read controller | Query, filter, rank, deduplicate, and return evidence spans | Returned records satisfy all filters |
| Compression | Summaries retain source links and version references | Compression never removes required provenance |
| Citation layer | Bind generated claims to retrieved record IDs and spans | Unsupported claims are distinguishable |
| Retention policy | FIFO, time-based, priority, legal/user deletion, and capacity policies | Policy decisions are deterministic and auditable |
| Conflict manager | Detect contradictory active records and expose conflict sets | Conflict is not silently averaged away |
| Recovery | Rebuild indexes from the event log and verify checksums | Recovered index matches canonical state |

A recommended training objective is:

```text
L = L_next + λ_retrieval L_retrieval
  + λ_write L_write + λ_citation L_citation
  + λ_conflict L_conflict + λ_memory_consistency L_consistency
```

The memory controller must support a no-memory mode for ablation. The model must not be allowed to access the evaluation answer through a memory record or cached retrieval artifact.

## Memory lifecycle

### Write path

The write path receives an event or document chunk, computes a deterministic content digest, extracts metadata, generates an embedding, evaluates novelty and retention policy, and either writes a new record, creates a new version, ignores the input, or quarantines it. Every decision receives a reason code.

### Read path

The read path accepts a query, time interval, causal filter, provenance constraint, and retrieval budget. It performs metadata filtering before vector ranking, returns record IDs and source spans, removes deleted or quarantined records, deduplicates versions, and exposes confidence and conflict state.

### Use path

The model receives retrieved evidence in a typed context separate from its own generated state. The decoder must be able to identify which output claims rely on which memory records. If the system cannot support exact claim-level attribution initially, it must report evidence at the answer or sentence level and label the limitation.

### Update and deletion path

Updates create a new version linked to the previous version. Deletion creates a tombstone and triggers index, cache, summary, and checkpoint handling according to policy. The harness must test both immediate and eventual deletion semantics, with the chosen guarantee documented.

## Evaluation harness

### Retrieval quality

Construct queries with known relevant records, distractors, duplicates, superseded versions, and conflicting claims. Measure precision@k, recall@k, mean reciprocal rank, nDCG, latency, and memory footprint. Compare exact and approximate indexes.

### Long-context utilization

Use long documents and event streams in which the answer depends on distant, noncontiguous evidence. Evaluate single-hop and multi-hop retrieval, query reformulation, distractor density, and context length. Include ChapterBreak-style sequence tasks and a locally generated equivalent with known evidence locations, while documenting the exact split and licensing.

### Provenance and citation integrity

For each generated claim, classify whether it is supported by retrieved evidence, contradicted by evidence, unsupported, or correctly abstained. Measure citation precision, citation recall, span overlap, and false-support rate. A fluent answer with incorrect evidence links is a failure.

### Staleness and temporal validity

Create records that are valid only during specific intervals, then query before, during, and after those intervals. The model must prefer current valid records when asked for current information and retrieve historical records only when the query requests history.

### Deletion and retention

Write a record, retrieve it, delete it, rebuild indexes, restart the process, and query again. Test deletion from active index, caches, summaries, and checkpointed memory. Test retention expiry and verify that expired records are handled according to policy.

### Conflict handling

Insert contradictory records with different sources, timestamps, and confidence. Evaluate whether the system exposes conflict, ranks evidence according to the declared policy, and avoids fabricating a false consensus.

### Memory poisoning and injection

Insert adversarial records that attempt to override system policy, exfiltrate hidden values, or instruct the model to ignore provenance. The memory layer must treat stored content as data, not executable policy. Retrieval must preserve source identity and policy separation.

## Pass/fail criteria

| Criterion | Pass condition | Failure condition |
|---|---|---|
| Canonical replay | Replaying the event log reconstructs identical active records and checksums | Recovery changes content, identity, or status |
| Retrieval correctness | Exact index returns all known relevant records under correct filters; approximate recall meets declared target | Deleted, quarantined, or out-of-window records are returned |
| Learned retrieval benefit | Learned controller improves a declared quality/latency metric over the no-memory baseline without leakage | Gains come from answer leakage, cached labels, or unsupported retrieval |
| Provenance | Retrieved records include valid source IDs, spans, versions, and confidence | Evidence is missing, mismatched, or fabricated |
| Citation integrity | Citation precision and unsupported-claim rate meet predefined thresholds | Fluent unsupported claims are counted as correct |
| Staleness | Temporal queries select records valid for the requested time | Superseded or expired records silently override valid records |
| Deletion | Deleted content is not retrievable after the declared deletion guarantee | Any active path returns deleted content |
| Conflict handling | Conflicts are exposed and policy-ranked | Contradictory records are silently merged or ignored |
| Poisoning resistance | Stored instructions cannot change system policy or execute actions | Memory content bypasses policy boundaries |
| Resource bounds | Index, cache, and retrieval latency remain within declared budgets | Memory grows without retention or recovery behavior |
| Ablation | No-memory, exact-index, learned-index, no-citation, and no-provenance variants are measured | Component contribution is not isolatable |

The stage passes only when both storage correctness and model-use correctness pass. An index that is correct but not useful, or a model that is useful but unverifiable, is not sufficient.

## Transition to Stage 5

Stage 5 may begin when memory records are versioned, recoverable, provenance-aware, and safe to use as an augmentation channel. The transition package must include the schema, replay/recovery logs, retrieval benchmark, citation report, deletion report, conflict report, poisoning tests, resource profile, and all memory leakage controls.

The memory interface must be stable enough that the language and code stage can train against it without coupling the model to a specific storage vendor. Exact retrieval must remain available as the evaluation oracle.

If the stage fails, the team must first repair canonical state, deletion, provenance, or leakage before increasing model size. Retrieval quality cannot be repaired by allowing the model to hallucinate unsupported answers.

## Exit report

The report must state the deletion guarantee, retention policy, retrieval budgets, exact-versus-approximate recall, citation definitions, stale-memory behavior, conflict policy, poisoning results, and the conditions under which the model abstains.

**Transition decision:** `PASS` authorizes Stage 5. `FAIL` requires remediation. `BLOCKED` is allowed only for distributed indexing; the local canonical log, exact oracle, and deletion tests must pass.

## References

[1]: ../CCT_EVOLUTION_PROPOSAL.md "CCT-ASE evolution proposal"

[2]: https://proceedings.neurips.cc/paper_files/paper/2023/hash/ebd82705f44793b6f9ade5a669d0f0bf-Abstract-Conference.html "Augmenting Language Models with Long-Term Memory"

[3]: ../Stages/03_Causal_Event_Learning.md "CCT Stage 3 causal event learning specification"
