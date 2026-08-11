# Stage 4 Expanded Gate Contract — Persistent Verifiable Memory

**Project:** CCT-ASE  
**Stage:** 4 — Persistent Verifiable Memory  
**Implementation:** Native C++20 only  
**Predecessor:** Stage 3 — Causal Event Learning  
**Transition:** Stage 5 only after explicit user approval

## Purpose and claim boundary

Stage 4 evaluates a local, exact, provenance-aware persistent memory subsystem that can append, version, retrieve, cite, supersede, quarantine, expire, delete, replay, and recover records. It demonstrates **persistent verifiable memory behavior on controlled native fixtures**. It does not establish long-term memory in a deployed language model, general factual reliability, autonomous tool use, or superintelligence.

The canonical source of truth is an append-only event log. Metadata filtering precedes exact vector ranking. Deleted, expired, superseded, and quarantined records are excluded from ordinary retrieval. Every returned hit carries its memory ID, version, source reference, evidence span, confidence, status, and checksum.

## Native memory contract

The public API is defined in `cpp/include/cct/memory.hpp` and implemented in `cpp/src/memory.cpp`.

| Component | Contract |
|---|---|
| `MemoryRecord` | Versioned content, embedding, causal/event links, logical validity interval, source/span, confidence, status, retention class, conflict group, checksum |
| `MemoryEventLog` | Append-only typed events with sequence number, previous checksum, event checksum, and deterministic replay |
| `PersistentMemory` | Canonical log, active-state reconstruction, exact metadata/vector indexes, update and tombstone semantics, snapshot/recovery |
| `MemoryEncoder` | Deterministic versioned content-plus-metadata encoding with fixed checksum behavior |
| `MemoryWriteController` | Deterministic write/update/ignore/quarantine decision with reason code and audit record |
| `MemoryReadController` | Time/event/source/status filtering, exact cosine ranking, version deduplication, budget enforcement, and evidence return |
| `CitationBinder` | Typed answer/sentence claim bindings to memory IDs, versions, spans, and support status |
| `RetentionPolicy` | Capacity, expiry, FIFO/priority selection, legal/user deletion, and deterministic policy audit |
| `ConflictManager` | Explicit conflict groups and ranked active alternatives; no silent averaging |

The chosen deletion guarantee is **immediate logical deletion plus replay/restart persistence**. A tombstone is appended, all active indexes are rebuilt, and deleted records cannot be retrieved or cited after the delete operation returns.

## Visibility and policy boundary

Memory content is data. It is never parsed as executable policy, never changes retention or authorization rules, and never overrides provenance requirements. Adversarial stored instructions remain ordinary content with their source identity intact.

The memory-augmented path receives a typed `EvidenceContext`; the no-memory path receives an empty context. The evaluator truth, answer labels, and hidden relevance annotations are stored outside model-visible contexts and are never inserted into the memory store.

## Deterministic benchmark fixtures

The Stage 4 gate uses a controlled corpus containing relevant records, distractors, duplicate versions, superseded records, conflicting claims, historical intervals, expired records, and poisoning strings. Each fixture has a stable source ID and evidence span. Queries specify text embedding, time window, source constraints, event constraints, retrieval budget, and whether historical records are requested.

The benchmark includes single-hop and multi-hop evidence joins over distant records. Exact search is the evaluation oracle. No approximate index is claimed; the interface leaves that optimization for a later stage.

## Declared thresholds

Thresholds are frozen before the final gate run and written to `artifacts/stage-4/cpp-gate/metrics.json`.

| Check | Pass condition |
|---|---|
| Schema/checksum integrity | Record and event-log round trips preserve all fields; tampered content, checksum, chain, and schema are rejected |
| Canonical replay/recovery | Replaying the append-only log and rebuilding indexes reproduces identical active state and checksums |
| Retrieval correctness | Relevant active records achieve precision@3 and recall@3 of `1.0` on the controlled fixture; deleted/quarantined records are never returned |
| Provenance/citation | Every returned hit has valid source ID, version, span, confidence, and checksum; supported citation precision is `1.0` |
| Version/staleness | Current queries select valid latest records; historical queries may select superseded records; expired records are excluded unless policy explicitly requests history |
| Deletion | Deleted content is absent immediately, after index rebuild, after snapshot recovery, and from citation binding |
| Conflict handling | Contradictory records remain an explicit conflict set and are not averaged; ranking follows confidence/time policy |
| Retention | Capacity and expiry policies make deterministic decisions and never delete legal-hold records |
| Poisoning resistance | Stored instruction text cannot alter policy, execute actions, bypass deletion, or fabricate provenance |
| Long-context/multi-hop | Distant two-record evidence join returns both required IDs within the retrieval budget and produces a supported citation |
| No-memory ablation | Empty memory mode produces no evidence IDs and cannot access stored answers; memory mode improves declared evidence recall |
| Resource bounds | Exact retrieval latency and resident record count remain within declared local budgets; deletion does not leave active stale copies |
| Reproducibility | Same seed and event log produce byte-identical reports; changed fixture seed changes the corpus fingerprint |

Every mandatory check must pass. No deferred correctness limitation is allowed at transition.

## Gate artifacts

The native executable is `cct_stage4_gate --output artifacts/stage-4/cpp-gate`. It writes:

| Artifact | Contents |
|---|---|
| `gate.json` | Stage, status, commit, clean-tree state, deletion guarantee, and approval requirement |
| `checks.json` | Machine-readable check records and durations |
| `metrics.json` | Frozen thresholds, measured values, units, and status |
| `report.md` | Methodology, retrieval/citation/deletion/conflict/poisoning results, ablations, and limitations |
| `memory_visible.json` | Model-visible evidence schema with no evaluator labels |
| `memory_truth.json` | Evaluator-only fixture inventory and audit boundary, without copied answer payloads |

The gate returns nonzero on any failed mandatory check. The release candidate must pass Stage 0 through Stage 4 validation from a clean tree before publication.

## Transition package

The Stage 4 package consists of the versioned memory schema, append-only log, replay/recovery implementation, exact retrieval oracle, write/read controller audit records, citation layer, retention/deletion policy, conflict and poisoning tests, no-memory ablation, resource profile, gate artifacts, and final commit SHA. A passing Stage 4 gate authorizes only **Stage 5 preparation after explicit approval**.

## Non-goals

This stage does not implement approximate nearest-neighbor indexing, distributed storage, full language-model training, autonomous tools, deliberation, or claims of superintelligence. Results are limited to the declared local native fixtures and explicit deletion/provenance semantics.
