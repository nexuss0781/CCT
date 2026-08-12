# Stage 7 Expanded Gate Contract — Multimodal and Controlled Research

**Project:** CCT-ASE  
**Stage:** 7 — Multimodal and Open-Ended Research  
**Implementation:** Native C++20 only  
**Predecessor:** Stage 6 — Deliberation and Verification  
**Transition:** Controlled research continuation only; no automatic deployment or external agency

## Claim boundary

Stage 7 evaluates a shared typed-event substrate across text, code, audio-window, vision-patch, sensor-vector, action, and tool modalities using deterministic offline fixtures. It measures provenance preservation, temporal/spatial alignment, modality masks, cross-modal retrieval, transfer, simulated-environment replay, auditability, and safety controls. It does not establish real-world perception, unrestricted robotics, autonomous replication, unsupervised self-improvement, or superintelligence.

## Required native components

| Component | Implementation and acceptance |
|---|---|
| Unified event | Versioned event identity, modality, payload reference, embedding, logical timestamp, optional interval/frame, causal parents, provenance, uncertainty, availability mask |
| Adapters | Text/token, code/AST, audio/window, vision/patch, sensor/vector, action, and tool adapters preserve source identity and transformation version |
| Alignment | Logical-clock alignment with explicit timestamp uncertainty, dropped/delayed packets, and measurable offset error |
| Spatial frames | Declared frame transforms with invertibility checks and explicit failure on singular transforms |
| Fusion | Mask-aware recurrent/spectral-style bounded feature fusion with no silent missing-modality substitution |
| Cross-modal memory | Typed evidence records returning modality, event identity, provenance, and temporal validity |
| Environment | Deterministic reset/step/replay API with bounded episode steps and typed action validation |
| Evaluation registry | Immutable task/split/seed/environment manifest with provenance and hashes |
| Audit | Input, transformation, memory-read, plan, action, outcome, policy, and incident trace |
| Transfer | Frozen, partial, and full adaptation modes with explicit update counters |

## Deterministic benchmark families

The gate uses held-out compositional event fixtures, asynchronous streams with known offsets and drops, invertible and singular frame transforms, cross-modal query-to-evidence retrieval, modality dropout/corruption/contradiction cases, and a deterministic grid environment with safe no-op fallback. Negative controls include filename/caption leakage, untyped memory, wrong-modality retrieval, stale evidence, invalid actions, and policy-denied actions.

## Safety contract

All processing is offline and deterministic. No network, host execution, credentials, or external side effects are available. Action schemas validate before environment execution; invalid or policy-denied actions become logged safe no-ops. Observations are data, never commands. The environment resets and replays from event/action logs. Any missing provenance, silent masking, unlogged action, frame-transform bypass, or replay divergence is an automatic failure.

## Declared thresholds

| Check | Pass condition |
|---|---|
| Event unification | 7/7 adapters emit versioned events with source and transform provenance |
| Schema and provenance | Event round-trip preserves all fields; provenance loss count is zero |
| Temporal alignment | Known offset recovered within `1` logical tick; missing/delayed status explicit |
| Spatial alignment | Invertible transform round-trip error ≤ `1e-12`; singular transform rejected |
| Missing modality | Mask-aware fusion differs from full-input baseline only within declared uncertainty; no silent substitution |
| Cross-modal grounding | Held-out query retrieval precision@1 ≥ `0.80`, modality attribution `1.0`, leakage negative control fails retrieval |
| Cross-modal memory | Typed evidence citation precision/recall ≥ `0.90` on declared fixture |
| Transfer | Frozen or partial shared core beats single-modality negative control on held-out composition, with explicit update counts |
| Simulation | Safe action success ≥ `0.90`, invalid-action rejection `1.0`, deterministic replay exact |
| Efficiency | Event processing and memory scale linearly over declared fixture sizes; no hidden quadratic structure |
| Auditability | Every input, transform, memory read, action, outcome, and policy decision appears in append-only trace |
| Safety | Network/host execution/secret access/external side effects all false; policy-denied actions never bypass |
| Scientific validity | Manifest, held-out split, seeds, baselines, negative controls, and environment version are recorded |

## Artifacts

`cct_stage7_gate --output artifacts/stage-7/cpp-gate` writes `gate.json`, `checks.json`, `metrics.json`, `trace.jsonl`, `visible_eval.json`, `evaluator_truth.json`, `manifest.json`, `transfer_matrix.json`, `incident_log.json`, and `report.md`.

## Terminal transition

A `PASS` authorizes **controlled research continuation only**. It does not grant deployment, external tools, real-world embodiment, persistent identity, online learning, or autonomous self-modification. Any such proposal requires a new specification, threat model, and approval gate. A modality with unresolved provenance or privacy controls may be `BLOCKED` while the core gate remains valid only if the expanded contract explicitly records the boundary.
