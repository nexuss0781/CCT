# Stage 3 Expanded Gate Contract — Causal Event Learning

**Project:** CCT-ASE  
**Stage:** 3 — Causal Event Learning  
**Implementation:** Native C++20 only  
**Predecessor:** Stage 2 — Efficient Sequence Core  
**Transition:** Stage 4 only after explicit user approval

## Purpose and claim boundary

Stage 3 tests whether the CCT-ASE sequence core can consume explicit event identity, temporal metadata, causal-parent structure, intervention markers, and uncertainty without leaking evaluator-only truth. The stage demonstrates **causal-structure-aware prediction on declared synthetic structural-equation families**. It does not demonstrate general causal understanding, causal discovery in the wild, language competence, or superintelligence.

The implementation is split into two auditable paths. The **event graph path** stores and validates versioned causal records, executes deterministic graph queries, and learns small structural equations from intervention data. The **temporal path** encodes model-visible event payloads, timestamps, uncertainty, provenance, and optionally parent summaries into the Stage 2 selective recurrent core. The evaluator-only ground truth is held in separate dataset structures and is never passed to the model-visible encoder.

## Native data contract

The public C++ API is defined in `cpp/include/cct/causal.hpp` and implemented in `cpp/src/causal.cpp`. It contains the following versioned records:

| Record | Required fields and invariant |
|---|---|
| `CausalEvent` | Stable `EventId`, schema version, payload, fixed-dimensional bounded coordinates, logical timestamp, sorted causal parents, intervention marker, uncertainty record, provenance record, and explicit unresolved-parent list |
| `Intervention` | Variable identifier, intervention value, and distinct mode (`observed`, `do`, or `counterfactual`) |
| `CausalEventStore` | Deterministic indexed storage with identity protection, immutable insertion, parent/child adjacency, causal past/future, topological ordering, cycle rejection, coordinate validation, and deterministic binary snapshot round-trip |
| `CausalDataset` | Model-visible event streams and evaluator-only structural equations, graph truth, exogenous values, and counterfactual targets stored in separate types |
| `GraphConditionedSequence` | Stage 2 recurrent temporal backbone with an explicit edge-channel switch; unavailable future events and future parents are masked from the encoder |
| `CausalEventLearner` | Intervention-effect estimation, structural-coefficient recovery, graph prediction, do-intervention prediction, counterfactual abduction-action-prediction, uncertainty, and abstention |

Historical events are immutable. Corrections are represented by a new event ID and provenance link. A missing parent is accepted only when the event explicitly carries that parent in `unresolved_parent_ids`; ordinary insertion of a missing parent is rejected.

## Controlled structural-equation families

The gate uses deterministic linear structural equations with bounded noise and one nonlinear held-out function family. A graph is a DAG with a declared topological order. For variable `i`:

```text
X_i = b_i + sum(B_i,j * X_j for j in Pa_i) + U_i
```

The held-out nonlinear family replaces one parent term with a bounded `tanh` transform while retaining a deterministic evaluator. Training and test families are separated by seed, coefficient family, topology, intervention value, graph size, and sequence length. A confounded observational split shares one exogenous variable between two observed nodes; randomized interventions remain the identification source.

The generator emits three strictly separate products:

| Product | May contain | Must not contain |
|---|---|---|
| Model-visible stream | Event IDs, payloads, coordinates, timestamps, parent metadata supplied by the current graph, intervention/uncertainty/provenance markers, and masked encoded features | True coefficient matrix, hidden exogenous noise, evaluator counterfactual target, or private graph-family label |
| Learner training labels | Observed/intervened payload targets and declared training intervention values | Test-world graph truth or hidden test noise |
| Evaluator truth | True graph, structural coefficients, exogenous values, intervention outcomes, counterfactual outcomes, and corruption manifest | Any data returned to the model-visible encoder |

## Objectives and controls

The native learner logs separate objective values for event prediction, edge recovery, temporal ordering, intervention effect prediction, counterfactual prediction, graph consistency, and abstention. The deterministic gate does not combine them into one opaque score.

Mandatory negative controls are:

| Control | Expected behavior |
|---|---|
| Shuffled-edge input | Edge F1 and intervention performance degrade materially relative to aligned edges |
| Sequence-only encoder | Temporal prediction may remain finite, but intervention effect and edge recovery must not match the graph-conditioned learner |
| Observation-only learner | Held-out intervention effect error is worse than the intervention-aware learner |
| Graph-only learner | Removing payloads prevents accurate numerical intervention outcomes |
| Future-leakage run | Unmasked future features may improve scores, but the masked run must remain finite and the evaluator must detect the information-path difference |
| Corrupted graph run | Deleting, reversing, and adding irrelevant edges produces measurable degradation or abstention; it must not be silently ignored |

## Declared Stage 3 thresholds

Thresholds are fixed before the final gate run and written into `artifacts/stage-3/cpp-gate/metrics.json`.

| Check | Pass condition |
|---|---|
| Schema integrity | All fields survive round-trip; malformed version, coordinates, duplicate IDs, and invalid parents are rejected |
| Graph safety | Cycles are rejected; causal past/future and topological order are deterministic; insertion-order permutations produce identical snapshots |
| Leakage control | Model-visible serialization contains no evaluator-only marker; future-parent masking changes the encoded feature and blocks future payload contribution |
| Temporal task | Masked graph-conditioned temporal outputs remain finite and agree across repeated deterministic runs |
| Structural recovery | Held-out graph edge precision, recall, and F1 are each at least `0.75`; topological violations are zero |
| Intervention task | Held-out intervention MSE is at least `40%` lower than the observation-only control and effect-direction accuracy is at least `0.80` |
| Counterfactual task | Held-out counterfactual MSE is at most `0.08`; irrelevant-variable permutation changes prediction by at most `1e-12` |
| Robustness | Edge deletion/reversal causes measurable degradation or abstention; all outputs remain finite |
| Calibration and abstention | Conflicting or incomplete graph evidence yields an explicit abstention on at least one non-identifiable query; known queries remain calibrated with finite confidence |
| Ablation integrity | Edge, temporal, intervention, uncertainty, and MIMO channels each produce an independent observable configuration or metric change |
| Reproducibility | Two identical seeds produce byte-identical reports; a changed seed produces a distinct dataset fingerprint while preserving qualitative pass status |

Every mandatory check must pass. No deferred limitation is permitted at the Stage 3 transition.

## Gate artifact contract

The native gate executable is `cct_stage3_gate --output artifacts/stage-3/cpp-gate`. It writes:

| Artifact | Contents |
|---|---|
| `gate.json` | Stage number, PASS/FAIL status, commit, dirty-tree state, transition authorization, and leakage boundary status |
| `checks.json` | One record per mandatory check with duration, status, and machine-readable details |
| `metrics.json` | Threshold, measured value, unit, and status for every declared metric |
| `report.md` | Human-readable methodology, baseline/control comparison, degradation evidence, calibration/abstention evidence, and explicit scope limits |
| `visible_input.json` | Canonical model-visible schema fingerprint and fields |
| `evaluator_truth.json` | Evaluator-only schema inventory and access audit; no payload truth is copied into the visible artifact |

The gate must return nonzero on any failed check. A clean tree and remote synchronization are required before publishing the Stage 3 checkpoint.

## Transition package

The Stage 3 transition package consists of the versioned schema and store API, deterministic synthetic generator, causal learner, graph-conditioned sequence wrapper, native regression tests, gate artifacts, ablation and negative-control metrics, robustness/corruption results, calibration and abstention results, representative failure cases, and the final commit SHA. Passing this contract authorizes only **Stage 4 preparation after explicit user approval**.

## Non-goals

This stage does not implement persistent verifiable memory, large-scale language/code training, deliberation, multimodal perception, live tools, autonomous research, or claims of superintelligence. Any result is valid only for the declared synthetic structural-equation distributions and native implementation paths.
