# Stage 3 — Causal Event Learning

**Project:** CCT-ASE  
**Stage ID:** 3  
**Predecessor:** Stage 2 — Efficient Sequence Core  
**Successor:** Stage 4 — Persistent Verifiable Memory  
**Status:** Implemented in native C++20; Stage 3 gate PASS with strict event metadata, temporal-policy leakage controls, transactional learner fitting, evaluator-truth separation, and nine mandatory checks; Stage 4 approval required.

## Purpose

Stage 3 adds structured event identity, temporal order, causal-parent links, interventions, and counterfactual prediction to the efficient sequence core. The stage tests whether CCT-ASE can use explicit causal structure rather than merely exploit sequential correlation.

The output is a causal-structure-aware learner, not a claim of human-level causal understanding. A causal graph supplied to the model is a representation and an inductive bias. The stage passes only if the model survives graph perturbation, intervention, confounding, and shuffled-edge ablations.

## Scope and non-goals

The stage includes a versioned event schema, native C++ event storage, DAG validation, causal queries, event-to-sequence encoding, graph-conditioned recurrent updates, intervention datasets, counterfactual objectives, leakage auditing, robustness checks, calibration, abstention, and causal evaluation. It does not implement persistent semantic memory, large-scale language modeling, deliberation, or live tools.

The initial graph family should be synthetic and fully controlled so that ground-truth structural equations, interventions, and counterfactual outcomes are known. Natural data may be added only when provenance and causal assumptions are documented.

## Event and graph contract

Each event must have a stable identifier and immutable provenance fields:

```text
Event {
    id: EventId,
    semantic_payload: Tensor,
    coordinates: CoordinateVector,
    timestamp: LogicalTime,
    causal_parents: Vec<EventId>,
    intervention: Optional<Intervention>,
    uncertainty: UncertaintyRecord,
    provenance: ProvenanceRecord
}
```

The store must enforce:

| Invariant | Required behavior |
|---|---|
| Identity | Event IDs are unique and stable across serialization |
| Coordinate validity | Coordinates have declared dimension, range, and coordinate system |
| Time validity | Timestamps have a declared ordering and tie policy |
| Parent validity | Every parent exists or the event is explicitly marked unresolved |
| Acyclicity | Causal-parent edges form a DAG unless cycles are explicitly represented as hypotheses outside the ground-truth graph |
| Provenance | Generated, observed, intervened, and retrieved events are distinguishable |
| Uncertainty | Unknown, estimated, and known values are not silently conflated |
| Immutability | Historical events cannot be changed without a new version or correction event |

The native implementation also rejects unsupported provenance, uncertainty, and intervention modes; requires unresolved parent IDs to be declared in the causal-parent list; validates finite payloads, coordinates, confidence, and intervention values; and publishes failed insertions transactionally. The encoder repeats model-visible validation for duplicate IDs, finite values, sorted unique parents, and temporal policy so callers cannot bypass store checks by constructing an ad hoc event stream.

The API must support insertion, exact lookup, parent/child lookup, causal past, causal future, intervention application, graph snapshotting, and deterministic export. Queries must return event IDs and provenance, not only vectors.

## Causal model contract

The benchmark generator must define structural equations of the form:

```text
X_i := f_i(Pa_i, U_i)
```

where `Pa_i` is the true parent set and `U_i` is exogenous noise. It must be possible to generate observational samples, interventions such as `do(X_j = c)`, and counterfactual pairs sharing the same exogenous variables.

The learner receives an event stream plus an explicit indication of which variables are observed, intervened, missing, or uncertain. It must predict target events, parent edges, intervention effects, or counterfactual outcomes according to the task configuration.

The model must not be allowed to inspect generator-only metadata such as the true graph, hidden noise, or counterfactual target during inference. The harness must enforce separate serialization paths for model-visible inputs and evaluator-only truth.

## Required implementation

| Component | Required implementation | Testable contract |
|---|---|---|
| Event schema | Versioned typed record with IDs, timestamps, parent links, provenance, and uncertainty | Round-trip preserves all fields and rejects malformed records |
| Graph store | Arena or indexed storage with deterministic parent/child adjacency | CRUD and queries are stable under insertion order |
| DAG validator | Cycle detection, missing-parent handling, and topological ordering | Invalid graphs produce structured errors |
| Causal encoder | Convert graph metadata and event payloads into model inputs | Masking prevents unavailable future information |
| Graph-conditioned core | Add causal edge messages or parent summaries to the Stage 2 recurrence | Edge channel can be disabled for ablation |
| Intervention adapter | Represent observed, do-intervened, and counterfactual contexts distinctly | Interventions cannot be confused with observations |
| Objectives | Edge prediction, temporal ordering, intervention effect, and counterfactual losses | Each objective is independently logged |
| Dataset generator | Produce graph families with held-out functions, topologies, confounders, and noise levels | Test splits are generated from independent seeds |
| Audit record | Record visible graph, intervention, model output, evidence, and evaluator truth separately | No hidden truth appears in model input; failed learner fits preserve the prior fitted model |

A recommended multi-task objective is:

```text
L = L_event + λ_edge L_edge + λ_time L_time
  + λ_intervention L_do + λ_cf L_counterfactual
  + λ_consistency L_graph
```

The graph consistency term must not force the model to agree with an incorrect supplied graph. When graph uncertainty is present, the model should represent uncertainty or compare alternative hypotheses rather than silently treating the graph as fact. The native learner rejects unsorted, duplicate, self, future, or otherwise cyclic parent hypotheses before fitting and publishes a fitted model only after all child regressions and finite-state checks succeed.

## Training and data protocol

The generator must create training, validation, and test families separated by more than random row splits. Hold out at least some structural functions, graph motifs, variable names, intervention values, and sequence lengths. Include confounded and unconfounded settings, missing observations, noisy timestamps, irrelevant edges, and deliberately corrupted graph metadata.

Training should begin with exact symbolic or discretized structural equations where the target is measurable. The model must first learn temporal order and graph edge reconstruction, then intervention prediction, then counterfactual consistency. Curriculum changes must be recorded in the experiment manifest.

The harness must include negative controls:

- A shuffled-edge input that destroys true graph alignment.
- An observation-only model with no intervention markers.
- A sequence-only model with no causal edges.
- A graph-only model with payloads removed.
- A model trained on a graph family that does not contain the test mechanism.

## Evaluation harness

### Structural recovery

Measure edge precision, recall, F1, structural Hamming distance, ancestor recovery, and topological-order violations. Metrics must be computed against the held-out ground-truth graph and reported by graph size and noise level.

### Intervention prediction

For each intervention, measure outcome error, calibration, and effect-direction accuracy. Evaluate both in-distribution and held-out intervention values. An observational-only baseline must be included to show whether the intervention channel contributes information.

### Counterfactual consistency

For paired worlds with shared exogenous variables, measure counterfactual outcome error and consistency under irrelevant changes. The model must not change a target solely because an unrelated variable or event ID was permuted.

### Temporal and causal masking

Inject future events and verify that a causal-time mask prevents information leakage. Compare masked and deliberately unmasked runs. The evaluator must fail if a model achieves suspiciously perfect performance only when future information is visible.

### Robustness and falsification

Perturb edge direction, delete parent links, add irrelevant edges, reorder independent events, add timestamp noise, and corrupt selected payloads. Report degradation curves. A model that claims causal reasoning but is unchanged by causal graph corruption requires investigation rather than automatic approval.

### Calibration and abstention

When the graph is incomplete or conflicting, evaluate confidence and abstention. The model must be able to state that a causal effect is not identifiable under the provided evidence instead of producing an unjustified point estimate.

## Pass/fail criteria

| Criterion | Pass condition | Failure condition |
|---|---|---|
| Schema integrity | Event and graph round trips preserve identity, provenance, time, and uncertainty | Fields are dropped, reordered, or silently coerced |
| Graph safety | Cycles, missing parents, and invalid coordinates are handled explicitly | Invalid graph is accepted as valid or causes silent corruption |
| Leakage control | Evaluator detects and prevents future truth, hidden noise, and counterfactual target leakage | Model-visible input contains evaluator-only metadata |
| Temporal task | Model meets predefined ordering and masking thresholds on held-out lengths | Performance collapses under modest length or timestamp variation |
| Edge task | Model beats sequence-only and shuffled-edge baselines by the declared margin | Edge predictions are no better than controls |
| Intervention task | Model predicts held-out intervention effects better than observational-only baseline with calibrated uncertainty | It memorizes observational correlations or fails intervention markers |
| Counterfactual task | Model satisfies error and irrelevant-change consistency thresholds | Counterfactuals are inconsistent or depend on irrelevant IDs |
| Robustness | Performance degradation under graph corruption is measurable and behavior remains finite | Model silently treats corrupted graphs as trustworthy |
| Ablation | Edge, temporal, intervention, and uncertainty channels have independent reports | Component contribution cannot be isolated |
| Reproducibility | Independent seeds reproduce the qualitative ranking of baselines | Results depend on one seed or one graph family |

A pass requires success on every mandatory task family, not only the easiest graph size. Thresholds must be declared before the final test run and stored with the benchmark artifact. The current native gate has **nine mandatory checks**, including an independent strict-contract failure-closure check for missing parents, invalid enums, same-time parents, duplicate encoder IDs, non-finite payloads and queries, transactional fit preservation, and invalid causal inputs.

## Transition to Stage 4

Stage 4 may begin when event identity and causal metadata are stable enough to serve as memory records. The transition package must include the versioned schema, graph-store API, generator code, leakage audit, intervention and counterfactual reports, ablations, calibration results, and a list of non-identifiable cases where abstention is required.

If the stage fails, the team must distinguish representation failure from causal-learning failure. It is not acceptable to add more model capacity until a sequence-only baseline, shuffled-edge control, and leakage audit have been run.

## Exit report

The report must include graph-family definitions, structural equations, split policy, visible/evaluator-only schemas, task metrics, calibration curves, corruption curves, baseline comparisons, seed variance, and representative failure cases. It must explicitly state that passing this stage demonstrates causal-structure-aware prediction on the tested distributions, not general causal understanding.

**Transition decision:** `PASS` authorizes Stage 4 preparation only after explicit approval. `FAIL` requires remediation. `BLOCKED` is allowed only for optional real-world datasets; the synthetic causal suite, evaluator-truth separation, and leakage audit must pass. The current gate records `PASS` with `approval_required: true`.

## References

[1]: ../CCT_EVOLUTION_PROPOSAL.md "CCT-ASE evolution proposal"

[2]: ../SPEC/Phase-1.md "CCT Phase 1 substrate specification"

[3]: ../Stages/02_Sequence_Core.md "CCT Stage 2 sequence core specification"
