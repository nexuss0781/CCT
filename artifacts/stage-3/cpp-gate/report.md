# Native C++ Stage 3 Gate Report

**Status:** `PASS`  
**Transition:** `Stage 4 preparation; approval required`  
**Implementation:** `native-cpp-causal-event-learning`  
**Commit:** `f4788ca8e8d4649e679d39c431ccb59a5b795243`  
**Dirty tree at gate execution:** `False`

## Methodology

The gate uses deterministic synthetic structural equations with four variables, confounded observational noise, held-out do-interventions, paired counterfactual worlds, a held-out nonlinear parent feature, and a separate evaluator-only truth structure. The model-visible schema excludes coefficients, exogenous noise, and counterfactual targets. The Stage 2 selective recurrent core is exercised through a graph-conditioned encoder with future-parent masking and loop/scan equivalence.

## Mandatory checks

| Check | Status | Duration (s) |
|---|---:|---:|
| schema_integrity_and_graph_safety | `PASS` | 0.000185009 |
| leakage_control_and_temporal_masking | `PASS` | 7.079e-05 |
| structural_edge_recovery | `PASS` | 0.000375068 |
| intervention_effect_prediction | `PASS` | 0.000585421 |
| counterfactual_consistency | `PASS` | 0.000324174 |
| robustness_and_abstention | `PASS` | 0.000706038 |
| strict_contract_failure_closure | `PASS` | 0.000449089 |
| ablation_integrity | `PASS` | 7.0778e-05 |
| reproducibility | `PASS` | 0.000374069 |

## Scope limits

A passing gate demonstrates causal-structure-aware prediction on the declared synthetic structural-equation distributions. It does not establish general causal understanding, causal discovery on real data, language competence, or superintelligence. Stage 4 implementation remains blocked until explicit user approval.
