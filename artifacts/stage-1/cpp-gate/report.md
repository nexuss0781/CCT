# Native C++ Stage 1 Gate Report

**Status:** `PASS`  
**Transition:** `Stage 2`  
**Implementation:** `native-cpp`  
**Commit:** `57c0c0082d9bc4c8de287c877e59e7ad2580f3eb`  
**Dirty tree:** `False`  
**Configuration hash:** `745b25a845e29fc7`

## Checks

| Check | Status | Duration (s) |
|---|---:|---:|
| transform_correctness | `PASS` | 0.000910939 |
| spectral_reference_operator_agreement | `PASS` | 0.000267947 |
| spectral_reference_rollout_agreement | `PASS` | 0.00134199 |
| manufactured_solution_accuracy | `PASS` | 0.00175749 |
| forced_manufactured_solution | `PASS` | 0.00104652 |
| temporal_convergence | `PASS` | 0.00298432 |
| energy_stability | `PASS` | 0.0204929 |
| cfl_rejection | `PASS` | 5.2272e-05 |
| analytic_finite_difference_gradients | `PASS` | 0.000498477 |
| boundary_residuals | `PASS` | 0.000740218 |
| serialization_round_trip | `PASS` | 0.000165437 |
| precision_policy | `PASS` | 1.2871e-05 |
| nonfinite_input_rejection_and_source_causality | `PASS` | 2.5168e-05 |
| schema_version_validation | `PASS` | 0.000127823 |
| performance_scaling | `PASS` | 0.00450709 |

## Transition policy

A `PASS` proves the native C++ Stage 1 implementation and harness are green. It authorizes Stage 2 preparation only; Stage 2 implementation requires explicit user approval.
