# Native C++ Stage 2 Gate Report

**Status:** `PASS`  
**Transition:** `Stage 3 preparation; approval required`  
**Implementation:** `native-cpp-selective-recurrent-core`  
**Commit:** `06b3a99e81fb021621b6036e60fea1f6b98033d2`  
**Dirty tree:** `False`

## Mandatory checks

| Check | Status | Duration (s) |
|---|---:|---:|
| reference_scan_equivalence | `PASS` | 0.00367198 |
| streaming_equivalence | `PASS` | 0.000431765 |
| gradient_correctness | `PASS` | 0.000224468 |
| long_horizon_stability | `PASS` | 0.0211006 |
| algorithmic_copy_and_delayed_recall | `PASS` | 1.0003e-05 |
| parity_associative_overwrite_suite | `PASS` | 0.780518 |
| checkpoint_recovery | `PASS` | 0.000545591 |
| state_lifecycle_and_recurrent_resume | `PASS` | 0.000246182 |
| adversarial_gate_clamp_equivalence | `PASS` | 0.000209031 |
| failure_closure | `PASS` | 9.0692e-05 |
| matched_baseline_training | `PASS` | 0.0288192 |
| complex_state_equivalence | `PASS` | 0.000181889 |
| normalization_and_checkpoint | `PASS` | 0.000355678 |
| segmented_mask_scan | `PASS` | 0.000149061 |
| ablation_integrity_contract | `PASS` | 0.000155371 |
| linear_scaling_and_decode_memory | `PASS` | 0.0532839 |

## Limitation-closure evidence

Complex state is enabled and passes loop/scan equivalence with real and imaginary state errors below the declared tolerance. RMS state/output normalization is enabled, ablated, checkpointed, and measured. Segmented masked scanning is implemented and agrees with the reference loop across multiple active segments. Dense causal attention, GRU, diagonal SSM, and CCT are all trained on the same deterministic task budget with loss, parameter count, state memory, and timing reported. Selective-gate and MIMO ablations are independently measured.

A `PASS` authorizes Stage 3 preparation only; Stage 3 implementation requires explicit user approval.
