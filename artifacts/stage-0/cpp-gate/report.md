# CCT Level 1 Stage L1-0 Baseline Report

**Status:** `PASS`  
**Commit:** `dc590fadf51b16553f601df8c608c17d5dcf14c5`  
**Configuration hash:** `8c1a11faf7fdc8d2827f333b79aa9d470fbdf97091feb303a0ebb6527e5e6fd3`  
**Tracked source/configuration tree dirty at execution:** `False`

| Check | Status | Duration (s) |
|---|---:|---:|
| event_lifecycle | `PASS` | 3.453e-06 |
| invalid_input_non_mutation | `PASS` | 4.9335e-05 |
| deterministic_native_path | `PASS` | 0.000740319 |
| configuration_validation_and_identity | `PASS` | 2.0178e-05 |
| configuration_and_injected_threshold | `PASS` | 4.209e-06 |
| repository_hygiene | `PASS` | 0.0322008 |

The gate proves only the native reproducible-baseline contract: configuration identity, deterministic numerical replay, failure-path non-mutation, and deliberate threshold-failure detection. It does not establish language-teacher capability.
