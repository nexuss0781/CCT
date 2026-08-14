# CCT Level 1 L1-0 Clean-Checkout Verification

**Release commit:** `dc590fadf51b16553f601df8c608c17d5dcf14c5`  
**Configuration hash:** `8c1a11faf7fdc8d2827f333b79aa9d470fbdf97091feb303a0ebb6527e5e6fd3`  
**Source/configuration tree:** clean at gate execution  
**Native toolchain:** GCC 13.3.0, CMake 3.28.3, FFTW3 3.3.10, Ubuntu Linux x86_64

## Replay results

| Check | Result |
|---|---:|
| Fresh Git clone at release commit | PASS |
| `make native-build` | PASS |
| `make native-test` | PASS — 39/39 CTest tests |
| `make stage0-gate` | PASS — six L1-0 checks |
| Expanded-warning CTest replay | PASS — 39/39 tests |
| Deliberately malformed L1-0 configurations | PASS — rejected without state mutation |
| Deliberately failing benchmark threshold | PASS — recorded as `FAIL` with diagnostics while the gate continued |
| Missing PkgConfig/FFTW configure replay | PASS — CMake exited nonzero with diagnostic output |

The Stage 0 artifact bundle contains `config.json`, `environment.json`, `tests.json`, `benchmark_record.json`, `manifest.json`, `gate.json`, `release_record.json`, `report.md`, `gate_envelope.json`, and this clean-checkout verification record. This evidence establishes the reproducible native baseline only; it does not establish language-teacher capability.
