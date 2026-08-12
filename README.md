# CCT-ASE Native C++ Research Prototype

The **Chrono-Causal Tapestry — Adaptive Spectral Engine (CCT-ASE)** is a research prototype for testing causal event fields and efficient spectral dynamics. The repository currently provides a reproducible **native C++20 numerical, sequence, causal-event, persistent-memory, small-scale language/code, bounded deliberation, and controlled multimodal substrate**; it does not claim to be a general language model or superintelligence system. The implemented Stage 0 through Stage 7 gates establish reproducibility, numerical correctness, recurrent trainability, efficiency measurement, leakage-controlled graph conditioning, intervention prediction, counterfactual evaluation, persistent checksummed memory, exact retrieval, provenance, deletion, conflict handling, poisoning resistance, checkpoint replay, bounded token learning, code-safety checks, independent verification, evidence abstention, deny-by-default offline tools, interruption/replay, typed multimodal events, temporal/spatial alignment, modality masking, cross-modal memory, deterministic simulation, transfer controls, and audit logging on declared fixtures.

> **Current status:** Stages 0 through 7 are implemented in native C++ and pass their mandatory gates. The roadmap now requires research review before any further continuation or external integration.

## Implemented scope

The active runtime is C++20 with CMake and FFTW3. The native library provides event/manifold storage, periodic spectral Laplacians, an independent finite-difference reference Laplacian, leapfrog and RK4 integration, periodic/Dirichlet/Neumann boundaries, CFL rejection, bounded local potentials, analytic one-step gradients, a real/complex selective recurrent sequence core, segmented prefix-scan execution, optional RMS normalization, checkpointing, trained dense-attention/GRU/diagonal-SSM comparators, deterministic algorithmic benchmark executables, versioned causal event storage, DAG queries, leakage-safe graph-conditioned recurrence, synthetic structural-equation generation, intervention/counterfactual prediction, robustness, abstention, checksummed append-only memory logs, exact metadata/vector retrieval, versioning, citations, retention/deletion, conflict sets, recovery, byte-fallback vocabulary, manifest-addressed fixtures, checkpoint-resumable token models, matched dense-attention/GRU/diagonal-SSM/CCT comparators, frozen-memory attribution, long-context diagnostics, static code-safety checks, bounded typed planning, independent arithmetic/graph/evidence verifiers, deny-by-default offline tool policy, interruption/resume, deterministic replay, and incident logging.

The implementation is deliberately split into clear reference paths and optimized paths. The spectral and finite-difference solvers are compared on identical inputs, and the sequence loop and prefix scan are compared on identical inputs, including complex state and segmented masks. Gates check manufactured solutions, temporal convergence, energy drift, boundary residuals, recurrent path equivalence, gradient agreement, checkpoint recovery, copy/parity/associative/overwrite learning, trained matched baselines, ablations, and measured scaling.

| Stage | Status | Native entry points |
|---|---|---|
| Stage 0 — Reproducible baseline | **PASS** | `cct_stage0_gate`, `Stages/00_Reproducible_Baseline.md` |
| Stage 1 — Differentiable numerical engine | **PASS** | `cct_native`, `cct_tests`, `cct_stage1_gate`, `Stages/01_Numerical_Engine.md` |
| Stage 2 — Efficient sequence core | **PASS** | `cct_sequence_tests`, `cct_stage2_gate`, `Stages/02_Sequence_Core.md` |
| Stage 3 — Causal event learning | **PASS** | `cct_causal_tests`, `cct_stage3_gate`, `Stages/03_Causal_Event_Learning.md` |
| Stage 4 — Persistent verifiable memory | **PASS** | `cct_memory_tests`, `cct_stage4_gate`, `Stages/04_Persistent_Verifiable_Memory.md` |
| Stage 5 — Language and code scaling | **PASS** | `cct_scaling_tests`, `cct_stage5_gate`, `Stages/05_Language_Code_Scaling.md` |
| Stage 6 — Deliberation and verification | **PASS** | `cct_deliberation_tests`, `cct_stage6_gate`, `Stages/06_Deliberation_Verification.md` |
| Stage 7 — Multimodal and controlled research | **PASS** | `cct_multimodal_tests`, `cct_stage7_gate`, `Stages/07_Multimodal_Open_Ended.md` |

## Requirements

A declared CPU build requires a C++20 compiler, CMake 3.20 or newer, `pkg-config`, and FFTW3 development headers and libraries. On Ubuntu, the dependencies can be installed with:

```bash
sudo apt-get update
sudo apt-get install -y g++ cmake pkg-config libfftw3-dev
```

The repository contains no active Python runtime, Python test suite, Python packaging path, or Python gate script. The older Rust crate remains in the repository as historical substrate material, but the active Stage 0/1/2 implementation and validation path is native C++.

## Build and validate

The canonical commands are:

```bash
make native-build
make native-test
make stage0-gate
make stage1-test
make stage1-gate
make stage2-test
make stage2-gate
make stage3-test
make stage3-gate
make stage4-test
make stage4-gate
make stage5-test
make stage5-gate
make stage6-test
make stage6-gate
make stage7-test
make stage7-gate
make ci-stage7
```

`make native-build` configures and compiles the C++ library and executables under `build-cpp/`. `make native-test` runs the CTest suite. The gate commands create machine-readable artifacts under `artifacts/stage-0/cpp-gate/`, `artifacts/stage-1/cpp-gate/`, `artifacts/stage-2/cpp-gate/`, `artifacts/stage-3/cpp-gate/`, and `artifacts/stage-4/cpp-gate/`. `make ci-stage4` executes the complete native Stage 4 pipeline, including all prior gates, and returns a nonzero status if any mandatory check fails. `make ci-stage5` extends this to the manifest-audited language/code scaling suite and Stage 5 gate. `make ci-stage6` adds the bounded deliberation, independent-verifier, offline policy, replay, interruption, and incident harness. `make ci-stage7` adds the terminal multimodal event, alignment, fusion, typed-memory, simulation, transfer, audit, and safety harness.

A clean build can also be invoked directly:

```bash
rm -rf build-cpp
cmake -S cpp -B build-cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpp --parallel 2
ctest --test-dir build-cpp --output-on-failure
```

## Native API surface

The public headers are in `cpp/include/cct/`. The event substrate is defined in `cct/event.hpp`. The numerical engine is defined in `cct/field.hpp` and exposes the following conceptual contract:

```cpp
cct::FieldState state = solver.initialize(phi0, psi0);
cct::FieldState next = solver.step(state, source, potential);
cct::Trajectory trajectory = solver.rollout(state, source_sequence, potential);
double loss = solver.operator_loss(prediction, target, mask);
```

The field equation under test is:

```text
∂²φ/∂t² = c² Δφ − V(x)φ + J(x,t)
```

The spectral implementation applies Fourier multipliers on periodic regular grids. The reference implementation uses an independent finite-difference stencil. Analytic source and potential gradients for the leapfrog one-step loss are checked against centered finite differences; this is a native training contract, not a claim that a complete optimizer or language model exists.

## Gate criteria

The Stage 1 and Stage 2 gates are intentionally stricter than build smoke tests. Stage 2 records the following mandatory checks:

| Check | Required result |
|---|---:|
| FFT round-trip correctness | PASS |
| Spectral/reference Laplacian agreement | PASS |
| Spectral/reference rollout agreement | PASS |
| Manufactured-solution accuracy | PASS |
| RK4 convergence order | PASS |
| Energy stability | PASS |
| CFL rejection | PASS |
| Analytic/finite-difference gradient agreement | PASS |
| Dirichlet and Neumann residuals | PASS |
| Configuration serialization | PASS |
| Measured subquadratic scaling | PASS |
| Reference/prefix-scan equivalence | PASS |
| Streaming and chunked equivalence | PASS |
| Sequence gradient finite differences | PASS |
| Long-horizon state stability | PASS |
| Copy and delayed-recall training | PASS |
| Parity/state tracking | PASS |
| Associative recall | PASS |
| Selective overwrite | PASS |
| Checkpoint recovery | PASS |
| Trained dense-attention/GRU/diagonal-SSM baselines | PASS |
| Complex-state equivalence | PASS |
| Normalization and checkpoint persistence | PASS |
| Segmented masked scan | PASS |
| Selective-gate/MIMO/normalization ablations | PASS |
| Linear scaling and constant decode state memory | PASS |

The strengthened Stage 2 gate requires **12 mandatory checks** plus limitation-closure metrics. A Stage 2 `PASS` authorizes Stage 3 preparation only. It does not authorize Stage 3 implementation without explicit user approval.

## Repository map

| Path | Purpose |
|---|---|
| `cpp/include/cct/` | Public native C++ headers |
| `cpp/src/` | Event, field, FFT, sequence, baseline, and numerical-engine implementations |
| `cpp/tests/cct_tests.cpp` | Stage 0/1 native regression suite |
| `cpp/tests/sequence_tests.cpp` | Stage 2 sequence regression suite, including complex, normalization, and segmented-mask tests |
| `cpp/include/cct/baselines.hpp` | Matched baseline public API |
| `cpp/src/baselines.cpp` | Dense attention, GRU, and diagonal SSM implementations |
| `cpp/include/cct/causal.hpp` | Versioned causal event, graph, encoder, dataset, and learner API |
| `cpp/src/causal.cpp` | Native causal graph store, generator, learner, and Stage 2 integration |
| `cpp/tests/causal_tests.cpp` | Stage 3 causal regression suite |
| `cpp/tools/stage3_gate.cpp` | Stage 3 artifact-producing gate |
| `Stages/03_Expanded_Gate_Contract.md` | Stage 3 thresholds, controls, and artifact contract |
| `cpp/include/cct/memory.hpp` | Persistent memory, log, retrieval, citation, and retention API |
| `cpp/src/memory.cpp` | Checksummed append-only memory implementation and causal-event adapter |
| `cpp/tests/memory_tests.cpp` | Stage 4 persistent-memory regression suite |
| `cpp/tools/stage4_gate.cpp` | Stage 4 artifact-producing gate |
| `Stages/04_Expanded_Gate_Contract.md` | Stage 4 thresholds, controls, and artifact contract |
| `cpp/include/cct/scaling.hpp` | Stage 5 vocabulary, model, trainer, checkpoint, and memory-augmentation API |
| `cpp/src/scaling.cpp` | Native Stage 5 model wrapper, metrics, checkpointing, and memory attribution |
| `cpp/tests/scaling_tests.cpp` | Stage 5 vocabulary, training, checkpoint, baseline, and memory regression suite |
| `cpp/tools/stage5_gate.cpp` | Stage 5 artifact-producing gate |
| `data/stage-5/manifests/stage5_manifest.txt` | Immutable Stage 5 provenance and SHA-256 manifest |
| `Stages/05_Expanded_Gate_Contract.md` | Stage 5 thresholds, controls, and artifact contract |
| `cpp/include/cct/deliberation.hpp` | Stage 6 bounded workspace, planner, verifier, tool, evidence, and trace API |
| `cpp/src/deliberation.cpp` | Native deliberation engine, independent verifiers, policy, replay, and serialization |
| `cpp/tests/deliberation_tests.cpp` | Stage 6 deliberation and safety regression suite |
| `cpp/tools/stage6_gate.cpp` | Stage 6 artifact-producing gate |
| `Stages/06_Expanded_Gate_Contract.md` | Stage 6 thresholds, controls, and artifact contract |
| `cpp/include/cct/multimodal.hpp` | Stage 7 typed multimodal events, adapters, alignment, fusion, environment, transfer, and audit API |
| `cpp/src/multimodal.cpp` | Native multimodal event store, adapters, alignment, fusion, deterministic environment, and audit implementation |
| `cpp/tests/multimodal_tests.cpp` | Stage 7 multimodal and controlled-environment regression suite |
| `cpp/tools/stage7_gate.cpp` | Stage 7 terminal artifact-producing gate |
| `Stages/07_Expanded_Gate_Contract.md` | Stage 7 thresholds, controls, and terminal artifact contract |
| `cpp/tools/stage0_gate.cpp` | Stage 0 artifact-producing gate |
| `cpp/tools/stage1_gate.cpp` | Stage 1 artifact-producing gate |
| `cpp/tools/stage2_gate.cpp` | Stage 2 artifact-producing gate |
| `RESEARCH_STAGE2.md` | Primary state-space design references |
| `Stages/` | Independent stage specifications and transition contracts |
| `SPEC/` | Historical and forward-looking mathematical specifications |
| `artifacts/` | Local generated gate reports; excluded from source control |

## Research limitations

The current implementation validates a numerical operator substrate, a deterministic real/complex selective recurrent core, a native causal-event learner, a local persistent verifiable memory subsystem, a small native language/code scaling benchmark, a bounded deliberation/verification harness, and a controlled multimodal event/simulation harness on declared offline fixtures. Stage 7 supports seven typed modalities, explicit provenance, temporal/spatial alignment, mask-aware fusion, typed cross-modal retrieval, deterministic grid replay, transfer metadata, and audit/safety records. It does not validate broad language competence, general multimodal understanding, open-ended reasoning, unrestricted code generation, real-world perception or robotics, repository-level engineering, distributed scaling, autonomous agency, external deployment, or superintelligence. Further work requires research review and a new specification for any external integration or broader autonomy.

## License

MIT License. See the repository license file for the applicable terms.
