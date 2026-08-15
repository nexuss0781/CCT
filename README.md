# CCT-ASE Native C++ Research Prototype

The **Chrono-Causal Tapestry — Adaptive Spectral Engine (CCT-ASE)** is a research prototype for testing causal event fields and efficient spectral dynamics. The repository currently provides a reproducible **native C++20 numerical, sequence, causal-event, persistent-memory, small-scale language/code, bounded deliberation, controlled multimodal, governed-corpus, tokenizer/representation, trainable next-token, CPU scaling, supervised-adaptation, preference-alignment, verified-retrieval, production-like inference-operations, and controlled-release substrate**; it does not claim to be a general language model or superintelligence system. The implemented Stage 0 through Stage 17 gates establish reproducibility, recurrent trainability, efficiency measurement, governance, real-source corpus controls, tokenizer and checkpoint integrity, supervised adaptation, preference alignment, verified retrieval, versioned serving, bounded pilot admission, human review, incident response, deletion, drift, rollback, and named approval controls on declared fixtures and a named low-risk release scope.

> **Current status:** Stages 0 through 17 are implemented in native C++ and pass their mandatory gates on declared fixtures. Stage 17 adds an immutable bounded-release scope, locked replay, shadow isolation, pilot allowlists and quotas, human oversight, SLO and safety monitoring, incident containment, deletion propagation, drift detection, rollback rehearsal, and technical/security/product/governance approvals. The terminal release is limited to its named task, tenants, users, region, data classes, permissions, and expiration; future expansion requires a new specification.

## Implemented scope

The active runtime is C++20 with CMake and FFTW3. The native library provides event/manifold storage, periodic spectral Laplacians, an independent finite-difference reference Laplacian, leapfrog and RK4 integration, periodic/Dirichlet/Neumann boundaries, CFL rejection, bounded local potentials, analytic one-step gradients, a real/complex selective recurrent sequence core, segmented prefix-scan execution, optional RMS normalization, checkpointing, trained dense-attention/GRU/diagonal-SSM comparators, deterministic algorithmic benchmark executables, versioned causal event storage, DAG queries, leakage-safe graph-conditioned recurrence, synthetic structural-equation generation, intervention/counterfactual prediction, robustness, abstention, checksummed append-only memory logs, exact metadata/vector retrieval, versioning, citations, retention/deletion, conflict sets, recovery, manifest-addressed fixtures, checkpoint-resumable token models, matched dense-attention/GRU/diagonal-SSM/CCT comparators, frozen-memory attribution, long-context diagnostics, static code-safety checks, bounded typed planning, independent arithmetic/graph/evidence verifiers, deny-by-default offline tool policy, interruption/resume, deterministic replay, incident logging, the Stage 10 tokenizer/representation API with byte, subword, hybrid, byte-fallback, source-offset, immutable-snapshot, throughput, packed-batch, and padded-batch paths, and the Stage 11 categorical next-token trainer with analytic CCT gradients, optimizer moments, scheduler state, matched controls, and deterministic checkpoint recovery.

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
| Stage 8 — Production NLP foundation and governance | **PASS — foundation only** | `cct_production_tests`, `cct_stage8_gate`, `Stages/08_Production_NLP_Roadmap.md` |
| Stage 9 — Governed data and corpus | **PASS — real-source corpus foundation** | `cct_corpus_tests`, `cct_stage9_gate`, `Stages/09_Governed_Data_Corpus.md` |
| Stage 10 — Tokenizer and representation | **PASS — hybrid candidate selected; training not authorized** | `cct_tokenizer_tests`, `cct_stage10_gate`, `Stages/10_Tokenizer_Representation.md` |
| Stage 11 — Trainable native NLP core | **PASS — bounded pilot; training authorization remains false** | `cct_nlp_trainer_tests`, `cct_stage11_gate`, `Stages/11_Trainable_Native_NLP_Core.md` |
| Stage 12 — Scaling and accelerator systems | **PASS — CPU-only declared scope; no GPU/cluster claim** | `cct_scaling_systems_tests`, `cct_stage12_gate`, `Stages/12_Scaling_Accelerator_Systems.md` |
| Stage 13 — Supervised fine-tuning and adapters | **PASS — bounded six-task pilot; training authorization remains false** | `cct_sft_tests`, `cct_stage13_gate`, `Stages/13_Supervised_Fine_Tuning_Adapters.md` |
| Stage 14 — Preference tuning and alignment | **PASS — bounded native alignment gate** | `cct_preference_tests`, `cct_stage14_gate`, `Stages/14_Preference_Tuning_Alignment.md` |
| Stage 15 — Verified retrieval and knowledge | **PASS — bounded native verified-grounding gate** | `cct_knowledge_tests`, `cct_stage15_gate`, `Stages/15_Verified_Retrieval_Knowledge.md` |
| Stage 16 — Production inference and operations | **PASS — bounded native operations gate** | `cct_inference_tests`, `cct_stage16_gate`, `Stages/16_Production_Inference_Operations.md` |
| Stage 17 — Controlled pilot and production release | **PASS — bounded production for named scope; future expansion requires a new specification** | `cct_release_tests`, `cct_stage17_gate`, `Stages/17_Controlled_Pilot_Production_Release.md` |

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
make stage8-test
make stage8-gate
make stage9-test
make stage9-gate
make stage10-test
make stage10-gate
make stage11-test
make stage11-gate
make stage12-test
make stage12-gate
make stage13-test
make stage13-gate
make ci-stage10
make ci-stage11
make ci-stage12
make ci-stage13
make stage14-test
make stage14-gate
make stage15-test
make stage15-gate
make ci-stage14
make ci-stage15
make stage16-test
make stage16-gate
make ci-stage16
make stage17-test
make stage17-gate
make ci-stage17

# One-command native C++20 focused-English competency training
cd /home/ubuntu/CCT
bash run.sh

# Optional legacy Track 1 qualification workflow
cd /home/ubuntu/CCT
CURRICULUM_MODE=0 bash run.sh

# Optional separate Colab GPU companion experiment
cd /home/ubuntu/CCT
bash colab-gpu/run.sh
```

The canonical root `bash run.sh` workflow downloads pinned FineWeb-Edu and OpenAssistant ranges, trains one competency session, persists checkpoint lineage, and waits for human mastery validation before advancing. Its detailed protocol is documented in `RUN_TRAINING.md` and `SPEC/Continual_Learning_Curriculum.md`. `CURRICULUM_MODE=0 bash run.sh` retains the legacy Track 1 real-data qualification path. `make native-build` configures and compiles the C++ library and executables under `build-cpp/`. `make native-test` runs the CTest suite. The gate commands create machine-readable artifacts beneath the corresponding `artifacts/stage-{id}/cpp-gate/` directories. `make ci-stage4` executes the complete native Stage 4 pipeline, including all prior gates, and returns a nonzero status if any mandatory check fails. `make ci-stage5` extends this to the manifest-audited language/code scaling suite and Stage 5 gate. `make ci-stage6` adds the bounded deliberation, independent-verifier, offline policy, replay, interruption, and incident harness. `make ci-stage7` adds the multimodal event, alignment, fusion, typed-memory, simulation, transfer, audit, and safety harness. `make ci-stage8` adds governance and application policy fixtures. `make ci-stage9` adds real-source corpus rights, privacy, deduplication, contamination, shard, and deletion controls. `make ci-stage10` adds the byte/subword/hybrid tokenizer comparison, immutable snapshot, provenance-offset, packed/padded causal-batch, version-compatibility, efficiency, and evaluator-isolation harness. `make ci-stage11` extends the chain with stable categorical next-token loss, analytic CCT gradients, optimizer scheduling, three-seed validation, repeated-corpus overfit, matched dense-attention/GRU/diagonal-SSM controls, checkpoint interruption/resume, evaluator-only exclusion, and fail-closed anomaly checks. `make ci-stage12` adds the 72-point CPU reference/fused matrix, parity, repeatability, ordered worker equivalence, resource thresholds, atomic checkpoint recovery, unavailable-accelerator rejection, and architecture decision record. `make ci-stage13` adds six governed task schemas, deterministic instruction formatting and target-only masks, three-seed full SFT, representative structured extraction, rank-1 adapter efficiency and isolation, merged/runtime parity, grounded citation and abstention checks, safety retention, deletion lineage, and artifact identity. Stage 13 passes all 8 mandatory gate checks; artifacts are written to `artifacts/stage-13/cpp-gate/`. `make ci-stage14` adds governed preference records, DPO-like pairwise optimization, verifier-weighted reranking, calibration, adversarial reward-hacking controls, blind review, and expert escalation. `make ci-stage15` adds real-source knowledge ingestion, lexical/vector/hybrid retrieval, tenant isolation, freshness, citation hashes, conflict abstention, poisoning isolation, deletion replay, version checks, auditability, efficiency, and grounded-review controls. `make ci-stage16` adds the versioned native serving binding, authentication and tenant policy, dynamic batching, deadline admission, streaming cancellation and backpressure, state/cache isolation and eviction, model routing, retrieval and verifier integration, p50/p95/p99 SLO measurement, circuit breaking, fault injection, redacted audit traces, canary shadowing, failed-promotion blocking, and rollback. `make ci-stage17` adds immutable release scope, locked offline parity, side-effect-free shadowing, pilot allowlists and quotas, human review and escalation, safety and SLO observations, incident containment and resume approval, deletion propagation, drift ownership, rollback rehearsal, four-role approval signatures, and the terminal bounded-release decision. These are declared-scope gate measurements, not unrestricted public deployment or evidence of general language-model quality.

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
| `cpp/include/cct/tokenizer.hpp` | Stage 10 tokenizer candidates, vocabulary/snapshot, offsets, provenance, and causal batches API |
| `cpp/src/tokenizer.cpp` | Native byte, subword, hybrid, byte fallback, snapshot, throughput, packed, and padded implementation |
| `cpp/tests/tokenizer_tests.cpp` | Stage 10 native tokenizer and batch regression suite |
| `cpp/tools/stage10_gate.cpp` | Stage 10 artifact-producing candidate, compatibility, provenance, and batch gate |
| `cpp/include/cct/nlp_trainer.hpp` | Stage 11 governed dataset, next-token model, optimizer, evaluation, and checkpoint API |
| `cpp/src/nlp_trainer.cpp` | Stage 11 CCT analytic trainer and matched native controls |
| `cpp/tests/nlp_trainer_tests.cpp` | Stage 11 objective, gradient, optimizer, checkpoint, and fail-closed regressions |
| `cpp/tools/stage11_gate.cpp` | Stage 11 real-source pilot, baseline, resume, contamination, and artifact gate |
| `cpp/include/cct/scaling_systems.hpp` | Stage 12 backend capabilities, scaling points, resource profiles, and atomic recovery API |
| `cpp/src/scaling_systems.cpp` | Stage 12 native CPU reference/fused runner, profiler, checksums, and atomic checkpoints |
| `cpp/tests/scaling_systems_tests.cpp` | Stage 12 backend, parity, scaling, repeatability, resource, and recovery regressions |
| `cpp/tools/stage12_gate.cpp` | Stage 12 72-point scaling, parity, recovery, and architecture-decision gate |
| `cpp/include/cct/sft.hpp` | Stage 13 task schemas, manifests, formatter, SFT model, adapters, registry, decoder, and retention API |
| `cpp/src/sft.cpp` | Stage 13 native supervised fine-tuning, adapter updates, authorization, structured validation, and serialization |
| `cpp/tests/sft_tests.cpp` | Stage 13 manifest, mask, learning, adapter, merge, structured-output, and fail-closed regressions |
| `cpp/tools/stage13_gate.cpp` | Stage 13 six-task SFT, adapter, retention, deletion, and artifact gate |
| `cpp/include/cct/preference.hpp` | Stage 14 governed preference, alignment, verifier, reranker, and blind-review API |
| `cpp/src/preference.cpp` | Stage 14 native DPO-like preference optimization and verifier implementation |
| `cpp/tests/preference_tests.cpp` | Stage 14 preference governance, training, safety, and review regressions |
| `cpp/tools/stage14_gate.cpp` | Stage 14 preference alignment, adversarial, calibration, and release gate |
| `cpp/include/cct/knowledge.hpp` | Stage 15 typed knowledge, retrieval, citation, grounding, and audit API |
| `cpp/src/knowledge.cpp` | Stage 15 native lexical/vector/hybrid knowledge plane and verifier |
| `cpp/tests/knowledge_tests.cpp` | Stage 15 retrieval, access, freshness, deletion, poisoning, and review regressions |
| `cpp/tools/stage15_gate.cpp` | Stage 15 verified retrieval, grounded answer, and release gate |
| `cpp/include/cct/inference.hpp` | Stage 16 versioned inference, state/cache, observability, SLO, and deployment API |
| `cpp/src/inference.cpp` | Stage 16 native serving, batching, streaming, policy, faults, canary, and rollback implementation |
| `cpp/tests/inference_tests.cpp` | Stage 16 API, batching, streaming, state, routing, fault, and SLO regressions |
| `cpp/tools/stage16_gate.cpp` | Stage 16 operational, security, SLO, canary, rollback, and release gate |
| `cpp/include/cct/release.hpp` | Stage 17 immutable scope, pilot, oversight, incident, deletion, drift, and approval API |
| `cpp/src/release.cpp` | Stage 17 native controlled-pilot and terminal release implementation |
| `cpp/tests/release_tests.cpp` | Stage 17 scope, phase, pilot, safety, incident, deletion, drift, and approval regressions |
| `cpp/tools/stage17_gate.cpp` | Stage 17 terminal bounded-release gate and artifacts |
| `run.sh` | Repository-root one-command native C++20 real-data training and qualification workflow |
| `RUN_TRAINING.md` | Native one-command training, evaluation, configuration, and output contract |
| `colab-gpu/run.sh` | Optional GPU data download, preparation, pretraining, SFT, checkpoint, and evaluation companion |
| `colab-gpu/native/prepare.cpp` | Native Wikimedia XML/OASST1 JSONL stream preparer with deterministic splits |
| `colab-gpu/native/cuda_train.cu` | Native CUDA recurrent CCT-family trainer, checkpointing, validation, and test evaluation |
| `colab-gpu/README.md` | Colab setup, data licenses, controls, artifacts, and claim boundary |
| `data/stage-10/tokenizer_snapshot.bin` | Tracked immutable Stage 10 tokenizer snapshot bound to Stages 11–13 |
| `Stages/11_Expanded_Gate_Contract.md` | Stage 11 thresholds, pilot protocol, controls, adversarial checks, and transition contract |
| `Stages/11_Trainable_Native_NLP_Core_Model_Card.md` | Stage 11 intended use, evidence, risks, and claim boundary |
| `Stages/12_Expanded_Gate_Contract.md` | Stage 12 scaling, parity, recovery, resource, and CPU-only transition contract |
| `Stages/13_Expanded_Gate_Contract.md` | Stage 13 SFT, adapter, structured-output, retention, deletion, and transition contract |
| `Stages/13_Supervised_Fine_Tuning_Adapters_Model_Card.md` | Stage 13 intended use, provenance, evidence, risks, and non-claims |
| `cpp/src/multimodal.cpp` | Native multimodal event store, adapters, alignment, fusion, deterministic environment, and audit implementation |
| `cpp/tests/multimodal_tests.cpp` | Stage 7 multimodal and controlled-environment regression suite |
| `cpp/tools/stage7_gate.cpp` | Stage 7 terminal artifact-producing gate |
| `Stages/07_Expanded_Gate_Contract.md` | Stage 7 thresholds, controls, and terminal artifact contract |
| `Stages/08_Production_NLP_Roadmap.md` | Master production sequence, dependencies, and universal gate protocol |
| `Stages/09_Governed_Data_Corpus.md` | Native real-source data rights, privacy, deduplication, contamination, deletion, and corpus release gate |
| `Stages/10_Tokenizer_Representation.md` | Implemented tokenizer, byte fallback, offsets, packed batches, and representation gate |
| `Stages/10_Expanded_Gate_Contract.md` | Stage 10 thresholds, adversarial fixtures, candidate comparison, snapshot, and transition contract |
| `Stages/10_Tokenizer_Model_Card.md` | Stage 10 tokenizer identity, intended use, provenance, evidence, compatibility, and limitations |
| `Stages/11_Trainable_Native_NLP_Core.md` | Next-token trainer, optimizer, checkpoints, and matched-baseline gate |
| `Stages/12_Scaling_Accelerator_Systems.md` | Scaling studies, accelerator path, distributed recovery, and architecture gate |
| `Stages/13_Supervised_Fine_Tuning_Adapters.md` | SFT, full tuning, adapters, structured outputs, and retention gate |
| `Stages/14_Preference_Tuning_Alignment.md` | Preference optimization, safety alignment, and human-review gate |
| `Stages/15_Verified_Retrieval_Knowledge.md` | Retrieval, citations, freshness, conflicts, deletion, and grounding gate |
| `Stages/16_Production_Inference_Operations.md` | Native serving, SLOs, security, observability, canaries, and rollback gate |
| `Stages/17_Controlled_Pilot_Production_Release.md` | Shadow, pilot, oversight, incident response, rollback, and bounded release gate |
| `cpp/tools/stage0_gate.cpp` | Stage 0 artifact-producing gate |
| `cpp/tools/stage1_gate.cpp` | Stage 1 artifact-producing gate |
| `cpp/tools/stage2_gate.cpp` | Stage 2 artifact-producing gate |
| `RESEARCH_STAGE2.md` | Primary state-space design references |
| `Stages/` | Independent stage specifications and transition contracts |
| `SPEC/` | Historical and forward-looking mathematical specifications |
| `artifacts/` | Local generated gate reports; excluded from source control |

## Research limitations

The current implementation validates a numerical operator substrate, a deterministic real/complex selective recurrent core, a native causal-event learner, a local persistent verifiable memory subsystem, a small native language/code scaling benchmark, a bounded deliberation/verification harness, a controlled multimodal event/simulation harness, a Stage 8 production-governance foundation, a Stage 9 governed-corpus foundation, a Stage 10 tokenizer/representation foundation, a Stage 11 bounded trainable next-token pilot, a Stage 12 CPU-only scaling/recovery pilot, and a Stage 13 bounded six-task supervised-adaptation pilot on declared real and application-shaped fixtures. A separate `colab-gpu/` companion now provides a strict native C++20/CUDA user-run experiment over a pinned Wikimedia shard and OASST1 export; it is not yet a formal Stage 14 gate or production evidence. Stages 14–17 remain **specifications only**: they define the remaining production NLP implementation and release path but are not evidence that those capabilities have been built. Stage 8 does not implement a production language model, trainer, tokenizer, serving system, or deployment. Stage 9 does not prove corpus representativeness, universal rights resolution, data factuality, or language-model quality. Stage 10 does not prove language-model quality, multilingual completeness, production-scale throughput, tokenizer optimality, safety behavior, or general intelligence; its selected hybrid snapshot is an immutable input contract. Stage 11 does not prove broad language competence, production training efficiency, Transformer superiority, factuality, safety, instruction following, grounding, distributed scaling, serving readiness, or general intelligence; `training_authorized` remains false. Stage 12 does not prove GPU/cluster scaling, reduced-precision hardware performance, large-model extrapolation, energy efficiency, or universal architecture superiority; `large_training_authorized` remains false. Stage 13 does not prove broad instruction following, factuality, preference alignment, high-impact safety, human-review equivalence, production serving, or autonomous workflows; `training_authorized` remains false. The project does not yet validate broad language competence, general multimodal understanding, open-ended reasoning, unrestricted code generation, real-world perception or robotics, repository-level engineering, distributed scaling, autonomous agency, external deployment, or superintelligence. Further work requires explicit production-stage approval and new evidence at every gate.

## License

MIT License. See the repository license file for the applicable terms.
