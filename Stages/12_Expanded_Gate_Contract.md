# Stage 12 Expanded Gate Contract

## Scaling and Accelerator Systems

**Predecessor:** Stage 11 — Trainable Native NLP Core
**Successor:** Stage 13 — Supervised Fine-Tuning and Adapters
**Gate scope:** Native CPU reference and fused paths on the declared sandbox hardware; no CUDA/HIP device is present.
**Transition rule:** A PASS validates the declared CPU scaling/control-plane path and does not claim GPU, cluster, or universal architectural superiority.

## 1. Declared environment and boundary

The Stage 12 release environment is the six-vCPU Intel Xeon virtual machine exposed by the sandbox, with approximately 4 GiB physical memory, GCC C++20, CMake, FFTW3, and no `nvcc`, `hipcc`, or visible accelerator device. The implementation must therefore provide and validate a complete CPU path rather than silently pretending an accelerator exists. An unavailable CUDA/HIP backend is a fail-closed capability report, not an accepted speedup claim.

The CPU path contains a scalar/reference execution mode and a fused/vector-friendly execution mode implemented in native C++20. Both paths must share the Stage 11 tokenizer/model/data identity and agree numerically. The gate does not download or execute an untrusted compiler, kernel, binary, or external training package.

## 2. Scaling matrix

The fixed pilot varies at least three model sizes, three training horizons, and two context lengths while keeping the data source, tokenizer snapshot, seed set, hardware class, and optimizer semantics declared. The matrix is intentionally small enough to finish on the available CPU and large enough to expose non-monotonic behavior.

| Dimension | Required points |
|---|---|
| Embedding/hidden width | 2/2, 4/4, 8/8 |
| Training horizon | 4, 8, 16 optimizer steps |
| Context length | 8, 16 |
| Seeds | 3 and 5 for repeatability |
| Backend | CPU reference and CPU fused |
| Data | Stage 11 tokenizer snapshot and governed pilot fixture |

Each point records model kind, backend, width, context, optimizer steps, seed, training token count, parameter count, state memory, wall time, tokens/sec, peak resident memory, initial/final loss, validation loss, and failure status. The gate must not report extrapolated large-model or GPU curves.

## 3. Required implementation contract

| Component | CPU implementation contract |
|---|---|
| Backend abstraction | Explicit `cpu_reference`, `cpu_fused`, and unavailable accelerator capability states |
| Kernel path | Embedding, recurrent forward, categorical loss, gradient/update, and evaluation path with common numerical contract |
| Precision | FP64/FP32 reference comparison and a declared reduced-precision probe where the compiler/runtime supports it; no silent claim of BF16/FP16 hardware acceleration |
| Batching | Deterministic microbatch/gradient-accumulation accounting with exact global token and optimizer-step counts |
| Parallelism | Single-worker and deterministic multi-worker simulation using partitioned data and ordered reduction; no false cluster claim |
| Checkpoint | Atomic temporary-file rename, manifest identity, model/optimizer/config hashes, cursor, and recovery status |
| Profiler | Monotonic wall time, CPU time, resident memory, tokens/sec, state bytes, parameters, and backend metadata |
| Failure recovery | Worker-loss simulation, storage interruption simulation, truncated checkpoint, and cursor replay |
| Artifact registry | Commit, tokenizer, dataset, backend, configuration, result, and checkpoint identities |

## 4. Mandatory gate checks and thresholds

| Check | Hard pass condition |
|---|---|
| Backend capability | CPU reference and CPU fused complete; unavailable accelerator is explicitly reported and never counted as a pass |
| Numerical parity | Reference/fused logits, gradients, and final loss agree within `1e-10` on the deterministic fixture |
| Scaling matrix | At least 36 declared points are attempted; all required CPU points complete with finite metrics |
| Curve integrity | Every point reports positive tokens/sec, positive parameter/state counts, exact token/step accounting, and no NaN/Inf |
| Repeatability | Same seed/config/backend reruns have loss and parameter checksums within `1e-12` |
| Context behavior | Both context lengths complete and report state/memory behavior without boundary loss |
| Horizon behavior | 4, 8, and 16-step runs complete with explicit loss trajectories and no unexplained failure |
| Data/compute accounting | Sum of point tokens, steps, cursors, and backend labels is internally consistent |
| Single/multi-worker equivalence | Ordered two-partition reduction matches single-worker fixture within `1e-12` |
| Recovery | Worker-loss and storage-interruption simulations resume from the last atomic checkpoint without duplicate/skip cursor |
| Atomicity | Truncated/incomplete checkpoint is rejected and the last committed checkpoint remains loadable |
| Memory | Resident-memory samples are positive and stay below 75% of declared available memory for accepted points |
| Throughput | Every accepted CPU point exceeds 100 tokens/sec; reference/fused speed ratio is measured, not assumed |
| Baseline fairness | CCT width/context/horizon and data budgets are recorded identically for reference/fused comparison |
| Prior regressions | Stage 0–11 complete CI remains green |
| Decision integrity | Report records CPU-only selection, unavailable accelerator state, negative results, and non-claims |

A point may be marked `BLOCKED` only when its declared optional reduced-precision or unavailable-accelerator capability cannot exist in the environment. Core CPU points must not be blocked. The overall Stage 12 gate is `PASS` only when all core checks pass and optional capability states are explicit.

## 5. Recovery protocol

The harness writes checkpoints using a temporary path, flushes and closes the file, then atomically renames it into the committed path. It records a manifest hash before and after the rename. A storage interruption leaves a partial temporary file but must not replace the last committed checkpoint. A worker-loss simulation interrupts after a declared cursor, loads the last committed checkpoint, and resumes with the same deterministic partition order. Final model parameters, optimizer state, cursor, token count, and checksums must match an uninterrupted run within `1e-12`.

## 6. Architecture decision record

The decision matrix compares CPU reference, CPU fused, and the Stage 11 CCT model/control path on numerical correctness, loss trajectory, tokens/sec, state memory, parameter count, implementation risk, and operational complexity. The result may select the CPU fused path for the declared environment while retaining the reference path as the correctness oracle. No decision may be based on throughput alone, and no absent GPU backend may be described as validated.

## 7. Required artifacts

The gate writes all artifacts under `artifacts/stage-12/cpp-gate/`:

| Artifact | Required contents |
|---|---|
| `checks.json` | Core/optional check status, thresholds, durations, and evidence |
| `scaling_points.json` | All width/context/horizon/seed/backend points |
| `curve_summary.json` | Loss, throughput, memory, and repeatability summaries |
| `backend_parity.json` | Reference/fused outputs, gradients, losses, and tolerances |
| `worker_equivalence.json` | Single/ordered-partition equivalence and accounting |
| `recovery_report.json` | Worker-loss, storage interruption, atomicity, and resume evidence |
| `resource_profile.json` | CPU/memory environment and point-level resources |
| `architecture_decision.json` | Selected CPU path, rejected/limited capabilities, rationale |
| `dataset_manifest.json` | Stage 11 tokenizer/dataset identities and point accounting |
| `incident_log.json` | Numerical, checkpoint, cursor, memory, and unavailable-backend incidents |
| `release_record.json` | Stage status, backend scope, authorization boundary, and approval requirement |
| `report.md` | Human-readable evidence and explicit limitations |

## 8. Transition decision

Stage 12 passes only if the declared CPU reference/fused path is numerically equivalent, the complete scaling matrix is reproducible and finite, single/multi-worker accounting agrees, atomic checkpoint recovery works under simulated faults, memory and throughput thresholds pass, and the architecture decision records that the validated scope is CPU-only. A PASS authorizes Stage 13 adaptation work within its own gate; it does not authorize large unbounded training expenditure, GPU/cluster claims, or production deployment.

## 9. Explicit non-claims

The Stage 12 CPU gate does not prove GPU acceleration, CUDA/HIP correctness, distributed cluster efficiency, BF16/FP16 hardware performance, large-model scaling, compute-optimality, Transformer replacement, production serving readiness, energy efficiency, or universal architecture superiority. Any later accelerator or cluster implementation requires a new backend-specific gate on the actual hardware.
