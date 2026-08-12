# Stage 12 — Scaling and Accelerator Systems
## Compute Curves, Native CPU Paths, and Recovery

**Predecessor:** Stage 11 — Trainable Native NLP Core  
**Successor:** Stage 13 — Supervised Fine-Tuning and Adapters  
**Status:** Implemented and gated

**Implementation:** Native C++20 CPU reference/fused scaling paths, resource profiler, atomic checkpoints, recovery harness, and gate

## Purpose

Stage 12 determines which execution path is supportable for the declared environment and measures whether the Stage 11 CCT trainer has reproducible compute, memory, throughput, parity, and recovery behavior beyond a single pilot point. The released scope is a complete CPU reference/fused path on a six-vCPU, approximately 4-GB sandbox with no visible CUDA/HIP toolchain or accelerator.

The validated result is a **CPU execution-path decision**, not a GPU, cluster, or universal architectural-superiority claim.

## Scope and non-goals

The implemented stage includes explicit backend capability states, CPU reference and CPU fused control paths, three model widths, two context lengths, three training horizons, two seeds, parameter checksums, wall/CPU time, resident-memory sampling, token/step accounting, deterministic repeated runs, ordered one/two-worker equivalence simulation, atomic checkpoint replacement, worker-loss simulation, storage-interruption preservation, corrupt-checkpoint rejection, architecture decision artifacts, and fail-closed unavailable-accelerator handling.

It does not include CUDA/HIP kernels, BF16/FP16 hardware acceleration, cluster collectives, real multi-node execution, energy metering, distributed storage, large-scale training, production serving, or a claim that CCT is universally superior to Transformers.

## Implemented scope

The native API is in `cpp/include/cct/scaling_systems.hpp` and `cpp/src/scaling_systems.cpp`. `ScalingRunner` probes backend capabilities, runs declared scaling points through the Stage 11 trainer, profiles CPU time and resident memory, checks parameter identity, writes atomic checkpoints through temporary-file rename, and loads checkpoints only with matching tokenizer/dataset hashes. The reference and fused CPU labels use the same deterministic Stage 11 numerical contract in this environment; the reference path is the correctness oracle and the fused path is the selected operational path for the declared CPU scope.

The gate is `cpp/tools/stage12_gate.cpp`. The native regression executable is `cct_scaling_systems_tests`. The tracked Stage 10 snapshot remains the input identity, and the Stage 11 governed data split remains the source of the Stage 12 pilot.

## Scaling study design

The fixed gate matrix contains 72 points: widths `2`, `4`, and `8`; context lengths `8` and `16`; training horizons `4`, `8`, and `16`; seeds `3` and `5`; and CPU reference/fused backends. Each point records parameters, context, horizon, seed, optimizer steps, active training tokens, initial/final training and validation loss, perplexity, wall time, CPU time, tokens/sec, peak resident memory, recurrent state bytes, parameter bytes, and a parameter checksum.

The Stage 12 dataset identity is `fd688b541ecfdbef2a2ae7393b7efac3c452aae287af1bb01585090d5bca1040`, and the tokenizer identity remains `902e5a44f372a3d972b6f21036d62d7878f1d6907805c841e49aa84297ba7b0a`. The matrix contains 36 reference points and 36 fused points. All accepted points are finite, report positive throughput and memory, stay below the declared 3-GiB resident-memory ceiling, and satisfy exact optimizer-step accounting.

## Backend and recovery contract

| Component | Implemented contract |
|---|---|
| Backend abstraction | `cpu_reference`, `cpu_fused`, `cuda_unavailable`, and `hip_unavailable` states |
| Numerical path | Shared Stage 11 CCT model/objective/optimizer contract with deterministic parameter checksums |
| Precision boundary | Native double-precision CPU path; no silent reduced-precision hardware claim |
| Worker simulation | Ordered one/two-worker control-plane equivalence check; no cluster claim |
| Profiler | Wall time, CPU time, tokens/sec, resident memory, state bytes, parameter bytes |
| Checkpoint | Temporary file, close, atomic rename, identity hash, committed-state preservation |
| Failure recovery | Simulated worker loss, storage interruption, truncated checkpoint, and identity mismatch |
| Artifact registry | Commit, tokenizer, dataset, backend, configuration, point, and checkpoint identities |

The atomic checkpoint test writes a temporary checkpoint, commits it by rename, creates a separately truncated temporary interruption, and verifies that the committed checkpoint remains unchanged. The recovery harness reloads the last valid state and rejects the corrupt artifact before model use.

## Mandatory gate checks

All eight Stage 12 gate checks passed.

| Check | Result | Evidence |
|---|---|---|
| Environment and Stage 11 identity | **PASS** | Exact tokenizer/dataset identities; CPU-only capability declaration |
| Backend capability fail-closed | **PASS** | CPU paths complete; CUDA/HIP execution rejected as unavailable |
| Scaling matrix and resource thresholds | **PASS** | All 72 points finite and accounted |
| Reference/fused numerical parity | **PASS** | Loss tolerance `1e-10`; parameter checksums equal |
| Repeatability and data/compute accounting | **PASS** | Same-seed loss/checksum equality within `1e-12` |
| Ordered worker equivalence | **PASS** | One/two-worker deterministic local simulation agrees |
| Atomic checkpoint/recovery | **PASS** | Worker-loss/storage interruption/corrupt checkpoint paths covered |
| Architecture decision integrity | **PASS** | CPU fused selected; CPU reference retained as oracle; absent GPU recorded |

The gate's hard thresholds require at least 72 accepted points, positive finite metrics, at least 100 tokens/sec per accepted point, resident memory below 3 GiB, exact horizon/step accounting, reference/fused loss agreement within `1e-10`, repeated-run agreement within `1e-12`, and explicit rejection of unavailable backends.

## Evaluation harness

The native regression suite covers capability boundary handling, reference/fused parity, resource fields, all six width/context combinations, deterministic repeated points, atomic commit behavior, temporary storage interruption, missing-checkpoint rejection, and unavailable CUDA rejection. The artifact-producing gate runs the full 72-point matrix, parity, repeated-run, ordered worker, recovery, memory, throughput, and architecture-decision checks.

## Deliverables

The stage delivers the native scaling API and implementation, Stage 12 expanded gate contract, scaling regression suite, artifact-producing gate, CMake/CTest integration, Makefile targets, 72-point scaling report, backend parity report, worker-equivalence report, recovery report, resource profile, dataset manifest, incident log, architecture decision, release record, and human-readable gate report under `artifacts/stage-12/cpp-gate/`.

The canonical commands are:

```bash
make stage12-test
make stage12-gate
make ci-stage12
```

`ci-stage12` runs the complete sequential Stage 0–11 chain and then Stage 12. A clean final run must be performed from the release commit after documentation and prior-gate compatibility changes.

## Pass/fail transition

Stage 12 passes because the declared CPU reference and fused paths are reproducible and numerically equivalent, the complete 72-point matrix is finite and resource-accounted, ordered worker simulation agrees, atomic checkpoint recovery preserves committed state, corrupt checkpoints fail closed, throughput and memory thresholds pass, and the architecture decision records absent accelerator capabilities instead of fabricating them.

The selected path for the declared environment is `cpu_fused`, with `cpu_reference` retained as the correctness oracle. A Stage 12 `PASS` authorizes Stage 13 adaptation work within its own specification. It does not authorize large unbounded training expenditure, a CUDA/HIP claim, a cluster claim, or production deployment.

## Explicit limitations

These results are bounded to the declared six-vCPU CPU environment and small model/data matrix. The reference and fused labels share the same numerical trainer contract in this release, so the gate proves parity and operational instrumentation rather than a hardware speedup. The worker test is an ordered local simulation, not multi-node distributed execution. No GPU, CUDA, HIP, BF16/FP16, energy, cluster-communication, or large-model extrapolation result is claimed. A future accelerator or distributed path requires new backend-specific implementation and gate evidence.

## References

[1]: https://arxiv.org/abs/2203.15556 "Training Compute-Optimal Large Language Models"
