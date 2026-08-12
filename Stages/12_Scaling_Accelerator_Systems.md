# Stage 12 — Scaling and Accelerator Systems
## Compute Curves, Native Accelerator Paths, and Distributed Recovery

**Predecessor:** Stage 11 — Trainable Native NLP Core  
**Successor:** Stage 13 — Supervised Fine-Tuning and Adapters  
**Status:** Specification; implementation not started  
**Implementation:** Native C++20 control plane with audited CUDA/HIP/vendor-library integrations where approved

## Purpose

Stage 12 determines whether CCT-ASE can scale beyond a research trainer and which architecture should receive production investment. It measures model/data/compute scaling, accelerator efficiency, memory behavior, distributed execution, failure recovery, and serving-relevant state costs against matched Transformer and hybrid controls.

## Scope and non-goals

The stage includes pilot scaling matrices, native accelerator kernels or bindings, mixed precision, gradient accumulation, data/model/pipeline parallel strategies as required, distributed checkpointing, elastic recovery, resource profiling, communication diagnostics, and architecture selection. It does not authorize a full production training run, public deployment, or a claim of universal architectural superiority.

## Scaling study design

The study must vary at least three model sizes, three token budgets or training horizons, and two context lengths within a fixed hardware envelope. Each point records parameters, active state size, training tokens, wall time, accelerator-hours, peak memory, utilization, validation loss, downstream proxy tasks, and failure count.

| Curve | Question |
|---|---|
| Loss vs tokens | Does additional data improve predictably? |
| Loss vs compute | How does CCT compare with Transformer under equal budget? |
| Quality vs parameters | What capacity is required for each workload? |
| Quality vs context | Does recurrent state retain useful long-range information? |
| Throughput vs batch | Does state/memory behavior support serving? |
| Memory vs sequence | Is the declared decode-state benefit realized? |
| Communication vs workers | Does distributed scaling remain efficient? |

Compute-optimal training research motivates treating model size and token count jointly under a budget [1], but the CCT curve must be measured independently because its architecture differs from the paper’s Transformer regime.

## Required implementation

| Component | Implementation | Contract |
|---|---|---|
| Backend abstraction | CPU, approved CUDA/HIP, and vendor-library interfaces | Same numerical contract across backends |
| Kernel path | Embedding, recurrence, projections, normalization, loss, optimizer | Kernel outputs match reference tolerance |
| Precision | FP32 reference plus BF16/FP16/INT8 experiments | Error and safety drift are measured |
| Parallelism | Data parallel first; additional modes only if needed | Global batch and optimizer semantics are explicit |
| Communication | Native collective wrapper and diagnostics | Failures are detected and logged |
| Checkpointing | Per-rank and global manifest with atomic commit | Restart does not duplicate or skip data |
| Scheduler | Resource-aware job configuration | Hardware and allocation are recorded |
| Profiler | Time, memory, kernel, communication, and I/O metrics | Profiles are attached to every scaling point |
| Failure recovery | Worker loss, preemption, storage interruption tests | Recovery reaches last committed state |
| Artifact registry | Model/data/backend/config identity | No artifact is accepted without hashes |

## Architecture decision matrix

The stage must compare CCT-only, matched Transformer, and hybrid candidates on quality, cost, latency, memory, implementation risk, and operational complexity. A candidate is not selected by one metric. The decision record must show why a candidate is retained, rejected, or limited to particular workloads.

## Evaluation harness

The harness must provide:

1. deterministic small scaling sweeps;
2. reference-vs-accelerator numerical parity;
3. mixed-precision loss/gradient drift;
4. single-worker vs multi-worker equivalence;
5. worker failure and checkpoint resume;
6. storage interruption and atomic checkpoint tests;
7. data-shard and optimizer-state accounting;
8. throughput and utilization measurement;
9. peak-memory and sequence-length curves;
10. communication scaling and straggler reports;
11. energy or accelerator-hour accounting where available;
12. matched CCT/Transformer/hybrid comparisons.

## Mandatory gate checks

| Check | Pass condition |
|---|---|
| Numerical parity | Accelerator and reference outputs/gradients remain within declared tolerance |
| Precision safety | Reduced precision does not cause unexplained divergence or unacceptable quality loss |
| Scaling curve | Loss and resource curves are measurable and reproducible across declared points |
| Data/compute accounting | Every token, step, worker, and accelerator-hour is attributable |
| Distributed equivalence | Single- and multi-worker runs agree within tolerance on the fixture |
| Recovery | Worker/preemption/storage faults resume from the last valid checkpoint |
| Memory | Peak memory and sequence-state behavior remain within declared bounds |
| Throughput | Backend speedup and utilization are measured against the CPU/reference path |
| Communication | Communication overhead and scaling limits are reported |
| Baseline fairness | CCT, Transformer, and hybrid use equal data/compute budgets |
| Regression | Prior Stage 0–11 gates remain green |
| Decision integrity | Architecture choice records negative results and limitations |

## Pass/fail transition

Stage 12 passes only when a reproducible scaling curve, a validated accelerator or CPU production path, and a recovery-capable training system exist. The gate must select a candidate architecture per declared workload or explicitly conclude that the CCT-only path is not yet viable.

A `PASS` authorizes Stage 13 adaptation work. It does not authorize a large unbounded training expenditure; every larger run requires an approved budget and updated risk review.

## Deliverables

The stage must deliver backend abstractions, reference and accelerator kernels/bindings, scaling-study configurations, profiler outputs, distributed checkpoint format, failure-recovery tests, architecture decision report, resource budget, regression suite, gate executable, and CI command.

## Explicit limitations

A scaling curve over small models may not extrapolate to large models. Hardware-specific results do not generalize across clusters. Accelerator speedup may be offset by engineering complexity or serving behavior. CCT’s production choice may be hybrid rather than CCT-only.

## References

[1]: https://arxiv.org/abs/2203.15556 "Training Compute-Optimal Large Language Models"
