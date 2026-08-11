# Stage 2 — Efficient Sequence Core

**Project:** CCT-ASE  
**Stage ID:** 2  
**Predecessor:** Stage 1 — Differentiable Numerical Engine  
**Successor:** Stage 3 — Causal Event Learning  
**Status:** Implemented in native C++; Stage 2 gate PASS

## Purpose

Stage 2 implements the first trainable CCT-ASE intelligence core: a content-selective recurrent state-space sequence model with parallel training and recurrent decoding. Its purpose is to establish whether CCT can process long sequences efficiently while retaining information that ordinary fixed-state recurrence often loses.

The stage must be evaluated against matched baselines. It is not enough to demonstrate that the recurrence runs or has linear asymptotic complexity. The core must show correct gradients, stable state evolution, competitive task accuracy, predictable memory use, and no hidden quadratic operation.

## Scope and non-goals

The implemented stage includes explicit dense input/output projections, stable real or complex diagonal state updates, a prefix-scan training path with segmented masks, recurrent single-step decoding, MIMO input/output dimensions, optional RMS normalization, checkpointing, fully trained matched micro-baselines, and deterministic copy, parity, associative-recall, and selective-overwrite tasks. It does not add causal DAG learning, external memory, large-scale language modeling, deliberation, or live tools.

The native reference implementation prioritizes clarity over speed. The prefix-scan path is enabled only after the reference loop, streaming path, gradient checks, checkpoint tests, complex path, normalization path, and segmented-mask tests pass. The implementation is in `cpp/include/cct/sequence.hpp`, `cpp/src/sequence.cpp`, `cpp/src/sequence_scan.cpp`, and `cpp/include/cct/baselines.hpp`/`cpp/src/baselines.cpp`; tests are in `cpp/tests/sequence_tests.cpp`; the expanded gate is `cpp/tools/stage2_gate.cpp`.

## Model contract

For a sequence `x[0:T]`, define a state `h[t]` and output `y[t]`:

```text
h_t = A_t ⊙ h_{t-1} + B_t ⊙ x_t + Bprev_t ⊙ x_{t-1}
y_t = C_t ⊙ h_t + D ⊙ x_t
```

where `A_t` is bounded so that the retained state is stable, and `B_t`, `Bprev_t`, and `C_t` may be content-dependent. The implementation must specify whether each tensor is scalar, diagonal, block-diagonal, low-rank, or dense. No undocumented broadcasting is allowed.

The public native API is:

```cpp
cct::SequenceState state = core.initial_state();
cct::SequenceOutput loop = core.forward(inputs, mask, &state);
cct::SequenceOutput scan = core.forward_scan(inputs, mask, &state);
cct::SequenceState next = core.step(input_t, state, &output_t);
```

The prefix-scan path and one-step decode path share the same real diagonal recurrence. The scan is compared with the reference loop on sequences through length 2048, and one-event streaming is compared with the full forward path through length 257.

## Required implementation

| Component | Required implementation | Contract |
|---|---|---|
| Input encoder | Accept an explicit fixed-width event/token feature vector; metadata channels can be included in `input_dim` without hidden broadcasting | Feature dimensions and missing-feature behavior are explicit |
| Transition parameterization | Use bounded decay or stable matrix parameterization; expose spectral-radius diagnostics | Invalid or unstable transition is rejected or regularized |
| Selective gates | Compute write, retain, and read controls from the current input using a documented projection | Gate values are finite and within declared ranges |
| Reference recurrence | Implement a pure scan or loop with explicit state updates | Reference outputs are reproducible and differentiable |
| Parallel training path | Implement an associative prefix scan for real and complex diagonal recurrence segments; masked positions preserve state and use a segmented scan | Matches reference outputs within tolerance across multiple mask boundaries |
| Decode path | Implement constant-state `step` API | Step-by-step outputs match batched path |
| Complex option | Enable complex state behind an explicit configuration flag with real/imaginary conventions and checkpoint persistence | Complex loop/scan/step paths are equivalent and finite |
| MIMO projection | Support multiple inputs and outputs without unbounded state growth | Parameter count and state size are reported |
| Normalization | Provide state/output RMS normalization disabled by default with exact on/off ablation and checkpoint persistence | Enabled path reaches declared RMS target and ablation is measurable |
| Output heads | Expose generic MIMO output projection for deterministic next-event objectives | Output projection does not alter recurrence state semantics |
| Checkpointing | Save model, optimizer, config, vocabulary/schema, and RNG state | Resume reproduces the next training step within tolerance |

## Reference baselines

The native Stage 2 gate trains four matched micro-baseline families—dense causal attention, GRU, diagonal SSM, and CCT-ASE—on the same deterministic input/output task budget. It reports loss before/after training, parameter count, state memory at length 4096, and forward timing. This is a controlled micro-comparison, not a universal claim over all Transformer implementations.

The purpose of the baselines is not to prove that one family always wins. It is to determine where CCT-ASE provides a quality, memory, latency, or length-extrapolation advantage and where it does not.

## Evaluation harness

Training begins with deterministic synthetic tasks before any natural-language corpus. The native harness fixes the seed, uses explicit SGD with global-norm clipping, records loss and accuracy, tests checkpoint/resume, measures throughput and state memory, emits machine-readable failure diagnostics, and requires the expanded parity, associative-recall, and selective-overwrite suite to reduce loss on held-out task variants.

The initial objective is next-symbol or next-event prediction. Auxiliary losses may include state consistency and stability penalties, but each term must be independently logged:

```text
L = L_next + λ_state L_state + λ_stab L_stability
```

The harness must run both teacher-forced training and free-running evaluation. A model that succeeds only under teacher forcing does not pass the stage.

## Algorithmic evaluation suite

### Copy and delayed recall

Generate sequences containing a payload, a delimiter, distractors, and a query requiring exact recall after a variable delay. Evaluate exact-match accuracy by sequence length and delay, not only aggregate accuracy.

### Associative recall

Present key-value pairs followed by a query key. Vary the number of pairs, collision rate, ordering, and delay. Report exact value accuracy and degradation as the memory load grows.

### Parity and state tracking

Evaluate parity, modular counters, alternating-state machines, and multi-register tracking. These tasks reveal whether the recurrence preserves discrete state rather than merely fitting local correlations.

### Selective overwrite

Present information marked as valid, invalid, superseded, or temporary. The model must retain valid state, overwrite superseded state, and ignore distractors. This tests content-selective retention.

### Length extrapolation

Train on lengths up to `L_train` and evaluate at `2L_train`, `4L_train`, and the maximum supported length. Report accuracy and state norm. The test must use held-out random seeds and held-out symbol combinations.

### Streaming equivalence

Run the same sequence in one batched call, fixed-size chunks, and one event at a time. Outputs at aligned positions must match within tolerance. This is mandatory for a usable recurrent engine.

## Numerical and gradient harness

For randomly initialized small models, compare the reference step loop, parallel scan, and optimized kernel. Use finite differences or an independent autodiff implementation for selected parameters. Test gradients through sequence length, metadata features, and state initialization.

Run long-horizon stability tests with random inputs and adversarial gate patterns. Record state norms, transition radii, gradient norms, and output saturation. The harness must fail on NaN, infinity, unexplained state explosion, or silent truncation.

## Complexity and efficiency harness

Measure training throughput, decode latency per token/event, peak memory, parameter count, activation memory, and checkpoint size across sequence lengths. Use lengths that expose hidden quadratic behavior. Fit scaling slopes only over a declared range and show raw data.

The primary efficiency targets are:

| Metric | Required comparison |
|---|---|
| Decode memory | CCT-ASE must not require a sequence-length KV cache |
| Decode latency | Report per-step latency at fixed state size and compare against baselines |
| Training scaling | No hidden pairwise sequence operation in the recurrent hot path |
| Quality efficiency | Compare task quality at equal parameter count and equal training compute |
| Length extrapolation | Report accuracy retention beyond the training length |

These are measured targets, not assumptions. A recurrence that is asymptotically linear but slower in wall-clock time must be reported honestly.

## Pass/fail criteria

| Criterion | Pass condition | Failure condition |
|---|---|---|
| Reference correctness | Reference recurrence passes shape, mask, state, and deterministic tests | State semantics are ambiguous or outputs depend on hidden mutable state |
| Path equivalence | Batched, chunked, and step-wise paths agree within configured tolerance | Streaming changes outputs materially without documented reason |
| Gradient correctness | Reference and optimized gradients agree on selected parameters | Missing, NaN, or materially inconsistent gradients |
| Stability | Long-horizon tests remain finite with bounded diagnostics under declared stable settings | State or gradients explode without a declared failure signal |
| Algorithmic capability | CCT-ASE passes the predefined minimum on copy, recall, state tracking, and selective overwrite | It fails a mandatory task or cannot extrapolate at all beyond training length |
| Baseline comparison | Report includes trained dense attention, GRU, diagonal SSM, and CCT with shared task budget, loss, parameters, memory, and timing | Results are compared only to an untrained or mismatched baseline |
| Efficiency | No O(T²) operation in the declared core; memory and latency are measured across length | Hidden quadratic allocation or unsupported complexity claim |
| Checkpoint recovery | Resume reproduces loss and parameter trajectory within tolerance on a deterministic micro-run | Resume silently changes optimizer, RNG, or model state |
| Ablation integrity | Complex state, MIMO, normalization, selective gates, and metadata-channel contributions are individually measured | Claimed contribution cannot be isolated |

The stage passes only when all mandatory criteria are satisfied. A quality improvement without path equivalence or gradient correctness is not a pass.

## Transition to Stage 3

Stage 3 may begin when the sequence core is a stable reusable module with documented interfaces, matched baseline results, and an evaluation report showing where it succeeds and fails. The model must support metadata channels needed for causal events, but those channels must not yet be treated as proven causal understanding.

The transition package includes the native model configuration schema, reference and prefix-scan implementations, segmented-mask logic, complex/normalization toggles, deterministic copy/parity/associative/overwrite generators, fully trained baseline metrics, scaling data, checkpoint-resume test, expanded ablation report, source research notes, and final limitation-closure evidence.

If the stage fails, the team must first determine whether the failure comes from recurrence mathematics, optimizer/training setup, data encoding, or evaluation leakage. New complexity must not be added merely to hide a failure on a basic state-tracking task.

## Exit report

The exit report contains per-task training and extrapolation results, trained matched-baseline losses, state memory and timing profiles, real/complex loop/scan equivalence results, gradient discrepancies, checkpoint recovery evidence, normalization and selective-gate ablations, segmented-mask evidence, and explicit micro-comparison scope.

**Transition decision:** `PASS` authorizes Stage 3 preparation. `FAIL` requires correction. The strengthened gate has **12 mandatory checks**; no Stage 2 limitation remains deferred. Stage 3 implementation still requires explicit user approval.

## References

[1]: ../CCT_EVOLUTION_PROPOSAL.md "CCT-ASE evolution proposal"

[2]: https://proceedings.iclr.cc/paper_files/paper/2026/hash/8abd2043b71a074278d5f687947bff9c-Abstract-Conference.html "Mamba-3: Improved Sequence Modeling using State Space Principles"
