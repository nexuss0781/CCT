# Stage 1 — Differentiable Numerical Engine

**Project:** CCT-ASE  
**Stage ID:** 1  
**Predecessor:** Stage 0 — Reproducible Baseline  
**Successor:** Stage 2 — Efficient Sequence Core  
**Status:** Implemented; Stage 1 gate PASS

## Purpose

Stage 1 implements a correct, differentiable numerical substrate for learned field and operator dynamics. It converts the current single FFT/filter path into a testable numerical engine with explicit discretization, boundary conditions, stable time stepping, learnable kernels, and verified gradients.

The output of this stage is not a language model. It is a reference-quality operator library that can later serve as an inductive-bias branch inside CCT-ASE. Numerical correctness is a hard prerequisite: a model cannot be credited with reasoning if its core propagation is unstable, aliased, or incorrectly differentiated.

The implementation is located in `causa_py/numerical_engine.py`, with regression coverage in `tests/test_stage1_numerical_engine.py` and the independent gate runner in `scripts/stage1_harness.py`. The final gate is recorded in `artifacts/stage-1/gate/` and must be regenerated whenever the implementation or thresholds change.

## Scope and non-goals

The stage includes a periodic spectral solver, a reference finite-difference solver, stable time integrators, boundary-condition interfaces, learnable potential parameterizations, automatic differentiation, mixed-precision checks, and numerical benchmark reporting. It does not implement the recurrent sequence model, causal event learning, long-term memory, language training, or tool use.

The initial implementation should use a mathematically explicit scalar or vector field with a defined spatial grid and time coordinate. The system must distinguish a physical simulation mode from a learned operator mode; neither may silently substitute for the other.

## Mathematical contract

For the first implementation, define the field evolution as a discretized second-order system:

```text
∂²φ/∂t² = c² Δφ − V_θ(x, t, φ) + J(x, t)
```

The periodic spectral path may compute the Laplacian using Fourier multipliers. The reference path must use an independently implemented finite-difference stencil or a manufactured-solution formulation so that the two implementations are not identical copies of the same bug.

The solver state is:

```text
State = (φ_t, ψ_t, t, solver_config, diagnostics)
ψ_t = ∂φ/∂t
```

The operator API must expose:

```python
state = solver.initialize(phi0, psi0, params)
state_next = solver.step(state, source, params)
trajectory = solver.rollout(state, source_sequence, params)
loss = solver.operator_loss(prediction, target, mask)
```

All functions used in training must be pure with respect to model state. Cached frequency grids and static solver metadata may be stored outside the differentiable state, but any learned parameter that affects output must be an explicit function argument.

## Required implementation

| Component | Required implementation | Testable contract |
|---|---|---|
| Frequency grid | Build correctly indexed spatial frequencies with documented normalization | Forward and inverse transform recover a random field within tolerance |
| Spectral Laplacian | Apply `-||k||²` multiplier with spacing and dtype handled explicitly | Matches reference stencil on low-frequency periodic fields within expected discretization error |
| Propagator | Implement source-to-field propagation with explicit time and spatial axes | Output shapes, causality convention, and dtype are invariant under JIT |
| Time integration | Implement leapfrog first; add RK4 and an implicit or semi-implicit option only after reference tests pass | Order of accuracy matches the declared scheme |
| Boundary conditions | Implement periodic first, then Dirichlet and Neumann through a reference path; treat absorbing layers as optional | Boundary residual is measured, not assumed |
| Learnable potential | Support bounded spectral coefficients and local potential functions | Parameters receive finite, nonzero gradients on a controlled loss |
| Stability checks | Compute CFL-like limits, finite-value checks, norm/energy diagnostics, and step rejection | Invalid configuration returns structured failure |
| Differentiation | Use automatic differentiation first; add custom VJP only when profiling proves necessary | AD and custom-VJP gradients agree |
| Mixed precision | Support FP32 reference; test BF16/FP16 only where stable | Reduced precision reports error and overflow behavior |
| Serialization | Save solver configuration and learned parameters with schema version | Load/save round trip preserves output within tolerance |

## Reference implementation requirements

The reference solver must be slow but clear. It should use explicit array operations and avoid custom kernels. The optimized solver may use JIT compilation, fused FFT operations, or custom kernels, but every optimized result must be compared against the reference on the same input and configuration.

The reference implementation must include at least these manufactured solutions:

```text
φ(x,t) = sin(k·x − ωt),       ω² = c²||k||² + V
φ(x,t) = exp(−αt) sin(k·x),   for a declared damped operator
φ(x,t) = known_forced_solution(x,t), with J derived analytically
```

The manufactured forcing must be computed from the chosen discretized equation rather than from an unrelated continuous formula when the test is intended to verify the discrete operator.

## Evaluation harness

### Accuracy tests

Compare spectral and reference results for constant fields, single Fourier modes, Gaussian packets, and superpositions. Use both absolute and relative error, because relative error is meaningless near zero. Record maximum norm, root-mean-square norm, phase error, amplitude error, and boundary residual.

### Convergence tests

Run a resolution sweep with spatial sizes and time steps chosen so that the stability condition is respected. Estimate empirical order from successive errors. A second-order scheme must exhibit approximately second-order behavior over the asymptotic range; pre-asymptotic points must be reported rather than silently discarded.

### Conservation and stability tests

For an undamped, source-free configuration, compute the declared discrete energy. Measure drift over a fixed horizon and over a long stress horizon. For damped systems, verify monotonic or bounded energy behavior according to the model definition. Run random stable initial states and verify no NaN, infinity, or unbounded norm growth within the declared test horizon.

### Gradient tests

For every learnable parameter group, compare automatic gradients against centered finite differences on small deterministic problems. Use parameter scales that avoid cancellation, report relative and absolute discrepancy, and test both source gradients and potential/kernel gradients. If a custom VJP is introduced, compare it against ordinary AD on the same graph.

### Boundary tests

For each boundary condition, measure the appropriate residual directly. Periodic boundaries must match across opposite faces. Dirichlet boundaries must meet the declared value. Neumann boundaries must meet the declared normal derivative within tolerance. Absorbing layers must be evaluated using reflection energy rather than visual inspection.

### JIT and batching tests

Verify that eager and JIT outputs match, static and dynamic shapes behave as documented, batched fields do not mix batch and spatial axes, and gradients remain finite after compilation. The harness must test a second shape after compilation to detect accidental shape capture.

## Performance harness

The performance harness must measure forward propagation, one step, rollout, backward pass, compilation time, steady-state time, peak memory, and achieved throughput. Compilation time must be separated from execution time. Each benchmark must include warm-up iterations and confidence intervals or a robust spread statistic.

The stage must report scaling over grid sizes and rollout lengths. It must not claim O(n log n) merely because FFT is used; it must show the fitted scaling range, constants, memory behavior, and hardware. Any dense tensor or hidden pairwise operation must be visible in the profile.

## Pass/fail criteria

| Criterion | Pass condition | Failure condition |
|---|---|---|
| Transform correctness | Forward/inverse transform error is below the configured FP32 tolerance on deterministic random fields | Normalization or axis conventions produce material reconstruction error |
| Analytic/manufactured accuracy | Error meets the declared target for the chosen resolution and scheme; the report includes both norms and phase error | Threshold is met only by an undocumented special case or reference and optimized paths disagree |
| Convergence | Empirical order is consistent with the declared spatial and temporal scheme over at least three asymptotic resolutions | Refinement fails to reduce error or order is materially below specification |
| Stability | No NaN/Inf and no unexplained norm blow-up in the stress suite; invalid CFL settings are rejected | Runtime silently proceeds with unstable settings |
| Energy behavior | Source-free undamped runs satisfy the declared drift bound; damped runs satisfy the declared monotonic/bounded condition | Energy drift is unbounded or diagnostic is absent |
| Gradient correctness | Automatic and finite-difference gradients agree within the declared tolerance on all learnable groups | Any trainable group has missing, NaN, or materially incorrect gradients |
| Boundary correctness | Boundary residuals meet per-condition tolerances | Boundary code is a no-op or is not tested directly |
| Precision behavior | FP32 reference is stable; lower precision either passes declared tolerance or is explicitly rejected for that operator | Overflow or silent accuracy collapse occurs |
| Performance integrity | Benchmark report separates compile/run time and shows no hidden quadratic hot path | Complexity claim is unsupported by measurements |

The default starting targets are the Phase 2 targets: roughly 10^-3 numerical error on declared analytic tests, convergence consistent with the selected scheme, energy drift below 0.01% over the declared horizon, and gradient discrepancy near 10^-5 on small checks [[2](#references)]. These are gate targets, not universal laws; any revision must include evidence and reviewer approval.

## Transition to Stage 2

Stage 2 may begin only after the solver has a `PASS` report, a reference-versus-optimized comparison, a gradient report, a stability report, and a benchmark artifact for every supported boundary condition. The public API must be frozen sufficiently that the sequence core can depend on it without importing implementation internals.

If the stage fails, the failing numerical property becomes a regression test. The team must not compensate for an unstable solver by adding model normalization or gradient clipping and then call the numerical stage complete. Such mitigations may be added later, but the underlying solver contract must remain explicit.

## Exit report

The exit report must include the exact equation/discretization, solver configuration, grid and time-step sweeps, error tables, convergence plots or data, energy traces, gradient comparisons, precision results, hardware profile, and unresolved limitations. It must state which claims apply only to periodic regular grids and which apply to irregular or learned operators.

**Transition decision:** `PASS` authorizes Stage 2. `FAIL` requires numerical remediation. `BLOCKED` is allowed only for an optional optimization path; the FP32 reference and test harness must pass before transition.

## References

[1]: ../CCT_EVOLUTION_PROPOSAL.md "CCT-ASE evolution proposal"

[2]: ../SPEC/Phase-2.md "CCT Phase 2 spectral solver specification"

[3]: https://jmlr.org/papers/v24/21-1524.html "Neural Operator: Learning Maps Between Function Spaces With Applications to PDEs"
