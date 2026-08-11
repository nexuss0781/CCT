# Stage 0 — Reproducible Baseline and Evaluation Harness

**Project:** CCT-ASE  
**Stage ID:** 0  
**Predecessor:** None  
**Successor:** Stage 1 — Numerical Engine  
**Status:** Specification; implementation not started

## Purpose

Stage 0 converts the current CCT repository from a conceptual prototype into a reproducible research package. No architectural capability claim may be evaluated until a clean machine can install the project, import the native and Python layers, execute deterministic tests, and produce machine-readable benchmark artifacts.

This stage does not add intelligence. It establishes the measurement system that prevents later improvements from being confused with environment changes, undefined behavior, data leakage, or accidental regressions.

## Scope and non-goals

The stage includes repository layout, pinned toolchains, packaging, native-extension loading, deterministic configuration, CI, baseline tests, benchmark schemas, artifact retention, and a minimal baseline report. It does not implement the spectral solver, selective recurrence, memory system, language training, tool use, or autonomous execution.

The stage must preserve the public behavior that is intentionally retained from the current prototype while making incomplete or aspirational APIs explicit. Placeholder behavior must either be implemented, marked as unsupported with a structured error, or removed from the public surface.

## Required implementation

| Area | Required implementation | Acceptance artifact |
|---|---|---|
| Repository structure | Add `src/`, `tests/`, `benchmarks/`, `configs/`, `scripts/`, `docs/`, and `artifacts/` conventions without duplicating package roots | Directory contract checked in CI |
| Rust toolchain | Add `rust-toolchain.toml`, lock dependencies, configure release and test profiles, and expose one canonical crate name | `cargo metadata --locked` succeeds |
| Python toolchain | Add a pinned supported-version matrix, lockfile or constraints file, and editable-install path | Clean virtual-environment installation log |
| Native extension | Align PyO3 module name, shared-library name, import path, and wheel metadata; remove dynamic import ambiguity | `python -c "import causa_py"` succeeds in a clean environment |
| Configuration | Define a versioned `CCTConfig` or equivalent schema for device, dtype, seed, dimensions, benchmark budget, and logging | Config round-trip test |
| Determinism | Seed Python, NumPy, JAX, Rust-side generators, and benchmark data generation; record versions and hardware | Repeated-run equality report |
| Test runner | Provide one command for unit, integration, property, numerical-smoke, and benchmark-schema checks | CI command exits zero on a clean checkout |
| Continuous integration | Test at least one CPU Python job and one native Rust job; GPU job may be optional initially but must be declared | CI status and stored logs |
| Benchmark schema | Store metric name, value, unit, seed, commit, config hash, hardware, timestamp, and pass/fail status as JSON | JSON schema validation |
| Documentation | Add build, test, benchmark, and troubleshooting instructions | Documentation smoke test |
| Security hygiene | Remove secrets, network-dependent test assumptions, generated binaries, and unreviewed executable downloads from the repository | Secret scan and clean-tree check |

## Implementation contract

The project must expose the following canonical commands or their exact documented equivalents:

```text
make install
make test
make lint
make typecheck
make benchmark-smoke
make report
```

`make install` must install from the repository into an isolated environment. `make test` must not silently skip tests because optional dependencies are missing; it must report capability-specific skips with a machine-readable reason. `make benchmark-smoke` must run only bounded tests suitable for CI and must never require a GPU. `make report` must combine test results and benchmark metadata into a versioned report without changing source files.

The native substrate must expose a single supported Python import path. If the extension is unavailable, importing the package must raise a clear installation error rather than silently loading a different module or a stale shared object.

Every benchmark must receive an explicit seed and configuration. The benchmark runner must refuse to compare results from different model configurations unless the report marks the comparison as non-equivalent.

## Baseline test suite

### Build and import tests

The build harness must test a clean checkout with no pre-existing build directory. It must verify that the Rust library compiles with locked dependencies, the Python package installs, the native extension imports, and the public `Event` and `Manifold` symbols resolve from the documented namespace.

### API smoke tests

The current event lifecycle must be tested for construction, field access, insertion, exact lookup, out-of-bounds behavior, representation, and serialization if serialization is retained. The tests must include negative coordinates, wrong coordinate dimensionality, empty vectors, duplicate insertion, and dimensions containing zero or one.

### Determinism tests

Run the same bounded numerical smoke test twice with identical configuration. The outputs must be byte-identical where deterministic execution is promised, or within a declared tolerance where device kernels are nondeterministic. The report must distinguish deterministic data generation from deterministic floating-point execution.

### Benchmark-runner tests

Use a fixed tiny workload to verify that the runner records all required metadata, rejects missing units, detects malformed metrics, and marks a failed threshold without crashing the entire report. A benchmark that fails a gate must produce a nonzero gate status but still emit its diagnostics.

### Property and mutation smoke tests

At minimum, test that insertion followed by lookup returns the inserted event, out-of-bounds coordinates never mutate state, and a failed operation leaves the object unchanged. A small mutation test should introduce a known API failure and verify that the suite detects it.

## Evaluation harness

The harness consists of four layers:

| Layer | Inputs | Outputs | Required behavior |
|---|---|---|---|
| Test executor | Test target, config, seed | Structured test records | Captures stdout, stderr, exit code, duration, and environment |
| Benchmark runner | Workload, implementation, threshold | Metric records | Warm-up, timed repetitions, uncertainty estimate, threshold evaluation |
| Artifact collector | Logs, profiles, configs, reports | Immutable artifact directory | Names artifacts by commit and config hash |
| Gate evaluator | Test and metric records | `PASS`, `FAIL`, or `BLOCKED` | Applies all mandatory criteria and explains every non-pass |

The canonical report must include commit ID, dirty-tree status, package versions, compiler versions, CPU/GPU information, operating-system information, configuration hash, random seed, test counts, benchmark values, threshold definitions, and gate decision.

## Pass/fail criteria

Stage 0 passes only if all mandatory criteria below are satisfied.

| Criterion | Pass condition | Failure condition |
|---|---|---|
| Clean installation | A fresh environment installs and imports the package on the declared CPU target | Manual path, stale build artifact, or undocumented dependency is required |
| Native build | Locked Rust build succeeds with no compiler errors or unresolved module-name mismatch | Build is unpinned, non-reproducible, or import path is ambiguous |
| Required tests | All mandatory unit and integration tests pass | Any mandatory test fails or is silently skipped |
| Error behavior | Invalid inputs return structured errors and do not mutate state | Panic, process crash, silent coercion, or state mutation occurs |
| Determinism | Repeated smoke runs match the declared tolerance and record deviations | Seed is ignored or metadata is missing |
| Gate evaluator | A deliberately failing threshold is reported as `FAIL` with diagnostics | Failure is swallowed or report remains `PASS` |
| CI | Clean-checkout CPU test and native build jobs complete successfully | CI depends on local state or has no retained logs |
| Repository hygiene | No secrets, generated binaries, or unexplained untracked artifacts are included | Hygiene check reports unresolved findings |

A `BLOCKED` result is allowed only for a declared optional platform job, never for the core build, import, or test path.

## Transition to Stage 1

Stage 1 may begin only when the Stage 0 report contains a signed or otherwise reviewable gate record with `status = PASS`. The transition package must include the clean-install log, test report, benchmark-schema validation report, CI links or exported logs, environment manifest, and a list of known limitations.

If Stage 0 fails, implementation stops at the failing boundary. The team must classify the failure as packaging, native build, API contract, nondeterminism, test weakness, or infrastructure. The failed test is added to regression coverage before the stage is retried.

## Exit report

The exit report must answer five questions in plain language:

1. Can a new contributor reproduce the package from a clean checkout?
2. Can a reviewer identify exactly which code and configuration produced a metric?
3. Do invalid inputs fail safely and predictably?
4. Does the harness detect an intentionally injected regression?
5. Which limitations remain before numerical claims are permitted?

**Transition decision:** `PASS` advances to Stage 1. `FAIL` requires remediation and rerun. `BLOCKED` is reserved for non-core infrastructure and does not authorize architectural development.

## References

[1]: ../CCT_EVOLUTION_PROPOSAL.md "CCT-ASE evolution proposal"

[2]: ../SPEC/Phase-1.md "CCT Phase 1 substrate specification"
