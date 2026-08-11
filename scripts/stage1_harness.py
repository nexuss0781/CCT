#!/usr/bin/env python3
"""Stage 1 numerical-engine evaluation and transition gate.

This harness is intentionally independent from pytest. It executes the
numerical contracts, records quantitative metrics and thresholds, and emits a
machine-readable Stage 1 gate artifact. A PASS authorizes preparation of Stage
2 but does not start Stage 2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from causa_py import (
    FiniteDifferenceSolver,
    SpectralSolver,
    StabilityError,
    UnsupportedPrecisionError,
    finite_difference_laplacian,
    frequency_grids,
    manufactured_mode,
    manufactured_mode_frequency,
    spectral_laplacian,
)

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CheckRecord:
    name: str
    status: str
    duration_seconds: float
    details: dict[str, Any]
    required: bool = True


@dataclass
class MetricRecord:
    name: str
    value: float
    unit: str
    threshold: str
    status: str
    details: dict[str, Any]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_value(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def config_hash(config: dict[str, Any]) -> str:
    raw = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def finite_difference_gradient(
    function: Callable[[jnp.ndarray], jnp.ndarray],
    parameters: jnp.ndarray,
    epsilon: float = 1e-3,
) -> np.ndarray:
    """Compute a centered finite-difference gradient on a small vector."""
    values = []
    for index in range(int(parameters.size)):
        plus = parameters.at[index].add(epsilon)
        minus = parameters.at[index].add(-epsilon)
        values.append(float((function(plus) - function(minus)) / (2.0 * epsilon)))
    return np.asarray(values, dtype=np.float64)


def run_check(name: str, function: Callable[[], dict[str, Any]], required: bool = True) -> CheckRecord:
    started = time.perf_counter()
    try:
        details = function() or {}
        return CheckRecord(name, "PASS", time.perf_counter() - started, details, required)
    except Exception as exc:  # noqa: BLE001 - gate must record all failures
        return CheckRecord(
            name,
            "FAIL",
            time.perf_counter() - started,
            {"error_type": type(exc).__name__, "error": str(exc)},
            required,
        )


def metric(
    name: str,
    value: float,
    unit: str,
    threshold: str,
    passed: bool,
    details: dict[str, Any] | None = None,
) -> MetricRecord:
    return MetricRecord(name, float(value), unit, threshold, "PASS" if passed else "FAIL", details or {})


def transform_check() -> dict[str, Any]:
    key = jax.random.key(101)
    field = jax.random.normal(key, (32, 24), dtype=jnp.float32)
    recovered = jnp.fft.ifftn(jnp.fft.fftn(field)).real
    error = float(jnp.max(jnp.abs(recovered - field)))
    if error >= 2e-6:
        raise AssertionError(f"transform error {error} >= 2e-6")
    return {"max_abs_error": error, "shape": list(field.shape)}


def operator_agreement_check() -> dict[str, Any]:
    shape = (64,)
    spacing = (1.0,)
    field = manufactured_mode(shape, spacing, (1,))
    spectral = spectral_laplacian(field, frequency_grids(shape, spacing))
    reference = finite_difference_laplacian(field, spacing, "periodic")
    error = float(jnp.max(jnp.abs(spectral - reference)))
    relative = float(jnp.linalg.norm(spectral - reference) / jnp.linalg.norm(spectral))
    if relative >= 2e-3:
        raise AssertionError(f"spectral/reference relative error {relative} >= 2e-3")
    return {"max_abs_error": error, "relative_l2_error": relative}


def cross_solver_rollout_check() -> dict[str, Any]:
    n = 64
    dt = 0.05
    phi0 = manufactured_mode((n,), (1.0,), (1,))
    psi0 = jnp.zeros((n,), dtype=jnp.float32)
    source_sequence = jnp.zeros((20, n), dtype=jnp.float32)
    spectral = SpectralSolver(shape=(n,), dt=dt, method="leapfrog")
    reference = FiniteDifferenceSolver(shape=(n,), dt=dt, method="leapfrog", boundary="periodic")
    spectral_trajectory = spectral.rollout(spectral.initialize(phi0, psi0), source_sequence)
    reference_trajectory = reference.rollout(reference.initialize(phi0, psi0), source_sequence)
    max_error = float(jnp.max(jnp.abs(spectral_trajectory.phi - reference_trajectory.phi)))
    if max_error >= 3e-3:
        raise AssertionError(f"spectral/reference rollout error {max_error} >= 3e-3")
    return {"max_abs_rollout_error": max_error, "steps": 20, "grid": n}


def manufactured_accuracy_check() -> dict[str, Any]:
    n = 64
    mode_number = 4
    dt = 0.05
    solver = SpectralSolver(shape=(n,), dt=dt, method="rk4", dtype=jnp.float64)
    phi0 = manufactured_mode((n,), (1.0,), (mode_number,), dtype=jnp.float64)
    omega = manufactured_mode_frequency((n,), (1.0,), (mode_number,), 1.0)
    x = jnp.arange(n, dtype=jnp.float64)
    psi0 = -omega * jnp.cos(2.0 * jnp.pi * mode_number * x / n)
    trajectory = solver.rollout(solver.initialize(phi0, psi0), jnp.zeros((20, n), dtype=jnp.float64))
    expected = jnp.sin(2.0 * jnp.pi * mode_number * x / n - omega * trajectory.time[-1])
    error = float(jnp.max(jnp.abs(trajectory.phi[-1] - expected)))
    if error >= 2e-3:
        raise AssertionError(f"manufactured solution error {error} >= 2e-3")
    return {"max_abs_error": error, "final_time": float(trajectory.time[-1]), "mode": mode_number}


def convergence_check() -> dict[str, Any]:
    errors = []
    for dt in (0.2, 0.1, 0.05):
        n = 64
        mode_number = 4
        solver = SpectralSolver(shape=(n,), dt=dt, method="rk4", dtype=jnp.float64)
        phi0 = manufactured_mode((n,), (1.0,), (mode_number,), dtype=jnp.float64)
        omega = manufactured_mode_frequency((n,), (1.0,), (mode_number,), 1.0)
        x = jnp.arange(n, dtype=jnp.float64)
        psi0 = -omega * jnp.cos(2.0 * jnp.pi * mode_number * x / n)
        steps = int(round(1.0 / dt))
        trajectory = solver.rollout(
            solver.initialize(phi0, psi0), jnp.zeros((steps, n), dtype=jnp.float64)
        )
        expected = jnp.sin(2.0 * jnp.pi * mode_number * x / n - omega * trajectory.time[-1])
        errors.append(float(jnp.sqrt(jnp.mean(jnp.square(trajectory.phi[-1] - expected)))))
    rates = [math.log(errors[i] / errors[i + 1], 2.0) for i in range(2)]
    if min(rates) <= 2.5:
        raise AssertionError(f"RK4 convergence rates {rates} do not exceed 2.5")
    return {"errors": errors, "rates": rates, "declared_order": 4}


def energy_check() -> dict[str, Any]:
    n = 64
    solver = SpectralSolver(shape=(n,), dt=0.05, method="leapfrog")
    phi0 = manufactured_mode((n,), (1.0,), (1,))
    omega = manufactured_mode_frequency((n,), (1.0,), (1,), 1.0)
    x = jnp.arange(n, dtype=jnp.float32)
    psi0 = -omega * jnp.cos(2.0 * jnp.pi * x / n)
    trajectory = solver.rollout(solver.initialize(phi0, psi0), jnp.zeros((400, n)))
    energies = np.asarray(
        [solver.energy(solver.initialize(phi, psi)) for phi, psi in zip(trajectory.phi, trajectory.psi)]
    )
    relative_drift = float((energies.max() - energies.min()) / energies[0])
    if not np.isfinite(energies).all() or relative_drift >= 2e-3:
        raise AssertionError(f"energy drift {relative_drift} is outside 2e-3 bound")
    return {"relative_drift": relative_drift, "initial_energy": float(energies[0]), "steps": 400}


def stability_check() -> dict[str, Any]:
    try:
        SpectralSolver(shape=(16, 16), spacing=1.0, wave_speed=1.0, dt=0.7)
    except StabilityError as exc:
        return {"rejected": True, "message": str(exc)}
    raise AssertionError("CFL-violating configuration was accepted")


def gradient_check() -> dict[str, Any]:
    solver = SpectralSolver(shape=(8,), dt=0.05, method="rk4", dtype=jnp.float64)
    phi0 = manufactured_mode((8,), (1.0,), (1,), dtype=jnp.float64)
    state = solver.initialize(phi0)
    source = jnp.zeros((8,), dtype=jnp.float64)
    raw = jnp.linspace(-0.4, 0.4, 8, dtype=jnp.float64)

    def objective(parameters: jnp.ndarray) -> jnp.ndarray:
        next_state = solver.step(state, source, parameters)
        return jnp.mean(jnp.square(next_state.phi))

    autodiff = np.asarray(jax.grad(objective)(raw), dtype=np.float64)
    numerical = finite_difference_gradient(objective, raw, epsilon=1e-4)
    absolute = float(np.max(np.abs(autodiff - numerical)))
    relative = float(np.linalg.norm(autodiff - numerical) / max(np.linalg.norm(numerical), 1e-12))
    if absolute >= 2e-5 and relative >= 2e-4:
        raise AssertionError(f"gradient mismatch abs={absolute}, rel={relative}")
    return {"max_abs_error": absolute, "relative_l2_error": relative, "parameter_count": int(raw.size)}


def source_gradient_check() -> dict[str, Any]:
    solver = SpectralSolver(shape=(8,), dt=0.05, method="rk4", dtype=jnp.float64)
    phi0 = manufactured_mode((8,), (1.0,), (1,), dtype=jnp.float64)
    state = solver.initialize(phi0)
    potential = jnp.full((8,), 0.1, dtype=jnp.float64)
    source = jnp.linspace(-0.2, 0.2, 8, dtype=jnp.float64)

    def objective(source_value: jnp.ndarray) -> jnp.ndarray:
        next_state = solver.step(state, source_value, potential)
        return jnp.mean(jnp.square(next_state.phi))

    autodiff = np.asarray(jax.grad(objective)(source), dtype=np.float64)
    numerical = finite_difference_gradient(objective, source, epsilon=1e-5)
    absolute = float(np.max(np.abs(autodiff - numerical)))
    relative = float(np.linalg.norm(autodiff - numerical) / max(np.linalg.norm(numerical), 1e-12))
    if absolute >= 2e-5 and relative >= 2e-4:
        raise AssertionError(f"source gradient mismatch abs={absolute}, rel={relative}")
    return {"max_abs_error": absolute, "relative_l2_error": relative, "parameter_count": int(source.size)}


def boundary_check() -> dict[str, Any]:
    dirichlet = FiniteDifferenceSolver(shape=(32,), dt=0.1, boundary="dirichlet")
    phi_dirichlet = jnp.sin(jnp.linspace(0.0, jnp.pi, 32))
    traj_dirichlet = dirichlet.rollout(
        dirichlet.initialize(phi_dirichlet), jnp.zeros((20, 32))
    )
    dirichlet_residual = float(np.max(np.abs(np.asarray(traj_dirichlet.phi[:, [0, -1]]))))

    neumann = FiniteDifferenceSolver(shape=(32,), dt=0.1, boundary="neumann")
    phi_neumann = jnp.cos(jnp.linspace(0.0, jnp.pi, 32))
    traj_neumann = neumann.rollout(neumann.initialize(phi_neumann), jnp.zeros((20, 32)))
    values = np.asarray(traj_neumann.phi)
    neumann_residual = float(
        max(np.max(np.abs(values[:, 0] - values[:, 1])), np.max(np.abs(values[:, -1] - values[:, -2])))
    )
    if dirichlet_residual >= 1e-6 or neumann_residual >= 1e-6:
        raise AssertionError(
            f"boundary residuals dirichlet={dirichlet_residual}, neumann={neumann_residual}"
        )
    return {"dirichlet_residual": dirichlet_residual, "neumann_residual": neumann_residual}


def jit_batch_check() -> dict[str, Any]:
    solver = SpectralSolver(shape=(16,), dt=0.05, method="rk4")
    phi0 = jnp.stack([manufactured_mode((16,), (1.0,), (1,)), manufactured_mode((16,), (1.0,), (2,))])
    state = solver.initialize(phi0)
    source = jnp.zeros((2, 16), dtype=jnp.float32)
    eager = solver.step(state, source)
    compiled = jax.jit(lambda current_state, current_source: solver.step(current_state, current_source))(state, source)
    eager_error = float(jnp.max(jnp.abs(eager.phi - compiled.phi)))
    eager_error = max(eager_error, float(jnp.max(jnp.abs(eager.psi - compiled.psi))))
    trajectory = solver.rollout(state, jnp.zeros((4, 2, 16), dtype=jnp.float32))
    if eager_error >= 2e-5 or trajectory.phi.shape != (5, 2, 16):
        raise AssertionError(f"JIT/batch mismatch={eager_error}, shape={trajectory.phi.shape}")

    other_solver = SpectralSolver(shape=(24,), dt=0.05)
    other_state = other_solver.initialize(jnp.zeros((24,), dtype=jnp.float32))
    other_compiled = jax.jit(lambda current_state: other_solver.step(current_state))(other_state)
    if other_compiled.phi.shape != (24,):
        raise AssertionError("second-shape JIT call returned an unexpected shape")
    return {"jit_max_abs_error": eager_error, "batch_shape": list(trajectory.phi.shape), "second_shape": [24]}


def precision_check() -> dict[str, Any]:
    try:
        SpectralSolver(shape=(16,), dtype=jnp.float16)
    except UnsupportedPrecisionError as exc:
        return {"float16": "explicitly_rejected", "message": str(exc)}
    raise AssertionError("unsupported float16 reference path was silently accepted")


def serialization_check(output: Path) -> dict[str, Any]:
    solver = SpectralSolver(shape=(16,), spacing=0.5, dt=0.1, method="rk4")
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "solver_config.json"
    solver.save_config(config_path)
    loaded = solver.load_config(config_path)
    expected = solver.config_dict()
    actual = loaded._asdict()
    if tuple(actual["shape"]) != tuple(expected["shape"]) or actual["method"] != expected["method"]:
        raise AssertionError(f"config round trip mismatch: {actual} != {expected}")
    return {"config_path": str(config_path), "schema_version": actual["schema_version"]}


def performance_check() -> dict[str, Any]:
    measurements = []
    for n in (32, 64, 128):
        solver = SpectralSolver(shape=(n,), dt=0.05, method="rk4")
        state = solver.initialize(jnp.zeros((n,), dtype=jnp.float32))
        compiled_fn = jax.jit(lambda current_state: solver.step(current_state))
        compile_started = time.perf_counter()
        compiled = compiled_fn(state)
        compiled.phi.block_until_ready()
        compile_seconds = time.perf_counter() - compile_started
        for _ in range(3):
            compiled = compiled_fn(state)
            compiled.phi.block_until_ready()
        run_started = time.perf_counter()
        for _ in range(10):
            compiled = compiled_fn(state)
            compiled.phi.block_until_ready()
        run_seconds = (time.perf_counter() - run_started) / 10.0
        measurements.append(
            {"n": n, "compile_seconds": compile_seconds, "steady_seconds": run_seconds, "cells": n}
        )
    times = np.asarray([item["steady_seconds"] for item in measurements])
    sizes = np.asarray([item["cells"] for item in measurements])
    slope = float(np.polyfit(np.log(sizes), np.log(np.maximum(times, 1e-12)), 1)[0])
    if not np.isfinite(slope) or slope >= 2.0:
        raise AssertionError(f"measured scaling slope {slope} is not subquadratic")
    return {"measurements": measurements, "log_log_slope": slope, "hot_path_claim": "subquadratic_measured"}


def environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "jax_version": jax.__version__,
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "commit": git_value("rev-parse", "HEAD"),
        "dirty_tree": bool(git_value("status", "--porcelain")),
        "timestamp_utc": now_utc(),
    }


def write_artifacts(output: Path, config: dict[str, Any], checks: list[CheckRecord], metrics: list[MetricRecord]) -> str:
    output.mkdir(parents=True, exist_ok=True)
    env = environment()
    status = "PASS" if all(check.status == "PASS" for check in checks if check.required) else "FAIL"
    manifest = {
        "stage": 1,
        "status": status,
        "transition": "Stage 2" if status == "PASS" else "STOP",
        "config": config,
        "config_hash": config_hash(config),
        "environment": env,
        "created_at_utc": now_utc(),
        "required_checks": len([check for check in checks if check.required]),
        "passed_checks": len([check for check in checks if check.status == "PASS"]),
        "failed_checks": len([check for check in checks if check.status == "FAIL"]),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "environment.json").write_text(json.dumps(env, indent=2, sort_keys=True) + "\n")
    (output / "checks.json").write_text(json.dumps([asdict(item) for item in checks], indent=2, sort_keys=True) + "\n")
    (output / "metrics.json").write_text(json.dumps([asdict(item) for item in metrics], indent=2, sort_keys=True) + "\n")
    gate = {
        "stage": 1,
        "status": status,
        "transition": "Stage 2" if status == "PASS" else "STOP",
        "commit": env["commit"],
        "dirty_tree": env["dirty_tree"],
        "config_hash": config_hash(config),
        "mandatory_checks": [asdict(item) for item in checks if item.required],
        "metrics": [asdict(item) for item in metrics],
        "created_at_utc": now_utc(),
        "approval_required": True,
    }
    (output / "gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Stage 1 Gate Report",
        "",
        f"**Status:** `{status}`",
        f"**Transition:** `{gate['transition']}`",
        f"**Commit:** `{env['commit']}`",
        f"**Dirty tree:** `{env['dirty_tree']}`",
        f"**Configuration hash:** `{gate['config_hash']}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Duration (s) |",
        "|---|---:|---:|",
    ]
    lines.extend(f"| {item.name} | `{item.status}` | {item.duration_seconds:.6f} |" for item in checks)
    lines.extend(["", "## Metrics", "", "| Metric | Value | Unit | Threshold | Status |", "|---|---:|---|---|---:|"])
    lines.extend(
        f"| {item.name} | {item.value:.8g} | {item.unit} | {item.threshold} | `{item.status}` |"
        for item in metrics
    )
    lines.extend(
        [
            "",
            "## Transition policy",
            "",
            "A `PASS` means Stage 1 implementation and its declared harness are green. It authorizes preparation of Stage 2 but does not authorize Stage 2 implementation without explicit user approval.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "stage-1" / "gate")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "stage": 1,
        "solver": "spectral-plus-finite-difference-reference",
        "seed": 101,
        "dtype_reference": "float32",
        "dtype_convergence": "float64",
        "threshold_version": "stage1-v1",
    }
    checks = [
        run_check("transform_correctness", transform_check),
        run_check("spectral_reference_operator_agreement", operator_agreement_check),
        run_check("spectral_reference_rollout_agreement", cross_solver_rollout_check),
        run_check("manufactured_solution_accuracy", manufactured_accuracy_check),
        run_check("temporal_convergence", convergence_check),
        run_check("energy_stability", energy_check),
        run_check("cfl_rejection", stability_check),
        run_check("autodiff_finite_difference_gradient", gradient_check),
        run_check("source_gradient", source_gradient_check),
        run_check("boundary_residuals", boundary_check),
        run_check("jit_batch_and_second_shape", jit_batch_check),
        run_check("precision_policy", precision_check),
        run_check("serialization_round_trip", lambda: serialization_check(args.output)),
        run_check("performance_scaling", performance_check),
    ]
    metrics: list[MetricRecord] = []
    for check in checks:
        if check.name == "transform_correctness" and check.status == "PASS":
            metrics.append(metric("transform_max_abs_error", check.details["max_abs_error"], "absolute", "< 2e-6", check.status == "PASS", check.details))
        elif check.name == "spectral_reference_operator_agreement" and check.status == "PASS":
            metrics.append(metric("operator_relative_l2_error", check.details["relative_l2_error"], "relative", "< 2e-3", check.status == "PASS", check.details))
        elif check.name == "spectral_reference_rollout_agreement" and check.status == "PASS":
            metrics.append(metric("spectral_reference_max_rollout_error", check.details["max_abs_rollout_error"], "absolute", "< 3e-3", check.status == "PASS", check.details))
        elif check.name == "manufactured_solution_accuracy" and check.status == "PASS":
            metrics.append(metric("manufactured_max_abs_error", check.details["max_abs_error"], "absolute", "< 2e-3", check.status == "PASS", check.details))
        elif check.name == "temporal_convergence" and check.status == "PASS":
            metrics.append(metric("minimum_rk4_convergence_rate", min(check.details["rates"]), "order", "> 2.5", check.status == "PASS", check.details))
        elif check.name == "energy_stability" and check.status == "PASS":
            metrics.append(metric("relative_energy_drift", check.details["relative_drift"], "relative", "< 2e-3", check.status == "PASS", check.details))
        elif check.name == "autodiff_finite_difference_gradient" and check.status == "PASS":
            metrics.append(metric("gradient_max_abs_error", check.details["max_abs_error"], "absolute", "< 2e-5 or relative < 2e-4", check.status == "PASS", check.details))
        elif check.name == "source_gradient" and check.status == "PASS":
            metrics.append(metric("source_gradient_max_abs_error", check.details["max_abs_error"], "absolute", "< 2e-5 or relative < 2e-4", check.status == "PASS", check.details))
        elif check.name == "performance_scaling" and check.status == "PASS":
            metrics.append(metric("log_log_scaling_slope", check.details["log_log_slope"], "slope", "< 2.0", check.status == "PASS", check.details))
    status = write_artifacts(args.output, config, checks, metrics)
    print(json.dumps({"status": status, "output": str(args.output)}, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
