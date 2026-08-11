from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


def _mode_problem(n: int = 64, dt: float = 0.05, method: str = "rk4", mode_number: int = 1, dtype=jnp.float32):
    shape = (n,)
    spacing = (1.0,)
    mode = (mode_number,)
    wave_speed = 1.0
    phi0 = manufactured_mode(shape, spacing, mode, dtype=dtype)
    omega = manufactured_mode_frequency(shape, spacing, mode, wave_speed)
    x = jnp.arange(n, dtype=dtype)
    psi0 = -omega * jnp.cos(2.0 * jnp.pi * mode_number * x / n)
    solver = SpectralSolver(
        shape=shape,
        spacing=spacing,
        wave_speed=wave_speed,
        dt=dt,
        method=method,
        dtype=dtype,
    )
    return solver, phi0, psi0, omega


def test_frequency_transform_round_trip():
    key = jax.random.key(0)
    field = jax.random.normal(key, (16, 12), dtype=jnp.float32)
    grids = frequency_grids((16, 12), (0.5, 0.75))
    transformed = jnp.fft.fftn(field)
    recovered = jnp.fft.ifftn(transformed).real
    np.testing.assert_allclose(np.asarray(recovered), np.asarray(field), rtol=2e-6, atol=2e-6)
    assert all(grid.shape == field.shape for grid in grids)


def test_spectral_and_reference_laplacians_agree_on_low_mode():
    shape = (64,)
    spacing = (1.0,)
    field = manufactured_mode(shape, spacing, (1,))
    spectral = spectral_laplacian(field, frequency_grids(shape, spacing))
    reference = finite_difference_laplacian(field, spacing, "periodic")
    np.testing.assert_allclose(np.asarray(spectral), np.asarray(reference), rtol=2e-3, atol=2e-5)


def test_manufactured_mode_rollout_accuracy():
    solver, phi0, psi0, omega = _mode_problem(n=64, dt=0.05, method="rk4")
    trajectory = solver.rollout(solver.initialize(phi0, psi0), source_sequence=jnp.zeros((20, 64)))
    final_time = float(trajectory.time[-1])
    x = jnp.arange(64, dtype=jnp.float32)
    expected = jnp.sin(2.0 * jnp.pi * x / 64.0 - omega * final_time)
    error = float(jnp.max(jnp.abs(trajectory.phi[-1] - expected)))
    assert error < 2e-3


def test_rk4_temporal_convergence():
    errors = []
    for dt in (0.2, 0.1, 0.05):
        solver, phi0, psi0, omega = _mode_problem(n=64, dt=dt, method="rk4", mode_number=4, dtype=jnp.float64)
        steps = int(round(1.0 / dt))
        trajectory = solver.rollout(solver.initialize(phi0, psi0), jnp.zeros((steps, 64), dtype=jnp.float64))
        final_time = float(trajectory.time[-1])
        x = jnp.arange(64, dtype=jnp.float64)
        expected = jnp.sin(2.0 * jnp.pi * 4 * x / 64.0 - omega * final_time)
        errors.append(float(jnp.sqrt(jnp.mean(jnp.square(trajectory.phi[-1] - expected)))))
    rate_a = np.log(errors[0] / errors[1]) / np.log(2.0)
    rate_b = np.log(errors[1] / errors[2]) / np.log(2.0)
    assert rate_a > 2.5
    assert rate_b > 2.5


def test_leapfrog_energy_drift_is_bounded():
    solver, phi0, psi0, _ = _mode_problem(n=64, dt=0.05, method="leapfrog")
    trajectory = solver.rollout(solver.initialize(phi0, psi0), jnp.zeros((200, 64)))
    energies = np.asarray(
        [solver.energy(solver.initialize(phi, psi)) for phi, psi in zip(trajectory.phi, trajectory.psi)]
    )
    relative_drift = float((energies.max() - energies.min()) / energies[0])
    assert np.isfinite(energies).all()
    assert relative_drift < 2e-3


def test_stability_rejects_cfl_violation():
    with pytest.raises(StabilityError, match="CFL"):
        SpectralSolver(shape=(16, 16), spacing=1.0, wave_speed=1.0, dt=0.7)


def test_gradient_is_finite_and_nonzero_for_potential():
    solver, phi0, psi0, _ = _mode_problem(n=16, dt=0.05, method="rk4")
    state = solver.initialize(phi0, psi0)
    source = jnp.zeros((16,))

    def objective(raw_potential):
        state_next = solver.step(state, source, raw_potential)
        return jnp.mean(jnp.square(state_next.phi))

    raw = jnp.full((16,), 0.2, dtype=jnp.float32)
    gradient = jax.grad(objective)(raw)
    assert np.isfinite(np.asarray(gradient)).all()
    assert float(jnp.linalg.norm(gradient)) > 0.0


def test_eager_and_jit_single_step_match():
    solver, phi0, psi0, _ = _mode_problem(n=32, dt=0.05, method="rk4")
    state = solver.initialize(phi0, psi0)
    source = jnp.zeros((32,))
    eager = solver.step(state, source)
    compiled = jax.jit(lambda s, x: solver.step(s, x))(state, source)
    np.testing.assert_allclose(np.asarray(eager.phi), np.asarray(compiled.phi), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.asarray(eager.psi), np.asarray(compiled.psi), rtol=2e-5, atol=2e-6)


def test_dirichlet_boundary_residual():
    solver = FiniteDifferenceSolver(shape=(32,), spacing=1.0, dt=0.1, boundary="dirichlet")
    phi0 = jnp.sin(jnp.linspace(0.0, jnp.pi, 32))
    state = solver.initialize(phi0, jnp.zeros_like(phi0))
    trajectory = solver.rollout(state, jnp.zeros((5, 32)))
    assert np.max(np.abs(np.asarray(trajectory.phi[:, [0, -1]]))) < 1e-6


def test_neumann_boundary_residual():
    solver = FiniteDifferenceSolver(shape=(32,), spacing=1.0, dt=0.1, boundary="neumann")
    phi0 = jnp.cos(jnp.linspace(0.0, jnp.pi, 32))
    state = solver.initialize(phi0, jnp.zeros_like(phi0))
    trajectory = solver.rollout(state, jnp.zeros((5, 32)))
    phi = np.asarray(trajectory.phi)
    assert np.max(np.abs(phi[:, 0] - phi[:, 1])) < 1e-6
    assert np.max(np.abs(phi[:, -1] - phi[:, -2])) < 1e-6


def test_reduced_precision_is_explicitly_rejected():
    with pytest.raises(UnsupportedPrecisionError, match="requires float32 or float64"):
        SpectralSolver(shape=(16,), dtype=jnp.float16)


def test_config_round_trip(tmp_path: Path):
    solver = SpectralSolver(shape=(16,), spacing=0.5, dt=0.1, method="rk4")
    path = tmp_path / "solver_config.json"
    solver.save_config(path)
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 1
    loaded = solver.load_config(path)
    assert loaded.shape == (16,)
    assert loaded.spacing == (0.5,)
    assert loaded.method == "rk4"


def test_operator_loss_mask_is_correct():
    solver = SpectralSolver(shape=(4,))
    prediction = jnp.array([1.0, 2.0, 4.0, 8.0])
    target = jnp.array([1.0, 0.0, 0.0, 0.0])
    mask = jnp.array([1.0, 0.0, 0.0, 0.0])
    assert float(solver.operator_loss(prediction, target, mask)) == pytest.approx(0.0)
