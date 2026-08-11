"""Stage 1 differentiable numerical engine for CCT-ASE.

The module deliberately contains two implementations:

* :class:`SpectralSolver` is the optimized periodic-grid implementation.
* :class:`FiniteDifferenceSolver` is an independent reference implementation
  supporting periodic, Dirichlet, and Neumann boundaries.

Both implement the same second-order field equation

    phi_tt = c**2 * Laplacian(phi) - V * phi + source

with a velocity-Verlet/leapfrog integrator and a classical RK4 option. The
reference implementation is intentionally explicit so it can detect mistakes
in the optimized spectral path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import jax
import jax.numpy as jnp

Array = jax.Array


class NumericalEngineError(ValueError):
    """Base error for invalid numerical-engine configuration or state."""


class StabilityError(NumericalEngineError):
    """Raised when a configured time step violates the declared CFL limit."""


class UnsupportedPrecisionError(NumericalEngineError):
    """Raised when a dtype is not supported by the reference operator."""


class FieldState(NamedTuple):
    """Pure differentiable field state used by scan and step functions."""

    phi: Array
    psi: Array
    time: Array
    step_index: Array


class Trajectory(NamedTuple):
    """Rollout result containing initial state followed by evolved states."""

    phi: Array
    psi: Array
    time: Array
    step_index: Array


class SolverConfig(NamedTuple):
    """Serializable solver configuration with an explicit schema version."""

    schema_version: int
    shape: tuple[int, ...]
    spacing: tuple[float, ...]
    wave_speed: float
    dt: float
    boundary: str
    method: str
    dtype: str
    cfl_safety: float


_SUPPORTED_BOUNDARIES = {"periodic", "dirichlet", "neumann"}
_SUPPORTED_METHODS = {"leapfrog", "rk4"}


def _as_shape(shape: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(value) for value in shape)
    if not result or any(value <= 1 for value in result):
        raise NumericalEngineError("shape must contain dimensions greater than one")
    return result


def _as_spacing(spacing: float | Sequence[float], ndim: int) -> tuple[float, ...]:
    if isinstance(spacing, (int, float)):
        values = (float(spacing),) * ndim
    else:
        values = tuple(float(value) for value in spacing)
    if len(values) != ndim or any(value <= 0.0 for value in values):
        raise NumericalEngineError("spacing must contain one positive value per axis")
    return values


def _real_dtype(dtype: Any) -> jnp.dtype:
    resolved = jnp.dtype(dtype)
    if resolved not in (jnp.float32, jnp.float64):
        raise UnsupportedPrecisionError(
            f"Stage 1 reference solver requires float32 or float64; got {resolved}"
        )
    if resolved == jnp.float64 and not jax.config.read("jax_enable_x64"):
        return jnp.dtype(jnp.float32)
    return resolved


def _spatial_axes(ndim: int) -> tuple[int, ...]:
    return tuple(range(-ndim, 0))


def cfl_limit(shape: Sequence[int], spacing: float | Sequence[float], wave_speed: float) -> float:
    """Return a conservative explicit-wave CFL limit."""
    _ = _as_shape(shape)
    spacing_values = _as_spacing(spacing, len(tuple(shape)))
    speed = float(wave_speed)
    if speed <= 0.0:
        raise NumericalEngineError("wave_speed must be positive")
    return min(spacing_values) / (speed * (len(spacing_values) ** 0.5))


def validate_stability(
    dt: float,
    shape: Sequence[int],
    spacing: float | Sequence[float],
    wave_speed: float,
    safety: float = 0.9,
) -> float:
    """Validate and return the configured step under a CFL safety margin."""
    if not 0.0 < float(safety) <= 1.0:
        raise NumericalEngineError("cfl safety must be in (0, 1]")
    limit = cfl_limit(shape, spacing, wave_speed)
    if float(dt) <= 0.0:
        raise StabilityError("dt must be positive")
    if float(dt) > limit * float(safety) + 1e-12:
        raise StabilityError(
            f"dt={dt} exceeds CFL safety limit={limit * float(safety):.8g}"
        )
    return limit


def apply_boundary_conditions(
    field: Array,
    boundary: str,
    value: float = 0.0,
    spatial_ndim: int | None = None,
) -> Array:
    """Apply a boundary condition to the final spatial axes of ``field``."""
    if boundary not in _SUPPORTED_BOUNDARIES:
        raise NumericalEngineError(f"unsupported boundary: {boundary}")
    ndim = spatial_ndim if spatial_ndim is not None else field.ndim
    axes = _spatial_axes(ndim)
    result = field
    if boundary == "periodic":
        return result
    for axis in axes:
        first = [slice(None)] * result.ndim
        last = [slice(None)] * result.ndim
        first[axis] = 0
        last[axis] = -1
        if boundary == "dirichlet":
            result = result.at[tuple(first)].set(value)
            result = result.at[tuple(last)].set(value)
        elif boundary == "neumann":
            near_first = [slice(None)] * result.ndim
            near_last = [slice(None)] * result.ndim
            near_first[axis] = 1
            near_last[axis] = -2
            result = result.at[tuple(first)].set(result[tuple(near_first)])
            result = result.at[tuple(last)].set(result[tuple(near_last)])
    return result


def _pad_for_boundary(field: Array, boundary: str, spatial_ndim: int) -> Array:
    if boundary == "periodic":
        return field
    mode = "constant" if boundary == "dirichlet" else "edge"
    pad_width = [(0, 0)] * (field.ndim - spatial_ndim) + [(1, 1)] * spatial_ndim
    return jnp.pad(field, pad_width, mode=mode)


def finite_difference_laplacian(
    field: Array,
    spacing: float | Sequence[float],
    boundary: str = "periodic",
    spatial_ndim: int | None = None,
) -> Array:
    """Compute an independent second-order finite-difference Laplacian."""
    ndim = spatial_ndim if spatial_ndim is not None else field.ndim
    spacing_values = _as_spacing(spacing, ndim)
    if boundary not in _SUPPORTED_BOUNDARIES:
        raise NumericalEngineError(f"unsupported boundary: {boundary}")
    if boundary == "periodic":
        result = jnp.zeros_like(field)
        for axis, dx in zip(_spatial_axes(ndim), spacing_values):
            result = result + (
                jnp.roll(field, -1, axis=axis)
                - 2.0 * field
                + jnp.roll(field, 1, axis=axis)
            ) / (dx**2)
        return result

    padded = _pad_for_boundary(field, boundary, ndim)
    result = jnp.zeros_like(field)
    for offset, axis in enumerate(_spatial_axes(ndim)):
        axis_in_padded = padded.ndim - ndim + offset
        center = [slice(None)] * padded.ndim
        plus = [slice(None)] * padded.ndim
        minus = [slice(None)] * padded.ndim
        center[axis_in_padded] = slice(1, -1)
        plus[axis_in_padded] = slice(2, None)
        minus[axis_in_padded] = slice(None, -2)
        result = result + (
            padded[tuple(plus)]
            - 2.0 * padded[tuple(center)]
            + padded[tuple(minus)]
        ) / (spacing_values[offset] ** 2)
    return apply_boundary_conditions(result, boundary, spatial_ndim=ndim)


def frequency_grids(
    shape: Sequence[int], spacing: float | Sequence[float]
) -> tuple[Array, ...]:
    """Build correctly indexed angular-frequency grids."""
    shape_values = _as_shape(shape)
    spacing_values = _as_spacing(spacing, len(shape_values))
    axes = [2.0 * jnp.pi * jnp.fft.fftfreq(n, d=dx) for n, dx in zip(shape_values, spacing_values)]
    return tuple(jnp.meshgrid(*axes, indexing="ij"))


def spectral_laplacian(
    field: Array,
    grids: tuple[Array, ...],
    spatial_ndim: int | None = None,
) -> Array:
    """Apply the periodic Laplacian through a Fourier multiplier."""
    ndim = spatial_ndim if spatial_ndim is not None else len(grids)
    axes = _spatial_axes(ndim)
    k_squared = jnp.asarray(sum(jnp.square(grid) for grid in grids), dtype=field.dtype)
    transformed = jnp.fft.fftn(field, axes=axes)
    return jnp.fft.ifftn(transformed * (-k_squared), axes=axes).real


def spectral_gradient_energy(
    field: Array,
    grids: tuple[Array, ...],
    spacing: float | Sequence[float],
    spatial_ndim: int | None = None,
) -> Array:
    """Return the integral of squared spectral gradients."""
    ndim = spatial_ndim if spatial_ndim is not None else len(grids)
    axes = _spatial_axes(ndim)
    gradients = []
    transformed = jnp.fft.fftn(field, axes=axes)
    for grid in grids:
        gradients.append(jnp.fft.ifftn(transformed * (1j * grid), axes=axes).real)
    density = sum(jnp.square(gradient) for gradient in gradients)
    volume_element = jnp.prod(jnp.asarray(_as_spacing(spacing, ndim)))
    return jnp.sum(density) * volume_element


def bounded_local_potential(raw: Array, max_value: float = 1.0) -> Array:
    """Map unconstrained local parameters to a positive bounded potential."""
    if max_value <= 0.0:
        raise NumericalEngineError("max_value must be positive")
    return float(max_value) * jax.nn.sigmoid(raw)


def bounded_spectral_potential(raw: Array, max_value: float = 1.0) -> Array:
    """Map real spectral coefficients to a positive bounded spatial potential."""
    if max_value <= 0.0:
        raise NumericalEngineError("max_value must be positive")
    raw_field = jnp.fft.ifftn(raw).real
    return bounded_local_potential(raw_field, max_value=max_value)


def _coerce_source(source: Array | None, shape: tuple[int, ...], dtype: jnp.dtype) -> Array:
    if source is None:
        return jnp.zeros(shape, dtype=dtype)
    value = jnp.asarray(source, dtype=dtype)
    if value.shape != shape and value.shape[-len(shape) :] != shape:
        raise NumericalEngineError(f"source must have trailing shape {shape}, got {value.shape}")
    return value


def _coerce_potential(potential: Array | float | None, shape: tuple[int, ...], dtype: jnp.dtype) -> Array:
    if potential is None:
        return jnp.zeros(shape, dtype=dtype)
    value = jnp.asarray(potential, dtype=dtype)
    if value.ndim == 0:
        return jnp.broadcast_to(value, shape)
    if value.shape != shape and value.shape[-len(shape) :] != shape:
        raise NumericalEngineError(f"potential must have trailing shape {shape}, got {value.shape}")
    return value


def _spectral_acceleration(
    phi: Array,
    source: Array,
    potential: Array,
    grids: tuple[Array, ...],
    wave_speed: float,
) -> Array:
    speed = jnp.asarray(wave_speed, dtype=phi.dtype)
    return speed**2 * spectral_laplacian(phi, grids) - potential * phi + source


def _finite_difference_acceleration(
    phi: Array,
    source: Array,
    potential: Array,
    spacing: tuple[float, ...],
    boundary: str,
    wave_speed: float,
) -> Array:
    speed = jnp.asarray(wave_speed, dtype=phi.dtype)
    return speed**2 * finite_difference_laplacian(phi, spacing, boundary) - potential * phi + source


def _leapfrog_step(
    phi: Array,
    psi: Array,
    source: Array,
    potential: Array,
    dt: float,
    acceleration_fn,
    boundary: str,
    boundary_value: float,
    spatial_ndim: int,
) -> tuple[Array, Array]:
    dt_value = jnp.asarray(dt, dtype=phi.dtype)
    acceleration = acceleration_fn(phi, source, potential)
    psi_half = psi + jnp.asarray(0.5, dtype=phi.dtype) * dt_value * acceleration
    phi_new = phi + dt_value * psi_half
    phi_new = apply_boundary_conditions(
        phi_new, boundary, value=boundary_value, spatial_ndim=spatial_ndim
    )
    acceleration_new = acceleration_fn(phi_new, source, potential)
    psi_new = psi_half + jnp.asarray(0.5, dtype=phi.dtype) * dt_value * acceleration_new
    psi_new = apply_boundary_conditions(
        psi_new, boundary, value=0.0, spatial_ndim=spatial_ndim
    )
    return phi_new, psi_new


def _rk4_step(
    phi: Array,
    psi: Array,
    source: Array,
    potential: Array,
    dt: float,
    acceleration_fn,
    boundary: str,
    boundary_value: float,
    spatial_ndim: int,
) -> tuple[Array, Array]:
    dt_value = jnp.asarray(dt, dtype=phi.dtype)
    half = jnp.asarray(0.5, dtype=phi.dtype)
    one_sixth_dt = dt_value / jnp.asarray(6.0, dtype=phi.dtype)

    def rhs(current_phi: Array, current_psi: Array) -> tuple[Array, Array]:
        return current_psi, acceleration_fn(current_phi, source, potential)

    k1_phi, k1_psi = rhs(phi, psi)
    k2_phi, k2_psi = rhs(phi + half * dt_value * k1_phi, psi + half * dt_value * k1_psi)
    k3_phi, k3_psi = rhs(phi + half * dt_value * k2_phi, psi + half * dt_value * k2_psi)
    k4_phi, k4_psi = rhs(phi + dt_value * k3_phi, psi + dt_value * k3_psi)
    phi_new = phi + one_sixth_dt * (k1_phi + 2.0 * k2_phi + 2.0 * k3_phi + k4_phi)
    psi_new = psi + one_sixth_dt * (k1_psi + 2.0 * k2_psi + 2.0 * k3_psi + k4_psi)
    phi_new = apply_boundary_conditions(
        phi_new, boundary, value=boundary_value, spatial_ndim=spatial_ndim
    )
    psi_new = apply_boundary_conditions(
        psi_new, boundary, value=0.0, spatial_ndim=spatial_ndim
    )
    return phi_new, psi_new


class _BaseSolver:
    """Shared public API for spectral and reference solvers."""

    operator_name = "base"

    def __init__(
        self,
        shape: Sequence[int],
        spacing: float | Sequence[float] = 1.0,
        wave_speed: float = 1.0,
        dt: float | None = None,
        boundary: str = "periodic",
        method: str = "leapfrog",
        dtype: Any = jnp.float32,
        cfl_safety: float = 0.9,
        boundary_value: float = 0.0,
    ) -> None:
        self.shape = _as_shape(shape)
        self.spacing = _as_spacing(spacing, len(self.shape))
        self.wave_speed = float(wave_speed)
        self.dt = float(dt if dt is not None else 0.25 * min(self.spacing) / self.wave_speed)
        self.boundary = boundary
        self.method = method
        self.dtype = _real_dtype(dtype)
        self.cfl_safety = float(cfl_safety)
        self.boundary_value = float(boundary_value)
        if boundary not in _SUPPORTED_BOUNDARIES:
            raise NumericalEngineError(f"unsupported boundary: {boundary}")
        if method not in _SUPPORTED_METHODS:
            raise NumericalEngineError(f"unsupported method: {method}")
        validate_stability(
            self.dt,
            self.shape,
            self.spacing,
            self.wave_speed,
            safety=self.cfl_safety,
        )

    def config(self) -> SolverConfig:
        return SolverConfig(
            schema_version=1,
            shape=self.shape,
            spacing=self.spacing,
            wave_speed=self.wave_speed,
            dt=self.dt,
            boundary=self.boundary,
            method=self.method,
            dtype=str(self.dtype),
            cfl_safety=self.cfl_safety,
        )

    def config_dict(self) -> dict[str, Any]:
        return self.config()._asdict()

    def save_config(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.config_dict(), indent=2, sort_keys=True) + "\n")

    @staticmethod
    def load_config(path: str | Path) -> SolverConfig:
        payload = json.loads(Path(path).read_text())
        required = set(SolverConfig._fields)
        missing = required.difference(payload)
        if missing:
            raise NumericalEngineError(f"missing config fields: {sorted(missing)}")
        return SolverConfig(
            schema_version=int(payload["schema_version"]),
            shape=tuple(int(value) for value in payload["shape"]),
            spacing=tuple(float(value) for value in payload["spacing"]),
            wave_speed=float(payload["wave_speed"]),
            dt=float(payload["dt"]),
            boundary=str(payload["boundary"]),
            method=str(payload["method"]),
            dtype=str(payload["dtype"]),
            cfl_safety=float(payload["cfl_safety"]),
        )

    def initialize(self, phi0: Array, psi0: Array | None = None, time: float = 0.0) -> FieldState:
        phi = jnp.asarray(phi0, dtype=self.dtype)
        if phi.shape[-len(self.shape) :] != self.shape:
            raise NumericalEngineError(f"phi0 must have trailing shape {self.shape}, got {phi.shape}")
        psi = jnp.zeros_like(phi) if psi0 is None else jnp.asarray(psi0, dtype=self.dtype)
        if psi.shape != phi.shape:
            raise NumericalEngineError(f"psi0 must match phi0 shape {phi.shape}, got {psi.shape}")
        phi = apply_boundary_conditions(
            phi, self.boundary, value=self.boundary_value, spatial_ndim=len(self.shape)
        )
        psi = apply_boundary_conditions(psi, self.boundary, value=0.0, spatial_ndim=len(self.shape))
        return FieldState(phi, psi, jnp.asarray(time, dtype=self.dtype), jnp.asarray(0, dtype=jnp.int32))

    def _acceleration(self, phi: Array, source: Array, potential: Array) -> Array:
        raise NotImplementedError

    def step(
        self,
        state: FieldState,
        source: Array | None = None,
        potential: Array | float | None = None,
    ) -> FieldState:
        source_value = _coerce_source(source, self.shape, self.dtype)
        potential_value = _coerce_potential(potential, self.shape, self.dtype)
        if self.method == "leapfrog":
            phi, psi = _leapfrog_step(
                state.phi,
                state.psi,
                source_value,
                potential_value,
                self.dt,
                self._acceleration,
                self.boundary,
                self.boundary_value,
                len(self.shape),
            )
        else:
            phi, psi = _rk4_step(
                state.phi,
                state.psi,
                source_value,
                potential_value,
                self.dt,
                self._acceleration,
                self.boundary,
                self.boundary_value,
                len(self.shape),
            )
        return FieldState(
            phi,
            psi,
            state.time + self.dt,
            state.step_index + 1,
        )

    def rollout(
        self,
        state: FieldState,
        source_sequence: Array | None = None,
        potential: Array | float | None = None,
        include_initial: bool = True,
    ) -> Trajectory:
        steps = 0 if source_sequence is None else int(jnp.asarray(source_sequence).shape[0])
        if source_sequence is None:
            source_sequence = jnp.zeros((0, *state.phi.shape), dtype=self.dtype)
        else:
            source_sequence = jnp.asarray(source_sequence, dtype=self.dtype)
            if source_sequence.ndim < len(self.shape) + 1 or tuple(source_sequence.shape[-len(self.shape) :]) != self.shape:
                raise NumericalEngineError(
                    f"source_sequence must have trailing shape (steps, ..., {self.shape}), got {source_sequence.shape}"
                )
        potential_value = _coerce_potential(potential, self.shape, self.dtype)

        def scan_step(carry: FieldState, source_t: Array) -> tuple[FieldState, FieldState]:
            next_state = self.step(carry, source_t, potential_value)
            return next_state, next_state

        final_state, states = jax.lax.scan(scan_step, state, source_sequence)
        del final_state, steps
        if include_initial:
            return Trajectory(
                phi=jnp.concatenate([state.phi[None, ...], states.phi], axis=0),
                psi=jnp.concatenate([state.psi[None, ...], states.psi], axis=0),
                time=jnp.concatenate([state.time[None], states.time], axis=0),
                step_index=jnp.concatenate([state.step_index[None], states.step_index], axis=0),
            )
        return Trajectory(states.phi, states.psi, states.time, states.step_index)

    def energy(self, state: FieldState, potential: Array | float | None = None) -> Array:
        potential_value = _coerce_potential(potential, self.shape, self.dtype)
        gradient_term = self.gradient_energy(state.phi)
        volume_element = jnp.prod(jnp.asarray(self.spacing, dtype=self.dtype))
        kinetic = 0.5 * jnp.sum(jnp.square(state.psi)) * volume_element
        potential_energy = 0.5 * jnp.sum(potential_value * jnp.square(state.phi)) * volume_element
        return kinetic + 0.5 * self.wave_speed**2 * gradient_term + potential_energy

    def gradient_energy(self, field: Array) -> Array:
        return jnp.sum(jnp.square(finite_difference_gradient(field, self.spacing, self.boundary)))

    def finite_values(self, state: FieldState) -> Array:
        return jnp.all(jnp.isfinite(state.phi)) & jnp.all(jnp.isfinite(state.psi))

    def operator_loss(self, prediction: Array, target: Array, mask: Array | None = None) -> Array:
        prediction = jnp.asarray(prediction)
        target = jnp.asarray(target)
        error = jnp.square(prediction - target)
        if mask is not None:
            mask_value = jnp.asarray(mask, dtype=error.dtype)
            return jnp.sum(error * mask_value) / jnp.maximum(jnp.sum(mask_value), 1.0)
        return jnp.mean(error)


class SpectralSolver(_BaseSolver):
    """Optimized periodic-grid solver using FFT differentiation."""

    operator_name = "spectral"

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("boundary", "periodic")
        if kwargs["boundary"] != "periodic":
            raise NumericalEngineError("SpectralSolver supports only periodic boundaries")
        super().__init__(*args, **kwargs)
        self.grids = tuple(jnp.asarray(grid, dtype=self.dtype) for grid in frequency_grids(self.shape, self.spacing))

    def _acceleration(self, phi: Array, source: Array, potential: Array) -> Array:
        return _spectral_acceleration(phi, source, potential, self.grids, self.wave_speed)

    def gradient_energy(self, field: Array) -> Array:
        return spectral_gradient_energy(field, self.grids, self.spacing)


class FiniteDifferenceSolver(_BaseSolver):
    """Independent reference solver supporting all Stage 1 boundaries."""

    operator_name = "finite_difference"

    def _acceleration(self, phi: Array, source: Array, potential: Array) -> Array:
        return _finite_difference_acceleration(
            phi, source, potential, self.spacing, self.boundary, self.wave_speed
        )

    def gradient_energy(self, field: Array) -> Array:
        gradients = finite_difference_gradient(field, self.spacing, self.boundary)
        volume_element = jnp.prod(jnp.asarray(self.spacing, dtype=field.dtype))
        return jnp.sum(jnp.square(gradients)) * volume_element


def finite_difference_gradient(
    field: Array,
    spacing: float | Sequence[float],
    boundary: str = "periodic",
) -> Array:
    """Return summed squared gradient density for reference diagnostics."""
    ndim = field.ndim
    spacing_values = _as_spacing(spacing, ndim)
    gradients = []
    if boundary == "periodic":
        for axis, dx in zip(_spatial_axes(ndim), spacing_values):
            gradients.append((jnp.roll(field, -1, axis) - jnp.roll(field, 1, axis)) / (2.0 * dx))
    else:
        padded = _pad_for_boundary(field, boundary, ndim)
        for offset, dx in enumerate(spacing_values):
            axis = padded.ndim - ndim + offset
            plus = [slice(None)] * padded.ndim
            minus = [slice(None)] * padded.ndim
            plus[axis] = slice(2, None)
            minus[axis] = slice(None, -2)
            gradients.append((padded[tuple(plus)] - padded[tuple(minus)]) / (2.0 * dx))
    return jnp.stack(gradients, axis=0)


def manufactured_mode(
    shape: Sequence[int],
    spacing: float | Sequence[float],
    mode: Sequence[int] = (1,),
    phase: float = 0.0,
    dtype: Any = jnp.float32,
) -> Array:
    """Construct a periodic Fourier mode with integer mode numbers."""
    shape_values = _as_shape(shape)
    spacing_values = _as_spacing(spacing, len(shape_values))
    mode_values = tuple(int(value) for value in mode)
    if len(mode_values) != len(shape_values):
        raise NumericalEngineError("mode must have one integer per spatial axis")
    coordinates = [jnp.arange(n, dtype=dtype) * dx for n, dx in zip(shape_values, spacing_values)]
    grids = jnp.meshgrid(*coordinates, indexing="ij")
    argument = sum(2.0 * jnp.pi * k * x / (n * dx) for k, x, n, dx in zip(mode_values, grids, shape_values, spacing_values))
    return jnp.sin(argument + phase)


def manufactured_mode_frequency(
    shape: Sequence[int], spacing: float | Sequence[float], mode: Sequence[int], wave_speed: float, potential: float = 0.0
) -> float:
    """Return the continuous Fourier-mode frequency for a periodic grid."""
    spacing_values = _as_spacing(spacing, len(tuple(shape)))
    wave_numbers = [2.0 * jnp.pi * int(k) / (n * dx) for k, n, dx in zip(mode, shape, spacing_values)]
    return float(jnp.sqrt(wave_speed**2 * sum(k * k for k in wave_numbers) + potential))


__all__ = [
    "FieldState",
    "FiniteDifferenceSolver",
    "NumericalEngineError",
    "SolverConfig",
    "SpectralSolver",
    "StabilityError",
    "Trajectory",
    "UnsupportedPrecisionError",
    "apply_boundary_conditions",
    "bounded_local_potential",
    "bounded_spectral_potential",
    "cfl_limit",
    "finite_difference_gradient",
    "finite_difference_laplacian",
    "frequency_grids",
    "manufactured_mode",
    "manufactured_mode_frequency",
    "spectral_laplacian",
    "validate_stability",
]
