"""Public Python interface for the CCT-ASE research prototype."""

try:
    from causa_native import Event, Manifold
except ImportError as exc:  # pragma: no cover - exercised in clean-install diagnostics
    raise ImportError(
        "The CCT native extension is unavailable. Build it with "
        "`make install-native` (or `maturin develop --manifest-path "
        "causa_core/Cargo.toml`)."
    ) from exc

from . import physics
from .numerical_engine import (
    FieldState,
    FiniteDifferenceSolver,
    NumericalEngineError,
    SolverConfig,
    SpectralSolver,
    StabilityError,
    Trajectory,
    UnsupportedPrecisionError,
    apply_boundary_conditions,
    bounded_local_potential,
    bounded_spectral_potential,
    cfl_limit,
    finite_difference_gradient,
    finite_difference_laplacian,
    frequency_grids,
    manufactured_mode,
    manufactured_mode_frequency,
    spectral_laplacian,
    validate_stability,
)

__all__ = [
    "Event",
    "Manifold",
    "physics",
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
