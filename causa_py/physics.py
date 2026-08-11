"""Deterministic baseline numerical helpers for the CCT prototype."""

from typing import Any, Dict

import jax
import jax.numpy as jnp

from . import Event, Manifold


@jax.jit
def resolve_system(source_field: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    """Apply a frequency-domain kernel to a real source field."""
    if source_field.ndim != kernel.ndim:
        raise ValueError("Source field and kernel must have the same rank.")
    transformed = jnp.fft.fftn(source_field)
    return jnp.fft.ifftn(transformed * kernel).real


def create_source_field(manifold: Manifold) -> jnp.ndarray:
    """Convert stored manifold events into a scalar source field.

    Each event contributes the sum of its causal-potential vector at its
    validated temporal coordinates. Multiple events at one coordinate are
    accumulated, although the current native manifold rejects duplicate cells.
    """
    source_grid = jnp.zeros(tuple(manifold.dimensions), dtype=jnp.float32)
    for event in manifold.events():
        coordinates = tuple(int(coordinate) for coordinate in event.temporal_tensor)
        potential = sum(float(value) for value in event.causal_potential_vector)
        source_grid = source_grid.at[coordinates].add(potential)
    return source_grid


def create_propagation_kernel(shape: tuple[int, ...], params: Dict[str, Any]) -> jnp.ndarray:
    """Create a normalized Gaussian low-pass kernel in FFT ordering."""
    if not shape or any(int(size) <= 0 for size in shape):
        raise ValueError("shape must contain positive dimensions")

    freq_grids = jnp.meshgrid(
        *[jnp.fft.fftfreq(int(size)) for size in shape], indexing="ij"
    )
    dist_sq = sum(grid**2 for grid in freq_grids)
    decay_rate = jnp.asarray(params.get("decay_rate", 0.1), dtype=jnp.float32)
    if bool(jnp.any(decay_rate < 0.0)):
        raise ValueError("decay_rate must be non-negative")
    kernel = jnp.exp(-decay_rate * dist_sq)
    return jnp.fft.ifftshift(kernel)
