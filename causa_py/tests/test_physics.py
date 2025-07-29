import jax.numpy as jnp
import pytest
from causa_py import Manifold, Event
from causa_py.physics import create_propagation_kernel, resolve_system

# Your tests here


def test_single_source_propagation():
    dims = (32, 32)
    manifold = Manifold(dimensions=list(dims))
    center_coords = [dims[0] // 2, dims[1] // 2]

    event = Event(
        semantic_vector=[], 
        temporal_tensor=center_coords, 
        causal_potential_vector=[1.0]
    )

    source_field = jnp.zeros(dims).at[tuple(center_coords)].set(sum(event.causal_potential_vector))
    kernel = create_propagation_kernel(dims, {'decay_rate': 0.1})
    final_field = resolve_system(source_field, kernel)

    max_pos = jnp.unravel_index(jnp.argmax(final_field), final_field.shape)
    assert tuple(int(i) for i in max_pos) == (center_coords[0], center_coords[1])
    assert 0 < final_field[max_pos] < 1.0
    assert jnp.isclose(final_field[0, 0], 0.0)

def test_superposition_principle():
    """
    Validates that the influence of two sources is the sum of their individual influences.
    This is a key property of linear systems like our field equation.
    """
    # 1. Setup
    dims = (32, 32)
    coords1 = (8, 8)
    coords2 = (24, 24)
    kernel_params = {'decay_rate': 0.1}
    kernel = create_propagation_kernel(dims, kernel_params)

    # 2. Run simulation for each source individually
    source1 = jnp.zeros(dims).at[coords1].set(1.0)
    source2 = jnp.zeros(dims).at[coords2].set(1.0)
    
    field1 = resolve_system(source1, kernel)
    field2 = resolve_system(source2, kernel)

    # 3. Run simulation for both sources together
    combined_source = source1 + source2
    combined_field = resolve_system(combined_source, kernel)

    # 4. Assertion:
    # The combined field should be approximately the sum of the individual fields,
    # allowing for small floating-point differences.
    assert jnp.allclose(combined_field, field1 + field2)
    
def test_kernel_normalization():
    dims = (32, 32)
    kernel = create_propagation_kernel(dims, {'decay_rate': 0.0})
    center_idx = tuple(d // 2 for d in dims)
    dc_component = kernel[center_idx]

    assert jnp.isclose(dc_component, 1.0)