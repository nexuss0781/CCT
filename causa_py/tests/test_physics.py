import pytest
import jax.numpy as jnp
from causa_py.physics import create_source_field, create_propagation_kernel, resolve_system
# This assumes we have a way to create a Manifold from Python,
# which we do from the previous sub-phase.
from causa_py.causa_py import Manifold, Event

def test_single_source_propagation():
    """
    Validates that influence from a single source propagates correctly.
    This is analogous to checking the Green's function of the system.
    """
    # 1. Setup: Create a 2D Manifold with one Event at the center
    dims = (32, 32)
    manifold = Manifold(dimensions=list(dims))
    center_coords = [dims[0] // 2, dims[1] // 2]
    
    event = Event(
        semantic_vector=[], 
        temporal_tensor=center_coords, 
        causal_potential_vector=[1.0] # A strong positive potential
    )
    # This is the inefficient part we will optimize later.
    # For a test, this is acceptable.
    source_field = jnp.zeros(dims).at[tuple(center_coords)].set(sum(event.causal_potential_vector))

    # 2. Create a standard Gaussian kernel
    kernel_params = {'decay_rate': 0.1}
    kernel = create_propagation_kernel(dims, kernel_params)

    # 3. Resolve the system
    final_field = resolve_system(source_field, kernel)

    # 4. Assertions:
    # The maximum influence should remain at the center
    max_pos = jnp.unravel_index(jnp.argmax(final_field), final_field.shape)
    assert max_pos == (center_coords[0], center_coords[1])

    # The influence at the center should be less than 1.0 (due to decay/spread)
    # but greater than 0.
    assert 0 < final_field[max_pos] < 1.0

    # The influence at a corner should be near zero, showing decay.
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
    # The combined field should be the sum of the individual fields,
    # demonstrating the principle of superposition.
    assert jnp.allclose(combined_field, field1 + field2)

def test_kernel_energy_conservation():
    """
    Validates that the kernel is normalized correctly. A simple kernel should
    mostly preserve the total 'energy' or 'influence' in the system.
    """
    dims = (32, 32)
    # A kernel with zero decay should not diminish the total influence
    kernel_params = {'decay_rate': 0.0}
    kernel = create_propagation_kernel(dims, kernel_params)

    # The DC component (at index 0,0) of a frequency-domain kernel
    # represents its effect on the total sum of the signal.
    # A value of 1.0 means the total sum is preserved.
    # Our ifftshift moves the DC component to the center.
    center_idx = tuple(d // 2 for d in dims)
    dc_component = kernel[center_idx]
    
    assert jnp.isclose(dc_component, 1.0)