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

def n():
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
    
@jax.jit
def resolve_system(source_field: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    """
    Resolves the entire system using the Fourier Causal Solver method.

    This function is JIT-compiled for maximum performance on accelerators.

    Args:
        source_field: The JAX array J(x) representing the initial causal potentials.
        kernel: The JAX array K(ω) representing the propagation laws in the
                frequency domain.

    Returns:
        A JAX array C(x) representing the final, superposed Causal Field
        in the spacetime domain.
    """
assert source_field.shape == kernel.shape, "Source field and kernel must have the same shape."

# Step 1: Forward FFT
# Transform the source field J(x) from the spacetime domain to the
# frequency domain J(ω).
# `fftn` performs a multi-dimensional Fast Fourier Transform.
source_field_freq = jnp.fft.fftn(source_field)

# Step 2: Kernel Application
# This is the core of the solver. The complex convolution in the spacetime
# domain becomes a simple element-wise multiplication in the frequency domain.
# C(ω) = J(ω) * K(ω)
causal_field_freq = source_field_freq * kernel

# Step 3: Inverse FFT
# Transform the resolved causal field C(ω) back from the frequency domain
# to the spacetime domain C(x).
# `ifftn` performs the inverse transform. We take the real part as the
# imaginary part should be negligible due to numerical precision effects.
final_causal_field = jnp.fft.ifftn(causal_field_freq).real

return final_causal_field