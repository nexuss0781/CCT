from __future__ import annotations

import numpy as np
import pytest


def test_canonical_import_surface():
    import causa_native
    import causa_py

    assert causa_native.__name__ == "causa_native"
    assert causa_py.Event is not None
    assert causa_py.Manifold is not None
    assert hasattr(causa_py, "physics")


def test_event_lifecycle_and_source_extraction():
    from causa_py import Event, Manifold
    from causa_py.physics import create_source_field

    manifold = Manifold([4, 4])
    manifold.place_event(Event([1.0], [1, 2], [0.5, 0.25]))

    retrieved = manifold.get_event([1, 2])
    assert retrieved is not None
    assert retrieved.temporal_tensor == [1, 2]
    assert manifold.filled_cells() == 1
    assert len(manifold.events()) == 1

    source = np.asarray(create_source_field(manifold))
    assert source.shape == (4, 4)
    assert np.isclose(source[1, 2], 0.75)
    assert np.isclose(source.sum(), 0.75)


def test_invalid_inputs_do_not_mutate_manifold():
    from causa_py import Event, Manifold

    manifold = Manifold([4, 4])
    with pytest.raises(ValueError, match="out of bounds"):
        manifold.place_event(Event([], [-1, 0], []))
    assert manifold.filled_cells() == 0

    with pytest.raises(ValueError, match="dimensionality"):
        manifold.get_event([1])
    assert manifold.filled_cells() == 0

    event = Event([], [1, 1], [])
    manifold.place_event(event)
    with pytest.raises(ValueError, match="already occupied"):
        manifold.place_event(event)
    assert manifold.filled_cells() == 1


def test_invalid_dimensions_are_rejected():
    from causa_py import Manifold

    with pytest.raises(ValueError, match="dimensions"):
        Manifold([])
    with pytest.raises(ValueError, match="dimensions"):
        Manifold([0, 4])


def test_fft_baseline_is_deterministic():
    import jax.numpy as jnp
    from causa_py.physics import create_propagation_kernel, resolve_system

    source = jnp.zeros((8, 8), dtype=jnp.float32).at[4, 4].set(1.0)
    kernel = create_propagation_kernel((8, 8), {"decay_rate": 0.1})
    first = np.asarray(resolve_system(source, kernel))
    second = np.asarray(resolve_system(source, kernel))
    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all()


def test_kernel_rejects_invalid_shape_and_decay():
    import jax
    from causa_py.physics import create_propagation_kernel

    with pytest.raises(ValueError, match="positive"):
        create_propagation_kernel((0, 8), {"decay_rate": 0.1})
    with pytest.raises(ValueError, match="non-negative"):
        create_propagation_kernel((8, 8), {"decay_rate": -0.1})

    # Ensure the JAX runtime itself is available to the Stage 0 package.
    assert jax.__version__
