import pytest
# We will import our compiled library. The name 'causa_py' is what we
# specified in our Cargo.toml's [lib] section.
import causa_py

def test_event_creation_and_properties():
    """
    Tests if an Event can be created from Python and its properties
    can be accessed and modified.
    """
    sem_vec = [0.1, 0.2]
    temp_ten = [10, 1]
    caus_pot = [0.5, -0.5]

    # Create the event
    event = causa_py.Event(sem_vec, temp_ten, caus_pot)

    # Assert properties are correctly set
    assert event.semantic_vector == sem_vec
    assert event.temporal_tensor == temp_ten
    assert event.causal_potential_vector == caus_pot

    # Test modification
    new_temp_ten = [11, 2]
    event.temporal_tensor = new_temp_ten
    assert event.temporal_tensor == new_temp_ten

def test_manifold_creation_and_event_lifecycle():
    """
    Tests the full lifecycle: creating a Manifold, placing an Event,
    retrieving it, and checking for empty cells.
    """
    # 1. Create Manifold
    manifold = causa_py.Manifold(dimensions=[100, 20])
    assert manifold.dimensions == [100, 20]
    # Verify __repr__ works
    assert "filled_cells: 0" in repr(manifold)

    # 2. Create and Place Event
    event_coords = [42, 15]
    event = causa_py.Event([1.0], event_coords, [0.0])
    
    # This calls the Rust method
    manifold.place_event(event)
    assert "filled_cells: 1" in repr(manifold)

    # 3. Retrieve and Validate Event
    retrieved_event = manifold.get_event(coordinates=event_coords)
    assert retrieved_event is not None
    assert retrieved_event.semantic_vector == [1.0]
    assert retrieved_event.temporal_tensor == event_coords

    # 4. Check Empty Cell
    empty_event = manifold.get_event(coordinates=[1, 1])
    assert empty_event is None

def test_manifold_error_handling():
    """
    Tests that our custom Rust errors are correctly translated
    into Python exceptions.
    """
    manifold = causa_py.Manifold(dimensions=[10, 10])

    # Test out-of-bounds retrieval
    with pytest.raises(ValueError) as excinfo:
        manifold.get_event(coordinates=[99, 99])
    # Assert that our custom error message from Rust is present
    assert "Coordinates [99, 99] are out of bounds" in str(excinfo.value)

    # Test out-of-bounds placement
    out_of_bounds_event = causa_py.Event([], [10, 0], [])
    with pytest.raises(ValueError) as excinfo:
        manifold.place_event(out_of_bounds_event)
    assert "Coordinates [10, 0] are out of bounds" in str(excinfo.value)
