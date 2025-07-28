import sys
sys.path.insert(0, '/content/CCT/causa_py')

import sys
sys.path.insert(0, '/content/CCT/causa_py')

import sys
sys.path.insert(0, '/content/CCT/causa_py') # Add the package to the path
import pytest
from causa_py import Manifold, Event

def test_event_creation_and_properties():
    sem_vec = [0.1, 0.2]
    temp_ten = [10, 1]
    caus_pot = [0.5, -0.5]
    event = Event(sem_vec, temp_ten, caus_pot)
    
    # Use pytest.approx for floating point comparisons
    assert event.semantic_vector == pytest.approx(sem_vec)
    assert event.temporal_tensor == temp_ten
    assert event.causal_potential_vector == pytest.approx(caus_pot)

    new_temp_ten = [11, 2]
    event.temporal_tensor = new_temp_ten
    assert event.temporal_tensor == new_temp_ten

def test_manifold_creation_and_event_lifecycle():
    manifold = Manifold(dimensions=[100, 20])
    assert manifold.dimensions == [100, 20]
    assert "filled_cells: 0" in repr(manifold)
    event_coords = [42, 15]
    event = Event([1.0], event_coords, [0.0])
    manifold.place_event(event)
    assert "filled_cells: 1" in repr(manifold)
    retrieved_event = manifold.get_event(coordinates=event_coords)
    assert retrieved_event is not None
    assert retrieved_event.semantic_vector == [1.0]
    assert retrieved_event.temporal_tensor == event_coords
    empty_event = manifold.get_event(coordinates=[1, 1])
    assert empty_event is None

def test_manifold_error_handling():
    manifold = Manifold(dimensions=[10, 10])
    with pytest.raises(ValueError) as excinfo:
        manifold.get_event(coordinates=[99, 99])
    assert "Coordinates [99, 99] are out of bounds" in str(excinfo.value)
    out_of_bounds_event = Event([], [10, 0], [])
    with pytest.raises(ValueError) as excinfo:
        manifold.place_event(out_of_bounds_event)
    assert "Coordinates [10, 0] are out of bounds" in str(excinfo.value)
