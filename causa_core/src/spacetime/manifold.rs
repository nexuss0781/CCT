use crate::error::CausaError;
use crate::spacetime::event::Event;
use ndarray::ArrayD;
use pyo3::prelude::*;

#[pyclass]
#[derive(Debug)]
pub struct Manifold {
    #[pyo3(get)]
    pub dimensions: Vec<usize>,
    // The grid is not directly exposed to Python for safety.
    // We will expose it via methods.
    grid: ArrayD<Option<Event>>,
}

#[pymethods]
impl Manifold {
    #[new]
    /// Creates a new Manifold with the given dimensions.
    pub fn new(dimensions: Vec<usize>) -> PyResult<Self> {
        // ndarray can fail if dimensions are invalid, so we return a PyResult
        let grid = ArrayD::from_elem(dimensions.as_slice(), None);
        Ok(Manifold { dimensions, grid })
    }

    /// Places an Event onto the Manifold at the coordinates specified
    /// within the Event's temporal_tensor.
    pub fn place_event(&mut self, event: Event) -> PyResult<()> {
        let coordinates: Vec<usize> = event
            .temporal_tensor
            .iter()
            .map(|&x| x as usize)
            .collect();

        // The get_mut method on the ndarray checks bounds automatically.
        let cell = self
            .grid
            .get_mut(coordinates.as_slice())
            .ok_or_else(|| CausaError::OutOfBounds(event.temporal_tensor.clone()))?;

        // Place the event in the grid.
        *cell = Some(event);

        Ok(())
    }

    /// Retrieves a clone of an Event from the specified coordinates.
    /// Returns None if the cell is empty.
    #[pyo3(text_signature = "($self, coordinates)")]
    pub fn get_event(&self, coordinates: Vec<isize>) -> PyResult<Option<Event>> {
        let coordinates_usize: Vec<usize> =
            coordinates.iter().map(|&x| x as usize).collect();

        let cell = self
            .grid
            .get(coordinates_usize.as_slice())
            .ok_or_else(|| CausaError::OutOfBounds(coordinates.clone()))?;

        // `cloned()` on Option<T> returns a new Option<T> with the inner value cloned.
        Ok(cell.as_ref().cloned())
    }

    fn __repr__(&self) -> String {
        format!(
            "Manifold(dimensions: {:?}, filled_cells: {})",
            self.dimensions,
            self.grid.iter().filter(|cell| cell.is_some()).count()
        )
    }
}
