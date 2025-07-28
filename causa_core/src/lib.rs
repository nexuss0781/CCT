use pyo3::prelude::*;
use ndarray::ArrayD;
use thiserror::Error;

// --- Error Definition ---
#[derive(Error, Debug)]
pub enum CausaError {
    #[error("Coordinates {0:?} are out of bounds.")]
    OutOfBounds(Vec<isize>),
}

impl From<CausaError> for PyErr {
    fn from(err: CausaError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}


// --- Event Definition ---
#[pyclass]
#[derive(Clone, Debug)]
pub struct Event {
    #[pyo3(get, set)]
    pub semantic_vector: Vec<f32>,
    #[pyo3(get, set)]
    pub temporal_tensor: Vec<isize>,
    #[pyo3(get, set)]
    pub causal_potential_vector: Vec<f32>,
}

#[pymethods]
impl Event {
    #[new]
    pub fn new(
        semantic_vector: Vec<f32>,
        temporal_tensor: Vec<isize>,
        causal_potential_vector: Vec<f32>,
    ) -> Self {
        Event { semantic_vector, temporal_tensor, causal_potential_vector }
    }

    fn __repr__(&self) -> String {
        format!(
            "Event(temporal_pos: {:?}, semantic_dim: {}, causal_dim: {})",
            self.temporal_tensor,
            self.semantic_vector.len(),
            self.causal_potential_vector.len()
        )
    }
}


// --- Manifold Definition ---
#[pyclass]
#[derive(Debug)]
pub struct Manifold {
    #[pyo3(get)]
    pub dimensions: Vec<usize>,
    grid: ArrayD<Option<Event>>,
}

#[pymethods]
impl Manifold {
    #[new]
    pub fn new(dimensions: Vec<usize>) -> PyResult<Self> {
        let grid = ArrayD::from_elem(dimensions.as_slice(), None);
        Ok(Manifold { dimensions, grid })
    }

    pub fn place_event(&mut self, event: Event) -> PyResult<()> {
        let coordinates: Vec<usize> = event.temporal_tensor.iter().map(|&x| x as usize).collect();
        let cell = self.grid.get_mut(coordinates.as_slice())
            .ok_or_else(|| CausaError::OutOfBounds(event.temporal_tensor.clone()))?;
        *cell = Some(event);
        Ok(())
    }

    #[pyo3(text_signature = "($self, coordinates)")]
    pub fn get_event(&self, coordinates: Vec<isize>) -> PyResult<Option<Event>> {
        let coordinates_usize: Vec<usize> = coordinates.iter().map(|&x| x as usize).collect();
        let cell = self.grid.get(coordinates_usize.as_slice())
            .ok_or_else(|| CausaError::OutOfBounds(coordinates.clone()))?;
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


// --- Python Module Definition ---
#[pymodule]
fn causa_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Event>()?;
    m.add_class::<Manifold>()?;
    Ok(())
}