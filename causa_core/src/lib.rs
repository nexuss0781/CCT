use ndarray::ArrayD;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CausaError {
    #[error("Coordinates {0:?} are out of bounds.")]
    OutOfBounds(Vec<isize>),
    #[error("Coordinates {0:?} have the wrong dimensionality; expected {1}.")]
    InvalidDimensionality(Vec<isize>, usize),
    #[error("Cell already occupied at coordinates {0:?}.")]
    CellOccupied(Vec<isize>),
    #[error("Manifold dimensions must contain at least one positive axis.")]
    InvalidDimensions,
}

impl From<CausaError> for PyErr {
    fn from(err: CausaError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}

fn checked_coordinates(
    coordinates: &[isize],
    dimensions: &[usize],
) -> Result<Vec<usize>, CausaError> {
    if coordinates.len() != dimensions.len() {
        return Err(CausaError::InvalidDimensionality(
            coordinates.to_vec(),
            dimensions.len(),
        ));
    }

    let mut checked = Vec::with_capacity(coordinates.len());
    for (&coordinate, &dimension) in coordinates.iter().zip(dimensions.iter()) {
        if coordinate < 0 || coordinate as usize >= dimension {
            return Err(CausaError::OutOfBounds(coordinates.to_vec()));
        }
        checked.push(coordinate as usize);
    }
    Ok(checked)
}

#[pyclass(get_all, set_all)]
#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    pub semantic_vector: Vec<f32>,
    pub temporal_tensor: Vec<isize>,
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
        Event {
            semantic_vector,
            temporal_tensor,
            causal_potential_vector,
        }
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
        if dimensions.is_empty() || dimensions.iter().any(|dimension| *dimension == 0) {
            return Err(CausaError::InvalidDimensions.into());
        }
        let grid = ArrayD::from_elem(dimensions.as_slice(), None);
        Ok(Manifold { dimensions, grid })
    }

    pub fn place_event(&mut self, event: Event) -> PyResult<()> {
        let coordinates = checked_coordinates(&event.temporal_tensor, &self.dimensions)?;
        let cell = self
            .grid
            .get_mut(coordinates.as_slice())
            .ok_or_else(|| CausaError::OutOfBounds(event.temporal_tensor.clone()))?;
        if cell.is_some() {
            return Err(CausaError::CellOccupied(event.temporal_tensor).into());
        }
        *cell = Some(event);
        Ok(())
    }

    pub fn get_event(&self, coordinates: Vec<isize>) -> PyResult<Option<Event>> {
        let checked = checked_coordinates(&coordinates, &self.dimensions)?;
        let cell = self
            .grid
            .get(checked.as_slice())
            .ok_or_else(|| CausaError::OutOfBounds(coordinates.clone()))?;
        Ok(cell.as_ref().cloned())
    }

    pub fn events(&self) -> Vec<Event> {
        self.grid
            .iter()
            .filter_map(|cell| cell.as_ref().cloned())
            .collect()
    }

    pub fn filled_cells(&self) -> usize {
        self.grid.iter().filter(|cell| cell.is_some()).count()
    }

    fn __repr__(&self) -> String {
        format!(
            "Manifold(dimensions: {:?}, filled_cells: {})",
            self.dimensions,
            self.filled_cells()
        )
    }
}

#[pymodule]
fn causa_native(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<Event>()?;
    module.add_class::<Manifold>()?;
    Ok(())
}
