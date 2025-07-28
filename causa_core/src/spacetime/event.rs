use pyo3::prelude::*;

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
    /// Creates a new Event. This constructor will be callable from Python.
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

    /// A representation of the Event for printing in Python (like __repr__)
    fn __repr__(&self) -> String {
        format!(
            "Event(temporal_pos: {:?}, semantic_dim: {}, causal_dim: {})",
            self.temporal_tensor,
            self.semantic_vector.len(),
            self.causal_potential_vector.len()
        )
    }
}
