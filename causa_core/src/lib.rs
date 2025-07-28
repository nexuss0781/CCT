use pyo3::prelude::*;

// Declare the modules
pub mod error;
pub mod spacetime;

// Import the structs we want to expose
use spacetime::{Event, Manifold};

// This function defines the Python module.
#[pymodule]
fn causa_py(_py: Python, m: &PyModule) -> PyResult<()> {
    // Add our Event and Manifold classes to the Python module `m`
    m.add_class::<Event>()?;
    m.add_class::<Manifold>()?;
    
    Ok(())
}
