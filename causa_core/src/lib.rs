use pyo3::prelude::*;

// Declare the modules we just created
pub mod error;
pub mod spacetime;

// Import the Event struct so we can use it here
use spacetime::Event;

// This function defines the Python module.
// The name `causa_py` must match the name in Cargo.toml.
#[pymodule]
fn causa_py(_py: Python, m: &PyModule) -> PyResult<()> {
    // Add our Event class to the Python module `m`
    m.add_class::<Event>()?;
    
    // We will add the Manifold class here in the next task.
    Ok(())
}
