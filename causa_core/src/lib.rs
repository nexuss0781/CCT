use pyo3::prelude::*;

// Declare the modules we just created
pub mod error;
pub mod spacetime;

// This function defines the Python module.
// The name `causa_py` must match the name in Cargo.toml.
#[pymodule]
fn causa_py(_py: Python, m: &PyModule) -> PyResult<()> {
    // We will add our classes (Manifold, Event) to the module `m` here later.
    Ok(())
}
