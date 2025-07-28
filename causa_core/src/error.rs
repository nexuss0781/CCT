use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CausaError {
    #[error("Coordinates {0:?} are out of bounds.")]
    OutOfBounds(Vec<isize>),
}

// This block allows our custom Rust error to be converted
// into a standard Python exception.
impl From<CausaError> for PyErr {
    fn from(err: CausaError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}
