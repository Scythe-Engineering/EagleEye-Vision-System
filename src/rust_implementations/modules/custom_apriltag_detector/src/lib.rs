//! Independent AprilTag detector ported from the validated overlap-cache reference.
//! Python operation and ABI 2 call the same Rust implementation; no detector fallback.

mod candidates;
mod decode;
mod detection;
mod detector;
mod families;
mod family_data;
mod ffi;
mod geometry;
mod image;
mod python;
mod workers;

use pyo3::prelude::*;

/// Register the operation using the existing project Rust-module build framework.
#[pymodule]
fn custom_apriltag_detector(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<python::CustomAprilTagDetector>()?;
    module.add_class::<python::NativeDetection>()?;
    module.add_function(wrap_pyfunction!(python::native_library_path, module)?)?;
    Ok(())
}
