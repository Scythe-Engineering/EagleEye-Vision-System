//! AprilTag detector exposed to Python through PyO3.

mod candidates;
mod decode;
mod detection;
mod detector;
mod families;
mod family_data;
mod frame;
mod geometry;
mod image;
mod profile;
mod python;
mod track;
mod workers;

use pyo3::prelude::*;

#[pymodule]
fn custom_apriltag_detector(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<python::CustomAprilTagDetector>()?;
    module.add_class::<detection::Detection>()?;
    #[cfg(feature = "stage-profile")]
    module.add_function(wrap_pyfunction!(stage_profile, module)?)?;
    Ok(())
}

/// Accumulated per-stage nanoseconds and candidate counts since the last call.
///
/// Totals are per thread: this reports and resets only the calling thread's, so
/// call it from the thread that ran detection.
#[cfg(feature = "stage-profile")]
#[pyfunction]
fn stage_profile() -> Vec<(String, u64)> {
    profile::take()
}
