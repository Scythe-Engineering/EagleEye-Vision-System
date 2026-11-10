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
mod python;
mod track;
mod workers;

use pyo3::prelude::*;

#[pymodule]
fn custom_apriltag_detector(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<python::CustomAprilTagDetector>()?;
    module.add_class::<detection::Detection>()?;
    Ok(())
}
