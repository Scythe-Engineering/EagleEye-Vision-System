//! Native calibrated fixed-heading 2D localization; Python only marshals inputs.
mod alignment;
mod calibration;
mod solver;

use calibration::Calibration;
use nalgebra::{Vector2, Vector3};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyInt, PyList, PyTuple},
};
use std::collections::{HashMap, HashSet};

#[pyclass(frozen)]
struct PnpLocalization2D {
    calibration: Calibration,
    geometry: HashMap<i64, [Vector3<f64>; 4]>,
}
/// Preserve missing_capture diagnostics for non-exact ints and i64 overflow.
/// The sentinel remains internal; no valid synchronized timestamp is <= 1.
fn capture_timestamp(value: &Bound<'_, PyAny>) -> PyResult<i64> {
    Ok(if value.is_exact_instance_of::<PyInt>() {
        value.extract::<i64>().unwrap_or(1)
    } else {
        1
    })
}

/// Read either actual Detection attributes or explicit dictionary fields.
fn field<'py>(object: &Bound<'py, PyAny>, name: &str) -> Option<Bound<'py, PyAny>> {
    if let Ok(dict) = object.cast::<PyDict>() {
        dict.get_item(name).ok().flatten()
    } else {
        object.getattr(name).ok()
    }
}
/// Unknown IDs and already-used IDs are discarded before touching corners.
fn parse_points(
    detections: &Bound<'_, PyAny>,
    geometry: &HashMap<i64, [Vector3<f64>; 4]>,
) -> Result<solver::Points, &'static str> {
    if !(detections.is_instance_of::<PyList>() || detections.is_instance_of::<PyTuple>()) {
        return Err("invalid_detections");
    }
    let mut seen = HashSet::new();
    let mut objects = Vec::new();
    let mut images = Vec::new();
    for detection in detections.try_iter().map_err(|_| "invalid_detections")? {
        let detection = detection.map_err(|_| "invalid_detections")?;
        let Some(id) = field(&detection, "tag_id") else {
            continue;
        };
        // Python map membership also accepts numeric integral floats and bools.
        let id = id.extract::<i64>().ok().or_else(|| {
            let value = id.extract::<f64>().ok()?;
            if value.is_finite()
                && value.fract() == 0.
                && value >= i64::MIN as f64
                && value < (i64::MAX as f64)
            {
                Some(value as i64)
            } else {
                None
            }
        });
        let Some(id) = id else {
            continue;
        };
        let Some(corners) = geometry.get(&id) else {
            continue;
        };
        if seen.contains(&id) {
            continue;
        }
        let image = field(&detection, "corners").ok_or("invalid_points")?;
        let rows = image
            .extract::<Vec<Vec<f64>>>()
            .or_else(|_| image.call_method0("tolist")?.extract::<Vec<Vec<f64>>>())
            .map_err(|_| "invalid_points")?;
        if rows.len() != 4
            || rows
                .iter()
                .any(|row| row.len() != 2 || !row.iter().all(|value| value.is_finite()))
        {
            return Err("invalid_points");
        }
        objects.extend_from_slice(corners);
        images.extend(rows.iter().map(|row| Vector2::new(row[0], row[1])));
        seen.insert(id);
    }
    if seen.is_empty() {
        return Err("no_mapped_tags");
    }
    Ok(solver::Points { objects, images })
}
/// Serialize exactly the three operation outputs, including accumulated diagnostics.
fn output<'py>(
    py: Python<'py>,
    diagnostics: Bound<'py, PyDict>,
    result: Result<solver::Solution, &str>,
) -> PyResult<Bound<'py, PyDict>> {
    let dictionary = PyDict::new(py);
    match result {
        Ok(solution) => {
            dictionary.set_item("camera_pose", solution.pose)?;
            dictionary.set_item("pose_meta", solution.meta)?;
            diagnostics.set_item("robot_xy_m", solution.xy)?;
            diagnostics.set_item("reason", "ok")?;
        }
        Err(reason) => {
            dictionary.set_item("camera_pose", py.None())?;
            dictionary.set_item("pose_meta", py.None())?;
            diagnostics.set_item("reason", reason)?;
        }
    }
    dictionary.set_item("diagnostics", diagnostics)?;
    Ok(dictionary)
}
/// Append exact/interpolated/nearest diagnostics with unchanged key names.
fn alignment_diagnostics(
    dictionary: &Bound<'_, PyDict>,
    aligned: &alignment::Alignment,
) -> PyResult<()> {
    dictionary.set_item("alignment", aligned.mode)?;
    dictionary.set_item(
        if aligned.gap {
            "gyro_gap_us"
        } else {
            "gyro_delta_us"
        },
        aligned.delta,
    )?;
    Ok(())
}
#[pymethods]
impl PnpLocalization2D {
    /// Own validated immutable calibration and mapped global corners.
    #[new]
    #[pyo3(signature=(camera_matrix, distortion_coefficients, apriltag_ids, apriltag_corners))]
    fn new(
        camera_matrix: Vec<f64>,
        distortion_coefficients: Vec<f64>,
        apriltag_ids: Vec<i64>,
        apriltag_corners: Vec<f64>,
    ) -> PyResult<Self> {
        let calibration = Calibration::new(&camera_matrix, &distortion_coefficients)
            .map_err(PyValueError::new_err)?;
        if apriltag_ids.len().checked_mul(12) != Some(apriltag_corners.len())
            || !apriltag_corners.iter().all(|value| value.is_finite())
        {
            return Err(PyValueError::new_err(
                "apriltag_corners must contain 12 finite values per ID",
            ));
        }
        let mut geometry = HashMap::new();
        for (id, coordinates) in apriltag_ids.into_iter().zip(apriltag_corners.chunks(12)) {
            let corners = std::array::from_fn(|index| {
                Vector3::new(
                    coordinates[index * 3],
                    coordinates[index * 3 + 1],
                    coordinates[index * 3 + 2],
                )
            });
            if geometry.insert(id, corners).is_some() {
                return Err(PyValueError::new_err("duplicate mapped ID"));
            }
        }
        Ok(Self {
            calibration,
            geometry,
        })
    }
    /// Align gyro, select mapped observations, and solve natively without fallback.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(capture_us, gyro_samples, detections, mounting_transform, refinement_iterations, gyro_max_gap_us, gyro_nearest_us))]
    fn solve<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = capture_timestamp)] capture_us: i64,
        gyro_samples: &Bound<'py, PyAny>,
        detections: &Bound<'py, PyAny>,
        mounting_transform: Option<Vec<f64>>,
        refinement_iterations: usize,
        gyro_max_gap_us: i64,
        gyro_nearest_us: i64,
    ) -> PyResult<Bound<'py, PyDict>> {
        if gyro_max_gap_us < 0 || gyro_nearest_us < 0 {
            return Err(PyValueError::new_err("gyro limits must be nonnegative"));
        }
        if refinement_iterations > 100 {
            return Err(PyValueError::new_err(
                "refinement_iterations must be between 0 and 100",
            ));
        }
        let diagnostics = PyDict::new(py);
        if capture_us <= 1 {
            return output(py, diagnostics, Err("missing_capture"));
        }
        diagnostics.set_item("capture_nt_us", capture_us)?;
        let aligned =
            match alignment::parse(gyro_samples, capture_us, gyro_max_gap_us, gyro_nearest_us) {
                Ok(value) => value,
                Err(reason) => return output(py, diagnostics, Err(reason)),
            };
        alignment_diagnostics(&diagnostics, &aligned)?;
        diagnostics.set_item("yaw_rad", aligned.yaw)?;
        let points = match parse_points(detections, &self.geometry) {
            Ok(value) => value,
            Err(reason) => return output(py, diagnostics, Err(reason)),
        };
        let mut condition = None;
        // All Python objects have been marshaled; immutable self and owned vectors
        // make the numerical kernel safe to run without Python's thread-state.
        let result = py.detach(|| {
            if !solver::nondegenerate(&points) {
                return Err("degenerate_geometry");
            }
            let mount = mounting_transform.as_deref().ok_or("invalid_mounting")?;
            solver::solve(
                &self.calibration,
                &points,
                aligned.yaw,
                mount,
                refinement_iterations,
                &mut condition,
            )
        });
        if let Some(value) = condition {
            diagnostics.set_item("condition_number", value)?;
        }
        output(py, diagnostics, result)
    }
}
/// Standalone native heading alignment for envelope and boundary parity checks.
#[pyfunction]
#[pyo3(signature=(samples,capture_us,max_gap_us,nearest_us))]
fn align_heading<'py>(
    py: Python<'py>,
    samples: &Bound<'py, PyAny>,
    #[pyo3(from_py_with = capture_timestamp)] capture_us: i64,
    max_gap_us: i64,
    nearest_us: i64,
) -> PyResult<(f64, Bound<'py, PyDict>)> {
    if max_gap_us < 0 || nearest_us < 0 {
        return Err(PyValueError::new_err("gyro limits must be nonnegative"));
    }
    let aligned = alignment::parse(samples, capture_us, max_gap_us, nearest_us)
        .map_err(PyValueError::new_err)?;
    let diagnostics = PyDict::new(py);
    alignment_diagnostics(&diagnostics, &aligned)?;
    Ok((aligned.yaw, diagnostics))
}
/// Register the drop-in native solver and optional boundary-test helper.
#[pymodule]
fn pnp_localization_2d(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PnpLocalization2D>()?;
    module.add_function(wrap_pyfunction!(align_heading, module)?)?;
    Ok(())
}
