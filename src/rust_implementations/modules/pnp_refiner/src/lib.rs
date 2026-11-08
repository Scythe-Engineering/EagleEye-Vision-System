//! Levenberg-Marquardt PnP pose refinement, a faster stand-in for OpenCV's
//! `solvePnPRefineLM` on the few dozen points of an AprilTag solve.

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod refine;

use refine::Camera;

#[pymodule]
fn pnp_refiner(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PnpRefiner>()?;
    Ok(())
}

/// Refines camera-from-object poses for one set of camera intrinsics.
#[pyclass(module = "pnp_refiner")]
struct PnpRefiner {
    camera: Camera,
}

/// Copy a C-contiguous float64 buffer, checking its element count.
fn read_f64(
    py: Python<'_>,
    array: &Bound<'_, PyAny>,
    name: &str,
    count: Option<usize>,
) -> PyResult<Vec<f64>> {
    let buffer = PyBuffer::<f64>::get(array)
        .map_err(|_| PyValueError::new_err(format!("{name} must be a float64 array")))?;
    if !buffer.is_c_contiguous() || count.is_some_and(|n| buffer.item_count() != n) {
        return Err(PyValueError::new_err(format!(
            "{name} must be C-contiguous{}",
            count.map_or(String::new(), |n| format!(" with {n} values"))
        )));
    }
    buffer.to_vec(py)
}

/// Overwrite a writable, C-contiguous float64 buffer of three values.
fn write_vec3(py: Python<'_>, array: &Bound<'_, PyAny>, values: [f64; 3]) -> PyResult<()> {
    let buffer = PyBuffer::<f64>::get(array)?;
    buffer.copy_from_slice(py, &values)
}

#[pymethods]
impl PnpRefiner {
    /// Args:
    ///     camera_matrix: 3x3 float64 intrinsics.
    ///     distortion: 0, 4, 5 or 8 float64 OpenCV distortion coefficients.
    ///
    /// Raises:
    ///     ValueError: For any other distortion model, so callers keep OpenCV.
    #[new]
    fn new(
        py: Python<'_>,
        camera_matrix: &Bound<'_, PyAny>,
        distortion: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let matrix = read_f64(py, camera_matrix, "camera_matrix", Some(9))?;
        let distortion = read_f64(py, distortion, "distortion", None)?;
        let camera = Camera::new(&matrix, &distortion).map_err(PyValueError::new_err)?;
        Ok(Self { camera })
    }

    /// Refine `rvec` and `tvec` in place, like `cv2.solvePnPRefineLM`.
    // Mirrors the OpenCV call's arguments, so the count is deliberate.
    #[allow(clippy::too_many_arguments)]
    ///
    /// Args:
    ///     object_points: (N, 3) float64 object-space points.
    ///     image_points: (N, 2) float64 observed pixels.
    ///     rvec: Writable 3-value float64 rotation vector, updated in place.
    ///     tvec: Writable 3-value float64 translation, updated in place.
    ///     max_iterations: Maximum LM steps; rejected steps count too.
    ///     epsilon: Stop once a step's norm falls below this.
    ///
    /// Returns:
    ///     False, with the pose untouched, when the starting pose puts a point
    ///     behind the camera or is not finite.
    fn refine(
        &self,
        py: Python<'_>,
        object_points: &Bound<'_, PyAny>,
        image_points: &Bound<'_, PyAny>,
        rvec: &Bound<'_, PyAny>,
        tvec: &Bound<'_, PyAny>,
        max_iterations: usize,
        epsilon: f64,
    ) -> PyResult<bool> {
        let object = read_f64(py, object_points, "object_points", None)?;
        let count = object.len() / 3;
        if object.len() % 3 != 0 || count == 0 {
            return Err(PyValueError::new_err(
                "object_points must have shape (N, 3)",
            ));
        }
        let image = read_f64(py, image_points, "image_points", Some(count * 2))?;
        let r = read_f64(py, rvec, "rvec", Some(3))?;
        let t = read_f64(py, tvec, "tvec", Some(3))?;

        let object: Vec<[f64; 3]> = (0..count)
            .map(|i| [object[3 * i], object[3 * i + 1], object[3 * i + 2]])
            .collect();
        let image: Vec<[f64; 2]> = (0..count)
            .map(|i| [image[2 * i], image[2 * i + 1]])
            .collect();
        let mut r = [r[0], r[1], r[2]];
        let mut t = [t[0], t[1], t[2]];
        if !refine::refine(
            &self.camera,
            &object,
            &image,
            &mut r,
            &mut t,
            max_iterations,
            epsilon,
        ) {
            return Ok(false);
        }
        write_vec3(py, rvec, r)?;
        write_vec3(py, tvec, t)?;
        Ok(true)
    }
}
