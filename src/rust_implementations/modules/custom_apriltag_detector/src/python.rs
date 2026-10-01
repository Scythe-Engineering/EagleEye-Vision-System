//! Native operation for the project's existing PyO3 module framework.

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;

use crate::detection::Detection;
use crate::detector::{image_span, mapped_image, Detector, Settings};
use crate::geometry::H;

#[pyclass(frozen, get_all, module = "custom_apriltag_detector")]
pub(crate) struct NativeDetection {
    family_index: u32,
    tag_id: i32,
    hamming: i32,
    rotation: i32,
    decision_margin: f64,
    corners: [f64; 8],
    center: [f64; 2],
    homography: [f64; 9],
}

impl From<Detection> for NativeDetection {
    /// Copy results into Python-owned values before the detector buffer is reused.
    fn from(detection: Detection) -> Self {
        Self {
            family_index: detection.family_index,
            tag_id: detection.tag_id,
            hamming: detection.hamming,
            rotation: detection.rotation,
            decision_margin: detection.decision_margin,
            corners: detection.corners,
            center: detection.center,
            homography: detection.homography,
        }
    }
}

#[pyclass(module = "custom_apriltag_detector")]
pub(crate) struct CustomAprilTagDetector {
    detector: Option<Detector>,
}

#[pymethods]
impl CustomAprilTagDetector {
    /// Construct the custom operation without calling another detector backend.
    #[new]
    #[pyo3(signature = (families="tag36h11", nthreads=1, quad_decimate=2.0, quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25))]
    fn new(
        families: &str,
        nthreads: usize,
        quad_decimate: f64,
        quad_sigma: f64,
        refine_edges: i32,
        decode_sharpening: f64,
    ) -> PyResult<Self> {
        if !matches!(refine_edges, 0 | 1) {
            return Err(PyValueError::new_err("refine_edges must be 0 or 1"));
        }
        let settings = Settings {
            nthreads,
            refine_edges: refine_edges != 0,
            quad_decimate,
            quad_sigma,
            decode_sharpening,
        };
        let detector = Detector::new(families, settings).map_err(PyValueError::new_err)?;
        Ok(Self {
            detector: Some(detector),
        })
    }

    /// Detect a two-dimensional uint8 buffer, releasing the GIL during native work.
    ///
    /// The exporter keeps pixels alive through the call. As with the existing
    /// ctypes adapter, callers must not mutate the pixels while detection runs.
    #[pyo3(signature = (gray, *, source_map=None, source_shape=None))]
    fn run(
        &mut self,
        py: Python<'_>,
        gray: &Bound<'_, PyAny>,
        source_map: Option<Vec<f64>>,
        source_shape: Option<(u32, u32)>,
    ) -> PyResult<Vec<NativeDetection>> {
        let detector = self
            .detector
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Eagle Tags detector is closed"))?;
        let buffer = PyBuffer::<u8>::get(gray)
            .map_err(|_| PyValueError::new_err("Eagle Tags requires a 2D uint8 grayscale image"))?;
        if buffer.dimensions() != 2
            || buffer.suboffsets().is_some()
            || buffer.strides()[1] != 1
            || buffer.strides()[0] < 0
        {
            return Err(PyValueError::new_err(
                "Eagle Tags requires positive row strides and contiguous pixels",
            ));
        }
        let height = buffer.shape()[0];
        let width = buffer.shape()[1];
        let stride = buffer.strides()[0] as usize;
        let span = image_span(width, height, stride).map_err(PyRuntimeError::new_err)?;
        let mapping: Option<H> = source_map
            .map(|map| {
                map.try_into()
                    .map_err(|_| PyValueError::new_err("source_map must have nine elements"))
            })
            .transpose()?;
        // SAFETY: The buffer exporter owns these validated positive-stride rows.
        // The buffer is kept alive until detached detection has completed. This
        // uses the same no-concurrent-mutation contract as the existing C ABI.
        let pixels = unsafe { slice::from_raw_parts(buffer.buf_ptr().cast::<u8>(), span) };
        let image = mapped_image(pixels, width, height, stride, mapping, source_shape)
            .map_err(PyRuntimeError::new_err)?;
        let detected = py
            .detach(|| catch_unwind(AssertUnwindSafe(|| detector.detect(image).to_vec())))
            .map_err(|_| PyRuntimeError::new_err("native detector panicked"))?;
        Ok(detected.into_iter().map(NativeDetection::from).collect())
    }

    /// Release workers and state exactly once; subsequent detection fails explicitly.
    fn close(&mut self) {
        self.detector = None;
    }
}

/// Return the actual extension file, including when maturin wraps it in a package.
#[pyfunction]
#[pyo3(pass_module)]
pub(crate) fn native_library_path(module: &Bound<'_, PyModule>) -> PyResult<String> {
    module.getattr("__file__")?.extract()
}
