use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;

use crate::detection::Detection;
use crate::detector::{image_span, mapped_image, Detector, Settings};
use crate::geometry::Homography;

#[pyclass(module = "custom_apriltag_detector")]
pub(crate) struct CustomAprilTagDetector {
    detector: Detector,
}

#[pymethods]
impl CustomAprilTagDetector {
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
        Ok(Self { detector })
    }

    /// Detect tags in a 2D uint8 buffer, releasing the GIL during native work.
    ///
    /// Callers must not mutate the pixels while detection runs.
    #[pyo3(signature = (gray, *, source_map=None, source_shape=None))]
    fn run(
        &mut self,
        py: Python<'_>,
        gray: &Bound<'_, PyAny>,
        source_map: Option<Vec<f64>>,
        source_shape: Option<(u32, u32)>,
    ) -> PyResult<Vec<Detection>> {
        let buffer = PyBuffer::<u8>::get(gray)
            .map_err(|_| PyValueError::new_err("detector requires a 2D uint8 grayscale image"))?;
        if buffer.dimensions() != 2
            || buffer.suboffsets().is_some()
            || buffer.strides()[1] != 1
            || buffer.strides()[0] < 0
        {
            return Err(PyValueError::new_err(
                "detector requires positive row strides and contiguous pixels",
            ));
        }
        let height = buffer.shape()[0];
        let width = buffer.shape()[1];
        let stride = buffer.strides()[0] as usize;
        let span = image_span(width, height, stride).map_err(PyValueError::new_err)?;
        let mapping: Option<Homography> = source_map
            .map(|map| {
                map.try_into()
                    .map_err(|_| PyValueError::new_err("source_map must have nine elements"))
            })
            .transpose()?;
        // SAFETY: The buffer exporter owns these validated positive-stride rows and
        // stays alive until detached detection has completed.
        let pixels = unsafe { slice::from_raw_parts(buffer.buf_ptr().cast::<u8>(), span) };
        let image = mapped_image(pixels, width, height, stride, mapping, source_shape)
            .map_err(PyValueError::new_err)?;
        let detector = &mut self.detector;
        py.detach(|| catch_unwind(AssertUnwindSafe(|| detector.detect(image).to_vec())))
            .map_err(|_| PyRuntimeError::new_err("native detector panicked"))
    }
}
