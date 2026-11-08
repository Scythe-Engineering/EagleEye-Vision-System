use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;

use crate::detection::Detection;
use crate::detector::{image_span, mapped_image, Detector, Settings};
use crate::frame::Frame;
use crate::geometry::{Homography, Quad};
use crate::image::Image;
use crate::track::{decode_quad as quad_from_corners, track_in_frame};

#[pyclass(module = "custom_apriltag_detector")]
pub(crate) struct CustomAprilTagDetector {
    detector: Detector,
}

/// A validated 2D uint8 buffer: positive row strides and contiguous pixels.
struct GrayBuffer {
    buffer: PyBuffer<u8>,
    width: usize,
    height: usize,
    stride: usize,
    span: usize,
}

impl GrayBuffer {
    fn new(gray: &Bound<'_, PyAny>) -> PyResult<Self> {
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
        Ok(Self {
            buffer,
            width,
            height,
            stride,
            span,
        })
    }

    fn image(
        &self,
        mapping: Option<Homography>,
        source_shape: Option<(u32, u32)>,
    ) -> PyResult<Image<'_>> {
        // SAFETY: The buffer exporter owns these validated positive-stride rows and
        // stays alive while `self` holds the buffer.
        let pixels =
            unsafe { slice::from_raw_parts(self.buffer.buf_ptr().cast::<u8>(), self.span) };
        mapped_image(
            pixels,
            self.width,
            self.height,
            self.stride,
            mapping,
            source_shape,
        )
        .map_err(PyValueError::new_err)
    }
}

/// A validated 2D grayscale or 3D packed-BGR uint8 frame with positive row strides.
struct FrameBuffer {
    buffer: PyBuffer<u8>,
    width: usize,
    height: usize,
    stride: usize,
    channels: usize,
    span: usize,
}

impl FrameBuffer {
    fn new(frame: &Bound<'_, PyAny>) -> PyResult<Self> {
        let invalid =
            || PyValueError::new_err("frame must be a 2D grayscale or 3D BGR uint8 image");
        let buffer = PyBuffer::<u8>::get(frame).map_err(|_| invalid())?;
        let (shape, strides) = (buffer.shape(), buffer.strides());
        let channels = match buffer.dimensions() {
            2 => 1,
            3 if shape[2] == 3 && strides[2] == 1 => 3,
            _ => return Err(invalid()),
        };
        if buffer.suboffsets().is_some() || strides[1] != channels as isize || strides[0] < 0 {
            return Err(PyValueError::new_err(
                "frame requires positive row strides and packed pixels",
            ));
        }
        let (height, width, stride) = (shape[0], shape[1], strides[0] as usize);
        // Validate the frame as a `channels`-times wider grayscale image.
        let span = image_span(width * channels, height, stride).map_err(PyValueError::new_err)?;
        Ok(Self {
            buffer,
            width,
            height,
            stride,
            channels,
            span,
        })
    }

    fn frame(&self) -> Frame<'_> {
        // SAFETY: The buffer exporter owns these validated positive-stride rows and
        // stays alive while `self` holds the buffer.
        let pixels =
            unsafe { slice::from_raw_parts(self.buffer.buf_ptr().cast::<u8>(), self.span) };
        Frame {
            pixels,
            width: self.width,
            height: self.height,
            stride: self.stride,
            channels: self.channels,
        }
    }
}

/// Convert eight output-order pixel-edge corner values into a decode-order quad.
fn decode_quad(corners: &[f64]) -> PyResult<Quad> {
    let corners: &[f64; 8] = corners
        .try_into()
        .ok()
        .filter(|values: &&[f64; 8]| values.iter().all(|value| value.is_finite()))
        .ok_or_else(|| PyValueError::new_err("corners must be eight finite values"))?;
    Ok(quad_from_corners(corners))
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
        let mapping: Option<Homography> = source_map
            .map(|map| {
                map.try_into()
                    .map_err(|_| PyValueError::new_err("source_map must have nine elements"))
            })
            .transpose()?;
        let gray = GrayBuffer::new(gray)?;
        let image = gray.image(mapping, source_shape)?;
        let detector = &mut self.detector;
        py.detach(|| catch_unwind(AssertUnwindSafe(|| detector.detect(image).to_vec())))
            .map_err(|_| PyRuntimeError::new_err("native detector panicked"))
    }

    /// Re-decode and grid-fit corners from another detector.
    ///
    /// `corners` holds eight values in pupil_apriltags order and pixel-edge
    /// coordinates. Returns None when the quad does not decode as the first family.
    #[pyo3(signature = (gray, corners))]
    fn refine(
        &mut self,
        py: Python<'_>,
        gray: &Bound<'_, PyAny>,
        corners: Vec<f64>,
    ) -> PyResult<Option<Detection>> {
        let quad = decode_quad(&corners)?;
        let gray = GrayBuffer::new(gray)?;
        let image = gray.image(None, None)?;
        let detector = &self.detector;
        py.detach(|| {
            catch_unwind(AssertUnwindSafe(|| {
                crate::decode::decode(
                    &quad,
                    &image,
                    &detector.selected[0],
                    0,
                    detector.settings.decode_sharpening,
                    true,
                )
            }))
        })
        .map_err(|_| PyRuntimeError::new_err("native refine panicked"))
    }

    /// Re-find a tracked tag of the first family from predicted corners.
    ///
    /// `corners` holds the predicted position in pupil_apriltags order and
    /// pixel-edge coordinates. Returns None unless `tag_id` decodes there.
    #[pyo3(signature = (gray, corners, tag_id))]
    fn track(
        &mut self,
        py: Python<'_>,
        gray: &Bound<'_, PyAny>,
        corners: Vec<f64>,
        tag_id: i32,
    ) -> PyResult<Option<Detection>> {
        let quad = decode_quad(&corners)?;
        let gray = GrayBuffer::new(gray)?;
        let image = gray.image(None, None)?;
        let detector = &self.detector;
        py.detach(|| {
            catch_unwind(AssertUnwindSafe(|| {
                crate::track::track(
                    &quad,
                    &image,
                    &detector.selected[0],
                    0,
                    detector.settings.decode_sharpening,
                    tag_id,
                )
            }))
        })
        .map_err(|_| PyRuntimeError::new_err("native track panicked"))
    }

    /// Re-find tracked tags of the first family around predicted full-frame corners.
    ///
    /// `predictions` holds eight OpenCV pixel-centre values per tag, in
    /// pupil_apriltags corner order, matching `tag_ids`. Each tag is confirmed in a
    /// grayscale window of `frame` (2D grayscale or packed BGR) under one GIL release,
    /// spread over the detector's `nthreads` workers when there are several.
    /// Returns per tag its full-frame pixel-centre corners, or None when unconfirmed.
    #[pyo3(signature = (frame, predictions, tag_ids))]
    fn track_many(
        &mut self,
        py: Python<'_>,
        frame: &Bound<'_, PyAny>,
        predictions: Vec<f64>,
        tag_ids: Vec<i32>,
    ) -> PyResult<Vec<Option<[f64; 8]>>> {
        if predictions.len() != 8 * tag_ids.len() {
            return Err(PyValueError::new_err(
                "predictions must hold eight values per tag ID",
            ));
        }
        let frame = FrameBuffer::new(frame)?;
        let frame = frame.frame();
        let detector = &self.detector;
        py.detach(|| {
            catch_unwind(AssertUnwindSafe(|| {
                let (predictions, _) = predictions.as_chunks::<8>();
                let items: Vec<_> = predictions.iter().zip(&tag_ids).collect();
                // Tags are independent, so the result does not depend on the thread count.
                detector.workers.map(
                    &items,
                    items.len() > 1,
                    |scratch: &mut Vec<u8>, &(predicted, &tag_id)| {
                        track_in_frame(
                            &frame,
                            predicted,
                            &detector.selected[0],
                            0,
                            detector.settings.decode_sharpening,
                            tag_id,
                            scratch,
                        )
                    },
                )
            }))
        })
        .map_err(|_| PyRuntimeError::new_err("native track panicked"))
    }
}
