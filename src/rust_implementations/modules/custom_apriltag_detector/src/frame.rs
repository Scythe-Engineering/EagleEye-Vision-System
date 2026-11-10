//! Borrowed full camera frames, read as grayscale one window at a time.

use crate::detector::mapped_image;
use crate::image::Image;

/// A grayscale or packed BGR uint8 frame with positive row strides.
#[derive(Clone, Copy)]
pub(crate) struct Frame<'a> {
    pub(crate) pixels: &'a [u8],
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) stride: usize,
    /// 1 for grayscale, 3 for BGR.
    pub(crate) channels: usize,
}

/// OpenCV's 8-bit BGR2GRAY: 15-bit fixed-point weights summing to 1 << 15.
#[inline(always)]
fn gray(b: u8, g: u8, r: u8) -> u8 {
    ((u32::from(b) * 3735 + u32::from(g) * 19235 + u32::from(r) * 9798 + (1 << 14)) >> 15) as u8
}

impl<'a> Frame<'a> {
    /// Grayscale view of `[x0, x1) x [y0, y1)`.
    ///
    /// Grayscale frames are borrowed in place; BGR windows are converted into
    /// `scratch` with OpenCV's exact BGR2GRAY arithmetic, touching only the window.
    pub(crate) fn gray_window<'b>(
        &self,
        (x0, y0, x1, y1): (usize, usize, usize, usize),
        scratch: &'b mut Vec<u8>,
    ) -> Result<Image<'b>, String>
    where
        'a: 'b,
    {
        let (width, height) = (x1 - x0, y1 - y0);
        if self.channels == 1 {
            let start = y0 * self.stride + x0;
            return mapped_image(
                &self.pixels[start..],
                width,
                height,
                self.stride,
                None,
                None,
            );
        }
        scratch.clear();
        scratch.reserve(width * height);
        for y in y0..y1 {
            let row = &self.pixels[y * self.stride + 3 * x0..y * self.stride + 3 * x1];
            let (row, _) = row.as_chunks::<3>();
            scratch.extend(row.iter().map(|&[b, g, r]| gray(b, g, r)));
        }
        mapped_image(scratch, width, height, width, None, None)
    }
}

#[cfg(test)]
mod tests {
    use super::{gray, Frame};

    #[test]
    fn windows_match_opencv_weights_and_bounds() {
        let (width, height, stride) = (7, 5, 24);
        let pixels: Vec<u8> = (0..stride * height).map(|i| (i * 89 % 256) as u8).collect();
        let frame = Frame {
            pixels: &pixels,
            width,
            height,
            stride,
            channels: 3,
        };
        let mut scratch = Vec::new();
        let image = frame.gray_window((2, 1, 6, 4), &mut scratch).unwrap();
        assert_eq!((image.width, image.height), (4, 3));
        let p = |x: usize, y: usize, c: usize| pixels[y * stride + 3 * x + c];
        assert_eq!(
            image.pixels[4 + 1],
            gray(p(3, 2, 0), p(3, 2, 1), p(3, 2, 2))
        );
        // Pure channels and white, from cv2.cvtColor(..., COLOR_BGR2GRAY).
        assert_eq!(
            [
                gray(255, 0, 0),
                gray(0, 255, 0),
                gray(0, 0, 255),
                gray(255, 255, 255)
            ],
            [29, 150, 76, 255]
        );
    }
}
