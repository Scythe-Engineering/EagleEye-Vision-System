//! Borrowed full camera frames, read as grayscale one window at a time.

use crate::detector::mapped_image;
use crate::geometry::{project, Homography};
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

    /// Grayscale at an integer pixel, or black outside the frame like OpenCV's
    /// constant border.
    #[inline(always)]
    fn gray_at(&self, x: i64, y: i64) -> f32 {
        if x < 0 || y < 0 || x >= self.width as i64 || y >= self.height as i64 {
            return 0.0;
        }
        let i = y as usize * self.stride + x as usize * self.channels;
        f32::from(if self.channels == 1 {
            self.pixels[i]
        } else {
            gray(self.pixels[i], self.pixels[i + 1], self.pixels[i + 2])
        })
    }

    /// Bilinearly resample a `side` x `side` grayscale crop into `out`, where crop
    /// pixel centre `(u, v)` samples frame point `frame_from_crop * (u, v, 1)`.
    pub(crate) fn warp_gray(&self, side: usize, frame_from_crop: &Homography, out: &mut Vec<u8>) {
        out.clear();
        out.reserve(side * side);
        for v in 0..side {
            for u in 0..side {
                let p = project(frame_from_crop, u as f64, v as f64);
                if !(p.x.is_finite() && p.y.is_finite()) {
                    out.push(0);
                    continue;
                }
                let (x, y) = (p.x.floor(), p.y.floor());
                let (a, b) = ((p.x - x) as f32, (p.y - y) as f32);
                let (x, y) = (x as i64, y as i64);
                let top = (1.0 - a) * self.gray_at(x, y) + a * self.gray_at(x + 1, y);
                let bottom = (1.0 - a) * self.gray_at(x, y + 1) + a * self.gray_at(x + 1, y + 1);
                out.push(((1.0 - b) * top + b * bottom + 0.5) as u8);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{gray, Frame};

    #[test]
    fn warp_samples_shifted_pixels_and_blacks_out_the_border() {
        let (width, height) = (6, 4);
        let pixels: Vec<u8> = (0..width * height).map(|i| (i * 10) as u8).collect();
        let frame = Frame {
            pixels: &pixels,
            width,
            height,
            stride: width,
            channels: 1,
        };
        // Crop (u, v) reads frame (u + 1.5, v + 1).
        let shift = [1.0, 0.0, 1.5, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let mut out = Vec::new();
        frame.warp_gray(5, &shift, &mut out);
        let at = |x: usize, y: usize| f32::from(pixels[y * width + x]);
        assert_eq!(out[0], ((at(1, 1) + at(2, 1)) / 2.0).round() as u8);
        // (5.5, 1) mixes the last column with the black border.
        assert_eq!(out[4], (at(5, 1) / 2.0).round() as u8);
        assert_eq!(out[5 * 4], 0);
    }

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
