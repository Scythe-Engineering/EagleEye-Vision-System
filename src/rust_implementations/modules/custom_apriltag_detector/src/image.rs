use crate::geometry::{max_f64, min_f64, Homography, Point};

/// Borrowed grayscale pixels with an optional integer-center physical-source map.
#[derive(Clone, Copy)]
pub(crate) struct Image<'a> {
    pub(crate) pixels: &'a [u8],
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) stride: usize,
    pub(crate) source_map: Option<Homography>,
    pub(crate) source_width: u32,
    pub(crate) source_height: u32,
    /// Per-row `[first, end)` integer pixels that `valid` accepts, when mapped.
    pub(crate) row_spans: Option<&'a [(usize, usize)]>,
}

impl Image<'_> {
    /// `valid` for an integer pixel, using precomputed row spans when available.
    #[inline(always)]
    pub(crate) fn valid_pixel(&self, x: usize, y: usize) -> bool {
        match self.row_spans {
            Some(spans) => {
                let (first, end) = spans[y];
                x >= first && x < end
            }
            None => self.valid(Point {
                x: x as f64,
                y: y as f64,
            }),
        }
    }

    /// Fill per-row spans of valid integer pixels. Returns false when a row's valid
    /// pixels are not contiguous (a projective pole crosses it) or nothing is mapped.
    pub(crate) fn compute_row_spans(&self, spans: &mut Vec<(usize, usize)>) -> bool {
        spans.clear();
        let Some(m) = self.source_map else {
            return false;
        };
        let width = self.width;
        let max_x = (self.source_width - 1) as f64;
        let max_y = (self.source_height - 1) as f64;
        for y in 0..self.height {
            let yf = y as f64;
            let ok = |x: usize| self.valid(Point { x: x as f64, y: yf });
            let cz = m[7].mul_add(yf, m[8]);
            let z_last = m[6].mul_add((width - 1) as f64, cz);
            let linear = cz.is_finite()
                && z_last.is_finite()
                && cz != 0.0
                && z_last != 0.0
                && (cz > 0.0) == (z_last > 0.0);
            if !linear {
                // Rare: scan the row and give up on spans if it is not one interval.
                let first = (0..width).find(|&x| ok(x)).unwrap_or(width);
                let end = (first..width).find(|&x| !ok(x)).unwrap_or(width);
                if (end..width).any(ok) {
                    spans.clear();
                    return false;
                }
                spans.push((first, end));
                continue;
            }
            // With z of fixed sign, each source bound is a half-line in x.
            let sign = if cz > 0.0 { 1.0 } else { -1.0 };
            let cx = m[1].mul_add(yf, m[2]);
            let cy = m[4].mul_add(yf, m[5]);
            let (mut lo, mut hi) = (0.0f64, (width - 1) as f64);
            for (a, b) in [
                (m[0], cx),
                (max_x * m[6] - m[0], max_x * cz - cx),
                (m[3], cy),
                (max_y * m[6] - m[3], max_y * cz - cy),
            ] {
                let (a, b) = (sign * a, sign * b);
                if a > 0.0 {
                    lo = lo.max(-b / a);
                } else if a < 0.0 {
                    hi = hi.min(-b / a);
                } else if b < 0.0 {
                    lo = f64::INFINITY;
                }
            }
            // Snap to integers, then confirm the boundaries with the exact test.
            let mut first = if lo.is_finite() && lo <= hi {
                (lo.ceil().max(0.0) as usize).min(width)
            } else {
                width
            };
            while first > 0 && ok(first - 1) {
                first -= 1;
            }
            while first < width && !ok(first) {
                first += 1;
            }
            let mut end = if first < width && hi.is_finite() {
                ((hi.floor().max(0.0) as usize) + 1).clamp(first, width)
            } else {
                first
            };
            while end < width && ok(end) {
                end += 1;
            }
            while end > first && !ok(end - 1) {
                end -= 1;
            }
            spans.push((first, end));
        }
        true
    }

    /// Test crop bounds and, when mapped, availability in the physical source.
    pub(crate) fn valid(&self, q: Point) -> bool {
        if !q.x.is_finite()
            || !q.y.is_finite()
            || q.x < 0.0
            || q.y < 0.0
            || q.x > (self.width - 1) as f64
            || q.y > (self.height - 1) as f64
        {
            return false;
        }
        let Some(m) = self.source_map else {
            return true;
        };
        let z = m[6].mul_add(q.x, m[7] * q.y) + m[8];
        let x = (m[0].mul_add(q.x, m[1] * q.y) + m[2]) / z;
        let y = (m[3].mul_add(q.x, m[4] * q.y) + m[5]) / z;
        x.is_finite()
            && y.is_finite()
            && x >= 0.0
            && y >= 0.0
            && x <= (self.source_width - 1) as f64
            && y <= (self.source_height - 1) as f64
    }

    /// Check all four physical corners and exclude projective poles inside the crop.
    pub(crate) fn fully_observed(&self) -> bool {
        let Some(m) = self.source_map else {
            return true;
        };
        let mut sign = 0;
        for q in [
            Point { x: 0.0, y: 0.0 },
            Point {
                x: (self.width - 1) as f64,
                y: 0.0,
            },
            Point {
                x: (self.width - 1) as f64,
                y: (self.height - 1) as f64,
            },
            Point {
                x: 0.0,
                y: (self.height - 1) as f64,
            },
        ] {
            let z = m[6].mul_add(q.x, m[7] * q.y) + m[8];
            if !z.is_finite() || z == 0.0 || !self.valid(q) {
                return false;
            }
            let current = if z > 0.0 { 1 } else { -1 };
            if sign != 0 && sign != current {
                return false;
            }
            sign = current;
        }
        true
    }

    /// Limit a unit ray segment to the observed crop and physical-source footprint.
    pub(crate) fn ray_limit(&self, edge: Point, delta: Point) -> f64 {
        if !self.valid(edge) {
            return 0.0;
        }
        let mut t = 1.0;
        let mut bound = |a: f64, b: f64| {
            if b < 0.0 {
                t = min_f64(t, -a / b);
            }
        };
        bound(edge.x, delta.x);
        bound((self.width - 1) as f64 - edge.x, -delta.x);
        bound(edge.y, delta.y);
        bound((self.height - 1) as f64 - edge.y, -delta.y);
        if let Some(m) = self.source_map {
            let z = m[6].mul_add(edge.x, m[7] * edge.y) + m[8];
            let dz = m[6].mul_add(delta.x, m[7] * delta.y);
            let sign = if z < 0.0 { -1.0 } else { 1.0 };
            bound(sign * z, sign * dz);
            for axis in 0..2 {
                let i = axis * 3;
                let v = m[i].mul_add(edge.x, m[i + 1] * edge.y) + m[i + 2];
                let dv = m[i].mul_add(delta.x, m[i + 1] * delta.y);
                let maximum = (if axis != 0 {
                    self.source_height
                } else {
                    self.source_width
                } - 1) as f64;
                bound(sign * v, sign * dv);
                bound(sign * (maximum * z - v), sign * (maximum * dz - dv));
            }
        }
        max_f64(0.0, t)
    }

    /// Whether every point of segment `a`-`b` lies in the unmapped interior, where
    /// `sample_interior` gives exactly what `sample` gives. Both ends must be finite
    /// and at least one pixel inside the right and bottom edges; the segment is then too.
    pub(crate) fn interior_segment(&self, a: Point, b: Point) -> bool {
        let inside = |q: Point| {
            q.x >= 0.0
                && q.y >= 0.0
                && q.x <= (self.width - 2) as f64
                && q.y <= (self.height - 2) as f64
        };
        self.source_map.is_none() && inside(a) && inside(b)
    }

    /// `sample` for a point known to be interior (see `interior_segment`): no validity
    /// test, and the neighbour clamps are no-ops there, so the arithmetic is identical.
    #[inline(always)]
    pub(crate) fn sample_interior(&self, q: Point) -> f64 {
        debug_assert!(self.interior_segment(q, q));
        let x = q.x as usize;
        let y = q.y as usize;
        let a = q.x - x as f64;
        let b = q.y - y as f64;
        let top = &self.pixels[y * self.stride + x..y * self.stride + x + 2];
        let bottom = &self.pixels[(y + 1) * self.stride + x..(y + 1) * self.stride + x + 2];
        let top = (1.0 - a).mul_add(top[0] as f64, a * top[1] as f64);
        let bottom = (1.0 - a).mul_add(bottom[0] as f64, a * bottom[1] as f64);
        (1.0 - b).mul_add(top, b * bottom)
    }

    /// Bilinearly sample, excluding and renormalizing unavailable physical contributors.
    #[inline(always)]
    pub(crate) fn sample(&self, q: Point) -> f64 {
        if !self.valid(q) {
            return f64::NAN;
        }
        let x = q.x as usize;
        let y = q.y as usize;
        let xx = (x + 1).min(self.width - 1);
        let yy = (y + 1).min(self.height - 1);
        let a = q.x - x as f64;
        let b = q.y - y as f64;
        if self.source_map.is_some()
            && !(self.valid_pixel(x, y)
                && self.valid_pixel(xx, y)
                && self.valid_pixel(x, yy)
                && self.valid_pixel(xx, yy))
        {
            let mut value = 0.0;
            let mut weight = 0.0;
            for (row, row_weight) in [(y, 1.0 - b), (yy, b)] {
                for (col, col_weight) in [(x, 1.0 - a), (xx, a)] {
                    if self.valid_pixel(col, row) {
                        let t = row_weight * col_weight;
                        value = t.mul_add(self.pixels[row * self.stride + col] as f64, value);
                        weight += t;
                    }
                }
            }
            return if weight > 0.0 {
                value / weight
            } else {
                f64::NAN
            };
        }
        let top = (1.0 - a).mul_add(
            self.pixels[y * self.stride + x] as f64,
            a * self.pixels[y * self.stride + xx] as f64,
        );
        let bottom = (1.0 - a).mul_add(
            self.pixels[yy * self.stride + x] as f64,
            a * self.pixels[yy * self.stride + xx] as f64,
        );
        (1.0 - b).mul_add(top, b * bottom)
    }
}

#[cfg(test)]
mod tests {
    use crate::detector::mapped_image;
    use crate::geometry::Point;

    #[test]
    fn interior_samples_match_checked_samples() {
        let (width, height, stride) = (37, 23, 40);
        let pixels: Vec<u8> = (0..stride * height).map(|i| (i * 97 % 251) as u8).collect();
        let image = mapped_image(&pixels, width, height, stride, None, None).unwrap();
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut unit = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..20_000 {
            // Include the exact interior bounds as well as random points.
            let q = Point {
                x: if unit() < 0.05 {
                    (width - 2) as f64
                } else {
                    unit() * (width - 2) as f64
                },
                y: if unit() < 0.05 {
                    0.0
                } else {
                    unit() * (height - 2) as f64
                },
            };
            assert!(image.interior_segment(q, q));
            assert_eq!(
                image.sample_interior(q).to_bits(),
                image.sample(q).to_bits()
            );
        }
        let edge = Point {
            x: (width - 1) as f64,
            y: 1.0,
        };
        assert!(!image.interior_segment(edge, edge));
    }
}
