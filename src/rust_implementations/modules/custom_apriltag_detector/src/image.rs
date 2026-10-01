use crate::geometry::{max_f64, min_f64, Point, Homography};

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
}

impl Image<'_> {
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
            && !(self.valid(Point {
                x: x as f64,
                y: y as f64,
            }) && self.valid(Point {
                x: xx as f64,
                y: y as f64,
            }) && self.valid(Point {
                x: x as f64,
                y: yy as f64,
            }) && self.valid(Point {
                x: xx as f64,
                y: yy as f64,
            }))
        {
            let mut value = 0.0;
            let mut weight = 0.0;
            for (row, row_weight) in [(y, 1.0 - b), (yy, b)] {
                for (col, col_weight) in [(x, 1.0 - a), (xx, a)] {
                    if self.valid(Point {
                        x: col as f64,
                        y: row as f64,
                    }) {
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
