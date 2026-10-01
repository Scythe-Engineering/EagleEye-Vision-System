use crate::image::Image;
use std::ops::{Add, Mul, Sub};

/// Double-precision image-space point or vector.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Point {
    pub(crate) x: f64,
    pub(crate) y: f64,
}

impl Add for Point {
    type Output = Self;
    /// Add vector components in native arithmetic order.
    fn add(self, b: Self) -> Self {
        Self {
            x: self.x + b.x,
            y: self.y + b.y,
        }
    }
}
impl Sub for Point {
    type Output = Self;
    /// Subtract vector components in native arithmetic order.
    fn sub(self, b: Self) -> Self {
        Self {
            x: self.x - b.x,
            y: self.y - b.y,
        }
    }
}
impl Mul<f64> for Point {
    type Output = Self;
    /// Scale both vector components.
    fn mul(self, s: f64) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
        }
    }
}

pub(crate) type Quad = [Point; 4];
pub(crate) type Homography = [f64; 9];

/// C++ std::min semantics, including returning the first operand on unordered comparison.
pub(crate) fn min_f64(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}
/// C++ std::max semantics, including returning the first operand on unordered comparison.
pub(crate) fn max_f64(a: f64, b: f64) -> f64 {
    if a < b {
        b
    } else {
        a
    }
}
/// Signed two-dimensional cross product.
pub(crate) fn cross(a: Point, b: Point) -> f64 {
    a.x * b.y - a.y * b.x
}
/// Two-dimensional dot product.
pub(crate) fn dot(a: Point, b: Point) -> f64 {
    a.x * b.x + a.y * b.y
}
/// Euclidean vector length without fused arithmetic.
pub(crate) fn norm(a: Point) -> f64 {
    dot(a, a).sqrt()
}

/// Direct unit-square to quadrilateral homography solve.
pub(crate) fn homography(q: &Quad) -> Option<Homography> {
    let dx = q[0].x - q[1].x + q[2].x - q[3].x;
    let dy = q[0].y - q[1].y + q[2].y - q[3].y;
    let a = q[1].x - q[2].x;
    let b = q[3].x - q[2].x;
    let c = q[1].y - q[2].y;
    let d = q[3].y - q[2].y;
    let det = a * d - b * c;
    if !det.is_finite() || det.abs() < 1e-9 {
        return None;
    }
    let g = (dx * d - b * dy) / det;
    let h = (a * dy - dx * c) / det;
    Some([
        q[1].x - q[0].x + g * q[1].x,
        q[3].x - q[0].x + h * q[3].x,
        q[0].x,
        q[1].y - q[0].y + g * q[1].y,
        q[3].y - q[0].y + h * q[3].y,
        q[0].y,
        g,
        h,
        1.0,
    ])
}

/// Project coordinates through a homography, retaining nonfinite pole results.
pub(crate) fn project(h: &Homography, x: f64, y: f64) -> Point {
    let z = h[6].mul_add(x, h[7] * y) + h[8];
    Point {
        x: (h[0].mul_add(x, h[1] * y) + h[2]) / z,
        y: (h[3].mul_add(x, h[4] * y) + h[5]) / z,
    }
}

/// Monotone-chain hull of finite points, with exact duplicate removal and native traversal.
pub(crate) fn hull(mut p: Vec<Point>) -> Vec<Point> {
    p.sort_unstable_by(|a, b| {
        a.x.partial_cmp(&b.x)
            .unwrap()
            .then_with(|| a.y.partial_cmp(&b.y).unwrap())
    });
    p.dedup_by(|a, b| a.x == b.x && a.y == b.y);
    if p.len() < 4 {
        return Vec::new();
    }
    let mut h = vec![Point::default(); 2 * p.len()];
    let mut n = 0;
    for &a in &p {
        while n >= 2 && cross(h[n - 1] - h[n - 2], a - h[n - 1]) <= 0.0 {
            n -= 1;
        }
        h[n] = a;
        n += 1;
    }
    let lower = n;
    // Native post-decrement traversal skips the last point, but includes the first.
    for i in (0..p.len() - 1).rev() {
        let a = p[i];
        while n > lower && cross(h[n - 1] - h[n - 2], a - h[n - 1]) <= 0.0 {
            n -= 1;
        }
        h[n] = a;
        n += 1;
    }
    h.truncate(n - 1);
    h
}

/// Normal-form supporting line: dot(n, point) = d.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Line {
    pub(crate) n: Point,
    pub(crate) d: f64,
}

/// Weighted covariance line fit, using independent native sin and cos evaluations.
pub(crate) fn fit(points: &[Point], weights: &[f64]) -> Option<Line> {
    if points.len() < 3 {
        return None;
    }
    let mut sum = 0.0;
    let mut mean = Point { x: 0.0, y: 0.0 };
    for i in 0..points.len() {
        sum += weights[i];
        mean.x = points[i].x.mul_add(weights[i], mean.x);
        mean.y = points[i].y.mul_add(weights[i], mean.y);
    }
    if sum <= 0.0 {
        return None;
    }
    mean = mean * (1.0 / sum);
    let mut xx = 0.0;
    let mut xy = 0.0;
    let mut yy = 0.0;
    for i in 0..points.len() {
        let a = points[i] - mean;
        xx = (weights[i] * a.x).mul_add(a.x, xx);
        xy = (weights[i] * a.x).mul_add(a.y, xy);
        yy = (weights[i] * a.y).mul_add(a.y, yy);
    }
    if xx + yy < 1e-8 {
        return None;
    }
    let angle = 0.5 * (2.0 * xy).atan2(xx - yy);
    let n = Point {
        x: -angle.sin(),
        y: angle.cos(),
    };
    Some(Line { n, d: dot(n, mean) })
}

/// Intersect nonparallel lines, rejecting nonfinite intersections.
pub(crate) fn intersect(a: Line, b: Line) -> Option<Point> {
    let det = cross(a.n, b.n);
    if det.abs() < 1e-5 {
        return None;
    }
    let p = Point {
        x: (a.d * b.n.y - a.n.y * b.d) / det,
        y: (a.n.x * b.d - a.d * b.n.x) / det,
    };
    if p.x.is_finite() && p.y.is_finite() {
        Some(p)
    } else {
        None
    }
}

/// Refine all four edges from observed gradient lobes; commit only a bounded verified quad.
pub(crate) fn refine(q: &mut Quad, im: &Image<'_>, radius: f64, polarity: i32) -> bool {
    let mut lines = [Line {
        n: Point::default(),
        d: 0.0,
    }; 4];
    let mut strength = [0.0; 4];
    for k in 0..4 {
        let delta = q[(k + 1) % 4] - q[k];
        let len = norm(delta);
        if len < 4.0 {
            return false;
        }
        let n = Point {
            x: delta.y / len,
            y: -delta.x / len,
        };
        let mut points = Vec::new();
        let mut weights = Vec::new();
        let steps = (len / 2.0) as i32;
        let steps = 6.max(128.min(steps));
        // Contract the frozen C++ point multiply-adds only in this sampling loop.
        for j in 0..steps {
            let phase = (j as f64 + 0.5) / steps as f64;
            let p = Point {
                x: delta.x.mul_add(phase, q[k].x),
                y: delta.y.mul_add(phase, q[k].y),
            };
            let mut best = 4.0;
            let mut offset = 0.0;
            let mut s = -radius;
            while s <= radius {
                let a = Point {
                    x: n.x.mul_add(s, p.x),
                    y: n.y.mul_add(s, p.y),
                };
                let g = polarity as f64
                    * (im.sample(Point {
                        x: n.x.mul_add(0.75, a.x),
                        y: n.y.mul_add(0.75, a.y),
                    }) - im.sample(Point {
                        x: (-n.x).mul_add(0.75, a.x),
                        y: (-n.y).mul_add(0.75, a.y),
                    }));
                if g > best {
                    best = g;
                    offset = s;
                }
                s += 0.5;
            }
            if best <= 4.0 {
                continue;
            }
            let mut mass = 0.0;
            let mut moment = 0.0;
            let mut t = -2.0;
            while t <= 2.0 {
                let s = offset + t;
                let a = Point {
                    x: n.x.mul_add(s, p.x),
                    y: n.y.mul_add(s, p.y),
                };
                let g = polarity as f64
                    * (im.sample(Point {
                        x: n.x.mul_add(0.75, a.x),
                        y: n.y.mul_add(0.75, a.y),
                    }) - im.sample(Point {
                        x: (-n.x).mul_add(0.75, a.x),
                        y: (-n.y).mul_add(0.75, a.y),
                    }));
                if g.is_finite() && g > 0.0 {
                    mass += g;
                    moment = s.mul_add(g, moment);
                }
                t += 0.25;
            }
            if mass > 0.0 {
                offset = moment / mass;
            }
            points.push(Point {
                x: n.x.mul_add(offset, p.x),
                y: n.y.mul_add(offset, p.y),
            });
            weights.push(best);
        }
        if points.len() < 3.max(steps / 4) as usize {
            return false;
        }
        let mut ordered = weights.clone();
        let middle = ordered.len() / 2;
        strength[k] = *ordered
            .select_nth_unstable_by(middle, |a, b| a.partial_cmp(b).unwrap())
            .1;
        let Some(line) = fit(&points, &weights) else {
            return false;
        };
        lines[k] = line;
        // One residual trim/refit prevents nearby payload edges pulling the line.
        let mut residuals: Vec<f64> = points
            .iter()
            .map(|&point| (dot(lines[k].n, point) - lines[k].d).abs())
            .collect();
        let middle = residuals.len() / 2;
        let median = *residuals
            .select_nth_unstable_by(middle, |a, b| a.partial_cmp(b).unwrap())
            .1;
        let cutoff = max_f64(0.5, 2.5 * median);
        let mut retained = 0;
        for i in 0..points.len() {
            if (dot(lines[k].n, points[i]) - lines[k].d).abs() > cutoff {
                weights[i] = 0.0;
            } else {
                retained += 1;
            }
        }
        if retained < 3 {
            return false;
        }
        let Some(line) = fit(&points, &weights) else {
            return false;
        };
        lines[k] = line;
    }
    let mut minimum = strength[0];
    let mut maximum = strength[0];
    for &s in &strength[1..] {
        minimum = min_f64(minimum, s);
        maximum = max_f64(maximum, s);
    }
    if minimum < 0.1 * maximum {
        return false;
    }
    let mut out = [Point::default(); 4];
    for k in 0..4 {
        let Some(p) = intersect(lines[(k + 3) % 4], lines[k]) else {
            return false;
        };
        out[k] = p;
        if norm(out[k] - q[k]) > radius * 3.0 {
            return false;
        }
    }
    *q = out;
    true
}

/// Recover four supporting lines from an ordered hull while ignoring corner chamfers.
pub(crate) fn fit_quad(boundary: &[Point]) -> Option<Quad> {
    if boundary.len() < 4 {
        return None;
    }
    let mut polygon = boundary.to_vec();
    while polygon.len() > 4 {
        let mut remove = 0;
        let mut best = f64::INFINITY;
        for j in 0..polygon.len() {
            let cost = cross(
                polygon[j] - polygon[(j + polygon.len() - 1) % polygon.len()],
                polygon[(j + 1) % polygon.len()] - polygon[j],
            )
            .abs();
            if cost < best {
                best = cost;
                remove = j;
            }
        }
        polygon.remove(remove);
    }
    let q: Quad = [polygon[0], polygon[1], polygon[2], polygon[3]];
    let mut area = 0.0;
    let mut valid = true;
    for k in 0..4 {
        area += cross(q[k], q[(k + 1) % 4]);
        if norm(q[(k + 1) % 4] - q[k]) < 3.0 {
            valid = false;
        }
    }
    if !valid || area < 25.0 {
        return None;
    }
    let mut hullarea = 0.0;
    for j in 0..boundary.len() {
        hullarea += cross(boundary[j], boundary[(j + 1) % boundary.len()]);
    }
    if area < hullarea * 0.8 {
        return None;
    }
    let mut lines = [Line {
        n: Point::default(),
        d: 0.0,
    }; 4];
    for k in 0..4 {
        let d = q[(k + 1) % 4] - q[k];
        let length = norm(d);
        let mut p = Vec::new();
        let mut weights = Vec::new();
        for j in 0..boundary.len() {
            let a = boundary[j];
            let b = boundary[(j + 1) % boundary.len()];
            let span = b - a;
            let span_length = norm(span);
            if dot(span, d) < 0.94 * span_length * length {
                continue;
            }
            let t = dot((a + b) * 0.5 - q[k], d) / (length * length);
            let distance = cross(d, (a + b) * 0.5 - q[k]).abs() / length;
            if (-0.05..=1.05).contains(&t) && distance < max_f64(1.5, length * 0.08) {
                p.push(a);
                p.push((a + b) * 0.5);
                p.push(b);
                weights.extend_from_slice(&[span_length / 3.0; 3]);
            }
        }
        lines[k] = Line {
            n: Point {
                x: -d.y / length,
                y: d.x / length,
            },
            d: 0.0,
        };
        lines[k].d = dot(lines[k].n, q[k]);
        if let Some(fitted) = fit(&p, &weights) {
            lines[k] = fitted;
        }
    }
    let mut fitted = q;
    for k in 0..4 {
        let allowance = max_f64(
            3.0,
            0.1 * min_f64(norm(q[(k + 1) % 4] - q[k]), norm(q[(k + 3) % 4] - q[k])),
        );
        if let Some(p) = intersect(lines[(k + 3) % 4], lines[k]) {
            if norm(p - q[k]) < allowance {
                fitted[k] = p;
            }
        }
    }
    Some(fitted)
}
