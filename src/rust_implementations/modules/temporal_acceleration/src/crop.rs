//! Square rectified-crop geometry for projected tag quadrilaterals.

/// A square rectified crop of a frame quad.
pub(crate) struct CropGeometry {
    pub(crate) side: i32,
    pub(crate) frame_from_crop: [f64; 9],
    pub(crate) crop_from_frame: [f64; 9],
}

/// Side length and both homographies for one perimeter-ordered quad.
///
/// Reproduces the NumPy/OpenCV code it replaces: the side is the longest edge in
/// f32, rounded up, limited to `max_side` and at least 2. The crop's corners keep
/// the quad's winding, because a mirrored AprilTag cannot be decoded. Returns None
/// for a degenerate quad.
pub(crate) fn crop_geometry(quad: &[[f32; 2]; 4], max_side: i32) -> Option<CropGeometry> {
    let mut longest = 0.0f32;
    for (i, a) in quad.iter().enumerate() {
        let b = quad[(i + 1) % 4];
        let (dx, dy) = (a[0] - b[0], a[1] - b[1]);
        longest = longest.max((dx * dx + dy * dy).sqrt());
    }
    if !longest.is_finite() {
        return None;
    }
    let side = (longest.ceil() as i32).min(max_side).max(2);
    let last = (side - 1) as f32;
    let mut corners = [[0.0, 0.0], [last, 0.0], [last, last], [0.0, last]];
    if oriented_area(quad) < 0.0 {
        corners.reverse();
    }
    Some(CropGeometry {
        side,
        frame_from_crop: perspective_transform(&corners, quad)?,
        crop_from_frame: perspective_transform(quad, &corners)?,
    })
}

/// Signed shoelace area, accumulated like cv2.contourArea(oriented=True).
fn oriented_area(quad: &[[f32; 2]; 4]) -> f64 {
    let mut previous = quad[3];
    let mut area = 0.0;
    for &point in quad {
        area += f64::from(previous[0]) * f64::from(point[1])
            - f64::from(previous[1]) * f64::from(point[0]);
        previous = point;
    }
    area * 0.5
}

/// Homography taking `source` to `target`, set up and solved like
/// cv2.getPerspectiveTransform: f32 products, then an 8x8 partial-pivot LU in f64.
fn perspective_transform(source: &[[f32; 2]; 4], target: &[[f32; 2]; 4]) -> Option<[f64; 9]> {
    let mut a = [[0.0f64; 8]; 8];
    let mut b = [0.0f64; 8];
    for i in 0..4 {
        let ([x, y], [u, v]) = (source[i], target[i]);
        let f = f64::from;
        a[i] = [f(x), f(y), 1.0, 0.0, 0.0, 0.0, f(-x * u), f(-y * u)];
        a[i + 4] = [0.0, 0.0, 0.0, f(x), f(y), 1.0, f(-x * v), f(-y * v)];
        b[i] = f(u);
        b[i + 4] = f(v);
    }
    for i in 0..8 {
        let pivot = (i..8).fold(i, |k, j| if a[j][i].abs() > a[k][i].abs() { j } else { k });
        if a[pivot][i].abs() < f64::EPSILON * 100.0 {
            return None;
        }
        a.swap(i, pivot);
        b.swap(i, pivot);
        let d = -1.0 / a[i][i];
        let pivot_row = a[i];
        for j in i + 1..8 {
            let alpha = a[j][i] * d;
            for (value, above) in a[j][i + 1..].iter_mut().zip(&pivot_row[i + 1..]) {
                *value += alpha * above;
            }
            b[j] += alpha * b[i];
        }
    }
    for i in (0..8).rev() {
        let mut s = b[i];
        for (coefficient, solved) in a[i][i + 1..].iter().zip(&b[i + 1..]) {
            s -= coefficient * solved;
        }
        b[i] = s / a[i][i];
    }
    Some([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], 1.0])
}

#[cfg(test)]
mod tests {
    use super::crop_geometry;

    fn project(m: &[f64; 9], x: f64, y: f64) -> [f64; 2] {
        let z = m[6] * x + m[7] * y + m[8];
        [
            (m[0] * x + m[1] * y + m[2]) / z,
            (m[3] * x + m[4] * y + m[5]) / z,
        ]
    }

    #[test]
    fn crop_corners_map_onto_the_quad_in_either_winding() {
        let quad = [[10.0, 20.0], [70.0, 10.0], [80.0, 80.0], [20.0, 90.0]];
        let mut reversed = quad;
        reversed.reverse();
        for (quad, first) in [(quad, quad[0]), (reversed, reversed[3])] {
            let crop = crop_geometry(&quad, 1000).unwrap();
            // Longest edge: hypot(10, 70) = 70.7, rounded up.
            assert_eq!(crop.side, 71);
            let corner = project(&crop.frame_from_crop, 0.0, 0.0);
            assert!((corner[0] - f64::from(first[0])).abs() < 1e-9);
            assert!((corner[1] - f64::from(first[1])).abs() < 1e-9);
        }
        assert_eq!(crop_geometry(&quad, 50).unwrap().side, 50);
        assert!(crop_geometry(&[[5.0, 5.0]; 4], 50).is_none());
    }
}
