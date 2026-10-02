//! OpenCV pinhole calibration: rational radial, tangential, thin-prism and tilt.
//!
//! Coefficient order is k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,tau_x,tau_y;
//! absent trailing coefficients are zero. With r²=x²+y², radial scale is
//! (1+k1*r²+k2*r⁴+k3*r⁶)/(1+k4*r²+k5*r⁴+k6*r⁶). Tangential terms are
//! (2*p1*x*y+p2*(r²+2*x²), p1*(r²+2*y²)+2*p2*x*y); thin-prism terms
//! are (s1*r²+s2*r⁴, s3*r²+s4*r⁴). Tilt uses OpenCV's signed Ry*Rx then
//! its Z-projection homography, followed by homogeneous division. Pixel
//! coordinates use fx/fy/cx/cy. Derivatives chain every stage analytically.
//! Reference: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};

pub struct Calibration {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub coefficients: [f64; 14],
    tilt: Matrix3<f64>,
    inverse_tilt: Matrix3<f64>,
}
impl Calibration {
    /// Validate calibration and construct OpenCV's tilted sensor homography.
    pub fn new(matrix: &[f64], distortion: &[f64]) -> Result<Self, &'static str> {
        if matrix.len() != 9
            || !matches!(distortion.len(), 0 | 4 | 5 | 8 | 12 | 14)
            || !matrix.iter().chain(distortion).all(|v| v.is_finite())
            || matrix[0] <= 0.0
            || matrix[4] <= 0.0
        {
            return Err("invalid calibration");
        }
        let mut coefficients = [0.0; 14];
        coefficients[..distortion.len()].copy_from_slice(distortion);
        let (sx, cx) = coefficients[12].sin_cos();
        let (sy, cy) = coefficients[13].sin_cos();
        let rotation_x = Matrix3::new(1., 0., 0., 0., cx, sx, 0., -sx, cx);
        let rotation_y = Matrix3::new(cy, 0., -sy, 0., 1., 0., sy, 0., cy);
        let rotation = rotation_y * rotation_x;
        let projection = Matrix3::new(
            rotation[(2, 2)],
            0.,
            -rotation[(0, 2)],
            0.,
            rotation[(2, 2)],
            -rotation[(1, 2)],
            0.,
            0.,
            1.,
        );
        let tilt = projection * rotation;
        let inverse_tilt = tilt.try_inverse().ok_or("invalid calibration")?;
        Ok(Self {
            fx: matrix[0],
            fy: matrix[4],
            cx: matrix[2],
            cy: matrix[5],
            coefficients,
            tilt,
            inverse_tilt,
        })
    }
    /// Distort normalized coordinates and return the exact normalized Jacobian.
    fn distort(&self, point: Vector2<f64>) -> (Vector2<f64>, Matrix2<f64>) {
        let x = point.x;
        let y = point.y;
        let r = x * x + y * y;
        let d = self.coefficients;
        let numerator = 1. + d[0] * r + d[1] * r * r + d[4] * r * r * r;
        let denominator = 1. + d[5] * r + d[6] * r * r + d[7] * r * r * r;
        let radial = numerator / denominator;
        let derivative = ((d[0] + 2. * d[1] * r + 3. * d[4] * r * r) * denominator
            - numerator * (d[5] + 2. * d[6] * r + 3. * d[7] * r * r))
            / (denominator * denominator);
        let prism_x = d[8] + 2. * d[9] * r;
        let prism_y = d[10] + 2. * d[11] * r;
        let distorted = Vector2::new(
            x * radial + 2. * d[2] * x * y + d[3] * (r + 2. * x * x) + d[8] * r + d[9] * r * r,
            y * radial + d[2] * (r + 2. * y * y) + 2. * d[3] * x * y + d[10] * r + d[11] * r * r,
        );
        let jacobian = Matrix2::new(
            radial + 2. * x * x * derivative + 2. * d[2] * y + 6. * d[3] * x + 2. * x * prism_x,
            2. * x * y * derivative + 2. * d[2] * x + 2. * d[3] * y + 2. * y * prism_x,
            2. * x * y * derivative + 2. * d[2] * x + 2. * d[3] * y + 2. * x * prism_y,
            radial + 2. * y * y * derivative + 6. * d[2] * y + 2. * d[3] * x + 2. * y * prism_y,
        );
        (distorted, jacobian)
    }
    /// OpenCV projectPoints uses fx/fy/cx/cy (not intrinsic skew).
    pub fn project(&self, point: Vector3<f64>) -> (Vector2<f64>, nalgebra::Matrix2x3<f64>) {
        let normalized = Vector2::new(point.x / point.z, point.y / point.z);
        let (distorted, jacobian) = self.distort(normalized);
        let tilted = self.tilt * Vector3::new(distorted.x, distorted.y, 1.);
        // Match OpenCV's homogeneous zero-denominator convention.
        let inverse_depth = if tilted.z != 0. { 1. / tilted.z } else { 1. };
        let tilt_jacobian = Matrix2::from_fn(|row, col| {
            (self.tilt[(row, col)] * tilted.z - tilted[row] * self.tilt[(2, col)])
                * inverse_depth
                * inverse_depth
        });
        let pixel_scale = Matrix2::new(self.fx, 0., 0., self.fy);
        let normalize = nalgebra::Matrix2x3::new(
            1. / point.z,
            0.,
            -normalized.x / point.z,
            0.,
            1. / point.z,
            -normalized.y / point.z,
        );
        (
            Vector2::new(
                self.fx * tilted.x * inverse_depth + self.cx,
                self.fy * tilted.y * inverse_depth + self.cy,
            ),
            pixel_scale * tilt_jacobian * jacobian * normalize,
        )
    }
    /// Match undistortPoints' default five fixed-point iterations, including its
    /// negative inverse-radial fallback and inverse tilted-sensor transform.
    pub fn undistort(&self, pixel: Vector2<f64>) -> Vector2<f64> {
        let original = Vector2::new((pixel.x - self.cx) / self.fx, (pixel.y - self.cy) / self.fy);
        let untilted = self.inverse_tilt * Vector3::new(original.x, original.y, 1.);
        let inverse_depth = if untilted.z != 0. {
            1. / untilted.z
        } else {
            1.
        };
        let origin = Vector2::new(untilted.x * inverse_depth, untilted.y * inverse_depth);
        let mut point = origin;
        let d = self.coefficients;
        for _ in 0..5 {
            let x = point.x;
            let y = point.y;
            let r = x * x + y * y;
            let inverse = (1. + d[5] * r + d[6] * r * r + d[7] * r * r * r)
                / (1. + d[0] * r + d[1] * r * r + d[4] * r * r * r);
            if inverse < 0. {
                return original;
            }
            let delta = Vector2::new(
                2. * d[2] * x * y + d[3] * (r + 2. * x * x) + d[8] * r + d[9] * r * r,
                d[2] * (r + 2. * y * y) + 2. * d[3] * x * y + d[10] * r + d[11] * r * r,
            );
            point = (origin - delta) * inverse;
        }
        point
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn models_and_pixel_jacobian() {
        for count in [0, 4, 5, 8, 12, 14] {
            let mut d = vec![0.001; count];
            if count == 14 {
                d[12] = 0.04;
                d[13] = -0.03;
            }
            let c = Calibration::new(&[700., 0., 320., 0., 710., 240., 0., 0., 1.], &d).unwrap();
            let point = Vector3::new(0.3, -0.2, 2.);
            let (pixel, jacobian) = c.project(point);
            assert!((c.undistort(pixel) - Vector2::new(0.15, -0.1)).norm() < 1e-10);
            for axis in 0..3 {
                let mut plus = point;
                let mut minus = point;
                plus[axis] += 1e-6;
                minus[axis] -= 1e-6;
                assert!(
                    ((c.project(plus).0 - c.project(minus).0) / 2e-6 - jacobian.column(axis))
                        .norm()
                        < 1e-6
                );
            }
        }
    }
}
