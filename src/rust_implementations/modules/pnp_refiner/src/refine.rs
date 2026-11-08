//! Levenberg-Marquardt refinement of a camera-from-object pose.
//!
//! Minimizes the summed squared pixel reprojection error over the six pose
//! parameters, like OpenCV's `solvePnPRefineLM`. Rotation steps are applied
//! multiplicatively (`R <- exp(delta) R`), which keeps the Jacobian simple and
//! avoids Rodrigues-derivative singularities; the result converges to the same
//! minimum as OpenCV's additive rotation-vector steps.

/// Pinhole intrinsics with OpenCV's rational radial and tangential distortion.
#[derive(Clone, Debug)]
pub struct Camera {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    /// k1, k2, p1, p2, k3, k4, k5, k6; trailing coefficients default to zero.
    pub dist: [f64; 8],
}

impl Camera {
    /// Build from a row-major 3x3 camera matrix and 0, 4, 5 or 8 distortion
    /// coefficients. Thin-prism and tilted models are not supported.
    pub fn new(camera_matrix: &[f64], distortion: &[f64]) -> Result<Self, String> {
        if camera_matrix.len() != 9 || camera_matrix.iter().any(|v| !v.is_finite()) {
            return Err("camera matrix must hold 9 finite values".into());
        }
        if !matches!(distortion.len(), 0 | 4 | 5 | 8) || distortion.iter().any(|v| !v.is_finite()) {
            return Err("distortion must hold 0, 4, 5 or 8 finite coefficients".into());
        }
        let mut dist = [0.0; 8];
        dist[..distortion.len()].copy_from_slice(distortion);
        Ok(Self {
            fx: camera_matrix[0],
            fy: camera_matrix[4],
            cx: camera_matrix[2],
            cy: camera_matrix[5],
            dist,
        })
    }

    /// Project a camera-space point, returning pixel coordinates and their
    /// derivatives with respect to the camera-space point. None if behind.
    fn project(&self, p: [f64; 3]) -> Option<([f64; 2], [[f64; 3]; 2])> {
        let [k1, k2, p1, p2, k3, k4, k5, k6] = self.dist;
        // Also rejects NaN depth.
        if p[2].is_nan() || p[2] <= f64::EPSILON {
            return None;
        }
        let inv_z = 1.0 / p[2];
        let x = p[0] * inv_z;
        let y = p[1] * inv_z;
        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        let numerator = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
        let denominator = 1.0 + k4 * r2 + k5 * r4 + k6 * r6;
        let radial = numerator / denominator;
        let d_numerator = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4;
        let d_denominator = k4 + 2.0 * k5 * r2 + 3.0 * k6 * r4;
        let d_radial =
            (d_numerator * denominator - numerator * d_denominator) / (denominator * denominator);

        let xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        let yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
        let dxd_dx = radial + 2.0 * x * x * d_radial + 2.0 * p1 * y + 6.0 * p2 * x;
        let dxd_dy = 2.0 * x * y * d_radial + 2.0 * p1 * x + 2.0 * p2 * y;
        let dyd_dx = 2.0 * x * y * d_radial + 2.0 * p1 * x + 2.0 * p2 * y;
        let dyd_dy = radial + 2.0 * y * y * d_radial + 6.0 * p1 * y + 2.0 * p2 * x;

        // d(x, y)/dp = [[1/z, 0, -x/z], [0, 1/z, -y/z]].
        let row = |scale: f64, d_dx: f64, d_dy: f64| {
            [
                scale * d_dx * inv_z,
                scale * d_dy * inv_z,
                -scale * (d_dx * x + d_dy * y) * inv_z,
            ]
        };
        Some((
            [self.fx * xd + self.cx, self.fy * yd + self.cy],
            [row(self.fx, dxd_dx, dxd_dy), row(self.fy, dyd_dx, dyd_dy)],
        ))
    }
}

type Mat3 = [[f64; 3]; 3];

fn mat_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            *value = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

fn mat_vec(m: &Mat3, v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Rotation vector to matrix (Rodrigues).
pub fn rotation_matrix(rvec: [f64; 3]) -> Mat3 {
    let theta = (rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2]).sqrt();
    if theta < f64::EPSILON {
        return [
            [1.0, -rvec[2], rvec[1]],
            [rvec[2], 1.0, -rvec[0]],
            [-rvec[1], rvec[0], 1.0],
        ];
    }
    let k = [rvec[0] / theta, rvec[1] / theta, rvec[2] / theta];
    let (s, c) = theta.sin_cos();
    let v = 1.0 - c;
    [
        [
            c + v * k[0] * k[0],
            v * k[0] * k[1] - s * k[2],
            v * k[0] * k[2] + s * k[1],
        ],
        [
            v * k[1] * k[0] + s * k[2],
            c + v * k[1] * k[1],
            v * k[1] * k[2] - s * k[0],
        ],
        [
            v * k[2] * k[0] - s * k[1],
            v * k[2] * k[1] + s * k[0],
            c + v * k[2] * k[2],
        ],
    ]
}

/// Rotation matrix to rotation vector, following OpenCV's handling near pi.
pub fn rotation_vector(r: &Mat3) -> [f64; 3] {
    let axis = [r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]];
    let s = 0.5 * (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let c = ((r[0][0] + r[1][1] + r[2][2] - 1.0) * 0.5).clamp(-1.0, 1.0);
    if s < 1e-5 {
        if c > 0.0 {
            return [0.0; 3];
        }
        // Near a half turn: recover the axis from the symmetric part.
        let rx = ((r[0][0] + 1.0) * 0.5).max(0.0).sqrt();
        let ry = ((r[1][1] + 1.0) * 0.5).max(0.0).sqrt() * if r[0][1] < 0.0 { -1.0 } else { 1.0 };
        let mut rz =
            ((r[2][2] + 1.0) * 0.5).max(0.0).sqrt() * if r[0][2] < 0.0 { -1.0 } else { 1.0 };
        if rx.abs() < ry.abs() && rx.abs() < rz.abs() && (r[1][2] > 0.0) != (ry * rz > 0.0) {
            rz = -rz;
        }
        let norm = (rx * rx + ry * ry + rz * rz).sqrt();
        let scale = c.acos() / norm;
        return [rx * scale, ry * scale, rz * scale];
    }
    let scale = s.atan2(c) / (2.0 * s);
    [axis[0] * scale, axis[1] * scale, axis[2] * scale]
}

/// Squared reprojection error and, when requested, the 6x6 normal equations
/// `J^T J` and gradient `J^T r` for parameters (rotation delta, translation).
struct Evaluation {
    energy: f64,
    normal: [[f64; 6]; 6],
    gradient: [f64; 6],
}

fn evaluate(
    camera: &Camera,
    object: &[[f64; 3]],
    image: &[[f64; 2]],
    rotation: &Mat3,
    translation: [f64; 3],
) -> Option<Evaluation> {
    let mut eval = Evaluation {
        energy: 0.0,
        normal: [[0.0; 6]; 6],
        gradient: [0.0; 6],
    };
    for (point, observed) in object.iter().zip(image) {
        let rotated = mat_vec(rotation, *point);
        let camera_point = [
            rotated[0] + translation[0],
            rotated[1] + translation[1],
            rotated[2] + translation[2],
        ];
        let (pixel, d_pixel) = camera.project(camera_point)?;
        for axis in 0..2 {
            let residual = pixel[axis] - observed[axis];
            let g = d_pixel[axis];
            // d(camera point)/d(rotation delta) = -[rotated]x, so the row is rotated x g.
            let rotation_row = cross(rotated, g);
            let row = [
                rotation_row[0],
                rotation_row[1],
                rotation_row[2],
                g[0],
                g[1],
                g[2],
            ];
            eval.energy += residual * residual;
            for i in 0..6 {
                eval.gradient[i] += row[i] * residual;
                for j in i..6 {
                    eval.normal[i][j] += row[i] * row[j];
                }
            }
        }
    }
    for i in 0..6 {
        for j in 0..i {
            eval.normal[i][j] = eval.normal[j][i];
        }
    }
    eval.energy.is_finite().then_some(eval)
}

/// Solve the symmetric positive-definite system `a x = b` by Cholesky.
fn solve_spd(mut a: [[f64; 6]; 6], b: [f64; 6]) -> Option<[f64; 6]> {
    for j in 0..6 {
        let diagonal = a[j][j] - a[j][..j].iter().map(|v| v * v).sum::<f64>();
        if diagonal.is_nan() || diagonal <= 0.0 {
            return None;
        }
        let diagonal = diagonal.sqrt();
        a[j][j] = diagonal;
        for i in j + 1..6 {
            let dot: f64 = a[i][..j].iter().zip(&a[j][..j]).map(|(p, q)| p * q).sum();
            a[i][j] = (a[i][j] - dot) / diagonal;
        }
    }
    let mut x = b;
    for i in 0..6 {
        for k in 0..i {
            x[i] -= a[i][k] * x[k];
        }
        x[i] /= a[i][i];
    }
    for i in (0..6).rev() {
        for k in i + 1..6 {
            x[i] -= a[k][i] * x[k];
        }
        x[i] /= a[i][i];
    }
    Some(x)
}

/// Refine `rvec`/`tvec` in place with at most `max_iterations` LM steps.
///
/// Every attempted step, accepted or not, counts as one iteration. Stops early
/// once a step's norm falls below `epsilon`. Returns false, leaving the pose
/// untouched, when the starting pose cannot be evaluated (a point is behind
/// the camera or a value is not finite).
pub fn refine(
    camera: &Camera,
    object: &[[f64; 3]],
    image: &[[f64; 2]],
    rvec: &mut [f64; 3],
    tvec: &mut [f64; 3],
    max_iterations: usize,
    epsilon: f64,
) -> bool {
    let mut rotation = rotation_matrix(*rvec);
    let mut translation = *tvec;
    let Some(mut current) = evaluate(camera, object, image, &rotation, translation) else {
        return false;
    };
    let mut lambda = 1e-3;
    for _ in 0..max_iterations {
        let mut damped = current.normal;
        for (i, row) in damped.iter_mut().enumerate() {
            row[i] += lambda * row[i].max(f64::EPSILON);
        }
        let negative_gradient = current.gradient.map(|g| -g);
        let Some(step) = solve_spd(damped, negative_gradient) else {
            lambda *= 10.0;
            continue;
        };
        if step.iter().map(|s| s * s).sum::<f64>().sqrt() < epsilon {
            break;
        }
        let trial_rotation = mat_mul(&rotation_matrix([step[0], step[1], step[2]]), &rotation);
        let trial_translation = [
            translation[0] + step[3],
            translation[1] + step[4],
            translation[2] + step[5],
        ];
        match evaluate(camera, object, image, &trial_rotation, trial_translation) {
            Some(trial) if trial.energy < current.energy => {
                rotation = trial_rotation;
                translation = trial_translation;
                current = trial;
                lambda = (lambda / 10.0).max(1e-12);
            }
            _ => lambda *= 10.0,
        }
    }
    *rvec = rotation_vector(&rotation);
    *tvec = translation;
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera() -> Camera {
        Camera::new(
            &[762.7, 0.0, 640.0, 0.0, 762.7, 400.0, 0.0, 0.0, 1.0],
            &[-0.06, 0.012, 0.0005, -0.0005, 0.0],
        )
        .unwrap()
    }

    fn tag() -> Vec<[f64; 3]> {
        vec![
            [-0.08, 0.08, 0.0],
            [0.08, 0.08, 0.0],
            [0.08, -0.08, 0.0],
            [-0.08, -0.08, 0.0],
        ]
    }

    fn project_all(
        camera: &Camera,
        object: &[[f64; 3]],
        rvec: [f64; 3],
        t: [f64; 3],
    ) -> Vec<[f64; 2]> {
        let r = rotation_matrix(rvec);
        object
            .iter()
            .map(|p| {
                let q = mat_vec(&r, *p);
                camera
                    .project([q[0] + t[0], q[1] + t[1], q[2] + t[2]])
                    .unwrap()
                    .0
            })
            .collect()
    }

    #[test]
    fn rotation_vector_round_trips() {
        for rvec in [
            [0.1, -0.2, 0.3],
            [2.0, 1.0, -0.5],
            [0.0, 3.1, 0.0],
            [1e-9, 0.0, 0.0],
        ] {
            let back = rotation_vector(&rotation_matrix(rvec));
            for i in 0..3 {
                assert!((back[i] - rvec[i]).abs() < 1e-7, "{rvec:?} -> {back:?}");
            }
        }
    }

    #[test]
    fn projection_jacobian_matches_finite_differences() {
        let camera = camera();
        let p = [0.3, -0.2, 2.0];
        let (_, jacobian) = camera.project(p).unwrap();
        let h = 1e-7;
        for axis in 0..3 {
            let mut plus = p;
            let mut minus = p;
            plus[axis] += h;
            minus[axis] -= h;
            let (a, _) = camera.project(plus).unwrap();
            let (b, _) = camera.project(minus).unwrap();
            for out in 0..2 {
                let numeric = (a[out] - b[out]) / (2.0 * h);
                assert!((numeric - jacobian[out][axis]).abs() < 1e-4 * numeric.abs().max(1.0));
            }
        }
    }

    #[test]
    fn refine_recovers_a_perturbed_pose() {
        let camera = camera();
        let object = tag();
        let true_r = [0.2, -0.3, 0.1];
        let true_t = [0.1, -0.05, 2.5];
        let image = project_all(&camera, &object, true_r, true_t);
        let mut rvec = [0.25, -0.25, 0.12];
        let mut tvec = [0.12, -0.03, 2.4];
        assert!(refine(
            &camera, &object, &image, &mut rvec, &mut tvec, 20, 1e-12
        ));
        for i in 0..3 {
            assert!((rvec[i] - true_r[i]).abs() < 1e-8);
            assert!((tvec[i] - true_t[i]).abs() < 1e-8);
        }
    }

    #[test]
    fn zero_iterations_leave_the_pose_unchanged() {
        let camera = camera();
        let object = tag();
        let image = project_all(&camera, &object, [0.2, -0.3, 0.1], [0.1, -0.05, 2.5]);
        let mut rvec = [0.25, -0.25, 0.12];
        let mut tvec = [0.12, -0.03, 2.4];
        assert!(refine(
            &camera, &object, &image, &mut rvec, &mut tvec, 0, 1e-9
        ));
        for i in 0..3 {
            assert!((rvec[i] - [0.25, -0.25, 0.12][i]).abs() < 1e-12);
        }
        assert_eq!(tvec, [0.12, -0.03, 2.4]);
    }

    #[test]
    fn a_pose_behind_the_camera_is_rejected() {
        let camera = camera();
        let object = tag();
        let image = project_all(&camera, &object, [0.0; 3], [0.0, 0.0, 2.0]);
        let mut rvec = [0.0; 3];
        let mut tvec = [0.0, 0.0, -2.0];
        assert!(!refine(
            &camera, &object, &image, &mut rvec, &mut tvec, 10, 1e-9
        ));
        assert_eq!(tvec, [0.0, 0.0, -2.0]);
    }
}
