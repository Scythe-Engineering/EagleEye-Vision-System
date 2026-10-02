//! Fixed-heading, level-robot XY least squares and actual-pixel refinement.
use crate::calibration::Calibration;
use nalgebra::{DMatrix, DVector, Matrix3, Vector2, Vector3};

pub struct Points {
    pub objects: Vec<Vector3<f64>>,
    pub images: Vec<Vector2<f64>>,
}
pub struct Solution {
    pub pose: Vec<f64>,
    pub meta: Vec<f64>,
    pub xy: Vec<f64>,
}
/// Match numpy matrix_rank's default singular-value tolerance on centered points.
pub fn nondegenerate(points: &Points) -> bool {
    fn rank(matrix: DMatrix<f64>) -> usize {
        // Finite inputs can still overflow while centering; never send NaNs to SVD.
        if !matrix.iter().all(|value| value.is_finite()) {
            return 0;
        }
        let dimension = matrix.nrows().max(matrix.ncols()) as f64;
        let svd = matrix.svd(false, false);
        let tolerance = svd.singular_values[0] * dimension * f64::EPSILON;
        svd.singular_values
            .iter()
            .filter(|value| **value > tolerance)
            .count()
    }
    let count = points.objects.len();
    let object_mean = points.objects.iter().copied().sum::<Vector3<f64>>() / count as f64;
    let image_mean = points.images.iter().copied().sum::<Vector2<f64>>() / count as f64;
    rank(DMatrix::from_fn(count, 3, |row, col| {
        points.objects[row][col] - object_mean[col]
    })) >= 2
        && rank(DMatrix::from_fn(count, 2, |row, col| {
            points.images[row][col] - image_mean[col]
        })) >= 2
}
/// SVD, not normal equations: preserve conditioning and numpy's lstsq cutoff.
fn least_squares(
    matrix: DMatrix<f64>,
    target: &DVector<f64>,
) -> Result<(Vector2<f64>, f64, usize), &'static str> {
    if !matrix
        .iter()
        .chain(target.iter())
        .all(|value| value.is_finite())
    {
        return Err("invalid_geometry");
    }
    let dimension = matrix.nrows().max(matrix.ncols()) as f64;
    let svd = matrix.svd(true, true);
    let tolerance = svd.singular_values[0] * dimension * f64::EPSILON;
    let condition = svd.singular_values[0] / svd.singular_values[1];
    let rank = svd.rank(tolerance);
    let answer = svd
        .solve(target, tolerance)
        .map_err(|_| "invalid_geometry")?;
    Ok((Vector2::new(answer[0], answer[1]), condition, rank))
}
struct Projection {
    residual: DVector<f64>,
    jacobian: DMatrix<f64>,
}
/// Evaluate residual and analytic XY Jacobian after full calibrated projection.
fn project(
    calibration: &Calibration,
    points: &Points,
    camera_rotation: &Matrix3<f64>,
    offset: Vector3<f64>,
    xy: Vector2<f64>,
) -> Result<Projection, &'static str> {
    let translation = -camera_rotation * (offset + Vector3::new(xy.x, xy.y, 0.));
    let mut residual = DVector::zeros(points.objects.len() * 2);
    let mut jacobian = DMatrix::zeros(points.objects.len() * 2, 2);
    for (index, object) in points.objects.iter().enumerate() {
        let point = camera_rotation * object + translation;
        if !point.z.is_finite() || point.z <= 1e-6 {
            return Err("behind_camera");
        }
        let (pixel, derivative) = calibration.project(point);
        let error = pixel - points.images[index];
        let xy_derivative = derivative * (-camera_rotation.fixed_columns::<2>(0));
        for row in 0..2 {
            residual[index * 2 + row] = error[row];
            for col in 0..2 {
                jacobian[(index * 2 + row, col)] = xy_derivative[(row, col)];
            }
        }
    }
    if !residual
        .iter()
        .chain(jacobian.iter())
        .all(|value| value.is_finite())
    {
        return Err("nonfinite_solution");
    }
    Ok(Projection { residual, jacobian })
}
/// Solve field-from-camera EDN pose for a robot at Z=0 and fixed NWU gyro yaw.
pub fn solve(
    calibration: &Calibration,
    points: &Points,
    yaw: f64,
    mount: &[f64],
    iterations: usize,
    condition: &mut Option<f64>,
) -> Result<Solution, &'static str> {
    if mount.len() != 16 || !mount.iter().all(|value| value.is_finite()) {
        return Err("invalid_mounting");
    }
    let (sine, cosine) = yaw.sin_cos();
    let robot_rotation = Matrix3::new(cosine, -sine, 0., sine, cosine, 0., 0., 0., 1.);
    let mount_rotation = Matrix3::from_fn(|row, col| mount[row * 4 + col]);
    let rotation = robot_rotation * mount_rotation;
    let offset = robot_rotation * Vector3::new(mount[3], mount[7], mount[11]);
    let camera_rotation = rotation.transpose();
    let mut coefficients = DMatrix::zeros(points.objects.len() * 2, 2);
    let mut targets = DVector::zeros(points.objects.len() * 2);
    for index in 0..points.objects.len() {
        let ray = calibration.undistort(points.images[index]);
        let rotated = camera_rotation * (points.objects[index] - offset);
        for row in 0..2 {
            targets[index * 2 + row] = rotated[row] - ray[row] * rotated.z;
            for col in 0..2 {
                coefficients[(index * 2 + row, col)] =
                    camera_rotation[(row, col)] - ray[row] * camera_rotation[(2, col)];
            }
        }
    }
    let (mut xy, initial_condition, rank) = least_squares(coefficients, &targets)?;
    *condition = Some(initial_condition);
    if rank != 2 || !initial_condition.is_finite() || initial_condition > 1e8 {
        return Err("ill_conditioned_geometry");
    }
    let mut projection = project(calibration, points, &camera_rotation, offset, xy)?;
    for _ in 0..iterations {
        let (mut step, _, _) =
            least_squares(projection.jacobian.clone(), &(-&projection.residual))?;
        if !step.iter().all(|value| value.is_finite()) {
            return Err("nonfinite_solution");
        }
        // ponytail: fixed 1 m ceiling; expose tuning only if benchmarks need it.
        step /= step.norm().max(1.);
        let mut improved = false;
        for scale in [1., 0.5, 0.25, 0.125, 0.0625, 0.03125] {
            if let Ok(candidate) = project(
                calibration,
                points,
                &camera_rotation,
                offset,
                xy + step * scale,
            ) {
                if candidate.residual.norm_squared() < projection.residual.norm_squared() {
                    xy += step * scale;
                    projection = candidate;
                    improved = true;
                    break;
                }
            }
        }
        if !improved || step.norm() < 1e-9 {
            break;
        }
    }
    let singular_values = projection.jacobian.svd(false, false).singular_values;
    let final_condition = singular_values[0] / singular_values[1];
    if !xy.iter().all(|value| value.is_finite())
        || final_condition > 1e8
        || !final_condition.is_finite()
    {
        return Err("ill_conditioned_geometry");
    }
    let camera_position = offset + Vector3::new(xy.x, xy.y, 0.);
    let tag_count = points.objects.len() / 4;
    let mean_distance = points
        .objects
        .chunks(4)
        .map(|corners| {
            let center = corners.iter().copied().sum::<Vector3<f64>>() / 4.;
            (camera_rotation * (center - camera_position)).norm()
        })
        .sum::<f64>()
        / tag_count as f64;
    let mean_error = projection
        .residual
        .as_slice()
        .chunks(2)
        .map(|error| error[0].hypot(error[1]))
        .sum::<f64>()
        / points.objects.len() as f64;
    let meta = vec![tag_count as f64, mean_distance, mean_error];
    if !meta.iter().all(|value| value.is_finite()) {
        return Err("nonfinite_solution");
    }
    let mut pose = vec![0.; 16];
    for row in 0..3 {
        for col in 0..3 {
            pose[row * 4 + col] = rotation[(row, col)];
        }
        pose[row * 4 + 3] = camera_position[row];
    }
    pose[15] = 1.;
    Ok(Solution {
        pose,
        meta,
        xy: vec![xy.x, xy.y],
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn geometry_quality_and_rank() {
        let calibration = Calibration::new(
            &[700., 0., 320., 0., 710., 240., 0., 0., 1.],
            &[0.01, -0.001, 0.0001, 0.0002, 0.],
        )
        .unwrap();
        let mount = vec![
            0., 0., 1., 0.2, -1., 0., 0., 0., 0., -1., 0., 0.5, 0., 0., 0., 1.,
        ];
        let rotation = Matrix3::from_fn(|r, c| mount[r * 4 + c]);
        let position = Vector3::new(1.2, -0.4, 0.5);
        let objects = vec![
            Vector3::new(5., -0.5, 0.2),
            Vector3::new(5., 0.5, 0.2),
            Vector3::new(5., 0.5, 1.2),
            Vector3::new(5., -0.5, 1.2),
        ];
        let images = objects
            .iter()
            .map(|point| {
                calibration
                    .project(rotation.transpose() * (point - position))
                    .0
            })
            .collect();
        let points = Points { objects, images };
        assert!(nondegenerate(&points));
        let solution = solve(&calibration, &points, 0., &mount, 10, &mut None).unwrap();
        assert!((solution.xy[0] - 1.).abs() < 1e-8 && (solution.xy[1] + 0.4).abs() < 1e-8);
        assert_eq!(solution.meta[0], 1.);
        assert!(solution.meta[2] < 1e-8);
        assert_eq!(
            project(
                &calibration,
                &points,
                &rotation.transpose(),
                Vector3::zeros(),
                Vector2::new(6., 0.)
            )
            .err(),
            Some("behind_camera")
        );
        let (_, condition, rank) = least_squares(
            DMatrix::from_row_slice(2, 2, &[1., 0., 2., 0.]),
            &DVector::from_vec(vec![1., 2.]),
        )
        .unwrap();
        assert_eq!(rank, 1);
        assert!(condition.is_infinite());
        let bad = Points {
            objects: points.objects,
            images: vec![Vector2::zeros(); 4],
        };
        assert!(!nondegenerate(&bad));
        let overflow = Points {
            objects: vec![Vector3::repeat(1e308); 4],
            images: vec![Vector2::repeat(1e308); 4],
        };
        assert!(!nondegenerate(&overflow));
    }
}
