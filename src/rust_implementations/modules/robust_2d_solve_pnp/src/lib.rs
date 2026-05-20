use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

#[pymodule]
fn robust_2d_solve_pnp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Robust2dSolvePnp>()?;
    Ok(())
}

#[derive(Clone)]
struct DetectionSample {
    tag_id: i32,
    image_corners: [[f64; 2]; 4],
    decision_margin: f64,
}

#[derive(Clone, Copy)]
struct PoseState {
    x: f64,
    y: f64,
    yaw: f64,
}

#[pyclass]
pub struct Robust2dSolvePnp {
    camera_matrix: [f64; 9],
    distortion_coefficients: Vec<f64>,
    robot_from_camera: [f64; 16],
    apriltag_corners_by_id: HashMap<i32, [[f64; 3]; 4]>,
    jump_threshold: f64,
    gyro_prior_weight: f64,
    max_iterations: usize,
    last_robot_state: Option<PoseState>,
    last_camera_pose: Option<[f64; 16]>,
    non_finite_count: usize,
}

#[pymethods]
impl Robust2dSolvePnp {
    #[new]
    #[pyo3(signature = (
        camera_matrix,
        distortion_coefficients,
        robot_from_camera,
        apriltag_ids,
        apriltag_corners,
        jump_threshold=2.0,
        gyro_prior_weight=1000000.0,
        max_iterations=20
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        camera_matrix: Vec<f64>,
        distortion_coefficients: Vec<f64>,
        robot_from_camera: Vec<f64>,
        apriltag_ids: Vec<i32>,
        apriltag_corners: Vec<f64>,
        jump_threshold: f64,
        gyro_prior_weight: f64,
        max_iterations: usize,
    ) -> PyResult<Self> {
        if camera_matrix.len() != 9 {
            return Err(PyValueError::new_err("camera_matrix must have 9 elements"));
        }
        if robot_from_camera.len() != 16 {
            return Err(PyValueError::new_err(
                "robot_from_camera must have 16 elements",
            ));
        }
        if apriltag_ids.len() * 12 != apriltag_corners.len() {
            return Err(PyValueError::new_err(
                "apriltag_corners must have 12 floats per tag",
            ));
        }

        let mut camera_matrix_array = [0.0; 9];
        camera_matrix_array.copy_from_slice(&camera_matrix);

        let mut robot_from_camera_array = [0.0; 16];
        robot_from_camera_array.copy_from_slice(&robot_from_camera);

        let mut apriltag_corners_by_id = HashMap::with_capacity(apriltag_ids.len());
        for (tag_index, tag_id) in apriltag_ids.iter().enumerate() {
            let corner_start = tag_index * 12;
            let mut corners = [[0.0; 3]; 4];
            for (corner_index, corner) in corners.iter_mut().enumerate() {
                let source_index = corner_start + corner_index * 3;
                *corner = [
                    apriltag_corners[source_index],
                    apriltag_corners[source_index + 1],
                    apriltag_corners[source_index + 2],
                ];
            }
            apriltag_corners_by_id.insert(*tag_id, corners);
        }

        Ok(Self {
            camera_matrix: camera_matrix_array,
            distortion_coefficients,
            robot_from_camera: robot_from_camera_array,
            apriltag_corners_by_id,
            jump_threshold: jump_threshold.max(0.01),
            gyro_prior_weight: gyro_prior_weight.max(0.0),
            max_iterations: max_iterations.max(1),
            last_robot_state: None,
            last_camera_pose: None,
            non_finite_count: 0,
        })
    }

    fn estimate_pose(
        &mut self,
        tag_ids: Vec<i32>,
        image_corners: Vec<f64>,
        decision_margins: Vec<f64>,
        gyro_yaw: Option<f64>,
    ) -> PyResult<Option<Vec<f64>>> {
        let samples = self.parse_samples(tag_ids, image_corners, decision_margins)?;
        if samples.is_empty() {
            return Ok(None);
        }

        let gyro_constraint =
            gyro_yaw.filter(|yaw| yaw.is_finite() && self.gyro_prior_weight > 0.0);
        let used_previous_seed = self.last_robot_state.is_some();
        let initial_state = self.initial_state(&samples, gyro_constraint);
        let mut solved_state = match self.optimize(&samples, initial_state, gyro_constraint) {
            Some(state) => state,
            None => return Ok(None),
        };
        let mut camera_pose = self.camera_pose_from_robot_state(solved_state);

        if used_previous_seed && self.exceeds_jump_threshold(&camera_pose) {
            self.clear_position_cache();
            let reset_seed = self.seed_from_detections(&samples, gyro_constraint);
            let Some(reset_state) = self.optimize(&samples, reset_seed, gyro_constraint) else {
                return Ok(None);
            };
            solved_state = reset_state;
            camera_pose = self.camera_pose_from_robot_state(solved_state);
        }

        if !camera_pose.iter().all(|value| value.is_finite()) {
            self.non_finite_count += 1;
            if self.non_finite_count >= 3 {
                self.clear_position_cache();
            }
            return Ok(None);
        }

        self.last_robot_state = Some(solved_state);
        self.last_camera_pose = Some(camera_pose);
        self.non_finite_count = 0;
        Ok(Some(camera_pose.to_vec()))
    }

    fn back_propagate_input(&mut self, input_transform: Vec<f64>) -> PyResult<()> {
        if input_transform.len() != 16 {
            return Err(PyValueError::new_err("Transform must have 16 elements"));
        }
        if !input_transform.iter().all(|value| value.is_finite()) {
            return Err(PyValueError::new_err(
                "Transform contains non-finite values",
            ));
        }
        let mut camera_pose = [0.0; 16];
        camera_pose.copy_from_slice(&input_transform);
        self.last_camera_pose = Some(camera_pose);
        Ok(())
    }

    fn clear_position_cache(&mut self) {
        self.last_robot_state = None;
        self.last_camera_pose = None;
        self.non_finite_count = 0;
    }

    fn update_config(&mut self, config: &Bound<'_, PyDict>) -> PyResult<()> {
        if let Some(value) = config.get_item("jump_threshold")? {
            self.jump_threshold = value.extract::<f64>()?.max(0.01);
        }
        if let Some(value) = config.get_item("gyro_prior_weight")? {
            self.gyro_prior_weight = value.extract::<f64>()?.max(0.0);
        }
        if let Some(value) = config.get_item("max_iterations")? {
            self.max_iterations = value.extract::<usize>()?.max(1);
        }
        Ok(())
    }
}

impl Robust2dSolvePnp {
    fn parse_samples(
        &self,
        tag_ids: Vec<i32>,
        image_corners: Vec<f64>,
        decision_margins: Vec<f64>,
    ) -> PyResult<Vec<DetectionSample>> {
        if tag_ids.len() * 8 != image_corners.len() {
            return Err(PyValueError::new_err(
                "image_corners must have 8 floats per detection",
            ));
        }
        if decision_margins.len() != tag_ids.len() {
            return Err(PyValueError::new_err(
                "decision_margins must match tag_ids length",
            ));
        }

        let mut samples = Vec::with_capacity(tag_ids.len());
        for (sample_index, tag_id) in tag_ids.iter().enumerate() {
            if !self.apriltag_corners_by_id.contains_key(tag_id) {
                continue;
            }

            let corner_start = sample_index * 8;
            let mut corners = [[0.0; 2]; 4];
            let mut corners_are_finite = true;
            for (corner_index, stored_corner) in corners.iter_mut().enumerate() {
                let source_index = corner_start + corner_index * 2;
                let corner = [image_corners[source_index], image_corners[source_index + 1]];
                corners_are_finite &= corner[0].is_finite() && corner[1].is_finite();
                *stored_corner = corner;
            }
            if !corners_are_finite {
                continue;
            }

            let decision_margin = decision_margins[sample_index];
            samples.push(DetectionSample {
                tag_id: *tag_id,
                image_corners: corners,
                decision_margin,
            });
        }
        Ok(samples)
    }

    fn initial_state(
        &self,
        samples: &[DetectionSample],
        gyro_constraint: Option<f64>,
    ) -> PoseState {
        match self.last_robot_state {
            Some(mut state) => {
                if let Some(gyro_yaw) = gyro_constraint {
                    state.yaw = gyro_yaw;
                }
                state
            }
            None => self.seed_from_detections(samples, gyro_constraint),
        }
    }

    fn seed_from_detections(
        &self,
        samples: &[DetectionSample],
        gyro_constraint: Option<f64>,
    ) -> PoseState {
        let yaw = gyro_constraint.unwrap_or(0.0);
        let robot_rotation = yaw_to_rotation(yaw);
        let robot_from_camera_rotation = mat3_from_transform(&self.robot_from_camera);
        let world_from_camera_rotation = mat3_mul(robot_rotation, robot_from_camera_rotation);
        let robot_from_camera_translation = transform_translation(&self.robot_from_camera);
        let fx = self.camera_matrix[0];
        let fy = self.camera_matrix[4];
        let cx = self.camera_matrix[2];
        let cy = self.camera_matrix[5];

        let mut x_sum = 0.0;
        let mut y_sum = 0.0;
        let mut sample_count = 0.0;

        for sample in samples {
            let Some(world_corners) = self.apriltag_corners_by_id.get(&sample.tag_id) else {
                continue;
            };
            let world_center = corners_center3(world_corners);
            let image_center = corners_center2(&sample.image_corners);
            let tag_size = average_world_side_length(world_corners);
            let pixel_size = average_image_side_length(&sample.image_corners);
            if tag_size <= 0.0
                || pixel_size <= 1.0
                || fx.abs() <= f64::EPSILON
                || fy.abs() <= f64::EPSILON
            {
                continue;
            }

            let depth = (fx.abs() * tag_size / pixel_size).max(0.1);
            let camera_center = [
                (image_center[0] - cx) * depth / fx,
                (image_center[1] - cy) * depth / fy,
                depth,
            ];
            let camera_translation = vec3_sub(
                world_center,
                mat3_mul_vec3(world_from_camera_rotation, camera_center),
            );
            let robot_translation = vec3_sub(
                camera_translation,
                mat3_mul_vec3(robot_rotation, robot_from_camera_translation),
            );
            if robot_translation[0].is_finite() && robot_translation[1].is_finite() {
                x_sum += robot_translation[0];
                y_sum += robot_translation[1];
                sample_count += 1.0;
            }
        }

        if sample_count > 0.0 {
            PoseState {
                x: x_sum / sample_count,
                y: y_sum / sample_count,
                yaw,
            }
        } else {
            PoseState {
                x: 0.0,
                y: 0.0,
                yaw,
            }
        }
    }

    fn optimize(
        &self,
        samples: &[DetectionSample],
        initial_state: PoseState,
        gyro_constraint: Option<f64>,
    ) -> Option<PoseState> {
        let yaw_is_fixed = gyro_constraint.is_some() && self.gyro_prior_weight > 0.0;
        let parameter_count = if yaw_is_fixed { 2 } else { 3 };
        let mut state = initial_state;
        if let Some(gyro_yaw) = gyro_constraint {
            state.yaw = gyro_yaw;
        }

        let mut damping = 1.0e-3;
        let mut residuals = self.residuals(samples, state)?;
        if residuals.len() < parameter_count {
            return None;
        }
        let mut current_cost = squared_norm(&residuals);

        for _ in 0..self.max_iterations {
            let mut jacobian_columns = Vec::with_capacity(parameter_count);
            for parameter_index in 0..parameter_count {
                let epsilon = if parameter_index == 2 { 1.0e-5 } else { 1.0e-4 };
                let perturbed_state = perturb_state(state, parameter_index, epsilon, yaw_is_fixed);
                let perturbed_residuals = self.residuals(samples, perturbed_state)?;
                let column = residuals
                    .iter()
                    .zip(perturbed_residuals.iter())
                    .map(|(base, perturbed)| (perturbed - base) / epsilon)
                    .collect::<Vec<_>>();
                jacobian_columns.push(column);
            }

            let (normal_matrix, gradient) =
                build_normal_equations(&jacobian_columns, &residuals, damping);
            let step =
                solve_linear_system(normal_matrix, gradient.iter().map(|value| -value).collect())?;

            let candidate_state = apply_step(state, &step, yaw_is_fixed, gyro_constraint);
            let Some(candidate_residuals) = self.residuals(samples, candidate_state) else {
                damping *= 10.0;
                continue;
            };
            let candidate_cost = squared_norm(&candidate_residuals);

            if candidate_cost < current_cost {
                state = candidate_state;
                residuals = candidate_residuals;
                current_cost = candidate_cost;
                damping = (damping * 0.4).max(1.0e-9);
                if step.iter().map(|value| value * value).sum::<f64>().sqrt() < 1.0e-6 {
                    break;
                }
            } else {
                damping *= 8.0;
            }
        }

        Some(state)
    }

    fn residuals(&self, samples: &[DetectionSample], state: PoseState) -> Option<Vec<f64>> {
        let world_from_camera = self.camera_pose_from_robot_state(state);
        let world_to_camera = invert_se3(&world_from_camera);
        let world_to_camera_rotation = mat3_from_transform(&world_to_camera);
        let world_to_camera_translation = transform_translation(&world_to_camera);
        let mut residuals = Vec::with_capacity(samples.len() * 8);

        for sample in samples {
            let Some(world_corners) = self.apriltag_corners_by_id.get(&sample.tag_id) else {
                continue;
            };
            let margin_weight = decision_margin_weight(sample.decision_margin);
            for (corner_index, world_corner) in world_corners.iter().enumerate() {
                let camera_point = vec3_add(
                    mat3_mul_vec3(world_to_camera_rotation, *world_corner),
                    world_to_camera_translation,
                );
                let projected = self.project(camera_point)?;
                let depth_weight = distance_weight(camera_point[2]);
                let residual_weight = (margin_weight * depth_weight).sqrt();
                residuals
                    .push((projected[0] - sample.image_corners[corner_index][0]) * residual_weight);
                residuals
                    .push((projected[1] - sample.image_corners[corner_index][1]) * residual_weight);
            }
        }

        if residuals.is_empty() || !residuals.iter().all(|value| value.is_finite()) {
            return None;
        }
        Some(residuals)
    }

    fn project(&self, camera_point: [f64; 3]) -> Option<[f64; 2]> {
        if !camera_point.iter().all(|value| value.is_finite()) || camera_point[2] <= 0.01 {
            return None;
        }
        let x_normalized = camera_point[0] / camera_point[2];
        let y_normalized = camera_point[1] / camera_point[2];
        let (x_distorted, y_distorted) =
            distort_brown_conrady(x_normalized, y_normalized, &self.distortion_coefficients);
        Some([
            self.camera_matrix[0] * x_distorted + self.camera_matrix[2],
            self.camera_matrix[4] * y_distorted + self.camera_matrix[5],
        ])
    }

    fn camera_pose_from_robot_state(&self, state: PoseState) -> [f64; 16] {
        let world_from_robot = yaw_translation_transform(state.x, state.y, state.yaw);
        mat4_mul(&world_from_robot, &self.robot_from_camera)
    }

    fn exceeds_jump_threshold(&self, candidate_pose: &[f64; 16]) -> bool {
        let Some(last_pose) = self.last_camera_pose else {
            return false;
        };
        let delta = [
            candidate_pose[3] - last_pose[3],
            candidate_pose[7] - last_pose[7],
            candidate_pose[11] - last_pose[11],
        ];
        vec3_norm(delta) > self.jump_threshold
    }
}

fn perturb_state(
    state: PoseState,
    parameter_index: usize,
    epsilon: f64,
    yaw_is_fixed: bool,
) -> PoseState {
    match (parameter_index, yaw_is_fixed) {
        (0, _) => PoseState {
            x: state.x + epsilon,
            ..state
        },
        (1, _) => PoseState {
            y: state.y + epsilon,
            ..state
        },
        (2, false) => PoseState {
            yaw: state.yaw + epsilon,
            ..state
        },
        _ => state,
    }
}

fn apply_step(
    state: PoseState,
    step: &[f64],
    yaw_is_fixed: bool,
    gyro_constraint: Option<f64>,
) -> PoseState {
    let yaw = if yaw_is_fixed {
        gyro_constraint.unwrap_or(state.yaw)
    } else {
        wrap_angle(state.yaw + step.get(2).copied().unwrap_or(0.0))
    };
    PoseState {
        x: state.x + step.first().copied().unwrap_or(0.0),
        y: state.y + step.get(1).copied().unwrap_or(0.0),
        yaw,
    }
}

fn build_normal_equations(
    jacobian_columns: &[Vec<f64>],
    residuals: &[f64],
    damping: f64,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let parameter_count = jacobian_columns.len();
    let mut normal_matrix = vec![vec![0.0; parameter_count]; parameter_count];
    let mut gradient = vec![0.0; parameter_count];

    for (row_index, residual) in residuals.iter().enumerate() {
        for (column_index, jacobian_column) in jacobian_columns.iter().enumerate() {
            let jacobian_value = jacobian_column[row_index];
            gradient[column_index] += jacobian_value * residual;
            for (other_column_index, other_jacobian_column) in jacobian_columns.iter().enumerate() {
                normal_matrix[column_index][other_column_index] +=
                    jacobian_value * other_jacobian_column[row_index];
            }
        }
    }

    for (diagonal_index, normal_row) in normal_matrix.iter_mut().enumerate() {
        normal_row[diagonal_index] += damping;
    }
    (normal_matrix, gradient)
}

fn solve_linear_system(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    let size = rhs.len();
    for pivot_index in 0..size {
        let mut best_row = pivot_index;
        for candidate_row in pivot_index + 1..size {
            if matrix[candidate_row][pivot_index].abs() > matrix[best_row][pivot_index].abs() {
                best_row = candidate_row;
            }
        }
        if matrix[best_row][pivot_index].abs() < 1.0e-12 {
            return None;
        }
        matrix.swap(pivot_index, best_row);
        rhs.swap(pivot_index, best_row);

        let pivot_value = matrix[pivot_index][pivot_index];
        for column_index in pivot_index..size {
            matrix[pivot_index][column_index] /= pivot_value;
        }
        rhs[pivot_index] /= pivot_value;

        for row_index in 0..size {
            if row_index == pivot_index {
                continue;
            }
            let factor = matrix[row_index][pivot_index];
            for column_index in pivot_index..size {
                matrix[row_index][column_index] -= factor * matrix[pivot_index][column_index];
            }
            rhs[row_index] -= factor * rhs[pivot_index];
        }
    }
    Some(rhs)
}

fn yaw_translation_transform(x: f64, y: f64, yaw: f64) -> [f64; 16] {
    let cos_yaw = yaw.cos();
    let sin_yaw = yaw.sin();
    [
        cos_yaw, -sin_yaw, 0.0, x, sin_yaw, cos_yaw, 0.0, y, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

fn yaw_to_rotation(yaw: f64) -> [[f64; 3]; 3] {
    let cos_yaw = yaw.cos();
    let sin_yaw = yaw.sin();
    [
        [cos_yaw, -sin_yaw, 0.0],
        [sin_yaw, cos_yaw, 0.0],
        [0.0, 0.0, 1.0],
    ]
}

fn mat4_mul(left: &[f64; 16], right: &[f64; 16]) -> [f64; 16] {
    let mut output = [0.0; 16];
    for row in 0..4 {
        for column in 0..4 {
            output[row * 4 + column] = (0..4)
                .map(|inner| left[row * 4 + inner] * right[inner * 4 + column])
                .sum();
        }
    }
    output
}

fn invert_se3(transform: &[f64; 16]) -> [f64; 16] {
    let rotation = mat3_from_transform(transform);
    let rotation_transpose = transpose3(rotation);
    let translation = transform_translation(transform);
    let inverse_translation = mat3_mul_vec3(
        rotation_transpose,
        [-translation[0], -translation[1], -translation[2]],
    );
    [
        rotation_transpose[0][0],
        rotation_transpose[0][1],
        rotation_transpose[0][2],
        inverse_translation[0],
        rotation_transpose[1][0],
        rotation_transpose[1][1],
        rotation_transpose[1][2],
        inverse_translation[1],
        rotation_transpose[2][0],
        rotation_transpose[2][1],
        rotation_transpose[2][2],
        inverse_translation[2],
        0.0,
        0.0,
        0.0,
        1.0,
    ]
}

fn mat3_from_transform(transform: &[f64; 16]) -> [[f64; 3]; 3] {
    [
        [transform[0], transform[1], transform[2]],
        [transform[4], transform[5], transform[6]],
        [transform[8], transform[9], transform[10]],
    ]
}

fn transform_translation(transform: &[f64; 16]) -> [f64; 3] {
    [transform[3], transform[7], transform[11]]
}

fn mat3_mul(left: [[f64; 3]; 3], right: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut output = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            output[row][column] = (0..3)
                .map(|inner| left[row][inner] * right[inner][column])
                .sum();
        }
    }
    output
}

fn mat3_mul_vec3(matrix: [[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    ]
}

fn transpose3(matrix: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [matrix[0][0], matrix[1][0], matrix[2][0]],
        [matrix[0][1], matrix[1][1], matrix[2][1]],
        [matrix[0][2], matrix[1][2], matrix[2][2]],
    ]
}

fn vec3_add(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn vec3_sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn vec3_norm(vector: [f64; 3]) -> f64 {
    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
}

fn corners_center3(corners: &[[f64; 3]; 4]) -> [f64; 3] {
    let mut center = [0.0; 3];
    for corner in corners {
        center[0] += corner[0];
        center[1] += corner[1];
        center[2] += corner[2];
    }
    [center[0] * 0.25, center[1] * 0.25, center[2] * 0.25]
}

fn corners_center2(corners: &[[f64; 2]; 4]) -> [f64; 2] {
    let mut center = [0.0; 2];
    for corner in corners {
        center[0] += corner[0];
        center[1] += corner[1];
    }
    [center[0] * 0.25, center[1] * 0.25]
}

fn average_world_side_length(corners: &[[f64; 3]; 4]) -> f64 {
    let side_a = vec3_norm(vec3_sub(corners[1], corners[0]));
    let side_b = vec3_norm(vec3_sub(corners[2], corners[1]));
    let side_c = vec3_norm(vec3_sub(corners[3], corners[2]));
    let side_d = vec3_norm(vec3_sub(corners[0], corners[3]));
    (side_a + side_b + side_c + side_d) * 0.25
}

fn average_image_side_length(corners: &[[f64; 2]; 4]) -> f64 {
    let mut side_sum = 0.0;
    for side_index in 0..4 {
        let next_index = (side_index + 1) % 4;
        let dx = corners[next_index][0] - corners[side_index][0];
        let dy = corners[next_index][1] - corners[side_index][1];
        side_sum += (dx * dx + dy * dy).sqrt();
    }
    side_sum * 0.25
}

fn distort_brown_conrady(x: f64, y: f64, coefficients: &[f64]) -> (f64, f64) {
    let k1 = coefficients.first().copied().unwrap_or(0.0);
    let k2 = coefficients.get(1).copied().unwrap_or(0.0);
    let p1 = coefficients.get(2).copied().unwrap_or(0.0);
    let p2 = coefficients.get(3).copied().unwrap_or(0.0);
    let k3 = coefficients.get(4).copied().unwrap_or(0.0);
    let radius_squared = x * x + y * y;
    let radial = 1.0
        + k1 * radius_squared
        + k2 * radius_squared * radius_squared
        + k3 * radius_squared * radius_squared * radius_squared;
    let x_distorted = x * radial + 2.0 * p1 * x * y + p2 * (radius_squared + 2.0 * x * x);
    let y_distorted = y * radial + p1 * (radius_squared + 2.0 * y * y) + 2.0 * p2 * x * y;
    (x_distorted, y_distorted)
}

fn decision_margin_weight(decision_margin: f64) -> f64 {
    if !decision_margin.is_finite() || decision_margin <= 0.0 {
        return 0.25;
    }
    (decision_margin / 50.0).clamp(0.15, 1.0)
}

fn distance_weight(depth: f64) -> f64 {
    if !depth.is_finite() || depth <= 0.0 {
        return 0.05;
    }
    (1.0 / (1.0 + 0.04 * depth * depth)).clamp(0.05, 1.0)
}

fn squared_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum()
}

fn wrap_angle(angle: f64) -> f64 {
    let two_pi = std::f64::consts::PI * 2.0;
    (angle + std::f64::consts::PI).rem_euclid(two_pi) - std::f64::consts::PI
}
