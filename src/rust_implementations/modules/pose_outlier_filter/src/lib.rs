use ndarray::{Array2, Array1, Axis, s};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

/// Pose outlier filter implemented in Rust for high performance.
///
/// This filter maintains a history of accepted poses and uses constant velocity
/// prediction with uncertainty growth to detect and reject outlier measurements.
#[pyclass]
#[derive(Clone)]
pub struct PoseOutlierFilter {
    /// Maximum number of accepted poses to keep in history
    #[pyo3(get, set)]
    history_size: usize,

    /// Base uncertainty for position predictions in meters
    #[pyo3(get, set)]
    base_sigma: f64,

    /// Rate at which uncertainty grows with consecutive rejections
    #[pyo3(get, set)]
    growth_rate: f64,

    /// Multiplier for uncertainty to create gating threshold
    #[pyo3(get, set)]
    gate_k: f64,

    /// Max rejections before gate relaxation
    #[pyo3(get, set)]
    max_consecutive_rejections: usize,

    /// Factor by which to relax gate when max rejections reached
    #[pyo3(get, set)]
    relax_factor: f64,

    /// Min samples needed for Mahalanobis gating
    #[pyo3(get, set)]
    min_samples_for_covariance: usize,

    /// Max angular difference in radians for acceptance
    #[pyo3(get, set)]
    angular_gate_threshold: f64,

    /// Smoothing factor for velocity estimates (0-1)
    #[pyo3(get, set)]
    velocity_smoothing_alpha: f64,

    /// Number of consecutive rejections to trigger full filter reset
    #[pyo3(get, set)]
    full_reset_threshold: usize,

    // Internal state
    accepted_poses: VecDeque<Array2<f64>>,
    accepted_timestamps: VecDeque<f64>,
    last_velocity: Array1<f64>,
    consecutive_rejections: usize,
    pos_uncertainty: f64,

    // Covariance tracking
    pose_covariance: Option<Array2<f64>>,

    // Rolling window statistics for covariance
    positions_window: VecDeque<Array1<f64>>,
    positions_sum: Array1<f64>,
    positions_outer_sum: Array2<f64>,
    positions_count: usize,

    // State flags
    has_previous_pose: bool,
    last_accepted_pose: Option<Array2<f64>>,
    last_accepted_timestamp: Option<f64>,
}

#[pymethods]
impl PoseOutlierFilter {
    #[new]
    #[pyo3(signature = (
        history_size=20,
        base_sigma=0.1,
        growth_rate=0.2,
        gate_k=3.0,
        max_consecutive_rejections=10,
        relax_factor=2.0,
        min_samples_for_covariance=15,
        angular_gate_threshold=0.5,
        velocity_smoothing_alpha=0.3,
        full_reset_threshold=10
    ))]
    fn new(
        history_size: usize,
        base_sigma: f64,
        growth_rate: f64,
        gate_k: f64,
        max_consecutive_rejections: usize,
        relax_factor: f64,
        min_samples_for_covariance: usize,
        angular_gate_threshold: f64,
        velocity_smoothing_alpha: f64,
        full_reset_threshold: usize,
    ) -> Self {
        PoseOutlierFilter {
            history_size,
            base_sigma,
            growth_rate,
            gate_k,
            max_consecutive_rejections,
            relax_factor,
            min_samples_for_covariance,
            angular_gate_threshold,
            velocity_smoothing_alpha,
            full_reset_threshold,

            accepted_poses: VecDeque::with_capacity(history_size),
            accepted_timestamps: VecDeque::with_capacity(history_size),
            last_velocity: Array1::zeros(3),
            consecutive_rejections: 0,
            pos_uncertainty: base_sigma,

            pose_covariance: None,

            positions_window: VecDeque::with_capacity(history_size),
            positions_sum: Array1::zeros(3),
            positions_outer_sum: Array2::zeros((3, 3)),
            positions_count: 0,

            has_previous_pose: false,
            last_accepted_pose: None,
            last_accepted_timestamp: None,
        }
    }

    /// Filter a pose measurement for outliers using predictive gating.
    ///
    /// Args:
    ///     pose: 4x4 homogeneous transformation matrix as a flat array (16 elements)
    ///
    /// Returns:
    ///     The pose as a flat array if accepted, None if rejected
    #[pyo3(signature = (pose))]
    fn run(&mut self, pose: Vec<f64>) -> PyResult<Option<Vec<f64>>> {
        if pose.len() != 16 {
            return Err(PyValueError::new_err("Pose must be a 16-element array representing a 4x4 matrix"));
        }

        let pose_array = Array2::from_shape_vec((4, 4), pose)
            .map_err(|_| PyValueError::new_err("Failed to reshape pose into 4x4 matrix"))?;

        let current_timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let current_position = pose_array.slice(s![..3, 3]).to_owned();

        // Extract rotation matrix for angular comparison
        let current_rotation = pose_array.slice(s![..3, ..3]).to_owned();

        // Initialize if this is the first pose
        if !self.has_previous_pose {
            self.initialize_first_pose(&pose_array, current_timestamp);
            return Ok(Some(pose_array.iter().cloned().collect()));
        }

        // Predict next pose using constant velocity model
        let predicted_position = self.predict_next_position(current_timestamp);

        // Calculate dynamic uncertainty
        let current_sigma = self.base_sigma * (1.0 + self.growth_rate * self.consecutive_rejections as f64);
        self.pos_uncertainty = current_sigma;

        // Calculate Euclidean distance to predicted position
        let position_error = (&current_position - &predicted_position).mapv(|x| x * x).sum().sqrt();

        // Calculate angular error if we have a previous accepted pose
        let angular_error = if let Some(ref last_pose) = self.last_accepted_pose {
            let last_rotation = last_pose.slice(s![..3, ..3]).to_owned();
            self.calculate_angular_error(&current_rotation, &last_rotation)
        } else {
            0.0
        };

        // Determine gate threshold
        let mut gate_threshold = self.gate_k * current_sigma;

        // Check if max consecutive rejections reached - relax gate or reset
        if self.consecutive_rejections >= self.max_consecutive_rejections {
            gate_threshold *= self.relax_factor;
        }

        // Apply gating criteria
        let position_accepted = position_error <= gate_threshold;
        let angular_accepted = angular_error <= self.angular_gate_threshold;

        if position_accepted && angular_accepted {
            // Accept the pose
            self.accept_pose(&pose_array, current_timestamp, &current_position);
            Ok(Some(pose_array.iter().cloned().collect()))
        } else {
            // Reject the pose
            self.reject_pose();
            Ok(None)
        }
    }

    /// Update configuration parameters
    #[pyo3(signature = (config))]
    fn update_config(&mut self, config: std::collections::HashMap<String, PyObject>, py: Python) -> PyResult<()> {
        use pyo3::FromPyObject;

        if let Some(val) = config.get("history_size") {
            self.history_size = val.extract(py)?;
            // Resize deques
            let old_poses: Vec<_> = self.accepted_poses.iter().cloned().collect();
            let old_timestamps: Vec<_> = self.accepted_timestamps.iter().cloned().collect();
            self.accepted_poses = old_poses.into_iter().rev().take(self.history_size).rev().collect();
            self.accepted_timestamps = old_timestamps.into_iter().rev().take(self.history_size).rev().collect();

            // Rebuild rolling covariance window
            let old_positions: Vec<_> = self.positions_window.iter().cloned().collect();
            self.positions_window = old_positions.into_iter().rev().take(self.history_size).rev().collect();

            if !self.positions_window.is_empty() {
                let stacked: Array2<f64> = Array2::from_shape_vec(
                    (self.positions_window.len(), 3),
                    self.positions_window.iter().flatten().cloned().collect::<Vec<_>>()
                ).unwrap();
                self.positions_sum = stacked.sum_axis(Axis(0));
                self.positions_outer_sum = stacked.t().dot(&stacked);
                self.positions_count = self.positions_window.len();
            } else {
                self.positions_sum = Array1::zeros(3);
                self.positions_outer_sum = Array2::zeros((3, 3));
                self.positions_count = 0;
            }
        }

        if let Some(val) = config.get("base_sigma") { self.base_sigma = val.extract(py)?; }
        if let Some(val) = config.get("growth_rate") { self.growth_rate = val.extract(py)?; }
        if let Some(val) = config.get("gate_k") { self.gate_k = val.extract(py)?; }
        if let Some(val) = config.get("max_consecutive_rejections") { self.max_consecutive_rejections = val.extract(py)?; }
        if let Some(val) = config.get("relax_factor") { self.relax_factor = val.extract(py)?; }
        if let Some(val) = config.get("min_samples_for_covariance") { self.min_samples_for_covariance = val.extract(py)?; }
        if let Some(val) = config.get("angular_gate_threshold") { self.angular_gate_threshold = val.extract(py)?; }
        if let Some(val) = config.get("velocity_smoothing_alpha") { self.velocity_smoothing_alpha = val.extract(py)?; }
        if let Some(val) = config.get("full_reset_threshold") { self.full_reset_threshold = val.extract(py)?; }

        Ok(())
    }
}

impl PoseOutlierFilter {
    fn initialize_first_pose(&mut self, pose: &Array2<f64>, timestamp: f64) {
        self.last_accepted_pose = Some(pose.clone());
        self.last_accepted_timestamp = Some(timestamp);
        self.has_previous_pose = true;

        // Add to history
        if self.accepted_poses.len() >= self.history_size {
            self.accepted_poses.pop_front();
            self.accepted_timestamps.pop_front();
        }
        self.accepted_poses.push_back(pose.clone());
        self.accepted_timestamps.push_back(timestamp);

        // Reset rejection counter
        self.consecutive_rejections = 0;
    }

    fn predict_next_position(&self, current_timestamp: f64) -> Array1<f64> {
        if let (Some(last_pose), Some(last_timestamp)) = (&self.last_accepted_pose, self.last_accepted_timestamp) {
            let last_position = last_pose.slice(s![..3, 3]);
            if current_timestamp > last_timestamp {
                let dt = current_timestamp - last_timestamp;
                return &last_position + &(&self.last_velocity * dt);
            } else {
                return last_position.to_owned();
            }
        }
        Array1::zeros(3)
    }

    fn calculate_angular_error(&self, rotation1: &Array2<f64>, rotation2: &Array2<f64>) -> f64 {
        // Calculate relative rotation matrix
        let relative_rotation = rotation1.dot(&rotation2.t());

        // Convert to axis-angle representation
        let trace = relative_rotation.diag().sum();
        if trace > 3.0 - 1e-6 {
            0.0
        } else if trace < -1.0 + 1e-6 {
            std::f64::consts::PI
        } else {
            let cos_theta = ((trace - 1.0) / 2.0).max(-1.0).min(1.0);
            cos_theta.acos()
        }
    }

    fn accept_pose(&mut self, pose: &Array2<f64>, timestamp: f64, position: &Array1<f64>) {
        // Calculate velocity from last accepted pose
        if let (Some(last_pose), Some(last_timestamp)) = (&self.last_accepted_pose, self.last_accepted_timestamp) {
            let dt = timestamp - last_timestamp;
            if dt > 1e-6 {
                let last_position = last_pose.slice(s![..3, 3]);
                let velocity = (position - &last_position) / dt;
                // Exponential smoothing of velocity
                self.last_velocity = self.velocity_smoothing_alpha * &velocity
                    + (1.0 - self.velocity_smoothing_alpha) * &self.last_velocity;
            }
        }

        // Update last accepted pose
        self.last_accepted_pose = Some(pose.clone());
        self.last_accepted_timestamp = Some(timestamp);

        // Add to history (bounded by history_size)
        if self.accepted_poses.len() >= self.history_size {
            self.accepted_poses.pop_front();
            self.accepted_timestamps.pop_front();
        }
        self.accepted_poses.push_back(pose.clone());
        self.accepted_timestamps.push_back(timestamp);

        // Reset rejection counter and uncertainty
        self.consecutive_rejections = 0;
        self.pos_uncertainty = self.base_sigma;

        // Update covariance (window accumulates from first sample)
        self.update_covariance(pose);
    }

    fn reject_pose(&mut self) {
        self.consecutive_rejections += 1;

        // If too many consecutive rejections, trigger full reset
        if self.consecutive_rejections >= self.full_reset_threshold {
            self.jump_reset(None, None);
        }
    }

    fn jump_reset(&mut self, pose: Option<&Array2<f64>>, timestamp: Option<f64>) {
        // Completely clear all state
        self.accepted_poses.clear();
        self.accepted_timestamps.clear();
        self.last_velocity = Array1::zeros(3);
        self.consecutive_rejections = 0;
        self.pos_uncertainty = self.base_sigma;
        self.pose_covariance = None;

        // Reset rolling covariance state
        self.positions_window.clear();
        self.positions_sum = Array1::zeros(3);
        self.positions_outer_sum = Array2::zeros((3, 3));
        self.positions_count = 0;

        // If we have a pose to start with, initialize with it
        if let (Some(p), Some(t)) = (pose, timestamp) {
            self.last_accepted_pose = Some(p.clone());
            self.last_accepted_timestamp = Some(t);
            self.has_previous_pose = true;

            // Add to history
            self.accepted_poses.push_back(p.clone());
            self.accepted_timestamps.push_back(t);
        } else {
            // Full reset - no pose to start with
            self.last_accepted_pose = None;
            self.last_accepted_timestamp = None;
            self.has_previous_pose = false;
        }
    }

    fn update_covariance(&mut self, pose: &Array2<f64>) {
        let position = pose.slice(s![..3, 3]).to_owned();

        // Maintain rolling window and running sums for O(1) covariance update
        if self.positions_window.len() == self.history_size {
            let oldest = self.positions_window.pop_front().unwrap();
            self.positions_sum -= &oldest;
            self.positions_outer_sum -= &oldest.clone().insert_axis(Axis(1)).dot(&oldest.insert_axis(Axis(0)));
            self.positions_count -= 1;
        }

        self.positions_window.push_back(position.clone());
        self.positions_sum += &position;
        self.positions_outer_sum += &position.clone().insert_axis(Axis(1)).dot(&position.insert_axis(Axis(0)));
        self.positions_count += 1;

        if self.positions_count >= self.min_samples_for_covariance.max(3) {
            let count = self.positions_count as f64;
            let mean = &self.positions_sum / count;
            let centered_outer = &self.positions_outer_sum / count - mean.clone().insert_axis(Axis(1)).dot(&mean.insert_axis(Axis(0)));
            // Unbiased estimator (divide by n-1)
            self.pose_covariance = Some(centered_outer * (count / (count - 1.0)));
        }
    }
}

/// Python module initialization
#[pymodule]
fn pose_outlier_filter(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PoseOutlierFilter>()?;
    Ok(())
}
