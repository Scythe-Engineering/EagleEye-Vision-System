use ndarray::{Array2, Array1, s};
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
        angular_gate_threshold: f64,
        velocity_smoothing_alpha: f64,
        full_reset_threshold: usize,
    ) -> Self {
        assert!(
            velocity_smoothing_alpha >= 0.0 && velocity_smoothing_alpha <= 1.0,
            "velocity_smoothing_alpha must be in [0, 1]"
        );
        assert!(base_sigma > 0.0, "base_sigma must be positive");
        assert!(gate_k > 0.0, "gate_k must be positive");
        assert!(relax_factor > 0.0, "relax_factor must be positive");
        assert!(history_size > 0, "history_size must be positive");
        assert!(
            max_consecutive_rejections > 0,
            "max_consecutive_rejections must be positive"
        );
        assert!(
            full_reset_threshold > 0,
            "full_reset_threshold must be positive"
        );

        PoseOutlierFilter {
            history_size,
            base_sigma,
            growth_rate,
            gate_k,
            max_consecutive_rejections,
            relax_factor,
            angular_gate_threshold,
            velocity_smoothing_alpha,
            full_reset_threshold,

            accepted_poses: VecDeque::with_capacity(history_size),
            accepted_timestamps: VecDeque::with_capacity(history_size),
            last_velocity: Array1::zeros(3),
            consecutive_rejections: 0,
            pos_uncertainty: base_sigma,

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
            .map_err(|e| PyValueError::new_err(format!("System time error: {}", e)))?
            .as_secs_f64();

        let current_position = pose_array.slice(s![..3, 3]).to_owned();

        // Extract rotation matrix for angular comparison
        let current_rotation = pose_array.slice(s![..3, ..3]).to_owned();

        // Initialize if this is the first pose
        if !self.has_previous_pose {
            self.initialize_first_pose(&pose_array, current_timestamp);
            return Ok(Some(pose_array.iter().cloned().collect()));
        }

        // Calculate dynamic uncertainty
        let current_sigma = self.base_sigma * (1.0 + self.growth_rate * self.consecutive_rejections as f64);
        self.pos_uncertainty = current_sigma;

        // Predict next position based on velocity
        let predicted_position = self.predict_next_position(current_timestamp);

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

        if let Some(val) = config.get("history_size") {
            let history_size_candidate: usize = val.extract(py)?;
            if history_size_candidate == 0 || history_size_candidate > 1000 {
                return Err(PyValueError::new_err("history_size must be between 1 and 1000"));
            }
            self.history_size = history_size_candidate;
            // Resize deques
            let old_poses: Vec<_> = self.accepted_poses.iter().cloned().collect();
            let old_timestamps: Vec<_> = self.accepted_timestamps.iter().cloned().collect();
            self.accepted_poses = old_poses.into_iter().rev().take(self.history_size).rev().collect();
            self.accepted_timestamps = old_timestamps.into_iter().rev().take(self.history_size).rev().collect();
        }

        if let Some(val) = config.get("base_sigma") {
            let base_sigma_candidate: f64 = val.extract(py)?;
            if base_sigma_candidate <= 0.0 {
                return Err(PyValueError::new_err("base_sigma must be greater than 0"));
            }
            self.base_sigma = base_sigma_candidate;
        }

        if let Some(val) = config.get("growth_rate") {
            let growth_rate_candidate: f64 = val.extract(py)?;
            if growth_rate_candidate < 0.0 {
                return Err(PyValueError::new_err("growth_rate must be non-negative"));
            }
            self.growth_rate = growth_rate_candidate;
        }

        if let Some(val) = config.get("gate_k") {
            let gate_k_candidate: f64 = val.extract(py)?;
            if gate_k_candidate <= 0.0 {
                return Err(PyValueError::new_err("gate_k must be greater than 0"));
            }
            self.gate_k = gate_k_candidate;
        }

        if let Some(val) = config.get("max_consecutive_rejections") {
            let max_consecutive_rejections_candidate: usize = val.extract(py)?;
            if max_consecutive_rejections_candidate == 0 {
                return Err(PyValueError::new_err("max_consecutive_rejections must be greater than 0"));
            }
            self.max_consecutive_rejections = max_consecutive_rejections_candidate;
        }

        if let Some(val) = config.get("relax_factor") {
            let relax_factor_candidate: f64 = val.extract(py)?;
            if relax_factor_candidate <= 0.0 {
                return Err(PyValueError::new_err("relax_factor must be greater than 0"));
            }
            self.relax_factor = relax_factor_candidate;
        }

        if let Some(val) = config.get("angular_gate_threshold") {
            let angular_gate_threshold_candidate: f64 = val.extract(py)?;
            if angular_gate_threshold_candidate < 0.0 || angular_gate_threshold_candidate > std::f64::consts::PI {
                return Err(PyValueError::new_err("angular_gate_threshold must be between 0 and π radians"));
            }
            self.angular_gate_threshold = angular_gate_threshold_candidate;
        }

        if let Some(val) = config.get("velocity_smoothing_alpha") {
            let velocity_smoothing_alpha_candidate: f64 = val.extract(py)?;
            if velocity_smoothing_alpha_candidate < 0.0 || velocity_smoothing_alpha_candidate > 1.0 {
                return Err(PyValueError::new_err("velocity_smoothing_alpha must be between 0 and 1"));
            }
            self.velocity_smoothing_alpha = velocity_smoothing_alpha_candidate;
        }

        if let Some(val) = config.get("full_reset_threshold") {
            let full_reset_threshold_candidate: usize = val.extract(py)?;
            if full_reset_threshold_candidate == 0 {
                return Err(PyValueError::new_err("full_reset_threshold must be greater than 0"));
            }
            self.full_reset_threshold = full_reset_threshold_candidate;
        }

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
                self.last_velocity = &velocity * self.velocity_smoothing_alpha
                    + &self.last_velocity * (1.0 - self.velocity_smoothing_alpha);
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

}

/// Python module initialization
#[pymodule]
fn pose_outlier_filter(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PoseOutlierFilter>()?;
    Ok(())
}
