import numpy as np
from collections import deque
import time
from typing import Optional


class PoseOutlierFilter:
    """Filter out pose outliers using predictive gating based on historical pose data.

    This filter maintains a history of accepted poses and uses constant velocity
    prediction with uncertainty growth to detect and reject outlier measurements.
    """

    def __init__(
        self,
        history_size: int = 20,
        base_sigma: float = 0.1,
        growth_rate: float = 0.2,
        gate_k: float = 3.0,
        max_consecutive_rejections: int = 10,
        relax_factor: float = 2.0,
        min_samples_for_covariance: int = 15,
        angular_gate_threshold: float = 0.5,
        velocity_smoothing_alpha: float = 0.3,
        full_reset_threshold: int = 10,
    ):
        """Initialize the pose outlier filter.

        Args:
            history_size (int): Maximum number of accepted poses to keep in history.
            base_sigma (float): Base uncertainty for position predictions in meters.
            growth_rate (float): Rate at which uncertainty grows with consecutive rejections.
            gate_k (float): Multiplier for uncertainty to create gating threshold.
            max_consecutive_rejections (int): Max rejections before gate relaxation.
            relax_factor (float): Factor by which to relax gate when max rejections reached.
            min_samples_for_covariance (int): Min samples needed for Mahalanobis gating.
            angular_gate_threshold (float): Max angular difference in radians for acceptance.
            velocity_smoothing_alpha (float): Smoothing factor for velocity estimates (0-1).
            full_reset_threshold (int): Number of consecutive rejections to trigger full filter reset.
        """
        self.history_size = history_size
        self.base_sigma = base_sigma
        self.growth_rate = growth_rate
        self.gate_k = gate_k
        self.max_consecutive_rejections = max_consecutive_rejections
        self.relax_factor = relax_factor
        self.min_samples_for_covariance = min_samples_for_covariance
        self.angular_gate_threshold = angular_gate_threshold
        self.velocity_smoothing_alpha = velocity_smoothing_alpha
        self.full_reset_threshold = full_reset_threshold

        # Core state variables
        self.accepted_poses = deque(maxlen=history_size)
        self.accepted_timestamps = deque(maxlen=history_size)
        self.last_velocity = np.zeros(3, dtype=float)
        self.consecutive_rejections = 0
        self.pos_uncertainty = base_sigma

        # Covariance tracking for Mahalanobis gating
        self.pose_covariance = None
        self.covariance_samples = []
        # Rolling window statistics for fast covariance computation
        self._positions_window = deque(maxlen=history_size)
        self._positions_sum = np.zeros(3, dtype=float)
        self._positions_outer_sum = np.zeros((3, 3), dtype=float)
        self._positions_count = 0

        # Internal state
        self._has_previous_pose = False
        self._last_accepted_pose = None
        self._last_accepted_timestamp = None

    def run(self, pose: np.ndarray) -> Optional[np.ndarray]:
        """Filter a pose measurement for outliers using predictive gating.

        Args:
            pose (np.ndarray): 4x4 homogeneous transformation matrix representing the pose.

        Returns:
            Optional[np.ndarray]: The pose if accepted, None if rejected as outlier.
        """
        current_timestamp = time.time()
        current_position = pose[:3, 3].copy()

        # Extract rotation matrix for angular comparison
        current_rotation = pose[:3, :3].copy()

        # Initialize if this is the first pose
        if not self._has_previous_pose:
            self._initialize_first_pose(pose, current_timestamp)
            return pose

        # Predict next pose using constant velocity model
        predicted_position = self._predict_next_position(current_timestamp)

        # Calculate dynamic uncertainty
        current_sigma = self.base_sigma * (
            1.0 + self.growth_rate * self.consecutive_rejections
        )
        self.pos_uncertainty = current_sigma

        # Calculate Euclidean distance to predicted position
        position_error = np.linalg.norm(current_position - predicted_position)

        # Calculate angular error if we have a previous accepted pose
        angular_error = 0.0
        if self._last_accepted_pose is not None:
            angular_error = self._calculate_angular_error(
                current_rotation, self._last_accepted_pose[:3, :3]
            )

        # Determine gate threshold
        gate_threshold = self.gate_k * current_sigma

        # Check if max consecutive rejections reached - relax gate or reset
        if self.consecutive_rejections >= self.max_consecutive_rejections:
            gate_threshold *= self.relax_factor

        # Apply gating criteria
        position_accepted = position_error <= gate_threshold
        angular_accepted = angular_error <= self.angular_gate_threshold

        if position_accepted and angular_accepted:
            # Accept the pose
            self._accept_pose(pose, current_timestamp, current_position)
            return pose
        else:
            # Reject the pose
            self._reject_pose()
            return None

    def _initialize_first_pose(self, pose: np.ndarray, timestamp: float) -> None:
        """Initialize the filter with the first pose measurement.

        Args:
            pose (np.ndarray): The first pose measurement.
            timestamp (float): Timestamp of the first measurement.
        """
        self._last_accepted_pose = pose.copy()
        self._last_accepted_timestamp = timestamp
        self._has_previous_pose = True

        # Add to history
        self.accepted_poses.append(pose.copy())
        self.accepted_timestamps.append(timestamp)

        # Reset rejection counter
        self.consecutive_rejections = 0

    def _predict_next_position(self, current_timestamp: float) -> np.ndarray:
        """Predict the next position using constant velocity model.

        Args:
            current_timestamp (float): Current timestamp for prediction.

        Returns:
            np.ndarray: Predicted position vector.
        """
        if (
            self._last_accepted_pose is None
            or self._last_accepted_timestamp is None
            or current_timestamp <= self._last_accepted_timestamp
        ):
            return np.zeros(3)

        dt = current_timestamp - self._last_accepted_timestamp
        last_position = self._last_accepted_pose[:3, 3]

        # Use constant velocity prediction
        predicted_position = last_position + self.last_velocity * dt
        return predicted_position

    def _calculate_angular_error(
        self, rotation1: np.ndarray, rotation2: np.ndarray
    ) -> float:
        """Calculate angular difference between two rotation matrices.

        Args:
            rotation1 (np.ndarray): First 3x3 rotation matrix.
            rotation2 (np.ndarray): Second 3x3 rotation matrix.

        Returns:
            float: Angular difference in radians.
        """
        # Calculate relative rotation matrix
        relative_rotation = rotation1 @ rotation2.T

        # Convert to axis-angle representation
        trace = np.trace(relative_rotation)
        if trace > 3.0 - 1e-6:  # Very small angle
            return 0.0
        elif trace < -1.0 + 1e-6:  # Large angle, approx pi
            return np.pi
        else:
            angle = np.arccos((trace - 1.0) / 2.0)
            return angle

    def _accept_pose(
        self, pose: np.ndarray, timestamp: float, position: np.ndarray
    ) -> None:
        """Accept a pose and update internal state.

        Args:
            pose (np.ndarray): The accepted pose.
            timestamp (float): Timestamp of the accepted pose.
            position (np.ndarray): Position vector of the accepted pose.
        """
        # Calculate velocity from last accepted pose
        if (
            self._last_accepted_pose is not None
            and self._last_accepted_timestamp is not None
        ):
            dt = timestamp - self._last_accepted_timestamp
            if dt > 1e-6:
                velocity = (position - self._last_accepted_pose[:3, 3]) / dt
                # Exponential smoothing of velocity
                self.last_velocity = (
                    self.velocity_smoothing_alpha * velocity
                    + (1.0 - self.velocity_smoothing_alpha) * self.last_velocity
                )

        # Update last accepted pose
        self._last_accepted_pose = pose.copy()
        self._last_accepted_timestamp = timestamp

        # Add to history
        self.accepted_poses.append(pose.copy())
        self.accepted_timestamps.append(timestamp)

        # Reset rejection counter and uncertainty
        self.consecutive_rejections = 0
        self.pos_uncertainty = self.base_sigma

        # Update covariance if we have enough samples
        self._update_covariance(pose)

    def _reject_pose(self) -> None:
        """Handle rejection of a pose measurement."""
        self.consecutive_rejections += 1

        # If too many consecutive rejections, trigger full reset
        if self.consecutive_rejections >= self.full_reset_threshold:
            self._jump_reset(None, None)  # No pose/timestamp needed for reset

    def _reset_filter(self) -> None:
        """Reset the filter state when too many consecutive rejections occur."""
        self.consecutive_rejections = 0
        self.pos_uncertainty = self.base_sigma
        self.last_velocity = np.zeros(3, dtype=float)
        self.pose_covariance = None
        self.covariance_samples = []
        # Reset rolling covariance state
        self._positions_window.clear()
        self._positions_sum[:] = 0.0
        self._positions_outer_sum[:] = 0.0
        self._positions_count = 0

        # Keep only the most recent pose if available
        if self.accepted_poses:
            last_pose = self.accepted_poses[-1]
            last_timestamp = self.accepted_timestamps[-1]

            # Clear and reinitialize with last pose
            self.accepted_poses.clear()
            self.accepted_timestamps.clear()
            self.accepted_poses.append(last_pose)
            self.accepted_timestamps.append(last_timestamp)

            self._last_accepted_pose = last_pose
            self._last_accepted_timestamp = last_timestamp

    def _jump_reset(
        self, pose: Optional[np.ndarray], timestamp: Optional[float]
    ) -> None:
        """Perform an aggressive reset when too many consecutive rejections occur.

        This method completely reinitializes the filter, clearing all history.
        If pose and timestamp are provided, it treats them as a new trajectory start.
        Used for handling video loops or consecutive rejection recovery.

        Args:
            pose (Optional[np.ndarray]): The pose that triggered the reset, or None.
            timestamp (Optional[float]): Timestamp of the pose, or None.
        """
        # Completely clear all state
        self.accepted_poses.clear()
        self.accepted_timestamps.clear()
        self.last_velocity = np.zeros(3, dtype=float)
        self.consecutive_rejections = 0
        self.pos_uncertainty = self.base_sigma
        self.pose_covariance = None
        self.covariance_samples = []
        # Reset rolling covariance state
        self._positions_window.clear()
        self._positions_sum[:] = 0.0
        self._positions_outer_sum[:] = 0.0
        self._positions_count = 0

        # If we have a pose to start with, initialize with it
        if pose is not None and timestamp is not None:
            self._last_accepted_pose = pose.copy()
            self._last_accepted_timestamp = timestamp
            self._has_previous_pose = True

            # Add to history
            self.accepted_poses.append(pose.copy())
            self.accepted_timestamps.append(timestamp)
        else:
            # Full reset - no pose to start with
            self._last_accepted_pose = None
            self._last_accepted_timestamp = None
            self._has_previous_pose = False

    def _update_covariance(self, pose: np.ndarray) -> None:
        """Update the pose covariance estimate for Mahalanobis gating.

        Args:
            pose (np.ndarray): The pose to add to covariance calculation.
        """
        if len(self.accepted_poses) < self.min_samples_for_covariance:
            return

        position = pose[:3, 3]
        # Maintain rolling window and running sums for O(1) covariance update
        if len(self._positions_window) == self._positions_window.maxlen:
            oldest = self._positions_window.popleft()
            self._positions_sum -= oldest
            self._positions_outer_sum -= np.outer(oldest, oldest)
            self._positions_count -= 1

        self._positions_window.append(position)
        self._positions_sum += position
        self._positions_outer_sum += np.outer(position, position)
        self._positions_count += 1

        if self._positions_count >= max(3, self.min_samples_for_covariance):
            count = float(self._positions_count)
            mean = self._positions_sum / count
            centered_outer = self._positions_outer_sum / count - np.outer(mean, mean)
            # Unbiased estimator (divide by n-1)
            self.pose_covariance = centered_outer * (count / (count - 1.0))

    def update_config(self, json_config: dict) -> None:
        """Update the configuration of the pose outlier filter.

        Args:
            json_config (dict): JSON configuration dictionary.
        """
        if "history_size" in json_config:
            new_size = json_config["history_size"]
            # Recreate deques with new size
            old_poses = list(self.accepted_poses)
            old_timestamps = list(self.accepted_timestamps)
            self.history_size = new_size
            self.accepted_poses = deque(old_poses[-new_size:], maxlen=new_size)
            self.accepted_timestamps = deque(
                old_timestamps[-new_size:], maxlen=new_size
            )
            # Rebuild rolling covariance window to new size
            old_positions = list(self._positions_window)[-new_size:]
            self._positions_window = deque(old_positions, maxlen=new_size)
            if old_positions:
                stacked = np.stack(old_positions, axis=0)
                self._positions_sum = stacked.sum(axis=0)
                self._positions_outer_sum = stacked.T @ stacked
                self._positions_count = len(old_positions)
            else:
                self._positions_sum = np.zeros(3, dtype=float)
                self._positions_outer_sum = np.zeros((3, 3), dtype=float)
                self._positions_count = 0

        if "base_sigma" in json_config:
            self.base_sigma = json_config["base_sigma"]
        if "growth_rate" in json_config:
            self.growth_rate = json_config["growth_rate"]
        if "gate_k" in json_config:
            self.gate_k = json_config["gate_k"]
        if "max_consecutive_rejections" in json_config:
            self.max_consecutive_rejections = json_config["max_consecutive_rejections"]
        if "relax_factor" in json_config:
            self.relax_factor = json_config["relax_factor"]
        if "min_samples_for_covariance" in json_config:
            self.min_samples_for_covariance = json_config["min_samples_for_covariance"]
        if "angular_gate_threshold" in json_config:
            self.angular_gate_threshold = json_config["angular_gate_threshold"]
        if "velocity_smoothing_alpha" in json_config:
            self.velocity_smoothing_alpha = json_config["velocity_smoothing_alpha"]
        if "full_reset_threshold" in json_config:
            self.full_reset_threshold = json_config["full_reset_threshold"]

    def visualize(self, frame: np.ndarray) -> None:
        """Visualize the pose outlier filter outputs.

        This operation returns pose estimation data (transform) only,
        so no frame visualization is available.

        Args:
            frame: Input frame (unused).

        Returns:
            None - no visualization available for transform-only operations.
        """
        return None
