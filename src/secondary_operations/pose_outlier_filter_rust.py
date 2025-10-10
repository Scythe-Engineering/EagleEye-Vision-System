import numpy as np
from typing import Optional

# Import the Rust module (built automatically)
try:
    from pose_outlier_filter import PoseOutlierFilter as RustPoseOutlierFilter
except ImportError:
    RustPoseOutlierFilter = None


class PoseOutlierFilterRust:
    """Rust-based pose outlier filter for high-performance pose filtering.

    This is a Python wrapper around a Rust implementation of the pose outlier filter.
    It provides the same interface as the Python version but with potentially better
    performance for computationally intensive operations.

    The filter maintains a history of accepted poses and uses constant velocity
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
        """Initialize the Rust-based pose outlier filter.

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
        if RustPoseOutlierFilter is None:
            raise ImportError(
                "Rust pose_outlier_filter module not available. "
                "Please build the Rust extension first."
            )

        self._rust_filter = RustPoseOutlierFilter(
            history_size=int(history_size),
            base_sigma=float(base_sigma),
            growth_rate=float(growth_rate),
            gate_k=float(gate_k),
            max_consecutive_rejections=int(max_consecutive_rejections),
            relax_factor=float(relax_factor),
            min_samples_for_covariance=int(min_samples_for_covariance),
            angular_gate_threshold=float(angular_gate_threshold),
            velocity_smoothing_alpha=float(velocity_smoothing_alpha),
            full_reset_threshold=int(full_reset_threshold),
        )

        # Store parameters for introspection
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

    def run(self, pose: np.ndarray) -> Optional[np.ndarray]:
        """Filter a pose measurement for outliers using predictive gating.

        Args:
            pose (np.ndarray): 4x4 homogeneous transformation matrix representing the pose.

        Returns:
            Optional[np.ndarray]: The pose if accepted, None if rejected as outlier.
        """
        # Convert numpy array to flat list for Rust
        pose_flat = pose.flatten().tolist()

        # Call Rust implementation
        result = self._rust_filter.run(pose_flat)

        if result is None:
            return None

        # Convert back to numpy array
        return np.array(result).reshape((4, 4))

    def update_config(self, json_config: dict) -> None:
        """Update the configuration of the pose outlier filter.

        Args:
            json_config (dict): JSON configuration dictionary.
        """
        self._rust_filter.update_config(json_config)

        # Update stored parameters
        if "history_size" in json_config:
            self.history_size = json_config["history_size"]
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
