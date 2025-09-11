from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from pupil_apriltags import Detection

from src.main_operations.modules.apriltags.utils.apriltag import Apriltag

from src.main_operations.modules.apriltags.ytd_camera_localization.math_processing import (
    estimate_tag_distance_and_horizontal_angle,
)


class YtdLocalization:
    """Implementation of multi-tag camera localization and fusion.

    See wrapper in `src/main_operations/definitions/ytd_camera_localization.py` for usage.
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        distortion_coefficients: np.ndarray,
        apriltag_map: Dict[int, Apriltag],
    ) -> None:
        """Initialize the YtdLocalization class.

        Args:
            camera_parameters_path (str): Path to the camera parameters file.
            apriltag_map (Dict[int, Apriltag]): Dictionary of apriltag objects.
        """
        self.apriltag_map = apriltag_map
        self.camera_yaw = 0
        self.camera_pitch = 0

        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients

    def set_attribute(self, attribute_name: str, value: Any) -> None:
        """Set an attribute of the YtdLocalization class.

        Args:
            attribute_name (str): Name of the attribute to set.
            value (Any): Value to set the attribute to.

        Raises:
            ValueError: If the attribute name is invalid.
        """
        if attribute_name == "camera_yaw":
            self.camera_yaw = value
        elif attribute_name == "camera_pitch":
            self.camera_pitch = value
        else:
            raise ValueError(f"Invalid attribute name: {attribute_name}")

    def estimate_pose_from_detections(
        self,
        detections: List[Detection] | None,
    ) -> Optional[Dict[str, Any]]:
        """Run the YtdLocalization class.

        Args:
            detections (List[Detection]): List of detections.

        Returns:
            Optional[Dict[str, Any]]: Dictionary of the fused position and transform.
        """

        if not detections:
            return None

        positions = np.zeros((len(detections), 2))
        for i, detection in enumerate(detections):
            distance, horizontal_angle = estimate_tag_distance_and_horizontal_angle(
                detection.corners,
                self.apriltag_map[detection.tag_id].local_corners,
                self.distortion_coefficients,
                self.camera_matrix,
                self.camera_pitch,
            )
            tag_global_center = self.apriltag_map[detection.tag_id].global_center

            visual_ray_yaw = np.pi - (self.camera_yaw - horizontal_angle)

            local_x_position = distance * np.cos(visual_ray_yaw)
            local_y_position = distance * np.sin(visual_ray_yaw)

            global_position = np.array(
                [
                    tag_global_center[0] + local_x_position,
                    tag_global_center[1] - local_y_position,
                ],
                dtype=np.float32,
            )

            positions[i, :] = global_position

        if positions.size == 0:
            return None

        median_pos = np.median(positions, axis=0)
        dists = np.linalg.norm(positions - median_pos, axis=1)
        med_dist = np.median(dists)
        abs_dev = np.abs(dists - med_dist)
        mad = np.median(abs_dev)

        if mad <= 1e-6:
            std = np.std(dists)
            if std <= 1e-6:
                inlier_mask = np.ones(len(dists), dtype=bool)
            else:
                inlier_mask = dists <= 3.0 * std
        else:
            inlier_mask = dists <= 3.0 * mad

        inlier_positions = positions[inlier_mask]

        if inlier_positions.size == 0:
            fused_position = median_pos
        else:
            fused_position = inlier_positions.mean(axis=0)

        cos_y = float(np.cos(self.camera_yaw - np.pi / 2))
        sin_y = float(np.sin(self.camera_yaw - np.pi / 2))

        rotation = np.array(
            [[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]]
        )

        transform = np.eye(4, dtype=float)
        transform[:3, :3] = rotation
        transform[:3, 3] = np.array([fused_position[0], fused_position[1], 0.0])

        return transform
