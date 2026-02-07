"""Dummy data helpers for operation smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class DummyApriltagDetection:
    """Lightweight AprilTag detection stub used in tests."""

    tag_id: int
    corners: np.ndarray


def dummy_frame() -> np.ndarray:
    """Create a dummy frame for tests."""

    width = int(os.environ.get("EAGLEEYE_DUMMY_FRAME_WIDTH", "640"))
    height = int(os.environ.get("EAGLEEYE_DUMMY_FRAME_HEIGHT", "480"))
    return np.zeros((height, width, 3), dtype=np.uint8)


def dummy_pose_matrix() -> np.ndarray:
    """Create a dummy 4x4 pose matrix."""

    return np.eye(4, dtype=float)


def dummy_color_detection() -> Dict[str, Any]:
    """Create a color detection dict with bbox and metadata."""

    return {
        "bbox": [0.4, 0.4, 0.6, 0.6],
        "class_id": 0,
        "color_name": "test",
        "area": 100.0,
    }


def dummy_detection_with_position() -> Dict[str, Any]:
    """Create a detection dict with a 3D position entry."""

    detection = dummy_color_detection()
    detection["position_3d"] = [1.0, 0.0, 2.0]
    return detection


def dummy_detections() -> List[Dict[str, Any]]:
    """Create a list of dummy detections."""

    return [dummy_color_detection()]


def dummy_detections_with_positions() -> List[Dict[str, Any]]:
    """Create a list of dummy detections with 3D positions."""

    return [dummy_detection_with_position()]


def dummy_apriltag_detections() -> List[DummyApriltagDetection]:
    """Create a list of dummy AprilTag detections."""

    corners = np.array(
        [[100.0, 100.0], [140.0, 100.0], [140.0, 140.0], [100.0, 140.0]],
        dtype=float,
    )
    return [DummyApriltagDetection(tag_id=1, corners=corners)]


def dummy_temporal_acceleration_input() -> Dict[str, Any]:
    """Create input payload for temporal acceleration preprocessor."""

    return {
        "frame": dummy_frame(),
        "camera_pose": dummy_pose_matrix(),
    }


def dummy_camera_adjust_input() -> np.ndarray:
    """Create input payload for camera adjust operation."""

    return dummy_frame()


def dummy_networktables_payload() -> Dict[str, Any]:
    """Create a payload compatible with Flatpack schemas."""

    return {"x": 1.0, "y": 2.0, "rotation": 0.1}


def dummy_robot_local_to_field_input() -> Dict[str, Any]:
    """Create a payload for RobotLocalToFieldTransform."""

    return {
        "detections": dummy_detections_with_positions(),
        "robot_pose": dummy_pose_matrix(),
    }


def dummy_device_input_data() -> None:
    """Device input is a data source and ignores input."""

    return None


def dummy_generic_input() -> np.ndarray:
    """Fallback input when an operation doesn't have a contract."""

    return dummy_frame()


def dummy_tag_filter_input() -> List[DummyApriltagDetection]:
    """Input for tag filter operations."""

    return dummy_apriltag_detections()


def dummy_detection_map_for_networktables() -> Tuple[List[Dict[str, Any]], str]:
    """Create data and data_path for NetworkTables publishing tests."""

    detections = [
        {
            "position_3d": {
                "x": 1.0,
                "y": 2.0,
                "z": 3.0,
                "roll": 0.1,
                "pitch": 0.2,
                "yaw": 0.3,
            }
        }
    ]
    return detections, "position_3d"


def dummy_apriltag_segments() -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Input for detect_apriltags when temporal acceleration is used."""

    frame = dummy_frame()
    crop = frame[0:100, 0:100]
    return [(crop, np.array([0, 0]))], frame
