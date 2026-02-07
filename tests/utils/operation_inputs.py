"""Operation input builders for run smoke tests."""

from __future__ import annotations

from typing import Any, Callable, Dict

from tests.utils import dummy_data


InputBuilder = Callable[[], Any]


def get_operation_input_builders() -> Dict[str, InputBuilder]:
    """Return operation-specific input builders."""

    return {
        "device_input": dummy_data.dummy_device_input_data,
        "detect_apriltags": dummy_data.dummy_apriltag_segments,
        "pnp_camera_localization": lambda: [],
        "temporal_acceleration_preprocessor_rust": dummy_data.dummy_temporal_acceleration_input,
        "color_threshold_detection": dummy_data.dummy_frame,
        "ground_plane_intersection": dummy_data.dummy_detections,
        "robot_local_to_field_transform": dummy_data.dummy_robot_local_to_field_input,
        "publish_to_networktables": dummy_data.dummy_networktables_payload,
        "robot_pose_output": dummy_data.dummy_pose_matrix,
        "detected_objects_output": dummy_data.dummy_detections_with_positions,
        "angle_to_objects": dummy_data.dummy_detections,
        "tag_filter": dummy_data.dummy_tag_filter_input,
        "get_networktables_value": dummy_data.dummy_device_input_data,
        "camera_adjust": dummy_data.dummy_camera_adjust_input,
        "fps_limiter": dummy_data.dummy_frame,
        "extract_pose": dummy_data.dummy_pose_matrix,
        "flatten_pose": dummy_data.dummy_pose_matrix,
        "pose_outlier_filter_rust": dummy_data.dummy_pose_matrix,
    }


def get_fallback_input() -> Any:
    """Return fallback input when no builder is available."""

    return dummy_data.dummy_generic_input()
