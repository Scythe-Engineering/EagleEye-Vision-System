import numpy as np
from typing import List, Dict, Any
from src.main_operations.definitions.base.base_class import OperationInstance


class AngleToObjects(OperationInstance):
    """Calculate horizontal angles to detected objects from color threshold detections.

    This operation takes color threshold detection results and computes the horizontal
    angle from camera center to each detected object. Results are sorted by detection
    area in descending order (largest first).

    Input: List[Dict[str, Any]] - Color threshold detection results with bbox as normalized 0-1 coordinates
    Output: List[Dict[str, Any]] - Objects with horizontal angles, sorted by area descending

    Each output dict contains:
        - angle_degrees: Horizontal angle in degrees from camera center (-180 to 180)
        - angle_radians: Horizontal angle in radians from camera center
        - bbox: Original bounding box [x1, y1, x2, y2] as normalized coordinates
        - class_id: Integer class identifier for the color
        - color_name: String name of detected color
        - area: Contour area used for sorting
    """

    def __init__(self, camera_fov_degrees: float = 60.0):
        """Initialize angle calculation operation.

        Args:
            camera_fov_degrees: Horizontal field of view in degrees
        """
        self.camera_fov_degrees = camera_fov_degrees
        self.fov_rad = np.deg2rad(camera_fov_degrees)

    def run(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate horizontal angles to detected objects.

        Args:
            detections: List of detection dictionaries from color threshold detection

        Returns:
            List of detection dictionaries with angle information, sorted by area descending
        """
        angled_objects = []

        for detection in detections:
            bbox = detection["bbox"]  # [x1, y1, x2, y2] as normalized 0-1 coordinates
            area = detection.get("area", 0)

            # Calculate center x coordinate as normalized value (0-1)
            center_x_norm = (bbox[0] + bbox[2]) / 2.0

            # Convert to centered normalized coordinate (-1 to 1)
            x_norm_centered = np.clip(2.0 * (center_x_norm - 0.5), -1.0, 1.0)

            # Calculate horizontal angle using pinhole camera model
            # Positive angle = object to the right of center
            # Negative angle = object to the left of center
            horizontal_angle_rad = np.arctan(
                x_norm_centered * np.tan(self.fov_rad / 2.0)
            )
            horizontal_angle_deg = np.rad2deg(horizontal_angle_rad)

            angled_object = {
                "angle_degrees": horizontal_angle_deg,
                "angle_radians": horizontal_angle_rad,
                "bbox": bbox,
                "class_id": detection["class_id"],
                "color_name": detection["color_name"],
                "area": area,
            }
            angled_objects.append(angled_object)

        # Sort by area in descending order (largest first)
        angled_objects.sort(key=lambda x: x["area"], reverse=True)

        return angled_objects
