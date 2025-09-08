from typing import Dict, Any, List, Optional

import numpy as np
from pupil_apriltags import Detection

from src.main_operations.modules.apriltags.utils.apriltag import Apriltag

from src.main_operations.modules.apriltags.ytd_camera_localization.ytd_localization import (
    YtdLocalization,
)
from src.main_operations.modules.apriltags.pnp_localization import PnpLocalization


class FusedLocalization:
    def __init__(
        self,
        camera_matrix: np.ndarray,
        distortion_coefficients: np.ndarray,
        apriltag_map: Dict[int, Apriltag],
    ) -> None:
        """Initialize the FusedLocalization class.

        Args:
            camera_parameters_path (str): Path to the camera parameters file.
            apriltag_map (Dict[int, Apriltag]): Dictionary of apriltag objects.
        """
        self.apriltag_map = apriltag_map

        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients

        self.ytd_localization = YtdLocalization(
            camera_matrix,
            distortion_coefficients,
            apriltag_map,
        )

        self.pnp_localization = PnpLocalization(
            camera_matrix,
            distortion_coefficients,
            apriltag_map,
        )

        self.counter = 0

    def set_attribute(self, attribute_name: str, value: Any) -> None:
        """Set an attribute of the FusedLocalization class.

        Args:
            attribute_name (str): Name of the attribute to set.
            value (Any): Value to set the attribute to.

        Raises:
            ValueError: If the attribute name is invalid.
        """
        self.ytd_localization.set_attribute(attribute_name, value)

    def run(
        self,
        detections: List[Detection] | None,
    ) -> Optional[Dict[str, Any]]:
        """Run the FusedLocalization class.

        Args:
            detections (List[Detection] | None): List of detections.

        Returns:
            Optional[Dict[str, Any]]: Dictionary of fused localization results.
        """
        self.counter += 1
        if len(detections) == 0:
            return None
        elif len(detections) == 1:
            return self.ytd_localization.estimate_pose_from_detections(detections)
        else:
            return self.pnp_localization.estimate_pose_from_detections(detections)
