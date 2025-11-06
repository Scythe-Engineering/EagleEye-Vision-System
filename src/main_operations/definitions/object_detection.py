import cv2
from threading import Lock
from typing import List, Optional, Dict, Any

import numpy as np

from src.main_operations.modules.object_detection.implementation import (
    ObjectDetectionImplementation,
)
from src.utils.device_management_utils.compute_pool import ComputePool


class ObjectDetectionDefinition:
    """Main operation definition for generic object detection.

    Input: np.ndarray BGR frame (H, W, 3) uint8.
    Output: list of detections: {"bbox": (x1, y1, x2, y2), "score": confidence, "class_id": class_id}
    where bbox coordinates are percentages (0-1) of image dimensions.
    """

    def __init__(
        self,
        model_path: Optional[str],
        device_id: Optional[str],
        compute_pool: ComputePool,
        post_processing_model_path: Optional[str] = None,
        target_width: int = 320,
        target_height: int = 320,
        conf_threshold: float = 0.25,
        max_detections: int = 100,
        is_grayscale: bool = False,
    ) -> None:
        """Construct the object detection operation.

        Args:
                model_path: Optional path to a device-compatible model (.dfp for compiled, .onnx for standalone). If None, CPU fallback is used.
                post_processing_model_path: Optional path to a ONNX post-processing model. If None, CPU fallback is used.
                device_id: Optional device identifier to fetch from `ComputePool`. If None, CPU fallback is used.
                compute_pool: Injected ComputePool instance to resolve compute devices.
                target_width: Target model width.
                target_height: Target model height.
                conf_threshold: Minimum confidence to keep detections.
                max_detections: Maximum number of detections to return.
                is_grayscale: Whether the model expects grayscale input (single channel) instead of RGB.
        """
        device = None
        if device_id is not None:
            device = compute_pool.get_compute_device(device_id)

        self.delegate = ObjectDetectionImplementation(
            model_path=model_path,
            device=device,
            target_width=target_width,
            target_height=target_height,
            conf_threshold=conf_threshold,
            max_detections=max_detections,
            post_processing_model_path=post_processing_model_path,
            is_grayscale=is_grayscale,
        )

        self.last_detections: Optional[List[Dict[str, Any]]] = None
        self.last_detections_lock: Lock = Lock()

    def run(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run object detection on a frame.

        Args:
                frame: BGR image, dtype=uint8.

        Returns:
                Detections as a list of {"bbox": (x1, y1, x2, y2), "score": confidence, "class_id": class_id}
                where bbox coordinates are percentages (0-1) of image dimensions.
        """
        detections = self.delegate.run(frame)

        with self.last_detections_lock:
            self.last_detections = detections

        return detections

    def visualize(self, frame: np.ndarray) -> np.ndarray:
        """Visualize object detections by drawing bounding boxes and labels on the frame.

        Args:
            frame: Input frame to draw detections on.

        Returns:
            Frame with object detections visualized.
        """
        with self.last_detections_lock:
            detections = self.last_detections

        if detections is not None and len(detections) > 0:
            height, width = frame.shape[:2]
            for detection in detections:
                x1_pct, y1_pct, x2_pct, y2_pct = detection["bbox"]
                confidence = detection["score"]
                class_id = detection["class_id"]

                # Convert percentages back to pixels
                x1 = int(x1_pct * width)
                y1 = int(y1_pct * height)
                x2 = int(x2_pct * width)
                y2 = int(y2_pct * height)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label with confidence and class ID
                label = f"Class {class_id}: {confidence:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        return frame
