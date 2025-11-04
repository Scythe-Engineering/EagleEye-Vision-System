from pathlib import Path
import traceback
from typing import List, Optional, Dict, Any

import numpy as np

from src.utils.device_management_utils.compute_device import ComputeDevice
from src.utils.colors import Colors
from src.main_operations.modules.object_detection.utils.yolov10.yolov10_ops import (
    YoloV10,
)


class ObjectDetectionImplementation:
    """Performs object detection on frames using YOLOv10-style postprocessing.

    This implementation loads a model onto a provided ComputeDevice and performs
    simplified YOLOv10-style preprocessing and postprocessing.

    Input frame format: np.ndarray BGR (H, W, 3), dtype=uint8.
    Output detections: list of {
        "bbox": (x1, y1, x2, y2),
        "score": confidence,
        "class_id": class_id,
    }.
    """

    def __init__(
        self,
        model_path: Optional[str],
        device: Optional[ComputeDevice],
        target_width: int = 416,
        target_height: int = 416,
        conf_threshold: float = 0.4,
        max_detections: int = 100,
        post_processing_model_path: Optional[str] = None,
        is_grayscale: bool = False,
    ) -> None:
        """Initialize the YOLOv10-style object detection implementation.

        Args:
            model_path: Path to model weights recognized by the device.
            device: Compute device capable of `load_model` and `run` calls.
            target_width: Target model input width in pixels.
            target_height: Target model input height in pixels.
            conf_threshold: Confidence threshold used for filtering detections.
            max_detections: Maximum number of detections to return.
            post_processing_model_path: Path to ONNX post-processing model.
            is_grayscale: Whether the model expects grayscale input (single channel) instead of RGB.
        """
        if target_width <= 0 or target_height <= 0:
            raise ValueError("target_width and target_height must be positive integers")
        if not (0.0 <= conf_threshold <= 1.0):
            raise ValueError("conf_threshold must be in [0.0, 1.0]")

        self.model_path = model_path
        self.device = device
        self.post_processing_model_path = post_processing_model_path
        self.is_grayscale = is_grayscale
        self.target_width = target_width
        self.target_height = target_height
        self.conf_threshold = conf_threshold
        self.max_detections = max_detections

        self.yolov10_ops = YoloV10(
            original_image_shape=None,  # set at inference time
            input_shape=(
                self.target_height,
                self.target_width,
                3 if not self.is_grayscale else 1,
            ),
            max_det=self.max_detections,
            conf_threshold=self.conf_threshold,
        )

        if self.device is not None and self.model_path is not None:
            self._load_model()

        # Register thread access for devices that support it (e.g., MX3)
        self.stream_idx: int = 0
        if self.device is not None and hasattr(self.device, "register_thread_access"):
            self.stream_idx = self.device.register_thread_access()
            print(
                f"{Colors.GREEN}Assigned stream index: {self.stream_idx}{Colors.RESET}"
            )

    def _load_model(self) -> None:
        """Load the model onto the device if available."""
        assert self.device is not None
        assert self.model_path is not None

        self.model_name = Path(self.model_path).stem

        try:
            from src.utils.device_management_utils.mx3_accelerator import MX3Accelerator

            if isinstance(self.device, MX3Accelerator):
                self.device.load_model(
                    self.model_path,
                    (self.target_height, self.target_width),
                    self.post_processing_model_path,
                )
            else:
                self.device.load_model(
                    self.model_path, (self.target_height, self.target_width)
                )
        except ImportError:
            self.device.load_model(
                self.model_path, (self.target_height, self.target_width)
            )

    def run(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run detection on a single frame using YOLOv7-style postprocessing.

        Args:
            frame: Input BGR frame.

        Returns:
            List of detections as {
                "bbox": (x1, y1, x2, y2),
                "score": confidence,
                "class_id": class_id,
            }.
        """
        input_tensor = self.yolov10_ops.preprocess(frame)
        outputs = self.device.run(
            self.model_name,
            input_tensor,
            (self.target_height, self.target_width),
            self.stream_idx,
        )

        try:
            return self.yolov10_ops.postprocess(outputs)
        except Exception as e:
            print(f"Error running model: {traceback.format_exc()}")
            raise e
