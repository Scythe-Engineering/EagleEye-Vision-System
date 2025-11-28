import json
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch


from src.main_operations.modules.apriltags.pre_processing.ai_acceleration.utils import (
    letterbox_image,
)
from src.utils.device_management_utils.compute_device import ComputeDevice


class PositionApriltagPreprocessor:
    """A class to handle AprilTag position-based preprocessing and inference.

    This class loads a trained position predictor model and performs inference on video frames
    to detect potential AprilTag locations using position and scale predictions.
    """

    def __init__(
        self,
        model_path: str,
        device: ComputeDevice,
        conf_threshold: float = 0.5,
        padding_factor: float = 0.3,
    ) -> None:
        """Initialize the position-based AprilTag preprocessor.

        Args:
            model_path: Path to the trained model weights file.
            device: The computation device (CPU/CUDA/MX3/CORAL).
            conf_threshold: Confidence threshold for predictions (valid range: 0.0 to 1.0 inclusive).
            padding_factor: Factor to pad around detected positions (valid range: >= 0.0).
        """
        if not (0.0 <= conf_threshold <= 1.0):
            raise ValueError(
                f"conf_threshold must be between 0.0 and 1.0 inclusive, got {conf_threshold}"
            )

        if padding_factor < 0.0:
            raise ValueError(
                f"padding_factor must be non-negative, got {padding_factor}"
            )

        self.model_path: str = model_path

        data_path: str = str(Path(model_path).with_suffix(".json"))
        try:
            with open(data_path, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {data_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in data file '{data_path}': {e}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid data structure in JSON file '{data_path}': {e}")

        self.target_width: int = data["target_width"]
        self.target_height: int = data["target_height"]
        self.grid_size: int = data.get(
            "grid_size", 40
        )  # Default grid size from predictor
        self.max_detections: int = int(data.get("max_detections", 12))

        # Validate config values
        if not isinstance(self.target_width, int) or self.target_width <= 0:
            raise ValueError(
                f"target_width must be a positive integer, got {self.target_width}"
            )
        if not isinstance(self.target_height, int) or self.target_height <= 0:
            raise ValueError(
                f"target_height must be a positive integer, got {self.target_height}"
            )
        if not isinstance(self.grid_size, int) or self.grid_size <= 0:
            raise ValueError(
                f"grid_size must be a positive integer, got {self.grid_size}"
            )
        if not isinstance(self.max_detections, int) or self.max_detections <= 0:
            raise ValueError(
                f"max_detections must be a positive integer, got {self.max_detections}"
            )

        self.conf_threshold: float = conf_threshold
        self.padding_factor: float = padding_factor
        self.device: ComputeDevice = device

        self.model_name: str = Path(model_path).stem

        # Preallocate RGB buffer for NHWC uint8 input (0-255)
        self.rgb_buffer: np.ndarray = np.zeros(
            (self.target_height, self.target_width, 3), dtype=np.uint8
        )
        self.rgb_tensor_buffer: torch.Tensor = torch.from_numpy(
            self.rgb_buffer
        ).unsqueeze(0)

        self.stream_idx: int = self.device.register_thread_access()
        print(f"Assigned stream index: {self.stream_idx}")

        # Keep track of letterbox mapping to original frame
        self.resized_width: int = 0
        self.resized_height: int = 0
        self.offset_x: int = 0
        self.offset_y: int = 0
        self.scale_factor: float = 1.0

        self._load_model()

    def _load_model(self) -> None:
        """Load and prepare the position predictor model for inference.

        Args:
            None

        Returns:
            None

        Raises:
            RuntimeError: If there's an error loading the model.
        """
        try:
            self.device.load_model(
                self.model_path, (self.target_height, self.target_width)
            )
        except Exception as _:
            raise RuntimeError(f"Error loading model: {traceback.format_exc()}")

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for the position predictor model.

        Args:
            frame: Input frame to preprocess.

        Returns:
            Preprocessed tensor ready for model input.
        """
        preprocessed_img, resized_size = letterbox_image(
            frame,
            (self.target_width, self.target_height),
            greyscale=False,
            return_resized_size=True,
        )

        self.resized_width, self.resized_height = (
            int(resized_size[0]),
            int(resized_size[1]),
        )
        self.offset_x = (self.target_width - self.resized_width) // 2
        self.offset_y = (self.target_height - self.resized_height) // 2

        original_h, original_w = frame.shape[0], frame.shape[1]
        max_model_dim = max(self.resized_width, self.resized_height)
        max_original_dim = max(original_w, original_h)
        self.scale_factor = (
            max_model_dim / float(max_original_dim) if max_original_dim > 0 else 1.0
        )

        # Convert BGR to RGB via channel swap into preallocated buffer
        self.rgb_buffer[:, :, :] = preprocessed_img[:, :, ::-1]

        return self.rgb_tensor_buffer

    def get_positions_and_scales(
        self, frame: np.ndarray
    ) -> list[tuple[float, float, float, float]]:
        """Get decoded position predictions (center and square size) from the device model.

        Args:
            frame: Input frame to process.

        Returns:
            List of (cx_px, cy_px, box_size_px, confidence) tuples in pixel units.
        """
        input_tensor = self._preprocess_frame(frame)

        logits = self.device.run(
            self.model_name,
            input_tensor,
            (self.target_height, self.target_width),
            self.stream_idx,
        )

        # Normalize logits shape to (H, W, 4) or (4, H, W)
        if logits.ndim == 4:
            if logits.shape[0] == 1:
                logits = logits[0]
            elif logits.shape[-1] == 1:
                logits = logits[..., 0]

        if logits.ndim != 3:
            return []

        # Prefer channel-first (4, H, W) to avoid implicit transposes
        if logits.shape[0] == 4:
            obj_logits = logits[0, ...]
            dx_hat = logits[1, ...]
            dy_hat = logits[2, ...]
            ds_hat = logits[3, ...]
        elif logits.shape[-1] == 4:
            obj_logits = logits[..., 0]
            dx_hat = logits[..., 1]
            dy_hat = logits[..., 2]
            ds_hat = logits[..., 3]
        else:
            channel_axes = [
                axis_idx
                for axis_idx, dim_size in enumerate(logits.shape)
                if dim_size == 4
            ]
            if channel_axes:
                if channel_axes[0] == 0:
                    obj_logits = logits[0, ...]
                    dx_hat = logits[1, ...]
                    dy_hat = logits[2, ...]
                    ds_hat = logits[3, ...]
                else:
                    logits = np.moveaxis(logits, channel_axes[0], -1)
                    obj_logits = logits[..., 0]
                    dx_hat = logits[..., 1]
                    dy_hat = logits[..., 2]
                    ds_hat = logits[..., 3]
            else:
                return []

        grid_h, grid_w = obj_logits.shape

        obj_probs = 1.0 / (1.0 + np.exp(-obj_logits))
        valid_mask = obj_probs > self.conf_threshold
        if not np.any(valid_mask):
            return []

        cell_w = self.target_width / float(grid_w)
        cell_h = self.target_height / float(grid_h)

        detections_input_space: list[tuple[float, float, float, float]] = []
        valid_indices = np.argwhere(valid_mask)
        for grid_i, grid_j in valid_indices:
            confidence = float(obj_probs[grid_i, grid_j])
            dx = 1.0 / (1.0 + np.exp(-float(dx_hat[grid_i, grid_j])))
            dy = 1.0 / (1.0 + np.exp(-float(dy_hat[grid_i, grid_j])))
            cx_in = (float(grid_j) + dx) * cell_w
            cy_in = (float(grid_i) + dy) * cell_h
            size_in = float(np.exp(float(ds_hat[grid_i, grid_j]))) * cell_w
            detections_input_space.append((cx_in, cy_in, size_in, confidence))

        # Map detections back to original frame coordinates
        frame_h, frame_w = frame.shape[0], frame.shape[1]
        detections_original: list[tuple[float, float, float, float]] = []
        for cx_in, cy_in, size_in, confidence in detections_input_space:
            x_resized = cx_in - float(self.offset_x)
            y_resized = cy_in - float(self.offset_y)
            if self.scale_factor <= 0.0:
                continue
            cx_orig = x_resized / self.scale_factor
            cy_orig = y_resized / self.scale_factor
            size_orig = size_in / self.scale_factor

            cx_orig = float(max(0.0, min(float(frame_w - 1), cx_orig)))
            cy_orig = float(max(0.0, min(float(frame_h - 1), cy_orig)))
            size_orig = float(max(1.0, min(float(min(frame_w, frame_h)), size_orig)))

            detections_original.append((cx_orig, cy_orig, size_orig, confidence))

        detections_original.sort(key=lambda d: d[3], reverse=True)
        detections_original = self._nms_square_boxes(
            detections_original, iou_threshold=0.3
        )
        return detections_original[: self.max_detections]

    def _nms_square_boxes(
        self, detections: list[tuple[float, float, float, float]], iou_threshold: float
    ) -> list[tuple[float, float, float, float]]:
        """Apply NMS on square boxes represented by center and size.

        Args:
            detections: List of (cx, cy, size, conf).
            iou_threshold: IoU threshold to suppress overlaps.

        Returns:
            Filtered detections after NMS.
        """
        if not detections:
            return detections

        kept: list[tuple[float, float, float, float]] = []
        boxes = []
        for cx, cy, size, conf in detections:
            half = size / 2.0
            x1 = cx - half
            y1 = cy - half
            x2 = cx + half
            y2 = cy + half
            boxes.append((x1, y1, x2, y2, conf, cx, cy, size))

        def iou(
            b1: tuple[float, float, float, float], b2: tuple[float, float, float, float]
        ) -> float:
            ax1, ay1, ax2, ay2 = b1
            bx1, by1, bx2, by2 = b2
            inter_x1 = max(ax1, bx1)
            inter_y1 = max(ay1, by1)
            inter_x2 = min(ax2, bx2)
            inter_y2 = min(ay2, by2)
            inter_w = max(0.0, inter_x2 - inter_x1)
            inter_h = max(0.0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            union = area_a + area_b - inter_area
            return inter_area / union if union > 0.0 else 0.0

        suppressed = [False] * len(boxes)
        for idx in range(len(boxes)):
            if suppressed[idx]:
                continue
            x1, y1, x2, y2, conf, cx, cy, size = boxes[idx]
            kept.append((cx, cy, size, conf))
            for jdx in range(idx + 1, len(boxes)):
                if suppressed[jdx]:
                    continue
                bx1, by1, bx2, by2, _, _, _, _ = boxes[jdx]
                if iou((x1, y1, x2, y2), (bx1, by1, bx2, by2)) > iou_threshold:
                    suppressed[jdx] = True
        return kept

    def _create_crop_region(
        self,
        center_x_px: float,
        center_y_px: float,
        size_px: float,
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int, int, int]:
        """Create a crop region around a detected position with padding.

        Args:
            center_x_px: Center x coordinate in pixels.
            center_y_px: Center y coordinate in pixels.
            size_px: Square size in pixels.
            frame_width: Original frame width.
            frame_height: Original frame height.

        Returns:
            Crop region as (left, top, right, bottom).
        """
        half_size = (size_px * (1.0 + self.padding_factor)) / 2.0

        left = max(0, int(center_x_px - half_size))
        top = max(0, int(center_y_px - half_size))
        right = min(frame_width, int(center_x_px + half_size))
        bottom = min(frame_height, int(center_y_px + half_size))

        return left, top, right, bottom

    def generate_cropped_images(
        self, frame: np.ndarray, detections: list[tuple[float, float, float, float]]
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[int, int, int, int]]]:
        """Generate cropped images from detections.

        Args:
            frame: Input frame to process.
            detections: List of (cx_px, cy_px, size_px, confidence) tuples.

        Returns:
            Tuple of (cropped_images_with_offsets, crop_regions).
        """
        cropped_images = []
        crop_regions = []

        for center_x_px, center_y_px, size_px, _ in detections:
            region = self._create_crop_region(
                center_x_px, center_y_px, size_px, frame.shape[1], frame.shape[0]
            )

            if region[2] <= region[0] or region[3] <= region[1]:
                continue

            cropped_images.append(frame[region[1] : region[3], region[0] : region[2]])
            crop_regions.append(region)

        return cropped_images, crop_regions

    def process_frame(
        self, frame: np.ndarray, _output_size: Optional[tuple[int, int]] = None
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[int, int, int, int]]]:
        """Process a single frame through the position predictor.

        Args:
            frame: Input frame to process.
            _output_size: Optional output size for scaling (not used in this implementation).

        Returns:
            Tuple of (cropped_images_with_offsets, crop_regions).
        """

        detections = self.get_positions_and_scales(frame)

        if not detections:
            frame_height, frame_width = frame.shape[:2]
            entire_frame_region = (0, 0, frame_width, frame_height)
            return [(frame, (0, 0))], [entire_frame_region]

        cropped_images, crop_regions = self.generate_cropped_images(frame, detections)

        cropped_images_with_offsets = [
            (img, (region[0], region[1]))
            for img, region in zip(cropped_images, crop_regions)
        ]

        return cropped_images_with_offsets, crop_regions

    def change_conf_threshold(self, conf_threshold: float) -> None:
        """Change the confidence threshold.

        Args:
            conf_threshold: Confidence threshold for predictions (0-1).
        """
        if not 0.0 <= conf_threshold <= 1.0:
            raise ValueError(
                f"conf_threshold must be between 0 and 1, got {conf_threshold}"
            )
        self.conf_threshold = conf_threshold

    def change_padding_factor(self, padding_factor: float) -> None:
        """Change the padding factor.

        Args:
            padding_factor: Factor to pad around detected positions.
        """
        if padding_factor < 0.0:
            raise ValueError(
                f"padding_factor must be non-negative, got {padding_factor}"
            )
        self.padding_factor = padding_factor
