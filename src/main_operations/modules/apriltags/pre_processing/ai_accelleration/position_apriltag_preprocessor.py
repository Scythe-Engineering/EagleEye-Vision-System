import json
import numpy as np
from typing import Optional
import traceback
import torch

from src.main_operations.modules.apriltags.pre_processing.ai_accelleration.utils import (
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
            conf_threshold: Confidence threshold for predictions (0-1).
            padding_factor: Factor to pad around detected positions (0.3 = 30% padding).
        """
        self.model_path: str = model_path

        data_path: str = model_path.split(".")[0] + ".json"
        try:
            with open(data_path, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {data_path}")

        self.target_width: int = data["target_width"]
        self.target_height: int = data["target_height"]
        self.grid_size: int = data.get(
            "grid_size", 40
        )  # Default grid size from predictor
        self.conf_threshold: float = conf_threshold
        self.padding_factor: float = padding_factor
        self.device: ComputeDevice = device

        self.model_name: str = model_path.split("/")[-1].split(".")[0]

        self.scaled_frame_buffer: np.ndarray = np.zeros(
            (self.target_height, self.target_width, 3), dtype=np.uint8
        )
        self.scaled_frame_tensor_buffer: torch.Tensor = torch.zeros(
            (1, 3, self.target_height, self.target_width), dtype=torch.float32
        )

        # Convert to grayscale for position predictor
        self.grayscale_buffer: np.ndarray = np.zeros(
            (self.target_height, self.target_width), dtype=np.uint8
        )
        self.grayscale_tensor_buffer: torch.Tensor = torch.zeros(
            (1, 1, self.target_height, self.target_width), dtype=torch.float32
        )

        self.stream_idx: int = self.device.register_thread_access()
        print(f"Assigned stream index: {self.stream_idx}")

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
        # Letterbox the frame
        self.scaled_frame_buffer = letterbox_image(
            frame, (self.target_width, self.target_height)
        )

        # Convert to tensor and normalize
        self.grayscale_tensor_buffer = (
            torch.from_numpy(self.scaled_frame_buffer.astype(np.float32)).unsqueeze(0)
            / 255.0
        )

        return self.grayscale_tensor_buffer

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

        # Execute model on device
        logits = self.device.run(
            self.model_name,
            input_tensor,
            (self.target_height, self.target_width),
            self.stream_idx,
        )

        # Expected shapes: (Gh, Gw, 4) or (4, Gh, Gw)
        if logits.ndim != 3:
            return []

        if logits.shape[-1] == 4:
            obj_logits = logits[..., 0]
            dx_hat = logits[..., 1]
            dy_hat = logits[..., 2]
            ds_hat = logits[..., 3]
        elif logits.shape[0] == 4:
            obj_logits = logits[0, ...]
            dx_hat = logits[1, ...]
            dy_hat = logits[2, ...]
            ds_hat = logits[3, ...]
        else:
            return []

        grid_h, grid_w = obj_logits.shape

        # Convert to probabilities
        obj_probs = 1.0 / (1.0 + np.exp(-obj_logits))

        # Thresholding
        valid_mask = obj_probs > self.conf_threshold
        if not np.any(valid_mask):
            return []

        frame_h, frame_w = frame.shape[0], frame.shape[1]
        cell_w = frame_w / float(grid_w)
        cell_h = frame_h / float(grid_h)

        detections: list[tuple[float, float, float, float]] = []
        valid_indices = np.argwhere(valid_mask)
        for i, j in valid_indices:
            conf = float(obj_probs[i, j])
            dx = 1.0 / (1.0 + np.exp(-float(dx_hat[i, j])))
            dy = 1.0 / (1.0 + np.exp(-float(dy_hat[i, j])))
            cx = (float(j) + dx) * cell_w
            cy = (float(i) + dy) * cell_h
            box_size = float(np.exp(float(ds_hat[i, j]))) * cell_w
            detections.append((cx, cy, box_size, conf))

        # Sort by confidence
        detections.sort(key=lambda d: d[3], reverse=True)

        # Optional simple NMS on square boxes
        detections = self._nms_square_boxes(detections, iou_threshold=0.3)

        # Keep top-k
        return detections[:12]

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

        # Create crop region
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

            # Skip invalid regions
            if region[2] <= region[0] or region[3] <= region[1]:
                continue

            cropped_images.append(frame[region[1] : region[3], region[0] : region[2]])
            crop_regions.append(region)

        return cropped_images, crop_regions

    def process_frame(
        self, frame: np.ndarray, output_size: Optional[tuple[int, int]] = None
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[int, int, int, int]]]:
        """Process a single frame through the position predictor.

        Args:
            frame: Input frame to process.
            output_size: Optional output size for scaling (not used in this implementation).

        Returns:
            Tuple of (cropped_images_with_offsets, crop_regions).
        """
        detections = self.get_positions_and_scales(frame)

        if not detections:
            return [], []

        cropped_images, crop_regions = self.generate_cropped_images(frame, detections)

        # Create list of (cropped_image, offset) tuples
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
        self.conf_threshold = conf_threshold

    def change_padding_factor(self, padding_factor: float) -> None:
        """Change the padding factor.

        Args:
            padding_factor: Factor to pad around detected positions.
        """
        self.padding_factor = padding_factor
