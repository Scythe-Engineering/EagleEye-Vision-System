from pathlib import Path
import traceback
from typing import List, Optional, Tuple

import numpy as np
import torch

from src.utils.device_management_utils.compute_device import ComputeDevice


class ObjectDetectionImplementation:
    """Performs object detection on frames using a device-backed model when available.

    This implementation optionally loads a model onto a provided `ComputeDevice`. If a model
    is loaded, frames are preprocessed to the target resolution and inference is executed
    through the device. If no model is provided, a simple CPU-based fallback detector is used
    that outputs prominent contour-based bounding boxes.

    Input frame format: np.ndarray BGR (H, W, 3), dtype=uint8.
    Output detections: list of (x1, y1, x2, y2, confidence, class_id).
    """

    def __init__(
        self,
        model_path: Optional[str],
        device: Optional[ComputeDevice],
        target_width: int = 640,
        target_height: int = 640,
        conf_threshold: float = 0.25,
        max_detections: int = 100,
    ) -> None:
        """Initialize the implementation.

        Args:
                model_path: Optional path to model weights recognized by the device. If None, CPU fallback is used.
                device: Optional compute device capable of `load_model` and `run` calls.
                target_width: Target model input width in pixels.
                target_height: Target model input height in pixels.
                conf_threshold: Confidence threshold used for filtering detections.
                max_detections: Maximum number of detections to return.
        """
        if target_width <= 0 or target_height <= 0:
            raise ValueError("target_width and target_height must be positive integers")
        if not (0.0 <= conf_threshold <= 1.0):
            raise ValueError("conf_threshold must be in [0.0, 1.0]")
        if max_detections <= 0:
            raise ValueError("max_detections must be positive")

        self.device: Optional[ComputeDevice] = device
        self.model_path: Optional[str] = model_path
        self.model_name: Optional[str] = None

        self.target_width: int = int(target_width)
        self.target_height: int = int(target_height)
        self.conf_threshold: float = float(conf_threshold)
        self.max_detections: int = int(max_detections)

        # Letterbox mapping fields
        self.resized_width: int = 0
        self.resized_height: int = 0
        self.offset_x: int = 0
        self.offset_y: int = 0
        self.scale_factor: float = 1.0

        # Preallocated buffers
        self.rgb_buffer: np.ndarray = np.zeros(
            (self.target_height, self.target_width, 3), dtype=np.uint8
        )
        self.rgb_tensor_buffer: torch.Tensor = torch.from_numpy(
            self.rgb_buffer
        ).unsqueeze(0)

        if self.device is not None and self.model_path is not None:
            self._load_model()

        # Register thread access for devices that support it (e.g., MX3)
        # For other devices, use default stream index 0
        self.stream_idx: int = 0
        if self.device is not None and hasattr(self.device, "register_thread_access"):
            self.stream_idx = self.device.register_thread_access()

    def _load_model(self) -> None:
        """Load the model onto the device if available."""
        assert self.device is not None
        assert self.model_path is not None

        self.model_name = Path(self.model_path).stem
        self.stream_idx = self.device.register_thread_access()
        self.device.load_model(self.model_path, (self.target_height, self.target_width))

    def _letterbox_image(
        self, frame: np.ndarray, dsize: Tuple[int, int]
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Resize with unchanged aspect ratio using padding.

        Args:
                frame: Input BGR frame.
                dsize: Target (width, height).

        Returns:
                Tuple of (image_with_padding, resized_image_size)
        """
        out_w, out_h = int(dsize[0]), int(dsize[1])
        in_h, in_w = frame.shape[:2]
        scale = (
            min(out_w / float(in_w), out_h / float(in_h))
            if in_w > 0 and in_h > 0
            else 1.0
        )
        new_w, new_h = int(round(in_w * scale)), int(round(in_h * scale))
        resized = (
            frame
            if (new_w == in_w and new_h == in_h)
            else (
                __import__("cv2").cv2.resize(
                    frame,
                    (new_w, new_h),
                    interpolation=__import__("cv2").cv2.INTER_LINEAR,
                )
            )
        )
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        offset_x = (out_w - new_w) // 2
        offset_y = (out_h - new_h) // 2
        canvas[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = resized
        self.resized_width, self.resized_height = new_w, new_h
        self.offset_x, self.offset_y = offset_x, offset_y
        # Compute mapping scale relative to the max dimension convention used elsewhere.
        max_model_dim = max(new_w, new_h)
        max_original_dim = max(in_w, in_h)
        self.scale_factor = (
            (max_model_dim / float(max_original_dim)) if max_original_dim > 0 else 1.0
        )
        return canvas, (new_w, new_h)

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Prepare frame for device inference by letterboxing and BGR->RGB channel swap."""
        preprocessed_img, _ = self._letterbox_image(
            frame, (self.target_width, self.target_height)
        )
        self.rgb_buffer[:, :, :] = preprocessed_img[:, :, ::-1]
        return self.rgb_tensor_buffer

    def _map_boxes_back(
        self, boxes_in_model: np.ndarray, frame_shape: Tuple[int, int, int]
    ) -> List[Tuple[int, int, int, int]]:
        """Map boxes from model input space back to original frame coordinates."""
        frame_h, frame_w = frame_shape[0], frame_shape[1]
        mapped: List[Tuple[int, int, int, int]] = []
        for x1_in, y1_in, x2_in, y2_in in boxes_in_model:
            x1_resized = float(x1_in) - float(self.offset_x)
            y1_resized = float(y1_in) - float(self.offset_y)
            x2_resized = float(x2_in) - float(self.offset_x)
            y2_resized = float(y2_in) - float(self.offset_y)
            if self.scale_factor <= 0.0:
                continue
            x1 = int(max(0.0, min(float(frame_w - 1), x1_resized / self.scale_factor)))
            y1 = int(max(0.0, min(float(frame_h - 1), y1_resized / self.scale_factor)))
            x2 = int(max(0.0, min(float(frame_w), x2_resized / self.scale_factor)))
            y2 = int(max(0.0, min(float(frame_h), y2_resized / self.scale_factor)))
            mapped.append((x1, y1, x2, y2))
        return mapped

    def _nms(
        self,
        boxes: List[Tuple[int, int, int, int]],
        scores: List[float],
        iou_threshold: float = 0.5,
    ) -> List[int]:
        """Non-maximum suppression returning kept indices."""
        if not boxes:
            return []
        # Convert to float for IoU computation
        boxes_f = [
            (float(x1), float(y1), float(x2), float(y2)) for x1, y1, x2, y2 in boxes
        ]
        keep: List[int] = []
        suppressed = [False] * len(boxes_f)
        for i in range(len(boxes_f)):
            if suppressed[i]:
                continue
            keep.append(i)
            ax1, ay1, ax2, ay2 = boxes_f[i]
            area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            for j in range(i + 1, len(boxes_f)):
                if suppressed[j]:
                    continue
                bx1, by1, bx2, by2 = boxes_f[j]
                inter_x1 = max(ax1, bx1)
                inter_y1 = max(ay1, by1)
                inter_x2 = min(ax2, bx2)
                inter_y2 = min(ay2, by2)
                inter_w = max(0.0, inter_x2 - inter_x1)
                inter_h = max(0.0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h
                area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
                union = area_a + area_b - inter_area
                iou = (inter_area / union) if union > 0.0 else 0.0
                if iou > iou_threshold:
                    suppressed[j] = True
        return keep

    def run(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float, int]]:
        """Run detection on a single frame.

        Args:
                frame: Input BGR frame.

        Returns:
                List of detections as (x1, y1, x2, y2, confidence, class_id).
        """
        if self.device is None or self.model_path is None or self.model_name is None:
            return self._cpu_fallback_detect(frame)

        input_tensor = self._preprocess(frame)
        logits = self.device.run(
            self.model_name,
            input_tensor,
            (self.target_height, self.target_width),
            self.stream_idx,
        )

        # Expect model to return detections in a common format if supported by device.
        # Fallback decoding assumes shape (N, 6): [x1, y1, x2, y2, conf, class]
        try:
            if isinstance(logits, torch.Tensor):
                logits = logits.detach().cpu().numpy()
            dets = np.asarray(logits)
            dets = dets.reshape((-1, dets.shape[-1]))
            if dets.shape[-1] < 6:
                return self._cpu_fallback_detect(frame)

            boxes_model = dets[:, 0:4]
            scores = dets[:, 4].astype(np.float32)
            classes = dets[:, 5].astype(np.int32)

            # Map boxes back to original frame
            mapped_boxes = self._map_boxes_back(boxes_model, frame.shape)

            # Filter by confidence and apply NMS
            filtered = [
                (b, float(s), int(c))
                for b, s, c in zip(mapped_boxes, scores, classes)
                if float(s) >= self.conf_threshold
            ]
            if not filtered:
                return []
            boxes_list = [b for b, _, _ in filtered]
            scores_list = [s for _, s, _ in filtered]
            keep = self._nms(boxes_list, scores_list, iou_threshold=0.5)
            kept = [filtered[i] for i in keep]
            kept = sorted(kept, key=lambda x: x[1], reverse=True)[: self.max_detections]
            return [(x1, y1, x2, y2, sc, cl) for (x1, y1, x2, y2), sc, cl in kept]
        except Exception:
            print(f"Error running model: {traceback.format_exc()}")
            return []
