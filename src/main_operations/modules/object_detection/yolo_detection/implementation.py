"""Synchronous Ultralytics YOLO detection for explicit CPU and CUDA devices."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from src.utils.device_registry import DeviceDescriptor, DeviceRegistry
from src.utils.model_library import ModelLibrary

Detection = dict[str, Any]
ModelFactory = Callable[[str], Any]


class ObjectDetectionImplementation:
    """Own and run one Ultralytics detection model on one selected device."""

    def __init__(
        self,
        model_id: str,
        device_id: str,
        device_registry: DeviceRegistry,
        model_library: ModelLibrary,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 100,
        image_size: int = 0,
        *,
        model_factory: ModelFactory | None = None,
    ) -> None:
        """Load a managed YOLO model for synchronous detection.

        Args:
            model_id: Stable managed model ID.
            device_id: Exact canonical device ID.
            device_registry: Startup hardware inventory.
            model_library: Managed model and artifact resolver.
            confidence_threshold: Minimum confidence in the inclusive range [0, 1].
            iou_threshold: Non-maximum-suppression IoU threshold in [0, 1].
            max_detections: Maximum detections returned for one frame.
            image_size: Optional square Ultralytics image-size override. Zero uses
                model/export metadata.
            model_factory: Optional Ultralytics-compatible factory for focused tests.

        Raises:
            ValueError: If configuration is invalid or MX3 is selected.
            RuntimeError: If the artifact cannot be loaded on the exact device.
        """
        self._validate_configuration(
            model_id=model_id,
            device_id=device_id,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            image_size=image_size,
        )
        self.model_id = model_id
        self.device_id = device_id
        self.device_descriptor = device_registry.get(device_id)
        if self.device_descriptor.device_type == "mx3":
            raise ValueError(
                "Synchronous Object Detection does not support MX3. "
                "Legacy MX3 inference was removed; use MX3 Async Object Detection "
                "when Stage 2 support is available."
            )
        if self.device_descriptor.device_type not in {"cpu", "cuda"}:
            raise ValueError(
                f"Unsupported object-detection device: {self.device_descriptor.device_id}"
            )

        self.model_metadata = model_library.get_model(model_id)
        self.resolved_artifact = model_library.resolve_artifact(model_id, device_id)
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = int(max_detections)
        self.image_size = int(image_size)
        self._onnx_cuda_verified = False

        factory = model_factory or self._load_ultralytics_factory()
        try:
            self.model = factory(str(self.resolved_artifact.path))
        except Exception as error:
            raise RuntimeError(
                f"Failed to load model {model_id!r} from "
                f"{self.resolved_artifact.path}: {error}"
            ) from error

        model_task = getattr(self.model, "task", None)
        if model_task != "detect":
            raise ValueError(
                f"Model {model_id!r} has unsupported task {model_task!r}; "
                "only Ultralytics YOLO detection models are supported"
            )

    @staticmethod
    def _validate_configuration(
        *,
        model_id: str,
        device_id: str,
        confidence_threshold: float,
        iou_threshold: float,
        max_detections: int,
        image_size: int,
    ) -> None:
        """Validate detector configuration before loading model artifacts."""
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("model_id is required")
        if not isinstance(device_id, str) or not device_id:
            raise ValueError("device_id is required")
        if not 0.0 <= float(confidence_threshold) <= 1.0:
            raise ValueError("confidence_threshold must be in [0.0, 1.0]")
        if not 0.0 <= float(iou_threshold) <= 1.0:
            raise ValueError("iou_threshold must be in [0.0, 1.0]")
        if isinstance(max_detections, bool) or int(max_detections) < 1:
            raise ValueError("max_detections must be a positive integer")
        if isinstance(image_size, bool) or int(image_size) < 0:
            raise ValueError("image_size must be zero or a positive integer")

    @staticmethod
    def _load_ultralytics_factory() -> ModelFactory:
        """Import and return the required Ultralytics YOLO constructor."""
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "The ultralytics package is required for Object Detection"
            ) from error
        return YOLO

    @staticmethod
    def _verify_onnx_provider_device(session: Any, device: DeviceDescriptor) -> None:
        """Verify ONNX Runtime's CUDA provider index when it exposes options."""
        get_provider_options = getattr(session, "get_provider_options", None)
        if not callable(get_provider_options):
            return
        provider_options = get_provider_options()
        cuda_options = provider_options.get("CUDAExecutionProvider", {})
        active_device = cuda_options.get("device_id")
        if active_device is not None and int(active_device) != device.physical_index:
            raise RuntimeError(
                "ONNX Runtime activated the wrong CUDA device: "
                f"expected {device.physical_index}, got {active_device}"
            )

    def update_live_settings(
        self,
        *,
        confidence_threshold: float | None = None,
        iou_threshold: float | None = None,
        max_detections: int | None = None,
    ) -> None:
        """Apply the per-inference settings that need no model reload.

        Args:
            confidence_threshold: New minimum confidence, when supplied.
            iou_threshold: New non-maximum-suppression IoU threshold, when supplied.
            max_detections: New per-frame detection cap, when supplied.

        Raises:
            ValueError: If a supplied value is outside its documented range.
        """
        if confidence_threshold is not None:
            if not 0.0 <= float(confidence_threshold) <= 1.0:
                raise ValueError("confidence_threshold must be in [0.0, 1.0]")
            self.confidence_threshold = float(confidence_threshold)
        if iou_threshold is not None:
            if not 0.0 <= float(iou_threshold) <= 1.0:
                raise ValueError("iou_threshold must be in [0.0, 1.0]")
            self.iou_threshold = float(iou_threshold)
        if max_detections is not None:
            if isinstance(max_detections, bool) or int(max_detections) < 1:
                raise ValueError("max_detections must be a positive integer")
            self.max_detections = int(max_detections)

    def run(self, frame: np.ndarray) -> list[Detection]:
        """Run synchronous inference and return normalized Python detections."""
        if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3):
            raise ValueError("Object Detection requires a NumPy image frame")
        if frame.shape[0] <= 0 or frame.shape[1] <= 0:
            raise ValueError("Object Detection received an empty frame")

        predict_arguments: dict[str, Any] = {
            "source": frame,
            "verbose": False,
            "conf": self.confidence_threshold,
            "iou": self.iou_threshold,
            "max_det": self.max_detections,
            "device": self.device_id,
        }
        if self.image_size:
            predict_arguments["imgsz"] = self.image_size

        try:
            results = self.model.predict(**predict_arguments)
        except Exception as error:
            raise RuntimeError(
                f"Inference failed for model {self.model_id!r} on "
                f"{self.device_id}: {error}"
            ) from error

        if self.resolved_artifact.slot == "onnx" and self.device_id.startswith("cuda:"):
            self._verify_ultralytics_onnx_provider()

        return self._normalize_results(results, frame.shape[1], frame.shape[0])

    def _verify_ultralytics_onnx_provider(self) -> None:
        """Verify Ultralytics' actual lazy ONNX backend did not fall back to CPU."""
        if self._onnx_cuda_verified:
            return
        predictor = getattr(self.model, "predictor", None)
        backend = getattr(predictor, "model", None)
        session = getattr(backend, "session", None)
        get_providers = getattr(session, "get_providers", None)
        if not callable(get_providers):
            raise RuntimeError(
                "Unable to verify the active ONNX Runtime provider used by "
                "Ultralytics; refusing possible CPU fallback"
            )
        active_providers = get_providers()
        if not active_providers or active_providers[0] != "CUDAExecutionProvider":
            raise RuntimeError(
                "Ultralytics ONNX inference did not activate CUDAExecutionProvider; "
                f"active providers: {active_providers}"
            )
        self._verify_onnx_provider_device(session, self.device_descriptor)
        self._onnx_cuda_verified = True

    def _normalize_results(
        self,
        results: Sequence[Any],
        frame_width: int,
        frame_height: int,
    ) -> list[Detection]:
        """Convert Ultralytics detection boxes to the stable output contract."""
        normalization = np.array(
            [frame_width, frame_height, frame_width, frame_height],
            dtype=np.float64,
        )
        detections: list[Detection] = []
        for result in results:
            if any(
                getattr(result, attribute, None) is not None
                for attribute in ("masks", "keypoints", "probs")
            ):
                raise ValueError(
                    "Only YOLO detection outputs are supported; received a "
                    "segmentation, pose, or classification result"
                )
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                raise ValueError(
                    "Model result does not expose detection boxes; arbitrary "
                    "ONNX detectors are not supported"
                )
            coordinates = self._as_float_array(getattr(boxes, "xyxy", None))
            confidences = self._as_float_array(getattr(boxes, "conf", None)).reshape(-1)
            class_ids = self._as_float_array(getattr(boxes, "cls", None)).reshape(-1)
            if coordinates.size == 0:
                continue
            if coordinates.ndim != 2 or coordinates.shape[1] != 4:
                raise ValueError("Malformed Ultralytics detection box output")
            if not len(coordinates) == len(confidences) == len(class_ids):
                raise ValueError(
                    "Ultralytics returned mismatched box, confidence, and class counts"
                )

            normalized_boxes = np.clip(coordinates / normalization, 0.0, 1.0).tolist()
            confidence_values = confidences.tolist()
            class_id_values = class_ids.astype(int).tolist()
            names = self._class_name_mapping(result)
            for box_index, normalized_box in enumerate(normalized_boxes):
                class_id = class_id_values[box_index]
                detection: Detection = {
                    "bbox": normalized_box,
                    "confidence": confidence_values[box_index],
                    "class_id": class_id,
                }
                class_name = names.get(class_id)
                if class_name is not None:
                    detection["class_name"] = str(class_name)
                detections.append(detection)
        return detections

    @staticmethod
    def _as_float_array(value: Any) -> np.ndarray:
        """Convert one batched tensor-like detection field into a float array."""
        if value is None:
            raise ValueError("Ultralytics detection boxes are missing a required field")
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value, dtype=np.float64)

    def _class_name_mapping(self, result: Any) -> dict[int, str]:
        """Resolve ordered library names before model-provided class labels."""
        if self.model_metadata.class_names:
            return {
                index: class_name
                for index, class_name in enumerate(self.model_metadata.class_names)
            }
        names = getattr(result, "names", None)
        if names is None:
            names = getattr(self.model, "names", None)
        if isinstance(names, Mapping):
            return {int(key): str(value) for key, value in names.items()}
        if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
            return {index: str(value) for index, value in enumerate(names)}
        return {}
