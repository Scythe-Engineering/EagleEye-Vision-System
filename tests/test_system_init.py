"""System initialization smoke tests without running the pipeline."""

from __future__ import annotations

import json
from time import sleep
from threading import current_thread
from pathlib import Path

from typing import Any, cast

import numpy as np
import pytest
from src.main_operations.modules.object_detection.yolo_detection.implementation import (
    ObjectDetectionImplementation,
)
from src.utils.camera_utils.camera_config_manager import CameraConfigRegistry
from src.utils.device_management_utils.async_compute_wrapper import AsyncComputeWrapper
from src.utils.device_management_utils.compute_device import ComputeDevice
from src.utils.device_management_utils.compute_pool import ComputePool
from src.utils.logging.logger import Logger
from tests.utils.dummy_dependencies import (
    DummyComputePool,
    FakeCameraThreadManager,
    FakeEagleEyeInterface,
    FakeNetworkTable,
)
from tests.utils.dummy_data import dummy_frame


class _FakeComputeDevice(ComputeDevice):
    """Compute device used to test async wrapper behavior."""

    def __init__(self, exception: BaseException | None = None) -> None:
        """Initialize the fake compute device.

        Args:
            exception: Optional exception raised during inference.
        """
        super().__init__(device_id="FAKE_001", device_type="FAKE")
        self.exception = exception
        self.loaded_model_paths: list[str] = []
        self.connected_streams: int | None = None
        self.stopped = False
        self.run_thread_names: list[str] = []

    def load_model(
        self,
        model_path: str,
        input_data_shape: tuple[int, int],
        post_processing_model_path: str | None = None,
        is_grayscale: bool = False,
    ) -> None:
        """Load a fake model.

        Args:
            model_path: Path to the model.
            input_data_shape: Shape of the input data.
            post_processing_model_path: Optional post-processing model path.
            is_grayscale: Whether the model expects grayscale input.
        """
        self.loaded_model_paths.append(model_path)

    def run(
        self,
        model_path: str,
        input_data: np.ndarray,
        input_data_shape: tuple[int, int],
        stream_idx: int,
    ) -> np.ndarray:
        """Run fake inference.

        Args:
            model_path: Path to the model.
            input_data: Input array.
            input_data_shape: Shape of the input data.
            stream_idx: Stream index.

        Returns:
            Input array incremented by one.
        """
        self.run_thread_names.append(current_thread().name)
        if self.exception is not None:
            raise self.exception
        return input_data + 1

    def connect_streams(self, num_streams: int) -> None:
        """Connect fake streams.

        Args:
            num_streams: Number of streams to connect.
        """
        self.connected_streams = num_streams

    def register_thread_access(self) -> int:
        """Register fake stream access.

        Returns:
            Fake stream index.
        """
        return 3

    def stop(self) -> None:
        """Stop the fake compute device."""
        self.stopped = True


class _FakeYoloOps:
    """YOLO ops stub for object detection async contract tests."""

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a fake frame.

        Args:
            frame: Input frame.

        Returns:
            Unchanged frame.
        """
        return frame

    def postprocess(self, outputs: np.ndarray) -> list[dict[str, Any]]:
        """Postprocess fake model output.

        Args:
            outputs: Fake model output.

        Returns:
            Single fake detection.
        """
        return [
            {
                "bbox": (0.0, 0.0, 1.0, 1.0),
                "score": float(outputs[0]),
                "class_id": 1,
            }
        ]


def test_pipeline_initialization_only(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    pipeline_config_path = project_root / "src" / "config" / "pipeline_config.json"
    with pipeline_config_path.open("r", encoding="utf-8") as handle:
        pipeline_config = json.load(handle)

    temp_config_path = tmp_path / "pipeline_config.json"
    temp_config_path.write_text(json.dumps(pipeline_config), encoding="utf-8")

    web_interface = FakeEagleEyeInterface()
    network_table = FakeNetworkTable()
    compute_pool = DummyComputePool()
    logger = Logger(log_directory="logs/test")
    camera_manager = FakeCameraThreadManager(default_frame=dummy_frame())
    camera_config_registry = CameraConfigRegistry()
    camera_manager.add_camera("basic_test")
    camera_manager.add_camera("FaceTime HD Camera")
    camera_manager.add_camera("test_camera")

    try:
        from src.config.utils.generate_all_pipelines import generate_all_pipelines
    except ImportError as exc:
        pytest.skip(f"system_init_optional: {exc}")

    pipelines = generate_all_pipelines(
        cast(Any, web_interface),
        cast(Any, compute_pool),
        cast(Any, network_table),
        cast(Any, camera_manager),
        camera_config_registry=camera_config_registry,
        logger=logger,
        pipeline_config=str(temp_config_path),
    )

    if not pipelines:
        pytest.skip("pipeline_init_optional: no pipelines created")

    for pipeline in pipelines.values():
        assert pipeline.operations, "Pipeline has no operations"
        assert pipeline.thread is None


def test_compute_pool_wraps_devices_with_async_contract() -> None:
    """Verify compute pool devices expose async callbacks and lifecycle forwarding."""
    compute_pool = ComputePool()
    fake_device = _FakeComputeDevice()

    compute_pool.add_compute_device(fake_device)
    wrapped_device = compute_pool.get_compute_device("FAKE_001")

    assert isinstance(wrapped_device, AsyncComputeWrapper)
    assert wrapped_device.register_thread_access() == 3

    wrapped_device.load_model("model.onnx", (1, 1))
    wrapped_device.connect_streams(2)
    compute_pool.stop_all_devices()

    assert fake_device.loaded_model_paths == ["model.onnx"]
    assert fake_device.connected_streams == 2
    assert fake_device.stopped


def test_async_compute_wrapper_emits_results_from_worker_thread() -> None:
    """Verify async requests return through on_result without caller-thread inference."""
    fake_device = _FakeComputeDevice()
    wrapped_device = AsyncComputeWrapper(fake_device)
    emitted_results = []
    wrapped_device.on_result(emitted_results.append)

    request_id = wrapped_device.on_frame("model", np.array([1]), (1, 1), 0)
    output_data = wrapped_device.wait_for_result(request_id, 1.0)
    wrapped_device.stop()

    assert output_data.tolist() == [2]
    assert emitted_results[0].request_id == request_id
    assert fake_device.run_thread_names[0].startswith("async-compute-")


def test_async_compute_wrapper_surfaces_device_and_callback_exceptions() -> None:
    """Verify worker and callback failures remain visible to pipeline callers."""
    fake_device = _FakeComputeDevice(exception=ValueError("device failed"))
    wrapped_device = AsyncComputeWrapper(fake_device)
    emitted_results = []
    wrapped_device.on_result(emitted_results.append)

    request_id = wrapped_device.on_frame("model", np.array([1]), (1, 1), 0)
    with pytest.raises(ValueError, match="device failed"):
        wrapped_device.wait_for_result(request_id, 1.0)

    assert isinstance(emitted_results[0].exception, ValueError)

    callback_failure_device = AsyncComputeWrapper(_FakeComputeDevice())

    def raise_callback_error(_result: Any) -> None:
        """Raise a callback error.

        Args:
            _result: Async result payload.
        """
        raise RuntimeError("callback failed")

    callback_failure_device.on_result(raise_callback_error)
    callback_request_id = callback_failure_device.on_frame(
        "model", np.array([1]), (1, 1), 0
    )

    with pytest.raises(RuntimeError, match="Async compute result callback failed"):
        callback_failure_device.wait_for_result(callback_request_id, 1.0)

    wrapped_device.stop()
    callback_failure_device.stop()


def test_object_detection_uses_async_device_callbacks() -> None:
    """Verify object detection queues inference through the async contract."""
    fake_device = _FakeComputeDevice()
    wrapped_device = AsyncComputeWrapper(fake_device)
    implementation = ObjectDetectionImplementation(
        model_path="model.dfp",
        device=wrapped_device,
        target_width=1,
        target_height=1,
    )
    implementation.yolov10_ops = _FakeYoloOps()

    first_result = implementation.run(np.array([1]))
    for _ in range(20):
        with implementation._async_result_lock:
            has_result = bool(implementation._latest_async_detections)
        if has_result:
            break
        sleep(0.01)
    second_result = implementation.run(np.array([1]))
    wrapped_device.stop()

    assert first_result == []
    assert second_result == [
        {"bbox": (0.0, 0.0, 1.0, 1.0), "score": 2.0, "class_id": 1}
    ]
