"""Focused device, model resolution, and synchronous detector tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.main_operations.definitions import object_detection
from src.main_operations.modules.object_detection.yolo_detection.implementation import (
    ObjectDetectionImplementation,
)
from src.utils.device_registry import (
    DeviceDescriptor,
    DeviceNotFoundError,
    DeviceRegistry,
)
from src.utils.model_library import (
    ArtifactError,
    ModelLibrary,
    ModelLibraryError,
    ModelReferencedError,
)


def _import_artifact(
    library: ModelLibrary,
    model_id: str,
    temporary_directory: Path,
    slot: str,
    suffix: str,
) -> Path:
    """Create and import a small placeholder artifact for a focused test.

    Args:
        library: Model library receiving the artifact.
        model_id: ID of the model to update.
        temporary_directory: Directory containing the source artifact.
        slot: Artifact slot to populate.
        suffix: Filename extension for the source artifact.

    Returns:
        Path: Source artifact path.
    """
    artifact_path = temporary_directory / f"source{suffix}"
    artifact_path.write_bytes(b"model")
    library.import_artifact(model_id, slot, artifact_path)
    return artifact_path


def test_registry_uses_only_canonical_ids() -> None:
    """Discover devices under their canonical IDs only."""
    registry = DeviceRegistry.discover(
        cuda_devices=["GPU A", "GPU B"],
        mx3_paths=["/dev/memx2", "/dev/memx0", "/dev/memx0_feature"],
    )

    assert [descriptor.device_id for descriptor in registry.descriptors()] == [
        "cpu",
        "cuda:0",
        "cuda:1",
        "mx3:0",
        "mx3:2",
    ]
    assert registry.get("cuda:1").display_name == "GPU B"
    with pytest.raises(DeviceNotFoundError):
        registry.get("GPU_1")


def test_model_library_resolves_priority_and_protects_references(
    tmp_path: Path,
) -> None:
    """Resolve device-specific artifacts and retain referenced models."""
    pipeline_path = tmp_path / "pipeline_config.json"
    library = ModelLibrary(tmp_path / "models", pipeline_config_path=pipeline_path)
    model = library.create_model("Detector", ["note"])
    _import_artifact(library, model.model_id, tmp_path, "pt", ".pt")
    _import_artifact(library, model.model_id, tmp_path, "onnx", ".onnx")
    _import_artifact(library, model.model_id, tmp_path, "engine", ".engine")

    assert library.resolve_artifact(model.model_id, "cpu").slot == "onnx"
    assert library.resolve_artifact(model.model_id, "cuda:3").slot == "engine"

    pipeline_path.write_text(
        '{"Vision": [{"action_params": {"model_id": "' + model.model_id + '"}}]}',
        encoding="utf-8",
    )
    assert library.references(model.model_id) == ("Vision",)
    with pytest.raises(ModelReferencedError):
        library.delete_model(model.model_id)


def test_mx3_resolution_requires_dfp_and_profile(tmp_path: Path) -> None:
    """Require both an MX3 artifact and profile before resolution."""
    library = ModelLibrary(tmp_path / "models")
    model = library.create_model("MX3 Detector")
    _import_artifact(library, model.model_id, tmp_path, "mx3_dfp", ".dfp")
    with pytest.raises(ArtifactError, match="profile"):
        library.resolve_artifact(model.model_id, "mx3:0")

    library.update_model(model.model_id, mx3_profile={"profile": "pending"})
    assert library.resolve_artifact(model.model_id, "mx3:0").slot == "mx3_dfp"


class _FakeBoxes:
    """Ultralytics-like batched detection boxes used by detector tests."""

    xyxy = np.array([[-10.0, 20.0, 220.0, 120.0]], dtype=np.float32)
    conf = np.array([0.875], dtype=np.float32)
    cls = np.array([0.0], dtype=np.float32)


class _FakeModel:
    """Ultralytics-like model that records prediction arguments."""

    task = "detect"

    def __init__(self) -> None:
        self.predict_arguments: dict[str, Any] | None = None

    def predict(self, **arguments: Any) -> list[Any]:
        """Return one deterministic detection result."""
        self.predict_arguments = arguments
        return [
            SimpleNamespace(
                boxes=_FakeBoxes(),
                masks=None,
                keypoints=None,
                probs=None,
                names={0: "fallback"},
            )
        ]


def test_detector_uses_exact_device_and_normalizes_python_output(
    tmp_path: Path,
) -> None:
    """Pass the exact CUDA ID and normalize detector outputs."""
    library = ModelLibrary(tmp_path / "models")
    model = library.create_model("Detector", ["note"])
    _import_artifact(library, model.model_id, tmp_path, "pt", ".pt")
    registry = DeviceRegistry(
        [
            DeviceDescriptor("cpu", "CPU", "cpu", None),
            DeviceDescriptor("cuda:1", "GPU B", "cuda", 1),
        ]
    )
    fake_model = _FakeModel()
    detector = ObjectDetectionImplementation(
        model_id=model.model_id,
        device_id="cuda:1",
        device_registry=registry,
        model_library=library,
        confidence_threshold=0.3,
        iou_threshold=0.5,
        max_detections=5,
        model_factory=lambda _path, **_kwargs: fake_model,
    )

    detections = detector.run(np.zeros((100, 200, 3), dtype=np.uint8))

    assert fake_model.predict_arguments is not None
    assert fake_model.predict_arguments["device"] == "cuda:1"
    assert fake_model.predict_arguments["conf"] == 0.3
    assert fake_model.predict_arguments["iou"] == 0.5
    assert fake_model.predict_arguments["max_det"] == 5
    assert detections == [
        {
            "bbox": [0.0, 0.2, 1.0, 1.0],
            "confidence": 0.875,
            "class_id": 0,
            "class_name": "note",
        }
    ]
    assert all(type(value) is float for value in detections[0]["bbox"])
    assert type(detections[0]["confidence"]) is float
    assert type(detections[0]["class_id"]) is int


class _FakeOnnxSession:
    """ONNX Runtime-like session attached to Ultralytics' lazy backend."""

    def get_providers(self) -> list[str]:
        """Return the active provider in execution priority order."""
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def get_provider_options(self) -> dict[str, dict[str, str]]:
        """Return the physical CUDA index selected by the provider."""
        return {"CUDAExecutionProvider": {"device_id": "2"}}


def test_cuda_onnx_provider_is_verified_after_first_inference(
    tmp_path: Path,
) -> None:
    """Verify the active ONNX CUDA provider after inference."""
    library = ModelLibrary(tmp_path / "models")
    model = library.create_model("ONNX Detector")
    _import_artifact(library, model.model_id, tmp_path, "onnx", ".onnx")
    registry = DeviceRegistry([DeviceDescriptor("cuda:2", "GPU C", "cuda", 2)])
    fake_model = _FakeModel()
    fake_model.predictor = SimpleNamespace(
        model=SimpleNamespace(session=_FakeOnnxSession())
    )
    detector = ObjectDetectionImplementation(
        model_id=model.model_id,
        device_id="cuda:2",
        device_registry=registry,
        model_library=library,
        model_factory=lambda _path, **_kwargs: fake_model,
    )

    detections = detector.run(np.zeros((100, 200, 3), dtype=np.uint8))

    assert fake_model.predict_arguments["device"] == "cuda:2"
    assert detections[0]["class_id"] == 0
    assert detector._onnx_cuda_verified is True


def test_synchronous_detector_rejects_legacy_mx3(tmp_path: Path) -> None:
    """Reject legacy MX3 artifacts in the synchronous detector."""
    library = ModelLibrary(tmp_path / "models")
    model = library.create_model("MX3", mx3_profile={"profile": "pending"})
    _import_artifact(library, model.model_id, tmp_path, "mx3_dfp", ".dfp")
    registry = DeviceRegistry([DeviceDescriptor("mx3:0", "MX3", "mx3", 0)])

    with pytest.raises(ValueError, match="does not support MX3"):
        ObjectDetectionImplementation(
            model_id=model.model_id,
            device_id="mx3:0",
            device_registry=registry,
            model_library=library,
            model_factory=lambda _path, **_kwargs: _FakeModel(),
        )


def test_empty_model_slot_skips_unloadable_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Library:
        def __init__(self) -> None:
            self.models: tuple[SimpleNamespace, ...] = ()
            self.missing: set[str] = set()

        def list_models(self) -> tuple[SimpleNamespace, ...]:
            return self.models

        def resolve_artifact(self, model_id: str, _device_id: str) -> object:
            if model_id in self.missing:
                raise ModelLibraryError("artifact unavailable")
            return object()

    class _Implementation:
        def __init__(self, *, model_id: str, **_kwargs: Any) -> None:
            if model_id == "unloadable":
                raise RuntimeError("model load failed")
            self.model_id = model_id

    library = _Library()
    monkeypatch.setattr(
        object_detection, "ObjectDetectionImplementation", _Implementation
    )
    detector = object_detection.ObjectDetectionDefinition(
        model_id="",
        device_id="cpu",
        device_registry=SimpleNamespace(),
        model_library=library,
    )
    library.models = (
        SimpleNamespace(model_id="unloadable"),
        SimpleNamespace(model_id="usable"),
    )
    detector._next_model_check = 0

    assert detector._load_available_model() is not None
    assert detector.model_id == "usable"

    with pytest.raises(RuntimeError, match="model load failed"):
        object_detection.ObjectDetectionDefinition(
            model_id="unloadable",
            device_id="cpu",
            device_registry=SimpleNamespace(),
            model_library=library,
        )

    library.missing.add("missing")
    with pytest.raises(ModelLibraryError, match="artifact unavailable"):
        object_detection.ObjectDetectionDefinition(
            model_id="missing",
            device_id="cpu",
            device_registry=SimpleNamespace(),
            model_library=library,
        )


def test_synchronous_detector_rejects_fractional_limits(tmp_path: Path) -> None:
    """A fractional limit must be rejected, not silently truncated."""
    library = ModelLibrary(tmp_path / "models")
    model = library.create_model("ONNX Detector")
    _import_artifact(library, model.model_id, tmp_path, "onnx", ".onnx")
    registry = DeviceRegistry([DeviceDescriptor("cpu", "CPU", "cpu", None)])

    with pytest.raises(ValueError, match="max_detections must be a positive integer"):
        ObjectDetectionImplementation(
            model_id=model.model_id,
            device_id="cpu",
            device_registry=registry,
            model_library=library,
            max_detections=1.9,
            model_factory=lambda _path, **_kwargs: _FakeModel(),
        )

    detector = ObjectDetectionImplementation(
        model_id=model.model_id,
        device_id="cpu",
        device_registry=registry,
        model_library=library,
        model_factory=lambda _path, **_kwargs: _FakeModel(),
    )
    with pytest.raises(ValueError, match="max_detections must be a positive integer"):
        detector.update_live_settings(max_detections=2.5)
