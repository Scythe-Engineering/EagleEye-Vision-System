"""Tests for managed test video files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.camera_utils import camera_thread_manager
from src.webui import web_server
from src.webui.web_server import EagleEyeInterface


class _FakeUpload:
    def __init__(self, filename: str, content: bytes = b"video") -> None:
        self.filename = filename
        self.content = content

    def save(self, destination: str) -> None:
        Path(destination).write_bytes(self.content)


class _FakeRequest:
    def __init__(
        self,
        files: dict[str, Any] | None = None,
        args: dict[str, Any] | None = None,
        form: dict[str, Any] | None = None,
        json_payload: dict[str, Any] | None = None,
    ) -> None:
        self.files = files or {}
        self.args = args or {}
        self.form = form or {}
        self._json_payload = json_payload

    def get_json(self, silent: bool = False) -> dict[str, Any] | None:
        return self._json_payload


class _Logger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def log(self, message: str) -> None:
        self.messages.append(message)


def _interface() -> EagleEyeInterface:
    interface = EagleEyeInterface.__new__(EagleEyeInterface)
    interface.log = lambda *_args, **_kwargs: None
    return interface


def _write_pipeline_config(src_dir: Path, payload: dict[str, Any]) -> None:
    config_dir = src_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "pipeline_config.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _setup_src_root(tmp_path: Path, monkeypatch: Any) -> Path:
    src_dir = tmp_path / "src"
    (src_dir / "utils" / "sim_videos").mkdir(parents=True, exist_ok=True)
    _write_pipeline_config(src_dir, {})
    monkeypatch.setattr(web_server, "src_path", str(src_dir))
    return src_dir


def test_get_test_videos_lists_mp4_and_pipeline_references(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    src_dir = _setup_src_root(tmp_path, monkeypatch)
    video_dir = src_dir / "utils" / "sim_videos"
    (video_dir / "basic_test.mp4").write_bytes(b"mp4")
    (video_dir / "basic_test_data.csv").write_text("ignored", encoding="utf-8")
    _write_pipeline_config(
        src_dir,
        {
            "VisionPipeline": [
                {
                    "action_name": "device_input",
                    "action_params": {"camera_bus_id": "basic_test"},
                }
            ]
        },
    )

    payload, status = _interface().get_test_videos()

    assert status == 200
    assert payload["videos"] == [
        {
            "filename": "basic_test.mp4",
            "bus_id": "basic_test",
            "size": 3,
            "modified": payload["videos"][0]["modified"],
            "pipeline_references": ["VisionPipeline"],
        }
    ]


def test_upload_test_video_accepts_mp4_and_rejects_invalid_inputs(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    src_dir = _setup_src_root(tmp_path, monkeypatch)
    interface = _interface()

    monkeypatch.setattr(
        web_server,
        "request",
        _FakeRequest(files={"file": _FakeUpload("new clip.mp4", b"new")}),
    )
    payload, status = interface.upload_test_video()

    assert status == 200
    assert payload["video"]["filename"] == "new_clip.mp4"
    assert (src_dir / "utils" / "sim_videos" / "new_clip.mp4").read_bytes() == b"new"

    monkeypatch.setattr(
        web_server,
        "request",
        _FakeRequest(files={"file": _FakeUpload("bad.txt")}),
    )
    payload, status = interface.upload_test_video()

    assert status == 400
    assert "Only .mp4" in payload["error"]

    monkeypatch.setattr(
        web_server,
        "request",
        _FakeRequest(files={"file": _FakeUpload("../evil.mp4")}),
    )
    payload, status = interface.upload_test_video()

    assert status == 400
    assert "Path separators" in payload["error"]


def test_upload_test_video_requires_explicit_overwrite(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    src_dir = _setup_src_root(tmp_path, monkeypatch)
    video_path = src_dir / "utils" / "sim_videos" / "existing.mp4"
    video_path.write_bytes(b"old")
    interface = _interface()

    monkeypatch.setattr(
        web_server,
        "request",
        _FakeRequest(files={"file": _FakeUpload("existing.mp4", b"new")}),
    )
    payload, status = interface.upload_test_video()

    assert status == 409
    assert payload["requires_overwrite"] is True
    assert video_path.read_bytes() == b"old"

    monkeypatch.setattr(
        web_server,
        "request",
        _FakeRequest(
            files={"file": _FakeUpload("existing.mp4", b"new")},
            form={"overwrite": "true"},
        ),
    )
    payload, status = interface.upload_test_video()

    assert status == 200
    assert video_path.read_bytes() == b"new"


def test_delete_test_video_reports_references_and_supports_force(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    src_dir = _setup_src_root(tmp_path, monkeypatch)
    video_path = src_dir / "utils" / "sim_videos" / "used.mp4"
    video_path.write_bytes(b"video")
    _write_pipeline_config(
        src_dir,
        {
            "UsedPipeline": [
                {
                    "action_name": "device_input.py",
                    "action_params": {"camera_bus_id": "used"},
                }
            ]
        },
    )
    interface = _interface()

    monkeypatch.setattr(web_server, "request", _FakeRequest())
    payload, status = interface.delete_test_video("used.mp4")

    assert status == 409
    assert payload["requires_force"] is True
    assert payload["pipeline_references"] == ["UsedPipeline"]
    assert video_path.exists()

    monkeypatch.setattr(web_server, "request", _FakeRequest(args={"force": "true"}))
    payload, status = interface.delete_test_video("used.mp4")

    assert status == 200
    assert payload["pipeline_references"] == ["UsedPipeline"]
    assert not video_path.exists()


def test_camera_initialization_preserves_system_and_video_cameras(
    monkeypatch: Any,
) -> None:
    manager = camera_thread_manager.CameraThreadManager.__new__(
        camera_thread_manager.CameraThreadManager
    )
    manager.web_interface = object()
    manager.logger = _Logger()

    system_cameras = [{"name": "Physical", "index": 0, "bus_id": "1"}]
    video_cameras = [{"name": "basic_test", "index": -1, "bus_id": "basic_test"}]

    monkeypatch.setattr(
        camera_thread_manager,
        "add_system_cameras",
        lambda *_args, **_kwargs: system_cameras,
    )
    monkeypatch.setattr(
        camera_thread_manager,
        "add_video_file_cameras",
        lambda *_args, **_kwargs: video_cameras,
    )

    camera_thread_manager.CameraThreadManager._initialize_cameras(manager)

    assert manager.known_cameras == system_cameras + video_cameras
