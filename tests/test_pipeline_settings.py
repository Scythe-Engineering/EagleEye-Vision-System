from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.webui.web_server_utils.pipeline_config_mixin import PipelineConfigMixin
from src.webui.web_server_utils.pipeline_settings_mixin import PipelineSettingsMixin


class _SettingsHarness(PipelineConfigMixin, PipelineSettingsMixin):
    def __init__(self, root: Path) -> None:
        self.root = root
        self._pipeline_settings_lock = threading.RLock()
        self.restart_required_for_config = False

    def _pipeline_config_path(self) -> str:
        return str(self.root / "pipeline_config.json")

    def _pipeline_settings_path(self) -> str:
        return str(self.root / "pipeline_settings.json")

    def log(self, _message: str) -> None:
        return None


def _write_pipeline_config(root: Path) -> None:
    (root / "pipeline_config.json").write_text(
        json.dumps({"vision": []}), encoding="utf-8"
    )


def test_pipeline_settings_default_to_enabled(tmp_path: Path) -> None:
    _write_pipeline_config(tmp_path)
    harness = _SettingsHarness(tmp_path)

    payload, status = harness.get_pipeline_settings("vision")

    assert status == 200
    assert payload == {"limit_frames_to_camera_capture_speed": True}


def test_pipeline_settings_are_persisted_and_require_restart(
    tmp_path: Path, monkeypatch: Any
) -> None:
    _write_pipeline_config(tmp_path)
    harness = _SettingsHarness(tmp_path)
    request = SimpleNamespace(
        get_json=lambda silent=True: {"limit_frames_to_camera_capture_speed": True}
    )
    monkeypatch.setattr(
        "src.webui.web_server_utils.pipeline_settings_mixin.request", request
    )

    payload, status = harness.save_pipeline_settings("vision")

    assert status == 200
    assert payload == {
        "limit_frames_to_camera_capture_speed": True,
        "restart_required": True,
    }
    assert harness.restart_required_for_config is True
    assert json.loads((tmp_path / "pipeline_settings.json").read_text()) == {
        "vision": {"limit_frames_to_camera_capture_speed": True}
    }


def test_pipeline_settings_reject_non_boolean_values(
    tmp_path: Path, monkeypatch: Any
) -> None:
    _write_pipeline_config(tmp_path)
    harness = _SettingsHarness(tmp_path)
    request = SimpleNamespace(
        get_json=lambda silent=True: {"limit_frames_to_camera_capture_speed": "true"}
    )
    monkeypatch.setattr(
        "src.webui.web_server_utils.pipeline_settings_mixin.request", request
    )

    payload, status = harness.save_pipeline_settings("vision")

    assert status == 400
    assert "Expected boolean field" in payload["error"]
    assert harness.restart_required_for_config is False


def test_unknown_pipeline_settings_return_not_found(tmp_path: Path) -> None:
    _write_pipeline_config(tmp_path)
    harness = _SettingsHarness(tmp_path)

    _payload, status = harness.get_pipeline_settings("missing")

    assert status == 404
