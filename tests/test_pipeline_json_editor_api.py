from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import src.webui.web_server_utils.pipeline_config_mixin as pipeline_config_module
from src.webui.web_server_utils.pipeline_config_mixin import PipelineConfigMixin


class _PipelineConfigHarness(PipelineConfigMixin):
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self._pipeline_settings_lock = threading.RLock()
        self.restart_required_for_config = False
        self.runtime_id = "test-runtime"

    def _pipeline_config_path(self) -> str:
        return str(self.config_path)


class _JsonRequest:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def get_json(self, silent: bool = False) -> dict[str, Any]:
        return self.payload


def test_raw_pipeline_json_can_be_loaded_when_malformed(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline_config.json"
    malformed_content = '{"test": [}'
    config_path.write_text(malformed_content, encoding="utf-8")

    payload, status = _PipelineConfigHarness(config_path).get_pipeline_config_json()

    assert status == 200
    assert payload["content"] == malformed_content
    assert len(payload["revision"]) == 64


def test_raw_pipeline_json_rejects_invalid_json(
    tmp_path: Path, monkeypatch: Any
) -> None:
    config_path = tmp_path / "pipeline_config.json"
    original_content = '{"test": []}\n'
    config_path.write_text(original_content, encoding="utf-8")
    monkeypatch.setattr(
        pipeline_config_module,
        "request",
        _JsonRequest(
            {
                "content": '{"test": [}',
                "revision": _PipelineConfigHarness._pipeline_config_revision(
                    original_content
                ),
            }
        ),
    )

    payload, status = _PipelineConfigHarness(config_path).save_pipeline_config_json()

    assert status == 400
    assert payload["line"] == 1
    assert payload["column"] > 0
    assert config_path.read_text(encoding="utf-8") == original_content


def test_raw_pipeline_json_saves_valid_object_atomically(
    tmp_path: Path, monkeypatch: Any
) -> None:
    config_path = tmp_path / "pipeline_config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    new_config = {"test": []}
    content = json.dumps(new_config, indent=2)
    monkeypatch.setattr(
        pipeline_config_module,
        "request",
        _JsonRequest(
            {
                "content": content,
                "revision": _PipelineConfigHarness._pipeline_config_revision("{}\n"),
            }
        ),
    )
    harness = _PipelineConfigHarness(config_path)

    payload, status = harness.save_pipeline_config_json()

    assert status == 200
    assert payload["restart_required"] is True
    assert len(payload["revision"]) == 64
    assert json.loads(config_path.read_text(encoding="utf-8")) == new_config
    assert list(tmp_path.glob(".pipeline_config.*.tmp")) == []


def test_raw_pipeline_json_rejects_stale_editor_revision(
    tmp_path: Path, monkeypatch: Any
) -> None:
    config_path = tmp_path / "pipeline_config.json"
    config_path.write_text('{"current": []}\n', encoding="utf-8")
    monkeypatch.setattr(
        pipeline_config_module,
        "request",
        _JsonRequest(
            {
                "content": '{"replacement": []}',
                "revision": _PipelineConfigHarness._pipeline_config_revision(
                    '{"old": []}\n'
                ),
            }
        ),
    )

    payload, status = _PipelineConfigHarness(config_path).save_pipeline_config_json()

    assert status == 409
    assert "changed while the editor was open" in payload["error"]
    assert json.loads(config_path.read_text(encoding="utf-8")) == {"current": []}
