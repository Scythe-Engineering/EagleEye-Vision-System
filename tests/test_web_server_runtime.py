"""Tests for WebUI runtime server selection."""

from __future__ import annotations

import gzip
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.webui.web_server import (
    DEFAULT_VIEW_STREAM_DOWNSCALE,
    EagleEyeInterface,
    VIEW_STREAM_DOWNSCALE_KEY,
    WEB_SERVER_HOST,
    WEB_SERVER_PORT,
)
from src.webui.web_server_utils.serve_static_files import STATIC_DIR


class _FakeThread:
    def __init__(
        self,
        target: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        daemon: bool = False,
    ) -> None:
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}
        self.daemon = daemon
        self.started = False

    def start(self) -> None:
        self.started = True


class _FakeApp:
    def __init__(self) -> None:
        self.routes: list[tuple[str, str, Any, dict[str, Any]]] = []
        self.after_request_funcs: list[Any] = []
        self.error_handlers: dict[Any, Any] = {}

    def add_url_rule(
        self, rule: str, endpoint: str, view_func: Any, **options: Any
    ) -> None:
        self.routes.append((rule, endpoint, view_func, options))

    def after_request(self, view_func: Any) -> Any:
        self.after_request_funcs.append(view_func)
        return view_func

    def errorhandler(self, exception_type: Any) -> Any:
        def _decorator(view_func: Any) -> Any:
            self.error_handlers[exception_type] = view_func
            return view_func

        return _decorator


class _RouteRegistrationInterface(EagleEyeInterface):
    def __getattr__(self, _name: str) -> Any:
        return lambda *args, **kwargs: None


class _FakeResponse:
    status_code = 404
    status = "404 NOT FOUND"


class _FakeRequest:
    endpoint = "frontend_missing"
    referrer = "http://localhost:5173/"
    remote_addr = "127.0.0.1"
    method = "GET"
    full_path = "/frontend-missing?asset=bundle"
    headers: dict[str, str] = {}


class _FakeCompressibleResponse:
    def __init__(
        self,
        data: bytes,
        mimetype: str = "application/json",
        is_streamed: bool = False,
    ) -> None:
        self.status_code = 200
        self.direct_passthrough = False
        self.is_streamed = is_streamed
        self.mimetype = mimetype
        self.headers: dict[str, str] = {}
        self._data = data

    def get_data(self) -> bytes:
        return self._data

    def set_data(self, data: bytes) -> None:
        self._data = data


def test_background_server_uses_threaded_wsgi(
    monkeypatch,
) -> None:
    interface = EagleEyeInterface.__new__(EagleEyeInterface)
    interface._serve_threaded_wsgi = lambda: None
    interface.log = lambda *_args, **_kwargs: None

    monkeypatch.setattr("src.webui.web_server.Thread", _FakeThread)

    EagleEyeInterface._start_background_server(interface)

    assert isinstance(interface.app_thread, _FakeThread)
    assert interface.app_thread.target == interface._serve_threaded_wsgi
    assert interface.app_thread.daemon is True
    assert interface.app_thread.started is True


def test_threaded_wsgi_server_uses_threaded_werkzeug(monkeypatch) -> None:
    calls: list[tuple[str, int, Any, bool]] = []
    served: list[bool] = []

    class _FakeServer:
        def serve_forever(self) -> None:
            served.append(True)

    def _fake_make_server(
        host: str, port: int, app: Any, threaded: bool
    ) -> _FakeServer:
        calls.append((host, port, app, threaded))
        return _FakeServer()

    interface = EagleEyeInterface.__new__(EagleEyeInterface)
    interface.app = object()
    interface._http_server = None

    monkeypatch.setattr("src.webui.web_server.make_server", _fake_make_server)

    EagleEyeInterface._serve_threaded_wsgi(interface)

    assert calls == [(WEB_SERVER_HOST, WEB_SERVER_PORT, interface.app, True)]
    assert served == [True]


def test_background_webp_route_serves_static_webp(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_send_from_directory(directory: str, filename: str) -> None:
        calls.append((directory, filename))

    interface = _RouteRegistrationInterface.__new__(_RouteRegistrationInterface)
    interface.app = _FakeApp()
    monkeypatch.setattr(
        "src.webui.web_server.send_from_directory", fake_send_from_directory
    )

    EagleEyeInterface._register_routes(interface)

    background_routes = [
        route
        for route in interface.app.routes
        if route[0] == "/background.webp" and route[1] == "background"
    ]
    assert len(background_routes) == 1

    _rule, _endpoint, view_func, _options = background_routes[0]
    view_func()

    background_dir = Path(STATIC_DIR).parent / "assets"
    assert calls == [(str(background_dir), "background.webp")]


def test_error_responses_are_logged_with_request_context(monkeypatch) -> None:
    app = _FakeApp()
    messages: list[str] = []
    interface = EagleEyeInterface.__new__(EagleEyeInterface)
    interface.app = app
    interface.log = messages.append

    EagleEyeInterface._register_error_handlers(interface)

    monkeypatch.setattr("src.webui.web_server.request", _FakeRequest())
    response = _FakeResponse()
    logged_response = app.after_request_funcs[0](response)

    assert logged_response is response
    assert messages == [
        "Serving error: 404 NOT FOUND for GET /frontend-missing?asset=bundle "
        "endpoint=frontend_missing remote_addr=127.0.0.1 "
        "referrer=http://localhost:5173/"
    ]


def test_unmatched_routes_are_logged_as_serving_errors(monkeypatch) -> None:
    app = _FakeApp()
    messages: list[str] = []
    interface = EagleEyeInterface.__new__(EagleEyeInterface)
    interface.app = app
    interface.log = messages.append

    EagleEyeInterface._register_error_handlers(interface)

    request = _FakeRequest()
    request.endpoint = None
    request.referrer = None
    request.full_path = "/missing-asset.js?"
    monkeypatch.setattr("src.webui.web_server.request", request)
    response = _FakeResponse()
    logged_response = app.after_request_funcs[0](response)

    assert logged_response is response
    assert messages == [
        "Serving error: 404 NOT FOUND for GET /missing-asset.js "
        "endpoint=<unmatched> remote_addr=127.0.0.1 referrer=-"
    ]


def test_general_conf_adds_default_view_stream_downscale(
    monkeypatch,
    tmp_path,
) -> None:
    general_conf_path = tmp_path / "general_conf.json"
    general_conf_path.write_text(
        json.dumps({"network_table_address": "10.0.0.62"}),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.webui.web_server.GENERAL_CONF_PATH", general_conf_path)

    interface = EagleEyeInterface.__new__(EagleEyeInterface)

    config = EagleEyeInterface._read_general_conf(interface)

    assert config["network_table_address"] == "10.0.0.62"
    assert config[VIEW_STREAM_DOWNSCALE_KEY] == DEFAULT_VIEW_STREAM_DOWNSCALE


def test_save_general_conf_updates_view_stream_downscale(
    monkeypatch,
    tmp_path,
) -> None:
    general_conf_path = tmp_path / "general_conf.json"
    general_conf_path.write_text(
        json.dumps({"network_table_address": "10.0.0.62"}),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.webui.web_server.GENERAL_CONF_PATH", general_conf_path)

    class _FakeJsonRequest:
        def get_json(self, silent: bool = False) -> dict[str, Any]:
            return {
                "network_table_address": "10.0.0.2",
                VIEW_STREAM_DOWNSCALE_KEY: 0.35,
            }

    interface = EagleEyeInterface.__new__(EagleEyeInterface)
    interface._general_conf_lock = threading.Lock()
    interface.view_stream_downscale = DEFAULT_VIEW_STREAM_DOWNSCALE

    monkeypatch.setattr("src.webui.web_server.request", _FakeJsonRequest())

    response, status_code = EagleEyeInterface.save_general_conf(interface)

    saved_config = json.loads(general_conf_path.read_text(encoding="utf-8"))
    assert status_code == 200
    assert response == {"message": "General configuration saved successfully"}
    assert saved_config["network_table_address"] == "10.0.0.2"
    assert saved_config[VIEW_STREAM_DOWNSCALE_KEY] == pytest.approx(0.35)
    assert interface.view_stream_downscale == pytest.approx(0.35)


def test_save_general_conf_rejects_invalid_view_stream_downscale(
    monkeypatch,
    tmp_path,
) -> None:
    general_conf_path = tmp_path / "general_conf.json"
    general_conf_path.write_text(
        json.dumps({"network_table_address": "10.0.0.62"}),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.webui.web_server.GENERAL_CONF_PATH", general_conf_path)

    class _FakeJsonRequest:
        def get_json(self, silent: bool = False) -> dict[str, Any]:
            return {VIEW_STREAM_DOWNSCALE_KEY: 1.5}

    interface = EagleEyeInterface.__new__(EagleEyeInterface)
    interface._general_conf_lock = threading.Lock()
    interface.view_stream_downscale = DEFAULT_VIEW_STREAM_DOWNSCALE

    monkeypatch.setattr("src.webui.web_server.request", _FakeJsonRequest())

    response, status_code = EagleEyeInterface.save_general_conf(interface)

    saved_config = json.loads(general_conf_path.read_text(encoding="utf-8"))
    assert status_code == 400
    assert response == {"error": "View stream downscale must be between 0.1 and 1.0"}
    assert VIEW_STREAM_DOWNSCALE_KEY not in saved_config
    assert interface.view_stream_downscale == DEFAULT_VIEW_STREAM_DOWNSCALE


def test_system_status_includes_network_table_connection(
    monkeypatch,
    tmp_path,
) -> None:
    general_conf_path = tmp_path / "general_conf.json"
    general_conf_path.write_text(
        json.dumps({"network_table_address": "10.0.0.2"}),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.webui.web_server.GENERAL_CONF_PATH", general_conf_path)

    class _FakeNetworkTableInstance:
        def isConnected(self) -> bool:  # noqa: N802
            return True

        def getConnections(self) -> list[object]:  # noqa: N802
            return [object()]

    interface = EagleEyeInterface.__new__(EagleEyeInterface)
    interface.network_table_instance = _FakeNetworkTableInstance()

    status = EagleEyeInterface._build_network_table_status(interface)

    assert status == {
        "status": "ok",
        "connected": True,
        "server": "10.0.0.2",
        "connection_count": 1,
    }


def test_update_camera_pose_publishes_camera_pose_event() -> None:
    interface = EagleEyeInterface.__new__(EagleEyeInterface)
    published_events: list[tuple[str, dict[str, Any]]] = []
    interface.available_cameras = {
        "Front Camera": {"bus_id": "cam0", "name": "Front_Camera"}
    }
    interface._publish_event = lambda event_name, payload: published_events.append(
        (event_name, payload)
    )
    interface.log = lambda *_args, **_kwargs: None

    EagleEyeInterface.update_camera_pose(interface, "cam0", np.eye(4, dtype=float))

    assert len(published_events) == 1
    event_name, payload = published_events[0]
    assert event_name == "update_camera_pose"
    assert payload["camera_bus_id"] == "cam0"
    assert payload["camera_name"] == "Front Camera"
    assert payload["transform_matrix"] == np.eye(4, dtype=float).tolist()
    assert isinstance(payload["timestamp_ms"], int)


def test_update_camera_pose_falls_back_to_bus_id_when_name_missing() -> None:
    interface = EagleEyeInterface.__new__(EagleEyeInterface)
    published_events: list[tuple[str, dict[str, Any]]] = []
    interface.available_cameras = {}
    interface._publish_event = lambda event_name, payload: published_events.append(
        (event_name, payload)
    )
    interface.log = lambda *_args, **_kwargs: None

    EagleEyeInterface.update_camera_pose(interface, "cam1", np.eye(4, dtype=float))

    assert len(published_events) == 1
    assert published_events[0][1]["camera_name"] == "cam1"


def test_update_camera_pose_skips_non_finite_values() -> None:
    interface = EagleEyeInterface.__new__(EagleEyeInterface)
    published_events: list[tuple[str, dict[str, Any]]] = []
    messages: list[str] = []
    interface.available_cameras = {}
    interface._publish_event = lambda event_name, payload: published_events.append(
        (event_name, payload)
    )
    interface.log = messages.append

    invalid_pose = np.eye(4, dtype=float)
    invalid_pose[0, 0] = np.nan

    EagleEyeInterface.update_camera_pose(interface, "cam2", invalid_pose)

    assert published_events == []
    assert messages == ["Skipping publish of camera transform due to non-finite values"]


def test_gzip_response_optimization_compresses_json(monkeypatch) -> None:
    interface = EagleEyeInterface.__new__(EagleEyeInterface)
    interface.app = _FakeApp()
    EagleEyeInterface._register_response_optimizations(interface)

    response = _FakeCompressibleResponse(json.dumps({"payload": "x" * 2048}).encode())
    request = _FakeRequest()
    request.headers = {"Accept-Encoding": "gzip"}

    monkeypatch.setattr("src.webui.web_server.request", request)
    optimized = interface.app.after_request_funcs[0](response)

    assert optimized.headers["Content-Encoding"] == "gzip"
    assert json.loads(gzip.decompress(optimized.get_data())) == {"payload": "x" * 2048}


def test_gzip_response_optimization_skips_streamed_responses(monkeypatch) -> None:
    interface = EagleEyeInterface.__new__(EagleEyeInterface)
    interface.app = _FakeApp()
    EagleEyeInterface._register_response_optimizations(interface)

    response = _FakeCompressibleResponse(
        b"x" * 2048,
        mimetype="text/plain",
        is_streamed=True,
    )
    request = _FakeRequest()
    request.headers = {"Accept-Encoding": "gzip"}

    monkeypatch.setattr("src.webui.web_server.request", request)
    optimized = interface.app.after_request_funcs[0](response)

    assert "Content-Encoding" not in optimized.headers
    assert optimized.get_data() == b"x" * 2048


def _restart_diff_interface(
    baseline: dict[str, list[dict[str, Any]]],
) -> EagleEyeInterface:
    """Create an interface shell with a supplied runtime config baseline."""
    interface = EagleEyeInterface.__new__(EagleEyeInterface)
    interface._runtime_pipeline_config_baseline = baseline
    interface.runtime_id = "test-runtime"
    interface.log = lambda *_args, **_kwargs: None
    return interface


def _device_input_operation(
    *,
    uuid: str = "op-1",
    camera_bus_id: str = "0",
    camera_type: str = "physical",
    position: dict[str, int] | None = None,
    connections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a minimal device_input operation config for restart diff tests."""
    return {
        "action_name": "device_input",
        "action_params": {
            "camera_bus_id": camera_bus_id,
            "camera_type": camera_type,
        },
        "position": position or {"x": 100, "y": 100},
        "uuid": uuid,
        "connections": connections or [],
    }


def test_restart_diff_ignores_operation_position_changes() -> None:
    baseline = {"Pipeline": [_device_input_operation()]}
    current = {
        "Pipeline": [
            _device_input_operation(position={"x": 400, "y": 250}),
        ],
    }

    state = _restart_diff_interface(baseline)._analyze_pipeline_restart_state(current)

    assert state["restart_required"] is False


def test_restart_diff_detects_operation_add_remove_and_connection_changes() -> None:
    baseline = {
        "Pipeline": [
            _device_input_operation(
                uuid="source",
                connections=[
                    {
                        "from_uuid": "source",
                        "from_port": "frame",
                        "to_uuid": "sink",
                        "to_port": "frame",
                        "data_type": "frame",
                        "is_default": False,
                        "custom_waypoints": None,
                    }
                ],
            ),
            _device_input_operation(uuid="sink"),
        ],
    }
    current = {
        "Pipeline": [
            _device_input_operation(
                uuid="source",
                connections=[
                    {
                        "from_uuid": "source",
                        "from_port": "frame",
                        "to_uuid": "new-sink",
                        "to_port": "frame",
                        "data_type": "frame",
                        "is_default": False,
                        "custom_waypoints": None,
                    }
                ],
            ),
            _device_input_operation(uuid="new-sink"),
        ],
    }

    state = _restart_diff_interface(baseline)._analyze_pipeline_restart_state(current)

    assert state["restart_required"] is True


def test_restart_diff_detects_restart_required_config_param_only() -> None:
    baseline = {"Pipeline": [_device_input_operation(camera_bus_id="0")]}
    current = {"Pipeline": [_device_input_operation(camera_bus_id="1")]}

    state = _restart_diff_interface(baseline)._analyze_pipeline_restart_state(current)

    assert state["restart_required"] is True


def test_restart_diff_ignores_live_updatable_config_param_changes() -> None:
    baseline = {"Pipeline": [_device_input_operation(camera_type="physical")]}
    current = {"Pipeline": [_device_input_operation(camera_type="video_file")]}

    state = _restart_diff_interface(baseline)._analyze_pipeline_restart_state(current)

    assert state["restart_required"] is False
