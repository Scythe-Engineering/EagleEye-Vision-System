"""Tests for WebUI runtime server selection."""

from __future__ import annotations

from typing import Any

from src.webui.web_server import EagleEyeInterface, WEB_SERVER_HOST, WEB_SERVER_PORT
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

    def add_url_rule(
        self, rule: str, endpoint: str, view_func: Any, **options: Any
    ) -> None:
        self.routes.append((rule, endpoint, view_func, options))


class _RouteRegistrationInterface(EagleEyeInterface):
    def __getattr__(self, _name: str) -> Any:
        return lambda *args, **kwargs: None


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

    def _fake_make_server(host: str, port: int, app: Any, threaded: bool) -> _FakeServer:
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

    assert calls == [(str(STATIC_DIR), "background.webp")]
