"""Tests for WebUI runtime server selection."""

from __future__ import annotations

from typing import Any

from src.webui.web_server import EagleEyeInterface, WEB_SERVER_HOST, WEB_SERVER_PORT


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


def test_background_server_uses_wsgi_fallback_for_threading_socketio(
    monkeypatch,
) -> None:
    interface = EagleEyeInterface.__new__(EagleEyeInterface)
    interface.socketio = type("SocketStub", (), {"async_mode": "threading"})()
    interface._serve_threaded_wsgi = lambda: None
    interface.log = lambda *_args, **_kwargs: None

    monkeypatch.setattr("src.webui.web_server.Thread", _FakeThread)

    EagleEyeInterface._start_background_server(interface)

    assert isinstance(interface.app_thread, _FakeThread)
    assert interface.app_thread.target == interface._serve_threaded_wsgi
    assert interface.app_thread.daemon is True
    assert interface.app_thread.started is True


def test_background_server_uses_socketio_run_when_async_backend_available(
    monkeypatch,
) -> None:
    def _fake_run(*_args: Any, **_kwargs: Any) -> None:
        return None

    interface = EagleEyeInterface.__new__(EagleEyeInterface)
    interface.app = object()
    interface.socketio = type(
        "SocketStub",
        (),
        {"async_mode": "gevent", "run": _fake_run},
    )()
    interface.log = lambda *_args, **_kwargs: None

    monkeypatch.setattr("src.webui.web_server.Thread", _FakeThread)

    EagleEyeInterface._start_background_server(interface)

    assert isinstance(interface.app_thread, _FakeThread)
    assert interface.app_thread.target == interface.socketio.run
    assert interface.app_thread.args == (interface.app,)
    assert interface.app_thread.kwargs == {
        "host": WEB_SERVER_HOST,
        "port": WEB_SERVER_PORT,
        "debug": False,
    }
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
