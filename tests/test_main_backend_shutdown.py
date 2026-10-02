"""Shutdown ordering without executing backend startup or touching hardware."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def load_backend_definitions() -> dict[str, Any]:
    """Load trusted backend definitions without startup or hardware side effects."""
    source = Path("src/main_backend.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in ("MainBackend", "main")
    ]
    namespace = {"__name__": __name__, "sys": sys}
    exec(  # noqa: S102 - trusted local definitions only; avoid hardware startup side effects
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[
                        ast.ImportFrom(
                            module="__future__",
                            names=[ast.alias(name="annotations")],
                            level=0,
                        ),
                        *definitions,
                    ],
                    type_ignores=[],
                )
            ),
            str(source),
            "exec",
        ),
        namespace,
    )
    return namespace


def test_shutdown_preserves_shared_resources_until_every_pipeline_closes() -> None:
    """Drain other pipelines after a failure, then safely retry shared teardown."""
    namespace = load_backend_definitions()
    backend = namespace["MainBackend"].__new__(namespace["MainBackend"])
    events: list[str] = []
    messages: list[str] = []
    fail_close = True

    def close_first() -> None:
        events.append("first.close")
        if fail_close:
            raise TimeoutError("worker still running; retry close")

    backend.logger = SimpleNamespace(log=messages.append)
    backend.web_interface = SimpleNamespace(
        mx3_compiler=SimpleNamespace(shutdown=lambda: events.append("compiler"))
    )
    backend.pipelines = {
        "first": SimpleNamespace(
            stop=lambda: events.append("first.stop"), close=close_first
        ),
        "second": SimpleNamespace(
            stop=lambda: events.append("second.stop"),
            close=lambda: events.append("second.close"),
        ),
    }
    backend.mx3_coordinator = SimpleNamespace(stop=lambda: events.append("mx3"))
    backend.camera_manager = SimpleNamespace(
        stop_all_cameras=lambda: events.append("cameras")
    )

    with pytest.raises(RuntimeError, match="retry shutdown before restarting"):
        backend.shutdown(restart_service=True)
    drained_events = [
        "compiler",
        "first.stop",
        "second.stop",
        "first.close",
        "second.close",
    ]
    assert events == drained_events
    assert any(
        "first" in message and "worker still running" in message for message in messages
    )
    assert "remain open" in messages[-1]

    fail_close = False
    events.clear()
    backend.shutdown()
    assert events == drained_events + ["mx3", "cameras"]


def test_startup_failure_survives_shutdown_failure() -> None:
    """Report cleanup failure without replacing the original construction error."""
    namespace = load_backend_definitions()
    startup_error = ValueError("NT startup failed")
    messages: list[str] = []

    def fail_startup() -> None:
        raise startup_error

    def fail_shutdown(_self: Any) -> None:
        raise TimeoutError("shutdown blocked")

    namespace["Colors"] = SimpleNamespace(YELLOW="", RESET="")
    namespace["ntcore"] = SimpleNamespace(
        NetworkTableInstance=SimpleNamespace(getDefault=fail_startup)
    )
    backend_class = namespace["MainBackend"]
    backend_class.shutdown = fail_shutdown
    with pytest.raises(ValueError) as caught:
        backend_class(SimpleNamespace(log=messages.append))
    assert caught.value is startup_error
    assert "shutdown blocked" in messages[-1]


@pytest.mark.parametrize(
    "runtime_error", [ValueError("runtime failed"), KeyboardInterrupt()]
)
def test_runtime_cleanup_preserves_active_failure(runtime_error: BaseException) -> None:
    """Propagate runtime errors, but expose shutdown failure after a handled interrupt."""
    namespace = load_backend_definitions()
    messages: list[str] = []

    def fail_shutdown() -> None:
        raise TimeoutError("shutdown blocked")

    def fail_runtime(_seconds: float) -> None:
        raise runtime_error

    namespace["logger"] = SimpleNamespace(log=messages.append)
    namespace["MainBackend"] = lambda logger: SimpleNamespace(shutdown=fail_shutdown)
    namespace["sleep"] = fail_runtime
    expected = (
        TimeoutError if isinstance(runtime_error, KeyboardInterrupt) else ValueError
    )
    with pytest.raises(expected) as caught:
        namespace["main"]()
    if expected is ValueError:
        assert caught.value is runtime_error
        assert "shutdown blocked" in messages[-1]
