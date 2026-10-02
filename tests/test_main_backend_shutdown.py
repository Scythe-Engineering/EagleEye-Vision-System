"""Shutdown ordering without executing backend startup or touching hardware."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_shutdown_preserves_shared_resources_until_every_pipeline_closes() -> None:
    """Drain other pipelines after a failure, then safely retry shared teardown."""
    source = Path("src/main_backend.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    backend_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MainBackend"
    )
    namespace = {"__name__": __name__}
    exec(  # noqa: S102 - trusted local class only; avoid hardware startup side effects
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[
                        ast.ImportFrom(
                            module="__future__",
                            names=[ast.alias(name="annotations")],
                            level=0,
                        ),
                        backend_class,
                    ],
                    type_ignores=[],
                )
            ),
            str(source),
            "exec",
        ),
        namespace,
    )
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
