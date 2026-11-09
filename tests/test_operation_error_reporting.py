"""Per-operation errors appear, clear, and reach the browser even if an event drops."""

from __future__ import annotations

import threading
from typing import Any

import pytest

from src.config.utils.flow_manager import FlowManager
from src.config.utils.operation import Connection, Operation
from src.config.utils.pipeline import Pipeline
from src.main_operations.definitions.base.base_class import OperationInstance
from src.utils.logging.logger import Logger
from src.webui.web_server import EagleEyeInterface
from src.webui.web_server_utils import pipeline_config_mixin
from tests.utils.dummy_dependencies import FakeEagleEyeInterface


class _Source(OperationInstance):
    def run(self, *_args: Any) -> int:
        return 1


class _Toggle(OperationInstance):
    """Fails while ``fail`` is set, like a live parameter edit breaking an op."""

    def __init__(self) -> None:
        self.fail = False

    def run(self, value: int) -> int:
        if self.fail:
            raise RuntimeError("bad parameter")
        return value + 1


def _error_pipeline() -> tuple[Pipeline, FlowManager, _Toggle, FakeEagleEyeInterface]:
    """Wire a real FlowManager to a pipeline's error callbacks."""
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.pipeline_name = "Errors"
    pipeline.web_interface = FakeEagleEyeInterface()
    pipeline.operation_errors = {}
    pipeline.operation_errors_lock = threading.Lock()

    toggle = _Toggle()
    source = Operation(
        _Source(), "src", "source", is_data_source=True, output_ports=("data",)
    )
    target = Operation(toggle, "toggle", "toggle", input_ports=("data",))
    Connection(source, "data", target, "data", "any")
    flow = FlowManager(
        {source.uuid: source, target.uuid: target},
        Logger(log_directory="logs/test"),
        on_operation_error=pipeline.record_operation_error,
        on_operation_success=pipeline.clear_operation_error,
        pipeline_name="Errors",
    )
    return pipeline, flow, toggle, pipeline.web_interface


def _run(flow: FlowManager) -> None:
    try:
        flow.run_flow()
    except ValueError:
        pass  # The pipeline thread logs and continues after an op error.


def test_error_appears_repeats_and_clears_on_next_success() -> None:
    pipeline, flow, toggle, web = _error_pipeline()
    _run(flow)
    assert pipeline.get_operation_errors() == []
    assert web.operation_errors == []  # Nothing to clear, nothing published.

    toggle.fail = True
    _run(flow)
    _run(flow)
    errors = pipeline.get_operation_errors()
    assert [(e["uuid"], e["count"]) for e in errors] == [("toggle", 2)]
    assert "bad parameter" in errors[0]["message"]
    assert web.operation_errors[-1]["errors"][0]["count"] == 2

    toggle.fail = False
    _run(flow)
    assert pipeline.get_operation_errors() == []
    assert web.operation_errors[-1]["errors"] == []
    published = len(web.operation_errors)
    _run(flow)
    assert len(web.operation_errors) == published


def test_initialization_error_clears_on_first_success() -> None:
    pipeline, flow, _toggle, web = _error_pipeline()
    pipeline.record_operation_init_error("toggle", "toggle", "failed to start")
    assert len(pipeline.get_operation_errors()) == 1

    _run(flow)
    assert pipeline.get_operation_errors() == []
    assert web.operation_errors[-1]["errors"] == []


def _interface(events: list[tuple[str, Any]], drop: list[bool]) -> EagleEyeInterface:
    interface = EagleEyeInterface.__new__(EagleEyeInterface)
    interface._pipeline_error_lock = threading.Lock()
    interface._pipeline_error_cache = {}
    interface._pipeline_error_dirty_pipelines = set()
    interface._pipeline_error_last_seq_sent = {}
    interface._pipeline_error_last_publish_ts = {}
    interface.pipeline_objects_callback = dict
    interface.log = lambda *_args, **_kwargs: None

    def publish(name: str, data: Any) -> None:
        if drop and drop.pop(0):
            return  # Simulates the SSE queue overflowing.
        events.append((name, data))

    interface._publish_event = publish
    return interface


def test_dropped_error_events_are_corrected_by_periodic_resend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [1000.0]
    monkeypatch.setattr(pipeline_config_mixin.time, "time", lambda: clock[0])
    events: list[tuple[str, Any]] = []
    error = {"uuid": "toggle", "name": "toggle", "message": "bad", "count": 1}
    # Drop the immediate event and the first batched resend.
    interface = _interface(events, drop=[True, True])

    interface.publish_operation_errors({"pipeline_name": "Errors", "errors": [error]})
    interface._publish_cached_pipeline_errors()
    assert events == []

    clock[0] += 0.5
    interface._publish_cached_pipeline_errors()
    assert events == []  # Throttled until the fallback interval passes.

    clock[0] += 0.6
    interface._publish_cached_pipeline_errors()
    assert events == [
        ("pipeline_operation_errors", {"pipeline_name": "Errors", "errors": [error]})
    ]

    # Clearing the last error is also repaired if its event is lost.
    events.clear()
    interface._publish_event = lambda *_args: None
    interface.publish_operation_errors({"pipeline_name": "Errors", "errors": []})
    interface._publish_cached_pipeline_errors()
    interface._publish_event = lambda name, data: events.append((name, data))
    clock[0] += 1.1
    interface._publish_cached_pipeline_errors()
    assert events == [
        ("pipeline_operation_errors", {"pipeline_name": "Errors", "errors": []})
    ]
