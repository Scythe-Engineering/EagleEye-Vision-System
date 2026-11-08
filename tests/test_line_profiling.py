from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

from src.config.utils.line_profiling import LineProfilingManager
from src.config.utils.operation import Operation
from src.main_operations.definitions.base.base_class import OperationInstance
from src.webui.web_server_utils.line_profiling_mixin import LineProfilingMixin


class _FakeProfiler:
    def __init__(self) -> None:
        self.functions: list[Any] = []

    def add_function(self, func: Any) -> None:
        self.functions.append(func)


def _make_function(name: str, module_name: str) -> Any:
    def _func() -> None:
        pass

    _func.__name__ = name
    _func.__qualname__ = name
    _func.__module__ = module_name
    return _func


def _make_module(module_name: str, *functions: Any) -> ModuleType:
    module = ModuleType(module_name)
    for func in functions:
        setattr(module, func.__name__, func)
    return module


def test_main_operation_line_profiling_discovers_reachable_main_modules() -> None:
    manager = LineProfilingManager()
    fake_profiler = _FakeProfiler()

    definition_func = _make_function(
        "definition_run", "src.main_operations.definitions.fake_operation"
    )
    linked_func = _make_function(
        "linked_run", "src.main_operations.modules.fake.implementation"
    )
    unrelated_func = _make_function("helper", "src.utils.fake")

    definition_module = _make_module(
        "src.main_operations.definitions.fake_operation", definition_func
    )
    linked_module = _make_module(
        "src.main_operations.modules.fake.implementation", linked_func
    )
    unrelated_module = _make_module("src.utils.fake", unrelated_func)
    sys.modules[definition_module.__name__] = definition_module
    sys.modules[linked_module.__name__] = linked_module
    sys.modules[unrelated_module.__name__] = unrelated_module

    class _LinkedDelegate:
        def run(self) -> None:
            pass

    _LinkedDelegate.__module__ = linked_module.__name__
    setattr(linked_module, "_LinkedDelegate", _LinkedDelegate)

    class _UnrelatedHelper:
        pass

    _UnrelatedHelper.__module__ = unrelated_module.__name__

    class _MainShim(OperationInstance):
        def __init__(self) -> None:
            self.delegate = _LinkedDelegate()
            self.helper = _UnrelatedHelper()

        def run(self, *_args: Any, **_kwargs: Any) -> None:
            pass

    _MainShim.__module__ = definition_module.__name__
    setattr(definition_module, "_MainShim", _MainShim)

    operation = Operation(_MainShim(), "main-op", "MainOp")
    manager._add_operation_functions(fake_profiler, operation)  # noqa: SLF001

    added_modules = {func.__module__ for func in fake_profiler.functions}
    assert definition_module.__name__ in added_modules
    assert linked_module.__name__ in added_modules
    assert unrelated_module.__name__ not in added_modules


def test_line_profiling_mixin_allows_main_and_secondary_operations() -> None:
    accepted: list[Operation] = []

    class _Pipeline:
        def __init__(self, operation: Operation) -> None:
            self.operation = operation

        def get_operation_by_uuid(self, _uuid: str) -> Operation:
            return self.operation

    class _Interface(LineProfilingMixin):
        def __init__(self, operation: Operation) -> None:
            self.pipeline_objects_callback = lambda: {"Pipeline": _Pipeline(operation)}

    class _MainOperation(OperationInstance):
        def run(self, *_args: Any, **_kwargs: Any) -> None:
            pass

    _MainOperation.__module__ = "src.main_operations.definitions.fake"
    main_operation = Operation(_MainOperation(), "op", "Op")

    from src.webui.web_server_utils import line_profiling_mixin

    original_start_session = line_profiling_mixin.line_profiling_manager.start_session
    line_profiling_mixin.line_profiling_manager.start_session = (  # type: ignore[method-assign]
        lambda _pipeline_name, operation: (accepted.append(operation) or {"success": True}, 200)
    )
    try:
        payload, status = _Interface(main_operation).start_line_profiling(
            "Pipeline", "op"
        )
    finally:
        line_profiling_mixin.line_profiling_manager.start_session = original_start_session  # type: ignore[method-assign]

    assert status == 200
    assert payload["success"] is True
    assert accepted == [main_operation]


class _CountingOperation(OperationInstance):
    def __init__(self) -> None:
        self.calls = 0

    def run(self, value: Any) -> Any:
        self.calls += 1
        return value


def test_line_profiling_starts_and_stops_between_operation_runs() -> None:
    """A session started mid-run profiles the next call and stops cleanly."""
    from src.config.utils.line_profiling import line_profiling_manager

    instance = _CountingOperation()
    operation = Operation(instance, "op-count", "count")
    other = Operation(_CountingOperation(), "op-other", "other")

    operation.run(1)
    assert not line_profiling_manager.is_active_for(operation.uuid)

    started, status = line_profiling_manager.start_session("Pipe", operation)
    assert status == 200 and started["success"]
    try:
        assert line_profiling_manager.is_active_for(operation.uuid)
        assert not line_profiling_manager.is_active_for(other.uuid)
        assert operation.run(2) == 2
        other.run(3)
        session = line_profiling_manager._active_session
        assert session is not None and session.call_count == 1
    finally:
        stopped, status = line_profiling_manager.stop_session("Pipe", operation.uuid)
    assert status == 200 and stopped["profiled_call_count"] == 1
    assert stopped["report_available"]

    assert not line_profiling_manager.is_active_for(operation.uuid)
    operation.run(4)
    assert instance.calls == 3
    report, _status = line_profiling_manager.get_report("Pipe", operation.uuid)
    assert report["profiled_call_count"] == 1
