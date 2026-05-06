from __future__ import annotations

import inspect
import io
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable

from line_profiler import LineProfiler

if TYPE_CHECKING:
    from src.config.utils.operation import Operation


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None) -> str:
    return dt.isoformat() if dt is not None else ""


@dataclass
class LineProfilingSession:
    pipeline_name: str
    operation_uuid: str
    operation_name: str
    profiler: LineProfiler
    start_time: datetime = field(default_factory=_utc_now)
    stop_time: datetime | None = None
    call_count: int = 0
    status: str = "running"
    report_text: str | None = None

    def public_status(self) -> dict[str, Any]:
        elapsed_end = self.stop_time or _utc_now()
        return {
            "status": self.status,
            "pipeline_name": self.pipeline_name,
            "operation_uuid": self.operation_uuid,
            "operation_name": self.operation_name,
            "start_time": _iso(self.start_time),
            "stop_time": _iso(self.stop_time),
            "elapsed_seconds": max(
                0.0, (elapsed_end - self.start_time).total_seconds()
            ),
            "profiled_call_count": self.call_count,
            "report_available": self.report_text is not None,
        }


class LineProfilingManager:
    """Runtime-only global manager for one active line_profiler session."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active_session: LineProfilingSession | None = None
        self._reports: dict[tuple[str, str], LineProfilingSession] = {}

    def start_session(self, pipeline_name: str, operation: "Operation") -> tuple[dict[str, Any], int]:
        with self._lock:
            if self._active_session is not None:
                return {
                    "success": False,
                    "error": "Another line profiling session is already active",
                    "active_session": self._active_session.public_status(),
                }, 409

            profiler = LineProfiler()
            self._add_module_functions(profiler, inspect.getmodule(operation.instance))
            session = LineProfilingSession(
                pipeline_name=pipeline_name,
                operation_uuid=operation.uuid,
                operation_name=operation.name,
                profiler=profiler,
            )
            self._active_session = session
            return {"success": True, **session.public_status()}, 200

    def stop_session(self, pipeline_name: str, operation_uuid: str) -> tuple[dict[str, Any], int]:
        with self._lock:
            session = self._active_session
            if session is None:
                report = self._reports.get((pipeline_name, operation_uuid))
                if report is not None:
                    return {"success": True, **report.public_status(), "report": report.report_text}, 200
                return {"success": False, "error": "No active line profiling session"}, 404
            if session.pipeline_name != pipeline_name or session.operation_uuid != operation_uuid:
                return {
                    "success": False,
                    "error": "Requested operation is not the active line profiling session",
                    "active_session": session.public_status(),
                }, 409
            self._active_session = None
            session.status = "stopped"
            session.stop_time = _utc_now()
            session.report_text = self._build_report(session)
            self._reports[(pipeline_name, operation_uuid)] = session
            return {"success": True, **session.public_status(), "report": session.report_text}, 200

    def is_active_for(self, operation_uuid: str) -> bool:
        with self._lock:
            return (
                self._active_session is not None
                and self._active_session.operation_uuid == operation_uuid
            )

    def profile_operation_call(self, operation: "Operation", call: Callable[[], Any]) -> Any:
        with self._lock:
            session = self._active_session
            if session is None or session.operation_uuid != operation.uuid:
                return call()
            session.call_count += 1
            profiler = session.profiler
        return profiler(call)()

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            if self._active_session is None:
                return {"status": "idle", "active_session": None}
            return {"status": "running", "active_session": self._active_session.public_status()}

    def get_report(self, pipeline_name: str, operation_uuid: str) -> tuple[dict[str, Any], int]:
        with self._lock:
            session = self._reports.get((pipeline_name, operation_uuid))
            if session is None:
                return {"success": False, "error": "No line profiling report available"}, 404
            return {"success": True, **session.public_status(), "report": session.report_text}, 200

    def _add_module_functions(self, profiler: LineProfiler, module: ModuleType | None) -> None:
        if module is None:
            return
        seen: set[Any] = set()
        for _, obj in inspect.getmembers(module):
            funcs = []
            if inspect.isfunction(obj):
                funcs = [obj]
            elif inspect.isclass(obj) and obj.__module__ == module.__name__:
                funcs = [m for _, m in inspect.getmembers(obj, inspect.isfunction)]
            for func in funcs:
                if func in seen or getattr(func, "__module__", None) != module.__name__:
                    continue
                seen.add(func)
                try:
                    profiler.add_function(func)
                except Exception:
                    pass

    def _build_report(self, session: LineProfilingSession) -> str:
        metadata = session.public_status()
        stream = io.StringIO()
        stream.write("EagleEye Line Profiling Report\n")
        stream.write(f"pipeline name: {session.pipeline_name}\n")
        stream.write(f"operation name: {session.operation_name}\n")
        stream.write(f"operation UUID: {session.operation_uuid}\n")
        stream.write(f"start time: {metadata['start_time']}\n")
        stream.write(f"stop time: {metadata['stop_time']}\n")
        stream.write(f"elapsed seconds: {metadata['elapsed_seconds']:.6f}\n")
        stream.write(f"profiled call count: {session.call_count}\n")
        stream.write("\n")
        session.profiler.print_stats(stream=stream)
        return stream.getvalue()


line_profiling_manager = LineProfilingManager()
