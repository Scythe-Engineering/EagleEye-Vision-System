"""Local, single-job compilation of managed ONNX models for MemryX MX3."""

from __future__ import annotations

import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from src.utils.model_library import ArtifactError, ModelLibrary, ModelMetadata
from src.utils.mx3_runtime import Mx3Profile, Mx3RuntimeError


class Mx3CompilerError(RuntimeError):
    """Raised for actionable local MX3 compilation failures."""


class Mx3CompilerBusyError(Mx3CompilerError):
    """Raised when a service already owns an unfinished compilation job."""


@dataclass(frozen=True, slots=True)
class Mx3CompilerSettings:
    """The deliberately small, safe subset of ``mx_nc`` compiler settings."""

    autocrop: bool = True
    num_chips: int = 4
    effort: str = "normal"
    target_fps: float | None = None


@dataclass(frozen=True, slots=True)
class Mx3CompileStatus:
    """Complete persistent snapshot of a compile job suitable for a popup UI."""

    state: str
    job_id: str | None
    model_id: str | None
    stage: str
    percent: float | None
    logs: tuple[str, ...]
    error: str | None = None
    references: tuple[str, ...] = ()
    restart_required: bool = False
    artifacts: Mapping[str, str] | None = None

    def to_dict(self, *, log_limit: int | None = None) -> dict[str, Any]:
        """Return a JSON-friendly copy, optionally limiting its compiler log tail."""
        logs = (
            self.logs
            if log_limit is None
            else self.logs[-log_limit:]
            if log_limit > 0
            else ()
        )
        return {
            "state": self.state,
            "job_id": self.job_id,
            "model_id": self.model_id,
            "stage": self.stage,
            "percent": self.percent,
            "logs": list(logs),
            "error": self.error,
            "references": list(self.references),
            "restart_required": self.restart_required,
            "artifacts": dict(self.artifacts) if self.artifacts is not None else None,
        }


_ANSI_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_PROGRESS_PATTERN = re.compile(r"(?<!\d)(\d{1,3}(?:\.\d+)?)\s*%")
_STAGE_PATTERN = re.compile(
    r"(?:stage|phase|step)\s*(?:\d+\s*(?:/|of)\s*\d+)?\s*[:\-]\s*(.+)",
    re.IGNORECASE,
)


def validate_compiler_settings(
    settings: Mx3CompilerSettings | Mapping[str, Any] | None,
) -> Mx3CompilerSettings:
    """Normalize and validate the only settings permitted to reach ``mx_nc``."""
    if settings is None:
        candidate: Mapping[str, Any] = {}
    elif isinstance(settings, Mx3CompilerSettings):
        candidate = {
            "autocrop": settings.autocrop,
            "num_chips": settings.num_chips,
            "effort": settings.effort,
            "target_fps": settings.target_fps,
        }
    elif isinstance(settings, Mapping):
        candidate = settings
    else:
        raise Mx3CompilerError("Compiler settings must be an object")
    unknown = set(candidate) - {"autocrop", "num_chips", "effort", "target_fps"}
    if unknown:
        raise Mx3CompilerError(
            f"Unsupported compiler setting(s): {', '.join(sorted(map(str, unknown)))}"
        )
    autocrop = candidate.get("autocrop", True)
    num_chips = candidate.get("num_chips", 4)
    effort = candidate.get("effort", "normal")
    target_fps = candidate.get("target_fps")
    if not isinstance(autocrop, bool):
        raise Mx3CompilerError("autocrop must be a boolean")
    if (
        isinstance(num_chips, bool)
        or not isinstance(num_chips, int)
        or not 1 <= num_chips <= 16
    ):
        raise Mx3CompilerError("num_chips must be an integer between 1 and 16")
    if not isinstance(effort, str) or effort not in {"lazy", "normal", "hard"}:
        raise Mx3CompilerError("effort must be one of: lazy, normal, hard")
    if target_fps is not None:
        if isinstance(target_fps, bool) or not isinstance(target_fps, (int, float)):
            raise Mx3CompilerError("target_fps must be a positive number")
        if not math.isfinite(float(target_fps)) or not 1 <= float(target_fps) <= 1000:
            raise Mx3CompilerError("target_fps must be between 1 and 1000")
        target_fps = float(target_fps)
    return Mx3CompilerSettings(autocrop, num_chips, effort, target_fps)


def build_mx_nc_command(
    executable: str | os.PathLike[str],
    onnx_path: str | os.PathLike[str],
    output_directory: str | os.PathLike[str],
    settings: Mx3CompilerSettings | Mapping[str, Any] | None = None,
) -> list[str]:
    """Build a shell-free ``mx_nc`` argv using only validated settings."""
    normalized = validate_compiler_settings(settings)
    command = [
        str(executable),
        "-m",
        str(Path(onnx_path)),
        "--dfp_fname",
        str(Path(output_directory) / "compiled.dfp"),
        "--num_chips",
        str(normalized.num_chips),
        "--effort",
        normalized.effort,
        "-v",
    ]
    if normalized.autocrop:
        command.append("--autocrop")
    if normalized.target_fps is not None:
        command.extend(("--target-fps", f"{normalized.target_fps:g}"))
    return command


def parse_compiler_progress(
    line: str,
    current_stage: str = "Starting compiler",
    current_percent: float | None = None,
) -> tuple[str, float | None]:
    """Extract the latest stage and bounded percentage from one compiler line."""
    clean_line = _ANSI_PATTERN.sub("", line).strip()
    percent = current_percent
    progress_match = _PROGRESS_PATTERN.search(clean_line)
    if progress_match is not None:
        value = float(progress_match.group(1))
        if 0 <= value <= 100:
            percent = value
    stage_match = _STAGE_PATTERN.search(clean_line)
    if stage_match is not None:
        stage = _PROGRESS_PATTERN.sub("", stage_match.group(1)).rstrip(" :-\t")
    elif progress_match is not None:
        stage = clean_line[: progress_match.start()].rstrip(" :-\t")
        stage = re.sub(r"^\[[^]]+\]\s*", "", stage).strip()
        if ":" in stage:
            stage = stage.split(":", 1)[0].strip()
    else:
        stage = ""
    if stage:
        return stage[:160], percent
    return current_stage, percent


def yolo26_profile_for_input(input_width: int, input_height: int) -> dict[str, Any]:
    """Return the supported MX3 YOLO26 runtime contract for a static ONNX input."""
    if (
        isinstance(input_width, bool)
        or isinstance(input_height, bool)
        or not isinstance(input_width, int)
        or not isinstance(input_height, int)
        or input_width < 1
        or input_height < 1
    ):
        raise Mx3CompilerError("YOLO26 input dimensions must be positive integers")
    profile = {
        "input_width": input_width,
        "input_height": input_height,
        "color_order": "rgb",
        "layout": "hwzc",
        "normalization": "zero_to_one",
        "use_model_shape": [False, True],
        "decoder": "yolo_nms_xyxy",
        "adjustable_controls": {"confidence": True, "max_detections": True},
        "max_inflight": 8,
    }
    return Mx3Profile.from_metadata(profile).to_metadata()


def detect_onnx_input_size(onnx_path: str | os.PathLike[str]) -> tuple[int, int]:
    """Read the static image width and height from the first ONNX model input."""
    path = Path(onnx_path)
    try:
        import onnx  # type: ignore[import-not-found]
    except ImportError:
        try:
            import onnxruntime  # type: ignore[import-not-found]

            dimensions = list(
                onnxruntime.InferenceSession(
                    str(path), providers=["CPUExecutionProvider"]
                )
                .get_inputs()[0]
                .shape
            )
        except Exception as error:
            raise Mx3CompilerError(
                "Cannot inspect ONNX input shape; install onnx or provide an MX3 profile"
            ) from error
    else:
        try:
            model = onnx.load(str(path), load_external_data=False)
            dimensions = [
                dimension.dim_value
                for dimension in model.graph.input[0].type.tensor_type.shape.dim
            ]
        except Exception as error:
            raise Mx3CompilerError(
                "Cannot inspect ONNX input shape; provide an MX3 profile"
            ) from error
    if len(dimensions) != 4 or not all(
        isinstance(value, int) and value > 0 for value in dimensions
    ):
        raise Mx3CompilerError(
            "YOLO26 compilation requires a static four-dimensional ONNX input"
        )
    if dimensions[1] in {1, 3, 4}:
        height, width = dimensions[2], dimensions[3]
    elif dimensions[3] in {1, 3, 4}:
        height, width = dimensions[1], dimensions[2]
    else:
        raise Mx3CompilerError("Cannot identify image channels in the ONNX input shape")
    return int(width), int(height)


class Mx3CompilerService:
    """Own one local ``mx_nc`` process at a time and retain its final status."""

    def __init__(
        self,
        model_library: ModelLibrary,
        *,
        compiler_path: str | os.PathLike[str] | None = None,
        max_log_lines: int = 300,
        process_factory: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        """Create an idle service using the deployed environment's compiler."""
        if max_log_lines < 1:
            raise ValueError("max_log_lines must be positive")
        self.model_library = model_library
        self.compiler_path = Path(compiler_path) if compiler_path is not None else None
        self.max_log_lines = max_log_lines
        self._process_factory = process_factory
        self._lock = threading.RLock()
        self._process: Any | None = None
        self._thread: threading.Thread | None = None
        self._cancel_requested = False
        self._shutting_down = False
        self._callback: Callable[[Mx3CompileStatus], None] | None = None
        self._status = Mx3CompileStatus("idle", None, None, "Idle", None, ())

    def status(self) -> Mx3CompileStatus:
        """Return the latest complete snapshot, including retained log lines."""
        with self._lock:
            return self._status

    def start_compile(
        self,
        model_id: str,
        settings: Mx3CompilerSettings | Mapping[str, Any] | None = None,
        *,
        profile: Mapping[str, Any] | None = None,
        overwrite: bool = False,
        callback: Callable[[Mx3CompileStatus], None] | None = None,
    ) -> Mx3CompileStatus:
        """Start compiling this model's existing ONNX artifact without queueing."""
        normalized_settings = validate_compiler_settings(settings)
        source = self._onnx_source(model_id)
        normalized_profile = self._normalize_profile(profile, source)
        model = self.model_library.get_model(model_id)
        if self._model_has_mx3_bundle(model) and not overwrite:
            raise Mx3CompilerError(
                "MX3 artifacts or profile already exist; explicit overwrite is required"
            )
        with self._lock:
            if self._shutting_down:
                raise Mx3CompilerError("MX3 compiler service is shutting down")
            if self._thread is not None and self._thread.is_alive():
                raise Mx3CompilerBusyError("An MX3 compilation job is already running")
            self._cancel_requested = False
            self._callback = callback
            self._status = Mx3CompileStatus(
                "running", uuid.uuid4().hex, model_id, "Preparing compiler", 0.0, ()
            )
            expected_bundle = (
                model.artifacts.get("mx3_dfp"),
                model.artifacts.get("mx3_postprocessor"),
                model.mx3_profile,
            )
            expected_onnx = (
                model.artifacts["onnx"],
                source.stat().st_size,
                source.stat().st_mtime_ns,
            )
            thread = threading.Thread(
                target=self._run_compile,
                args=(
                    model_id,
                    source,
                    normalized_settings,
                    normalized_profile,
                    overwrite,
                    expected_bundle,
                    expected_onnx,
                ),
                daemon=True,
                name="Mx3Compiler",
            )
            self._thread = thread
            thread.start()
            snapshot = self._status
        return snapshot

    def cancel(self) -> Mx3CompileStatus:
        """Request cancellation and asynchronously stop the compiler process group."""
        with self._lock:
            if self._status.state not in {"running", "cancelling"}:
                return self._status
            self._cancel_requested = True
            self._status = self._replace_status(state="cancelling", stage="Cancelling")
            process = self._process
            snapshot = self._status
        self._notify(snapshot)
        if process is not None:
            self._terminate_process_group(process)
            timer = threading.Timer(5.0, self._force_kill_process_group, (process,))
            timer.daemon = True
            timer.start()
        return snapshot

    def shutdown(self, timeout: float = 12.0) -> None:
        """Cancel an active job and wait for its worker to reap the compiler."""
        with self._lock:
            self._shutting_down = True
            self.cancel()
            thread = self._thread
        if thread is None or thread is threading.current_thread():
            return
        thread.join(timeout)
        if thread.is_alive():
            raise Mx3CompilerError("MX3 compiler did not stop during backend shutdown")

    def resolve_compiler_path(self) -> Path:
        """Resolve configured ``mx_nc``, then the UV Python sibling, then PATH."""
        candidates: list[Path] = []
        if self.compiler_path is not None:
            candidates.append(self.compiler_path)
        executable_name = "mx_nc.exe" if os.name == "nt" else "mx_nc"
        candidates.append(Path(sys.executable).parent / executable_name)
        path_match = shutil.which(executable_name)
        if path_match:
            candidates.append(Path(path_match))
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        requested = (
            str(self.compiler_path)
            if self.compiler_path is not None
            else executable_name
        )
        raise Mx3CompilerError(
            f"mx_nc was not found ({requested}); install it in the deployed UV environment"
        )

    def _onnx_source(self, model_id: str) -> Path:
        """Resolve only the managed ONNX artifact used as a compiler source."""
        try:
            artifact = self.model_library.resolve_artifact(model_id, "cpu")
        except ArtifactError as error:
            raise Mx3CompilerError(
                f"Cannot compile model {model_id}: {error}"
            ) from error
        if artifact.slot != "onnx" or artifact.path.suffix.lower() != ".onnx":
            raise Mx3CompilerError(
                "MX3 compilation requires an existing ONNX model artifact"
            )
        return artifact.path

    @staticmethod
    def _model_has_mx3_bundle(model: ModelMetadata) -> bool:
        """Return whether any MX3 artifact or profile would require overwrite."""
        return (
            "mx3_dfp" in model.artifacts
            or "mx3_postprocessor" in model.artifacts
            or model.mx3_profile is not None
        )

    def _normalize_profile(
        self, profile: Mapping[str, Any] | None, onnx_source: Path
    ) -> dict[str, Any]:
        """Validate supplied runtime metadata or derive the YOLO26 guided default."""
        if profile is None:
            width, height = detect_onnx_input_size(onnx_source)
            return yolo26_profile_for_input(width, height)
        try:
            return Mx3Profile.from_metadata(profile).to_metadata()
        except Mx3RuntimeError as error:
            raise Mx3CompilerError(f"Invalid MX3 profile: {error}") from error

    def _run_compile(
        self,
        model_id: str,
        onnx_source: Path,
        settings: Mx3CompilerSettings,
        profile: Mapping[str, Any],
        overwrite: bool,
        expected_bundle: tuple[str | None, str | None, Mapping[str, Any] | None],
        expected_onnx: tuple[str, int, int],
    ) -> None:
        """Run one compiler subprocess and install its output only on full success."""
        process: Any | None = None
        try:
            executable = self.resolve_compiler_path()
            with tempfile.TemporaryDirectory(
                prefix="mx3-compile-"
            ) as temporary_directory:
                output_directory = Path(temporary_directory)
                source_signature = (
                    onnx_source.stat().st_size,
                    onnx_source.stat().st_mtime_ns,
                )
                compiler_input = output_directory / onnx_source.name
                shutil.copy2(onnx_source, compiler_input)
                command = build_mx_nc_command(
                    executable, compiler_input, output_directory, settings
                )
                self._publish(stage="Starting mx_nc", percent=0.0)
                process = self._start_process(command)
                with self._lock:
                    self._process = process
                    cancelled = self._cancel_requested
                if cancelled:
                    self._terminate_process_group(process)
                stdout = getattr(process, "stdout", None)
                if stdout is not None:
                    for raw_line in stdout:
                        self._publish_output(str(raw_line))
                return_code = process.wait()
                with self._lock:
                    cancelled = self._cancel_requested
                if cancelled:
                    self._publish(state="cancelled", stage="Cancelled", percent=None)
                    return
                if return_code != 0:
                    raise Mx3CompilerError(f"mx_nc exited with status {return_code}")
                dfp_path, post_path = self._find_outputs(output_directory)
                current_signature = (
                    onnx_source.stat().st_size,
                    onnx_source.stat().st_mtime_ns,
                )
                if current_signature != source_signature:
                    raise Mx3CompilerError(
                        "The ONNX artifact changed while it was compiling; results were not installed"
                    )
                with self._lock:
                    cancelled = self._cancel_requested
                    self._status = self._replace_status(
                        state="cancelled" if cancelled else "installing",
                        stage="Cancelled" if cancelled else "Installing MX3 bundle",
                        percent=None if cancelled else 100.0,
                    )
                    install_snapshot = self._status
                self._notify(install_snapshot)
                if cancelled:
                    return
                metadata, references = self.model_library.install_mx3_bundle(
                    model_id,
                    dfp_path,
                    post_path,
                    profile,
                    overwrite=overwrite,
                    expected_bundle=expected_bundle,
                    expected_onnx=expected_onnx,
                )
                self._publish(
                    state="succeeded",
                    stage="Installed MX3 bundle",
                    percent=100.0,
                    references=references,
                    restart_required=bool(references),
                    artifacts=dict(metadata.artifacts),
                )
        except Exception as error:
            with self._lock:
                cancelled = self._cancel_requested
            if cancelled:
                self._publish(state="cancelled", stage="Cancelled", percent=None)
            else:
                self._publish(
                    state="failed",
                    stage="Failed",
                    error=str(error) or type(error).__name__,
                )
        finally:
            if process is not None:
                self._stop_process(process)
            with self._lock:
                self._process = None

    def _start_process(self, command: list[str]) -> Any:
        """Start ``mx_nc`` with an isolated process group and never a shell."""
        kwargs: dict[str, Any] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
            "shell": False,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        return self._process_factory(command, **kwargs)

    def _find_outputs(self, output_directory: Path) -> tuple[Path, Path | None]:
        """Find exactly one DFP and at most one post ONNX, excluding pre ONNX."""
        dfps = sorted(
            path for path in output_directory.rglob("*.dfp") if path.is_file()
        )
        posts = sorted(
            path
            for path in output_directory.rglob("*_post.onnx")
            if path.is_file() and not path.name.lower().endswith("_pre.onnx")
        )
        if len(dfps) != 1:
            raise Mx3CompilerError(
                f"mx_nc produced {len(dfps)} DFP files; expected exactly one"
            )
        if len(posts) > 1:
            raise Mx3CompilerError("mx_nc produced multiple postprocessor ONNX files")
        return dfps[0], posts[0] if posts else None

    def _publish_output(self, raw_line: str) -> None:
        """Append one bounded compiler log line and update its parsed progress."""
        line = _ANSI_PATTERN.sub("", raw_line).rstrip("\r\n")
        with self._lock:
            previous_stage = self._status.stage
            previous_percent = self._status.percent
            stage, percent = parse_compiler_progress(
                line, previous_stage, previous_percent
            )
            logs = (*self._status.logs, line)[-self.max_log_lines :]
            self._status = self._replace_status(stage=stage, percent=percent, logs=logs)
            snapshot = self._status
            should_notify = (
                stage != previous_stage
                or percent != previous_percent
                or _PROGRESS_PATTERN.search(line) is None
            )
        if should_notify:
            self._notify(snapshot)

    def _publish(self, **changes: Any) -> None:
        """Replace status fields and notify the subscriber with the full snapshot."""
        with self._lock:
            self._status = self._replace_status(**changes)
            snapshot = self._status
        self._notify(snapshot)

    def _replace_status(self, **changes: Any) -> Mx3CompileStatus:
        """Create a complete status snapshot by replacing selected current fields."""
        values = self._status.to_dict()
        values.update(changes)
        logs = values["logs"]
        values["logs"] = tuple(logs)
        references = values["references"]
        values["references"] = tuple(references)
        artifacts = values["artifacts"]
        if artifacts is not None:
            values["artifacts"] = dict(artifacts)
        return Mx3CompileStatus(**values)

    def _notify(self, snapshot: Mx3CompileStatus) -> None:
        """Call the current subscriber without allowing UI failures to stop a job."""
        with self._lock:
            callback = self._callback
        if callback is not None:
            try:
                callback(snapshot)
            except Exception:
                pass

    def _terminate_process_group(self, process: Any) -> None:
        """Send a graceful termination signal to the isolated compiler group."""
        if getattr(process, "poll", lambda: None)() is not None:
            return
        try:
            if os.name == "nt":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                kill_process_group = getattr(os, "killpg")
                get_process_group = getattr(os, "getpgid")
                kill_process_group(get_process_group(process.pid), signal.SIGTERM)
        except (AttributeError, OSError):
            try:
                process.terminate()
            except (AttributeError, OSError):
                return

    def _force_kill_process_group(self, process: Any) -> None:
        """Force-kill the isolated compiler group if graceful cancellation failed."""
        if getattr(process, "poll", lambda: None)() is not None:
            return
        try:
            if os.name != "nt":
                kill_process_group = getattr(os, "killpg")
                get_process_group = getattr(os, "getpgid")
                kill_process_group(get_process_group(process.pid), signal.SIGKILL)
            else:
                process.kill()
        except (AttributeError, OSError):
            try:
                process.kill()
            except (AttributeError, OSError):
                return

    def _stop_process(self, process: Any) -> None:
        """Terminate, escalate after five seconds, and reap a live compiler."""
        poll = getattr(process, "poll", None)
        wait = getattr(process, "wait", None)
        if not callable(poll) or not callable(wait) or poll() is not None:
            return
        self._terminate_process_group(process)
        try:
            wait(timeout=5)
            return
        except TypeError:
            return
        except subprocess.TimeoutExpired:
            pass
        self._force_kill_process_group(process)
        try:
            wait(timeout=5)
        except (subprocess.TimeoutExpired, TypeError):
            return
