"""Focused tests for local MX3 compiler primitives and bundle commits."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from src.utils.model_library import ArtifactError, ModelLibrary
from src.utils.mx3_compiler import (
    Mx3CompileStatus,
    Mx3CompilerBusyError,
    Mx3CompilerService,
    build_mx_nc_command,
    parse_compiler_progress,
    yolo26_profile_for_input,
)


def _library_with_onnx(tmp_path: Path) -> tuple[ModelLibrary, str]:
    """Create a managed model with a tiny existing ONNX source artifact."""
    library = ModelLibrary(
        tmp_path / "models", pipeline_config_path=tmp_path / "pipelines.json"
    )
    model = library.create_model("Test model")
    source = tmp_path / "source.onnx"
    source.write_bytes(b"onnx")
    library.import_artifact(model.model_id, "onnx", source)
    return library, model.model_id


def test_progress_and_fixed_command_validation(tmp_path: Path) -> None:
    """Progress parsing and argv construction retain no untrusted arguments."""
    stage, percent = parse_compiler_progress("[INFO] Stage 2/4: Placement 42.5%")
    assert stage == "Placement"
    assert percent == 42.5
    stage, percent = parse_compiler_progress(
        "\x1b[1AAssembling DFP : \x1b[32m■■■■\x1b[0m 97.8%"
    )
    assert stage == "Assembling DFP"
    assert percent == 97.8
    command = build_mx_nc_command(
        tmp_path / "mx_nc",
        tmp_path / "model.onnx",
        tmp_path / "out",
        {"autocrop": False, "num_chips": 2, "effort": "hard", "target_fps": 30},
    )
    assert command[1:3] == ["-m", str(tmp_path / "model.onnx")]
    assert "--autocrop" not in command
    assert command[-2:] == ["--target-fps", "30"]
    with pytest.raises(Exception, match="Unsupported compiler"):
        build_mx_nc_command("mx_nc", "a.onnx", "out", {"raw_args": "--bad"})


def test_status_serialization_can_limit_sse_log_tail() -> None:
    """SSE snapshots can stay small while status requests retain complete logs."""
    status = Mx3CompileStatus(
        "running", "job", "model", "Compiling", 50.0, ("one", "two", "three")
    )
    assert status.to_dict()["logs"] == ["one", "two", "three"]
    assert status.to_dict(log_limit=2)["logs"] == ["two", "three"]
    assert status.to_dict(log_limit=0)["logs"] == []


def test_service_allows_one_live_job(tmp_path: Path) -> None:
    """A service rejects a second request rather than creating a compile queue."""
    library, model_id = _library_with_onnx(tmp_path)
    service = Mx3CompilerService(library)
    release = threading.Event()

    def blocked_run(*_args: Any) -> None:
        """Keep the service running until the test releases it."""
        release.wait(2)

    service._run_compile = blocked_run  # type: ignore[method-assign]
    profile = yolo26_profile_for_input(640, 640)
    service.start_compile(model_id, profile=profile)
    with pytest.raises(Mx3CompilerBusyError):
        service.start_compile(model_id, profile=profile)
    release.set()
    assert service._thread is not None
    service._thread.join(2)


def test_cancel_terminates_fake_process(tmp_path: Path) -> None:
    """Cancellation reaches a spawned compiler and ends with persistent status."""
    library, model_id = _library_with_onnx(tmp_path)
    executable = tmp_path / "mx_nc"
    executable.write_text("fake", encoding="utf-8")
    started = threading.Event()
    stopped = threading.Event()

    class FakeProcess:
        """Small blocking Popen replacement used to exercise cancellation."""

        pid = 999999
        stdout: tuple[str, ...] = ()

        def poll(self) -> None:
            """Report the process as live until the service terminates it."""
            return None

        def terminate(self) -> None:
            """Release the waiting fake compiler."""
            stopped.set()

        def wait(self) -> int:
            """Wait until termination and return a non-zero result."""
            started.set()
            stopped.wait(2)
            return -15

    service = Mx3CompilerService(
        library,
        compiler_path=executable,
        process_factory=lambda *_a, **_k: FakeProcess(),
    )
    service.start_compile(model_id, profile=yolo26_profile_for_input(640, 640))
    assert started.wait(2)
    assert service.cancel().state == "cancelling"
    assert service._thread is not None
    service._thread.join(2)
    assert service.status().state == "cancelled"


def test_successful_job_installs_compiler_outputs(tmp_path: Path) -> None:
    """A successful compiler job installs its generated bundle and profile."""
    library, model_id = _library_with_onnx(tmp_path)
    executable = tmp_path / "mx_nc"
    executable.write_text("fake", encoding="utf-8")

    class FakeProcess:
        """Completed compiler process exposing one progress line."""

        pid = 1
        stdout = ("Compiling Model: 45%\n",)

        def wait(self) -> int:
            """Report compiler success."""
            return 0

    def process_factory(command: list[str], **_kwargs: Any) -> FakeProcess:
        """Create the output files requested by the fixed compiler argv."""
        output = Path(command[command.index("--dfp_fname") + 1])
        output.write_bytes(b"dfp")
        Path(command[command.index("-m") + 1]).with_name(
            "source_post.onnx"
        ).write_bytes(b"post")
        return FakeProcess()

    service = Mx3CompilerService(
        library, compiler_path=executable, process_factory=process_factory
    )
    status = service.start_compile(model_id, profile=yolo26_profile_for_input(320, 320))
    assert status.state == "running"
    assert service._thread is not None
    service._thread.join(2)

    final = service.status()
    assert final.state == "succeeded"
    artifact = library.resolve_artifact(model_id, "mx3:0")
    assert artifact.path.read_bytes() == b"dfp"
    assert artifact.postprocessor_path is not None
    assert artifact.postprocessor_path.read_bytes() == b"post"


def test_library_startup_removes_only_stale_generated_artifacts(
    tmp_path: Path,
) -> None:
    """Startup removes superseded generated files but preserves manifest artifacts."""
    library, model_id = _library_with_onnx(tmp_path)
    model_dir = library.root / model_id
    stale = model_dir / "mx3_dfp-stale.dfp"
    stale.write_bytes(b"stale")
    active_source = tmp_path / "active.dfp"
    active_source.write_bytes(b"active")
    active, _ = library.install_mx3_bundle(
        model_id, active_source, None, yolo26_profile_for_input(320, 320)
    )
    active_path = library.root / active.artifacts["mx3_dfp"]

    ModelLibrary(library.root, pipeline_config_path=tmp_path / "pipelines.json")

    assert not stale.exists()
    assert active_path.exists()


def test_bundle_commit_is_atomic_on_input_failure(tmp_path: Path) -> None:
    """A failed replacement leaves the previously committed MX3 bundle current."""
    library, model_id = _library_with_onnx(tmp_path)
    dfp = tmp_path / "result.dfp"
    post = tmp_path / "result_post.onnx"
    dfp.write_bytes(b"dfp-one")
    post.write_bytes(b"post-one")
    profile = yolo26_profile_for_input(640, 640)
    first, _ = library.install_mx3_bundle(model_id, dfp, post, profile)
    first_dfp = library.resolve_artifact(model_id, "mx3:0").path
    assert first_dfp.read_bytes() == b"dfp-one"

    with pytest.raises(ArtifactError):
        library.install_mx3_bundle(
            model_id, tmp_path / "missing.dfp", None, profile, overwrite=True
        )
    current = library.get_model(model_id)
    assert current.artifacts["mx3_dfp"] == first.artifacts["mx3_dfp"]
    assert first_dfp.exists()

    expected_bundle = (
        first.artifacts.get("mx3_dfp"),
        first.artifacts.get("mx3_postprocessor"),
        first.mx3_profile,
    )
    dfp.write_bytes(b"dfp-two")
    second, _ = library.install_mx3_bundle(model_id, dfp, post, profile, overwrite=True)
    with pytest.raises(ArtifactError, match="changed while compilation"):
        library.install_mx3_bundle(
            model_id,
            dfp,
            post,
            profile,
            overwrite=True,
            expected_bundle=expected_bundle,
        )
    assert library.get_model(model_id).artifacts == second.artifacts
