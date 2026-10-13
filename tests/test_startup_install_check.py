"""Unit tests for startup install validation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from src.startup.install_check import StartupInstallChecker, WEBUI_REQUIRED_ASSETS


class _LoggerStub:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def log(self, message: str) -> None:
        self.messages.append(message)


def _make_checker(tmp_path: Path) -> StartupInstallChecker:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "package.json").write_text("{}", encoding="utf-8")
    (repo_root / "vite.config.js").write_text("export default {}", encoding="utf-8")

    webui_dir = repo_root / "src" / "webui"
    (webui_dir / "js").mkdir(parents=True)
    (webui_dir / "css").mkdir()
    (webui_dir / "html").mkdir()
    (webui_dir / "static").mkdir()
    (webui_dir / "index.html").write_text("<html></html>", encoding="utf-8")
    (webui_dir / "style.css").write_text("body {}", encoding="utf-8")
    (webui_dir / "js" / "main.js").write_text("console.log('x')", encoding="utf-8")
    (webui_dir / "css" / "a.css").write_text(".a {}", encoding="utf-8")
    (webui_dir / "html" / "a.html").write_text("<div></div>", encoding="utf-8")
    # Temporary checker tests must never touch a real UV_PROJECT_ENVIRONMENT:
    # fake sync creates an empty interpreter file under this isolated repo.
    project_environment = os.environ.pop("UV_PROJECT_ENVIRONMENT", None)
    try:
        return StartupInstallChecker(logger=_LoggerStub(), repo_root=repo_root)
    finally:
        if project_environment is not None:
            os.environ["UV_PROJECT_ENVIRONMENT"] = project_environment


def test_webui_build_required_when_assets_missing(tmp_path: Path) -> None:
    checker = _make_checker(tmp_path)
    assert checker._webui_build_required() is True
    assert checker._missing_webui_assets() == list(WEBUI_REQUIRED_ASSETS)


def test_webui_build_not_required_when_assets_are_present_and_newer(
    tmp_path: Path,
) -> None:
    checker = _make_checker(tmp_path)
    latest_source_mtime = max(
        path.stat().st_mtime for path in checker._webui_source_files()
    )

    for asset_name in WEBUI_REQUIRED_ASSETS:
        asset_path = checker.static_dir / asset_name
        asset_path.write_text(asset_name, encoding="utf-8")
        newer_time = latest_source_mtime + 10
        os.utime(asset_path, (newer_time, newer_time))

    assert checker._missing_webui_assets() == []
    assert checker._webui_build_required() is False


def test_npm_install_ready_requires_gltf_transform_cli(tmp_path: Path) -> None:
    checker = _make_checker(tmp_path)
    checker.vite_bin.parent.mkdir(parents=True)
    checker.vite_bin.write_text("", encoding="utf-8")
    (checker.node_modules_dir / "three").mkdir()
    (checker.node_modules_dir / "three" / "package.json").write_text(
        "{}", encoding="utf-8"
    )

    assert checker._npm_install_ready() is False

    checker.gltf_transform_bin.write_text("", encoding="utf-8")
    (checker.node_modules_dir / "@gltf-transform" / "cli").mkdir(parents=True)
    (
        checker.node_modules_dir / "@gltf-transform" / "cli" / "package.json"
    ).write_text("{}", encoding="utf-8")

    assert checker._npm_install_ready() is True


def test_ensure_uv_environment_runs_sync_only_when_imports_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checker = _make_checker(tmp_path)
    calls: list[list[str]] = []
    current_python = Path(sys.executable).resolve()
    state = {"current_ready": False}

    monkeypatch.setattr("src.startup.install_check.which", lambda name: f"/usr/bin/{name}")

    def fake_imports(python_executable: Path) -> bool:
        return python_executable.resolve() == current_python and state["current_ready"]

    def fake_run(command: list[str], **_kwargs):
        calls.append(command)
        checker.venv_python.parent.mkdir(parents=True, exist_ok=True)
        checker.venv_python.write_text("", encoding="utf-8")
        state["current_ready"] = True
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(checker, "_imports_available", fake_imports)
    monkeypatch.setattr(checker, "_check_wpilib_runtime", lambda *_args: None)
    monkeypatch.setattr(checker, "_run_command", fake_run)

    assert checker._ensure_uv_environment() is True
    assert calls == [["uv", "sync", "--locked", "--group", "wpilib2026"]]


def test_repo_venv_interpreter_detection_accepts_unresolved_venv_python(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checker = _make_checker(tmp_path)
    checker.venv_python.parent.mkdir(parents=True, exist_ok=True)
    checker.venv_python.write_text("", encoding="utf-8")

    monkeypatch.setattr("src.startup.install_check.sys.prefix", str(checker.venv_dir))

    assert checker._is_repo_venv_interpreter(checker.venv_python) is True


def test_startup_reads_persisted_alpha_for_sync(tmp_path: Path) -> None:
    checker = _make_checker(tmp_path)
    (checker.repo_root / ".wpilib-variant").write_text("2027-alpha7\n")
    assert checker._wpilib_variant() == "2027-alpha7"
    assert checker._uv_sync_command(checker._wpilib_variant()) == [
        "uv", "sync", "--locked", "--no-group", "wpilib2026",
        "--group", "wpilib2027-alpha7", "--no-build-package", "pyntcore",
        "--no-build-package", "robotpy-wpimath",
    ]


def test_startup_rejects_invalid_persisted_variant(tmp_path: Path) -> None:
    import pytest
    checker = _make_checker(tmp_path)
    (checker.repo_root / ".wpilib-variant").write_text("2027\n")
    with pytest.raises(RuntimeError, match="Invalid WPILib variant"):
        checker._wpilib_variant()


def test_startup_alpha_linux_requires_glibc_239(monkeypatch) -> None:
    import pytest
    monkeypatch.setattr("src.startup.install_check.platform.system", lambda: "Linux")
    monkeypatch.setattr("src.startup.install_check.platform.libc_ver", lambda: ("glibc", "2.36"))
    with pytest.raises(RuntimeError, match="glibc >= 2.39"):
        StartupInstallChecker._check_wpilib_platform("2027-alpha7")
    StartupInstallChecker._check_wpilib_platform("2026")
    monkeypatch.setattr("src.startup.install_check.platform.libc_ver", lambda: ("glibc", "2.39"))
    StartupInstallChecker._check_wpilib_platform("2027-alpha7")


def test_startup_rejects_runtime_different_from_persisted_choice(tmp_path: Path, monkeypatch) -> None:
    import pytest
    checker = _make_checker(tmp_path)
    (checker.repo_root / ".wpilib-variant").write_text("2027-alpha7\n")
    monkeypatch.setattr(checker, "_run_command", lambda command, **kwargs:
        subprocess.CompletedProcess(command, 0, '{"api":"2026","ntcore":"2026.2.2","wpimath":"2026.2.2"}', ""))
    with pytest.raises(RuntimeError, match="does not match runtime API"):
        checker._check_wpilib_runtime(Path(sys.executable), "2027-alpha7")


def test_startup_reports_actual_binding_api(tmp_path: Path, monkeypatch) -> None:
    checker = _make_checker(tmp_path)
    monkeypatch.setattr(checker, "_run_command", lambda command, **kwargs:
        subprocess.CompletedProcess(command, 0, '{"api":"2027","ntcore":"2027.0.0a7","wpimath":"2027.0.0a7"}', ""))
    checker._check_wpilib_runtime(Path(sys.executable), "2026")
    assert any("WPILib API 2027: pyntcore=2027.0.0a7" in message for message in checker.logger.messages)
    assert any("unmanaged developer environment; no persisted install selection" in message
               for message in checker.logger.messages)
    assert not any("selection: 2026" in message for message in checker.logger.messages)


def test_startup_honors_isolated_uv_project_environment(tmp_path: Path, monkeypatch) -> None:
    environment = tmp_path / "isolated-alpha"
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", str(environment))
    checker = StartupInstallChecker(logger=_LoggerStub(), repo_root=tmp_path)
    assert checker.venv_dir == environment
    assert checker.venv_python == environment / "bin" / "python"


def test_startup_reports_installed_binding_versions() -> None:
    from src.utils import wpilib_compat as wpi

    checker = StartupInstallChecker(
        logger=_LoggerStub(), repo_root=Path(__file__).resolve().parents[1]
    )
    variant = "2027-alpha7" if wpi.API == "2027" else "2026"
    checker._check_wpilib_runtime(Path(sys.executable), variant)
    assert any(
        f"WPILib API {wpi.API}: pyntcore={wpi.NT_VERSION}, robotpy-wpimath={wpi.MATH_VERSION}"
        in message for message in checker.logger.messages
    )
