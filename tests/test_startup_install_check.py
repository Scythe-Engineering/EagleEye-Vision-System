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
    return StartupInstallChecker(logger=_LoggerStub(), repo_root=repo_root)


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
        state["current_ready"] = True
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(checker, "_imports_available", fake_imports)
    monkeypatch.setattr(checker, "_run_command", fake_run)

    assert checker._ensure_uv_environment() is True
    assert calls == [["uv", "sync"]]


def test_repo_venv_interpreter_detection_accepts_unresolved_venv_python(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checker = _make_checker(tmp_path)
    checker.venv_python.parent.mkdir(parents=True, exist_ok=True)
    checker.venv_python.write_text("", encoding="utf-8")

    monkeypatch.setattr("src.startup.install_check.sys.prefix", str(checker.venv_dir))

    assert checker._is_repo_venv_interpreter(checker.venv_python) is True
