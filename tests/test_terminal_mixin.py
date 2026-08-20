from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.webui.web_server_utils import terminal_mixin
from src.webui.web_server_utils.terminal_mixin import TerminalMixin


pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="terminal session requires bash"
)


class _TerminalHarness(TerminalMixin):
    pass


def _queue_command(monkeypatch: Any, command: str) -> None:
    """Make the next execute_terminal_command call read this command."""
    monkeypatch.setattr(
        terminal_mixin,
        "request",
        SimpleNamespace(get_json=lambda silent=True: {"command": command}),
    )


@pytest.mark.skipif(
    os.name != "posix",
    reason="Git Bash reports MSYS paths that Windows cannot resolve back to a directory",
)
def test_cd_persists_between_commands(tmp_path: Path, monkeypatch: Any) -> None:
    """Carry a changed directory into the next command."""
    harness = _TerminalHarness()
    harness._terminal_cwd = str(tmp_path)
    (tmp_path / "nested").mkdir()

    _queue_command(monkeypatch, "cd nested")
    payload, status = harness.execute_terminal_command()
    assert status == 200
    assert payload["success"] is True

    _queue_command(monkeypatch, "pwd")
    payload, status = harness.execute_terminal_command()
    assert status == 200
    assert Path(payload["output"]).name == "nested"
    assert Path(harness._terminal_cwd).name == "nested"


def test_failing_command_reports_exit_code(tmp_path: Path, monkeypatch: Any) -> None:
    """Return a failed command's exit code."""
    harness = _TerminalHarness()
    harness._terminal_cwd = str(tmp_path)

    _queue_command(monkeypatch, "exit 3")
    payload, status = harness.execute_terminal_command()

    assert status == 200
    assert payload["exit_code"] == 3
    assert payload["success"] is False
    assert harness._terminal_cwd == str(tmp_path)


def test_empty_command_is_rejected(monkeypatch: Any) -> None:
    """Reject commands containing only whitespace."""
    harness = _TerminalHarness()

    _queue_command(monkeypatch, "   ")
    payload, status = harness.execute_terminal_command()

    assert status == 400
    assert payload["error"] == "Command is required"


def test_non_object_payload_is_rejected(monkeypatch: Any) -> None:
    """Reject valid JSON that is not an object."""
    harness = _TerminalHarness()
    monkeypatch.setattr(
        terminal_mixin,
        "request",
        SimpleNamespace(get_json=lambda silent=True: ["echo", "hello"]),
    )

    payload, status = harness.execute_terminal_command()

    assert status == 400
    assert payload["error"] == "Command is required"


def test_deleted_working_directory_resets_to_home(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Recover when the session's working directory was deleted."""
    harness = _TerminalHarness()
    deleted_cwd = tmp_path / "deleted"
    deleted_cwd.mkdir()
    harness._terminal_cwd = str(deleted_cwd)
    deleted_cwd.rmdir()

    _queue_command(monkeypatch, "pwd")
    payload, status = harness.execute_terminal_command()

    assert status == 200
    assert payload["cwd"] == terminal_mixin.TERMINAL_HOME
