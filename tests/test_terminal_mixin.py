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


def _queue_command(
    monkeypatch: Any, command: str, sudo_password: str | None = None
) -> None:
    """Make the next execute_terminal_command call read this command.

    Args:
        monkeypatch: Pytest fixture used to replace the request object.
        command: Command supplied by the simulated request.
        sudo_password: Optional password supplied by the simulated request.
    """
    body = {"command": command}
    if sudo_password is not None:
        body["sudo_password"] = sudo_password
    monkeypatch.setattr(
        terminal_mixin,
        "request",
        SimpleNamespace(get_json=lambda silent=True: body),
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


def test_sudo_password_uses_stdin_without_echoing(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Pass a sudo password through stdin without including it in output.

    Args:
        tmp_path: Temporary directory used for the fake sudo executable.
        monkeypatch: Pytest fixture used to configure the test environment.
    """
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sudo = fake_bin / "sudo"
    fake_sudo.write_text(
        "#!/bin/sh\n"
        '[ "$1" = "-S" ] || exit 90\n'
        "shift\n"
        '[ "$1" = "-p" ] || exit 91\n'
        "shift 2\n"
        'IFS= read -r password || exit 92\n'
        '[ "$password" = "test-password" ] || exit 93\n'
        'printf "password accepted\\n"\n',
        encoding="utf-8",
    )
    fake_sudo.chmod(0o700)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")

    harness = _TerminalHarness()
    harness._terminal_cwd = str(tmp_path)
    _queue_command(monkeypatch, "sudo true", sudo_password="test-password")

    payload, status = harness.execute_terminal_command()

    assert status == 200
    assert payload["success"] is True
    assert payload["output"] == "password accepted"
    assert "test-password" not in payload["output"]
    assert "test-password" not in payload["error"]


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
