"""Unit tests for Rust builder environment detection."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

from src.rust_implementations.build import RustModuleBuilder


def test_get_clean_env_prepends_user_rustup_bin(monkeypatch) -> None:
    """The builder should search rustup's default install path first."""
    builder = RustModuleBuilder(Path("/tmp/rust_implementations"))

    cargo_bin = Path("/home/pi/.cargo/bin")
    local_bin = Path("/home/pi/.local/bin")
    system_paths = ["/usr/bin", "/bin"]
    monkeypatch.setenv("PATH", os.pathsep.join(system_paths))
    monkeypatch.setenv("CONDA_PREFIX", "/tmp/conda")
    monkeypatch.setattr(
        builder,
        "_get_additional_bin_dirs",
        lambda: [cargo_bin, local_bin],
    )

    env = builder._get_clean_env()

    assert "CONDA_PREFIX" not in env
    assert env["PATH"].split(os.pathsep)[:4] == [
        str(cargo_bin),
        str(local_bin),
        *system_paths,
    ]


def test_check_dependencies_finds_cargo_in_added_rustup_path(monkeypatch) -> None:
    """A rustup toolchain should still be found when the parent PATH omits it."""
    builder = RustModuleBuilder(Path("/tmp/rust_implementations"))

    cargo_bin = Path("/home/pi/.cargo/bin")
    cargo_executable = str(cargo_bin / "cargo")
    expected_path = os.pathsep.join((str(cargo_bin), "/usr/bin", "/bin"))
    monkeypatch.setenv("PATH", os.pathsep.join(("/usr/bin", "/bin")))
    monkeypatch.setattr(
        builder,
        "_get_additional_bin_dirs",
        lambda: [cargo_bin],
    )

    def fake_which(executable: str, path: str | None = None) -> str | None:
        if executable == "cargo" and path and str(cargo_bin) in path.split(os.pathsep):
            return cargo_executable
        return None

    maturin_version_command = [*builder._maturin_command(), "--version"]
    calls: list[tuple[list[str], str | None]] = []

    def fake_run(command: list[str], **kwargs):
        env = kwargs.get("env", {})
        calls.append((command, env.get("PATH")))

        if command == [cargo_executable, "--version"]:
            return subprocess.CompletedProcess(command, 0, stdout="cargo 1.0", stderr="")
        if command == maturin_version_command:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="maturin 1.0",
                stderr="",
            )
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr("src.rust_implementations.build.shutil.which", fake_which)
    monkeypatch.setattr("src.rust_implementations.build.subprocess.run", fake_run)

    assert builder.check_dependencies() is True
    assert calls == [
        ([cargo_executable, "--version"], expected_path),
        (maturin_version_command, expected_path),
    ]
