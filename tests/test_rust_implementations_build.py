"""Unit tests for Rust builder environment detection."""

from __future__ import annotations

from pathlib import Path
import subprocess

from src.rust_implementations.build import RustModuleBuilder


def test_get_clean_env_prepends_user_rustup_bin(monkeypatch) -> None:
    """The builder should search rustup's default install path first."""
    builder = RustModuleBuilder(Path("/tmp/rust_implementations"))

    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("CONDA_PREFIX", "/tmp/conda")
    monkeypatch.setattr(
        builder,
        "_get_additional_bin_dirs",
        lambda: [Path("/home/pi/.cargo/bin"), Path("/home/pi/.local/bin")],
    )

    env = builder._get_clean_env()

    assert "CONDA_PREFIX" not in env
    assert env["PATH"].split(":")[:4] == [
        "/home/pi/.cargo/bin",
        "/home/pi/.local/bin",
        "/usr/bin",
        "/bin",
    ]


def test_check_dependencies_finds_cargo_in_added_rustup_path(monkeypatch) -> None:
    """A rustup toolchain should still be found when the parent PATH omits it."""
    builder = RustModuleBuilder(Path("/tmp/rust_implementations"))

    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(
        builder,
        "_get_additional_bin_dirs",
        lambda: [Path("/home/pi/.cargo/bin")],
    )

    def fake_which(executable: str, path: str | None = None) -> str | None:
        if executable == "cargo" and path and "/home/pi/.cargo/bin" in path.split(":"):
            return "/home/pi/.cargo/bin/cargo"
        return None

    calls: list[tuple[list[str], str | None]] = []

    def fake_run(command: list[str], **kwargs):
        env = kwargs.get("env", {})
        calls.append((command, env.get("PATH")))

        if command == ["/home/pi/.cargo/bin/cargo", "--version"]:
            return subprocess.CompletedProcess(command, 0, stdout="cargo 1.0", stderr="")
        if command == ["uv", "run", "maturin", "--version"]:
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
        (["/home/pi/.cargo/bin/cargo", "--version"], "/home/pi/.cargo/bin:/usr/bin:/bin"),
        (["uv", "run", "maturin", "--version"], "/home/pi/.cargo/bin:/usr/bin:/bin"),
    ]
