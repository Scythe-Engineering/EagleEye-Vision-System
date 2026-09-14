"""Unit tests for the portable Blender match batch runner."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from benchmarks.blender import render_match


def _scenes(tmp_path: Path) -> Path:
    """Create a compact valid route document suitable for non-rendering tests."""
    path = tmp_path / "scenes.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "routes": [
                    {"name": "approach", "duration_s": 1},
                    {"name": "turn", "duration_s": 2},
                ],
            }
        )
    )
    return path


def test_list_expands_combined_realistic_jobs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Listing is deterministic and does not require Blender or an output directory."""
    assert (
        render_match.main(
            [
                "--scenes",
                str(_scenes(tmp_path)),
                "--output",
                str(tmp_path / "out"),
                "--list",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.splitlines() == [
        "approach:combined-realistic",
        "turn:combined-realistic",
    ]


def test_invalid_duplicate_assignment_fails_before_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Overlapping machine assignments are rejected before any job directory exists."""
    output = tmp_path / "out"
    assert (
        render_match.main(
            [
                "--scenes",
                str(_scenes(tmp_path)),
                "--output",
                str(output),
                "--jobs",
                "approach:combined-realistic,approach:combined-realistic",
            ]
        )
        == 2
    )
    assert "duplicate" in capsys.readouterr().err
    assert not output.exists()


def test_clean_job_is_rejected_before_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Historical clean catalog entries cannot be selected for new runs."""
    output = tmp_path / "out"
    assert (
        render_match.main(
            [
                "--scenes",
                str(_scenes(tmp_path)),
                "--output",
                str(output),
                "--jobs",
                "approach:clean",
            ]
        )
        == 2
    )
    assert "invalid job" in capsys.readouterr().err
    assert not output.exists()


def test_preflight_forwards_sparse_indices_and_quality_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sparse preflight uses its own root and invoke only the generator."""
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, stdout="Blender test")
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(render_match.subprocess, "run", fake_run)
    output = tmp_path / "out"
    assert (
        render_match.main(
            [
                "--scenes",
                str(_scenes(tmp_path)),
                "--output",
                str(output),
                "--jobs",
                "approach:combined-realistic",
                "--frame-indices",
                "0,119",
                "--samples",
                "32",
                "--denoise",
                "--persistent-data",
            ]
        )
        == 0
    )
    assert len(commands) == 1
    command = commands[0]
    assert (
        "--frame-indices" in command
        and command[command.index("--frame-indices") + 1] == "0,119"
    )
    assert str(output / "preflight" / "approach" / "combined-realistic") in command
    assert command[command.index("--samples") + 1] == "32"
    assert "--denoise" in command
    assert "--persistent-data" in command
    assert not (
        output / "approach" / "combined-realistic" / "combined-realistic.mkv"
    ).exists()


def test_completed_job_is_skipped_only_when_signature_and_outputs_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid completion marker prevents unnecessary Blender work on rerun."""
    scenes = _scenes(tmp_path)
    output = tmp_path / "out"
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, stdout="Blender test")
        calls.append(command)
        job_dir = output / "approach" / "combined-realistic"
        if "package.py" not in " ".join(command):
            (job_dir / "truth.jsonl").write_text("{}\n")
            (job_dir / "provenance.json").write_text("{}\n")
            (job_dir / "settings.json").write_text("{}\n")
        else:
            video = Path(command[command.index("--output") + 1])
            video.write_bytes(b"video")
            video.with_suffix(".mkv.manifest.json").write_text(
                json.dumps(
                    {
                        "video": video.name,
                        "pixel_round_trip": True,
                        "byte_size": 5,
                        "video_sha256": hashlib.sha256(b"video").hexdigest(),
                        "encoding_command": ["ffmpeg", video.name],
                    }
                )
            )
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(render_match.subprocess, "run", fake_run)
    argv = [
        "--scenes",
        str(scenes),
        "--output",
        str(output),
        "--jobs",
        "approach:combined-realistic",
    ]
    assert render_match.main(argv) == 0
    assert len(calls) == 2
    assert render_match.main(argv) == 0
    assert len(calls) == 2
    (output / "approach/combined-realistic/combined-realistic.mkv").write_bytes(
        b"truncated"
    )
    (output / "approach/combined-realistic/combined-realistic.partial.mkv").write_bytes(
        b"interrupted"
    )
    assert render_match.main(argv) == 0
    assert len(calls) == 4
    assert (
        output / "approach/combined-realistic/combined-realistic.mkv"
    ).read_bytes() == b"video"
