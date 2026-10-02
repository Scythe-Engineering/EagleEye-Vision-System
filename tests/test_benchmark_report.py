"""Focused tests for benchmark result persistence and offline reports."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.report import (
    BOUNDARY,
    RunWriter,
    write_diagnostic_images,
)


def _metadata() -> dict[str, object]:
    return {
        "dataset_id": "fixture",
        "dataset_release": "1",
        "dataset_manifest_sha256": "a" * 64,
        "graph_sha256": {"full-frame": "b" * 64},
        "calibration_sha256": ["c" * 64],
        "map_sha256": ["d" * 64],
        "revision": "abc",
        "git_dirty": False,
        "os": "test",
        "machine": "test",
        "cpu": "test",
        "logical_cpus": 1,
        "thread_environment": {},
        "dependencies": {},
        "mode": "accuracy",
        "boundary": BOUNDARY,
    }


def test_report_escapes_content_and_has_inline_svg(tmp_path: Path) -> None:
    writer = RunWriter.create(
        tmp_path / "run", {**_metadata(), "label": "<script>alert(1)</script>"}
    )
    writer.write_frame(
        {
            "frame_index": 0,
            "timestamp_ns": 0,
            "completed": True,
            "metrics": {
                "pose_available": True,
                "robot_pose": {"translation_3d_m": 0.1},
            },
        }
    )
    writer.finish(
        {"note": "<img src=x onerror=1>"},
        [{"clip": "a&b", "configuration": "full-frame", "variant": "clean"}],
    )
    page = (tmp_path / "run" / "index.html").read_text()
    assert "<script>alert" not in page and "&lt;script&gt;" in page
    assert "<img src=x" not in page and "&lt;img src=x" in page
    assert "<svg" in page and "cdn" not in page.lower()
    assert "&quot;complete&quot;" in page and "&quot;incomplete&quot;" not in page


def test_report_separates_series_stages_and_uses_eligible_size_denominators(
    tmp_path: Path,
) -> None:
    """Charts label each stream and never count excluded truth in detection recall."""
    writer = RunWriter.create(tmp_path / "run", _metadata())
    common = {
        "clip": "clip-<unsafe>",
        "repeat": 2,
        "truth": {"T_field_from_robot": [1, 0, 0, 1, 0, 1, 0, 2] + [0] * 8},
        "metrics": {
            "robot_pose": {"translation_3d_m": 0.25},
            "pose_available": True,
            "detection": {
                "by_tag": [
                    {"eligible": True, "projected_size_px": 12, "detected": True},
                    {"eligible": True, "projected_size_px": 12, "detected": False},
                    # This would inflate the bin if reports used every rendered tag.
                    {"eligible": False, "projected_size_px": 12, "detected": True},
                ]
            },
        },
    }
    writer.write_frame(
        {
            **common,
            "configuration": "full-frame",
            "frame_index": 0,
            "processing_ns": 2_000_000,
            "decode_ns": 500_000,
            "scheduled_completion_latency_ns": 3_000_000,
            "handoff_completion_latency_ns": 4_000_000,
            "delivery_lateness_ns": -250_000,
        }
    )
    writer.write_frame(
        {
            **common,
            "configuration": "temporal",
            "frame_index": 1,
            "processing_ns": 3_000_000,
            "decode_ns": 600_000,
        }
    )
    writer.finish({}, [])
    page = (tmp_path / "run" / "index.html").read_text()

    assert "Detection recall by projected tag size" in page
    assert "Eligible truth tags only" in page
    assert page.count("total binned: 2") == 2 and page.count(">1/2<") == 2
    assert "total binned: 4" not in page
    assert "clip=clip-&lt;unsafe&gt; · config=full-frame" in page
    assert "clip=clip-&lt;unsafe&gt; · config=temporal" in page
    assert "clip=clip-<unsafe>" not in page
    assert page.count('class="legend"') >= 2
    assert "Pipeline execution time by frame" not in page
    assert "Decode time by frame" not in page
    assert "Scheduled completion latency by frame" not in page


def test_report_keeps_clips_separate_and_shows_provisional_data(tmp_path: Path) -> None:
    """Real pilot data gets useful bins, and missing poses are not joined across gaps."""
    writer = RunWriter.create(tmp_path / "run", _metadata())
    for clip in ("clean", "combined"):
        for frame in range(3):
            writer.write_frame(
                {
                    "clip": clip,
                    "configuration": "full-frame",
                    "frame_index": frame,
                    "processing_ns": 1_000_000,
                    "metrics": {
                        "pose_available": frame != 1,
                        "robot_pose": None if frame == 1 else {"translation_3d_m": 0.1},
                        "detection": {
                            "by_tag": [
                                {
                                    "eligible": False,
                                    "category": "provisional",
                                    "projected_size_px": 24,
                                    "detected": frame != 1,
                                }
                            ]
                        },
                    },
                }
            )
    writer.finish({}, [])
    page = (tmp_path / "run" / "index.html").read_text()
    assert page.count('class="clip-plots"') == 2
    assert "No eligible truth tags" in page
    assert "Provisional detection match rate" in page
    assert page.count(">2/3<") == 2
    assert "This is not eligible recall" in page
    import re

    error_charts = re.findall(
        r"<h2>Robot translation error</h2>(.*?)</svg>", page, re.DOTALL
    )
    assert len(error_charts) == 2
    assert all(
        chart.count("<circle") == 2 and "<polyline" not in chart
        for chart in error_charts
    )
    assert page.index("Plots by clip") < page.index(
        "Provenance and resolved configurations"
    )


def test_writes_bounded_diagnostic_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selected frames are decoded after measurement and capped."""
    import cv2
    import numpy as np

    class Capture:
        """Minimal random-access test capture."""

        def set(self, _key: int, _value: int) -> bool:
            """Accept a frame seek."""
            return True

        def read(self) -> tuple[bool, np.ndarray]:
            """Return one test image."""
            return True, np.zeros((200, 240, 3), dtype=np.uint8)

        def release(self) -> None:
            """Release the fake capture."""

    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: Capture())
    images = []

    def save_image(path: str, image: np.ndarray) -> bool:
        images.append(image.copy())
        Path(path).write_bytes(b"jpeg")
        return True

    monkeypatch.setattr(cv2, "imwrite", save_image)
    records = [
        {
            "clip": "pilot",
            "configuration": "full_frame",
            "frame_index": index,
            "failure": True,
            "truth": {
                "tags": [
                    {"corners_px": [[150, 100], [190, 100], [190, 140], [150, 140]]}
                ]
            },
            "output": {
                "detections": [
                    {
                        "tag_id": 9,
                        "corners": [[40, 100], [80, 100], [80, 140], [40, 140]],
                    }
                ],
                "search_regions": [[[20, 80], [100, 80], [100, 160], [20, 160]]],
            },
        }
        for index in range(5)
    ]
    written = write_diagnostic_images(
        tmp_path, records, {"pilot": tmp_path / "pilot.mkv"}, limit=2
    )
    assert len(written) == 2
    assert all((tmp_path / path).is_file() for path in written)
    assert all(tuple(image[100, 40]) == (0, 255, 0) for image in images)
    assert all(tuple(image[80, 20]) == (0, 0, 255) for image in images)
    assert all(tuple(image[100, 150]) == (0, 0, 0) for image in images)
