"""Focused tests for benchmark result persistence and offline reports."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.report import (
    BOUNDARY,
    RunWriter,
    collect_provenance,
    file_sha256,
    native_provenance,
    write_diagnostic_images,
)


def test_provenance_identifies_uncommitted_solver_sources(tmp_path: Path) -> None:
    """A dirty revision alone must not identify the code used for a comparison."""
    manifest = tmp_path / "manifest.json"
    graph = tmp_path / "graph.json"
    manifest.write_text("{}")
    graph.write_text("[]")
    metadata = collect_provenance(manifest, {"comparison": graph})
    for source in (
        "src/utils/timestamped_samples.py",
        "src/rust_implementations/build.py",
        "src/rust_implementations/modules/pnp_localization_2d/Cargo.toml",
        "src/rust_implementations/modules/pnp_localization_2d/Cargo.lock",
        "src/rust_implementations/modules/pnp_localization_2d/src/lib.rs",
        "src/config/utils/flow_manager.py",
        "src/config/utils/pipeline.py",
        "src/config/utils/thread_object.py",
    ):
        assert metadata["source_sha256"][source] == file_sha256(
            Path(__file__).resolve().parents[1] / source
        )
    assert metadata["graph_sha256"]["comparison"] == file_sha256(graph)


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


def test_native_provenance_records_only_loaded_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hash genuine native inputs and the loaded binary, not nearby build artifacts."""
    import sys
    from importlib.machinery import EXTENSION_SUFFIXES
    from types import SimpleNamespace

    root = tmp_path / "repo"
    native = root / "src/rust_implementations"
    crate = native / "modules/example"
    (crate / "src").mkdir(parents=True)
    (native / "build.py").write_text("# builder")
    (crate / "Cargo.toml").write_text(
        '[package]\nname = "example"\nversion = "1.2.3"\n'
    )
    (crate / "Cargo.lock").write_text("# lock")
    (crate / "src/lib.rs").write_text("// native")
    (crate / "target").mkdir()
    (crate / "target/unused.rs").write_text("// build output")
    binary = tmp_path / ("example" + EXTENSION_SUFFIXES[0])
    binary.write_bytes(b"loaded native binary")
    monkeypatch.setitem(
        sys.modules,
        "example",
        SimpleNamespace(__file__=str(binary), __version__="1.2.3"),
    )
    sources, builds = native_provenance(root)
    assert len(sources) == 4
    assert not any("target" in source for source in sources)
    assert builds["example"]["version"] == "1.2.3"
    assert builds["example"]["loaded_binary"]["sha256"] == file_sha256(binary)
    assert builds["example"]["loaded_binary"]["version"] == "1.2.3"
    monkeypatch.delitem(sys.modules, "example")
    assert native_provenance(root)[1]["example"]["loaded_binary"] is None


def test_finished_report_refreshes_late_loaded_native_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Initial metadata collection precedes lazy production operation imports."""
    import json
    import sys
    from importlib.machinery import EXTENSION_SUFFIXES
    from types import SimpleNamespace

    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    metadata = collect_provenance(manifest, {})
    writer = RunWriter.create(tmp_path / "run", metadata)
    binary = tmp_path / ("pnp_localization_2d" + EXTENSION_SUFFIXES[0])
    binary.write_bytes(b"late loaded extension")
    monkeypatch.setitem(
        sys.modules,
        "pnp_localization_2d",
        SimpleNamespace(__file__=str(tmp_path / "__init__.py")),
    )
    monkeypatch.setitem(
        sys.modules,
        "pnp_localization_2d.pnp_localization_2d",
        SimpleNamespace(__file__=str(binary), __version__="fixture"),
    )
    writer.finish({}, [])
    saved = json.loads((writer.directory / "run.json").read_text())
    assert saved["native_builds"]["pnp_localization_2d"]["loaded_binary"][
        "sha256"
    ] == file_sha256(binary)


@pytest.mark.parametrize(
    "crate_name",
    ["pnp_localization_2d", "temporal_acceleration", "pose_outlier_filter"],
)
def test_native_provenance_discovers_loaded_package_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crate_name: str
) -> None:
    """Maturin wrappers identify their loaded extension, never a nearby disk binary."""
    import sys
    from importlib.machinery import EXTENSION_SUFFIXES
    from types import ModuleType

    from benchmarks import report

    native = tmp_path / "src/rust_implementations"
    crate = native / "modules" / crate_name
    crate.mkdir(parents=True)
    (native / "build.py").write_text("# builder")
    (crate / "Cargo.toml").write_text(
        f'[package]\nname = "{crate_name}"\nversion = "1.2.3"\n'
    )
    package_dir = tmp_path / "site-packages" / crate_name
    package_dir.mkdir(parents=True)
    wrapper_file = package_dir / "__init__.py"
    wrapper_file.write_text("raise AssertionError('provenance must not import code')")
    binary = package_dir / (crate_name + EXTENSION_SUFFIXES[0])
    binary.write_bytes(b"actual loaded extension")
    (package_dir / ("unused" + EXTENSION_SUFFIXES[0])).write_bytes(b"not loaded")
    wrapper = ModuleType(crate_name)
    wrapper.__file__ = str(wrapper_file)
    wrapper.__path__ = [str(package_dir)]
    extension_name = f"{crate_name}.{crate_name}"
    extension = ModuleType(extension_name)
    extension.__file__ = str(binary)
    monkeypatch.setitem(sys.modules, crate_name, wrapper)
    monkeypatch.setitem(sys.modules, extension_name, extension)

    def installed_version(distribution_name: str) -> str:
        """Return the installed version rather than the source manifest's version."""
        assert distribution_name == crate_name
        return "1.2.2"

    monkeypatch.setattr(report.metadata, "version", installed_version)
    build = native_provenance(tmp_path)[1][crate_name]
    assert build["version"] == "1.2.3"
    assert build["loaded_binary"] == {
        "module": extension_name,
        "path": str(binary.resolve()),
        "sha256": file_sha256(binary),
        "version": "1.2.2",
    }
    monkeypatch.delitem(sys.modules, extension_name)
    assert native_provenance(tmp_path)[1][crate_name]["loaded_binary"] is None
