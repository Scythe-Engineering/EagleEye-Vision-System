"""Explicit local Blender/FFmpeg smoke regression in clean subprocesses."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.blender.generate import verify_fmap_landmarks


def test_renderer_checks_separated_welded_map_landmarks() -> None:
    root = Path(__file__).resolve().parents[1]
    fmap = json.loads(
        (
            root
            / "src/webui/assets/fields/2026/apriltag_maps/FE-2026-_REBUILTTM_Playing_Field.fmap"
        ).read_text()
    )
    result = verify_fmap_landmarks(fmap)
    assert set(result["positions_centered_m"]) == {"1", "29"}
    assert result["separation_m"] > 10


@pytest.mark.skipif(
    os.environ.get("EAGLEEYE_RUN_RENDERED_TESTS") != "1",
    reason="set EAGLEEYE_RUN_RENDERED_TESTS=1 to run real Blender smoke render",
)
def test_blender_smoke_round_trips_pixels(tmp_path: Path) -> None:
    """Render three frames, round-trip FFV1, and run both real pipelines."""
    root = Path(__file__).resolve().parents[1]
    blender = os.environ.get("EAGLEEYE_BLENDER", "/opt/homebrew/bin/blender")
    ffmpeg = os.environ.get("EAGLEEYE_FFMPEG", "ffmpeg")
    output = tmp_path / "render"
    clean_env = {
        "HOME": os.environ.get("HOME", str(tmp_path)),
        "PATH": os.environ.get("PATH", ""),
        "TMPDIR": str(tmp_path),
        "PYTHONNOUSERSITE": "1",
        "LC_ALL": "C",
    }
    subprocess.run(
        [
            blender,
            "--background",
            "--python-exit-code",
            "1",
            "--python",
            str(root / "benchmarks/blender/generate.py"),
            "--",
            "--output",
            str(output),
            "--recipe",
            str(root / "benchmarks/blender/smoke-recipe.json"),
            "--start-x",
            "15",
            "--start-y",
            "4.0345",
            "--yaw-degrees",
            "180",
            "--speed",
            "0",
        ],
        cwd=root,
        env=clean_env,
        check=True,
        timeout=300,
    )
    video = output / "clean.mkv"
    subprocess.run(
        [
            sys.executable,
            str(root / "benchmarks/blender/package.py"),
            "--frames",
            str(output / "frames"),
            "--output",
            str(video),
            "--fps",
            "120",
            "--ffmpeg",
            ffmpeg,
        ],
        cwd=root,
        env=clean_env,
        check=True,
        timeout=120,
    )
    manifest = json.loads((output / "clean.mkv.manifest.json").read_text())
    truth = [
        json.loads(line) for line in (output / "truth.jsonl").read_text().splitlines()
    ]
    assert manifest["pixel_round_trip"] is True
    assert manifest["resolution"] == [1280, 800]
    assert manifest["postprocessing"]["overscan_margin_px"] == [0, 0]
    assert manifest["frame_count"] == len(truth) == 3
    assert [row["frame_index"] for row in truth] == [0, 1, 2]
    assert [row["timestamp_ns"] for row in truth] == [0, 8_333_333, 16_666_667]
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["camera"]["opencv_camera_to_robot_nwu"][:3] == [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
    ]
    assert provenance["trajectory_validation"]["passed"] is True
    assert provenance["field_mesh"]["enabled"] is True
    assert provenance["field_mesh"]["object_count"] >= 30
    assert provenance["field_mesh"]["root_count"] == 1
    assert provenance["mesh_alignment"]["applied_to"] == "import roots only"
    assert provenance["mesh_alignment"]["perimeter_validation"]["errors_m"][0] < 0.20
    assert (
        provenance["mesh_alignment"]["fmap_reference_validation"]["separation_m"] > 10
    )
    assert provenance["mesh_alignment"]["tag_surface_validation"]["count"] == 32
    assert provenance["mesh_alignment"]["tag_surface_validation"]["max_error_m"] < 0.002
    tags = {tag["id"]: tag for tag in truth[0]["tags"]}
    assert tags[9]["front_facing"] and tags[10]["front_facing"]
    settings = json.loads((output / "settings.json").read_text())
    assert settings["resolved"]["field_mesh"]["enabled"] is True

    first_truth = truth[0]
    flat_intrinsics = first_truth["calibration"]["camera_matrix"]
    calibration = output / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "camera_matrix": [
                    flat_intrinsics[0:3],
                    flat_intrinsics[3:6],
                    flat_intrinsics[6:9],
                ],
                "distortion_coefficients": first_truth["calibration"][
                    "distortion_coefficients"
                ],
            }
        )
    )
    events = output / "events.json"
    events.write_text("{}\n")

    files = {
        "smoke/clean.mkv": video,
        "smoke/calibration.json": calibration,
        "smoke/truth.jsonl": output / "truth.jsonl",
        "smoke/events.json": events,
    }
    assets = []
    cache = tmp_path / "cache"
    for relative, source in files.items():
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assets.append(
            {
                "path": relative,
                "size": source.stat().st_size,
                "sha256": digest,
                "roles": ["smoke"],
            }
        )
        cached = cache / digest[:2] / digest / relative
        cached.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, cached)

    fmap = (
        root
        / "src/webui/assets/fields/2026/apriltag_maps/FE-2026-_REBUILTTM_Playing_Field.fmap"
    )
    glb = (
        root
        / "src/webui/assets/fields/2026/field_files/FE-2026-_REBUILTTM_Playing_Field.glb"
    )
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": "rendered-smoke",
                "release": "local-test",
                "description": "Ephemeral rendered regression",
                "generation": {
                    "git_revision": "test",
                    "blender_version": provenance["blender"],
                    "render_engine": provenance["render_engine"],
                    "render_settings": {"samples": 1},
                    "random_seeds": {"render": 0},
                    "color_pipeline": "AgX",
                    "ffmpeg_version": manifest["ffmpeg"],
                    "encoding_settings": "FFV1 level 3 bgr0",
                },
                "sources": [
                    {
                        "name": "2026 fmap",
                        "sha256": hashlib.sha256(fmap.read_bytes()).hexdigest(),
                    },
                    {
                        "name": "2026 field mesh",
                        "sha256": hashlib.sha256(glb.read_bytes()).hexdigest(),
                    },
                ],
                "calibration_profiles": {
                    "synthetic": {
                        "camera_matrix": [
                            flat_intrinsics[0:3],
                            flat_intrinsics[3:6],
                            flat_intrinsics[6:9],
                        ],
                        "distortion": first_truth["calibration"][
                            "distortion_coefficients"
                        ],
                    }
                },
                "mounting_transforms": {
                    "synthetic": {
                        "pitch": 0.0,
                        "yaw": 0.0,
                        "roll": 0.0,
                        "x_offset": 0.25,
                        "y_offset": 0.0,
                        "z_offset": 0.5,
                    }
                },
                "assets": assets,
                "clips": [
                    {
                        "id": "clean-smoke",
                        "scenario_id": "three-frame-smoke",
                        "trajectory_id": "smoke-trajectory",
                        "camera_id": "synthetic",
                        "variant_id": "clean",
                        "severity": {},
                        "roles": ["smoke"],
                        "width": 1280,
                        "height": 800,
                        "pixel_format": "bgr0",
                        "frame_rate_num": 120,
                        "frame_rate_den": 1,
                        "frame_count": 3,
                        "timestamp_convention": "exposure_midpoint_ns",
                        "video": "smoke/clean.mkv",
                        "calibration": "smoke/calibration.json",
                        "ground_truth": "smoke/truth.jsonl",
                        "events": "smoke/events.json",
                    }
                ],
                "detection_policy_version": "pilot-provisional-v1",
            }
        )
    )
    results = tmp_path / "results"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks",
            "run",
            "--dataset",
            str(dataset),
            "--subset",
            "smoke",
            "--cache-dir",
            str(cache),
            "--pipeline",
            "both",
            "--output",
            str(results),
        ],
        cwd=root,
        env=clean_env,
        check=True,
        timeout=180,
    )
    summary = json.loads((results / "summary.json").read_text())
    assert summary["failed"] == 0
    assert summary["completed"] == 6
    with gzip.open(results / "frames.jsonl.gz", "rt") as stream:
        scored = [json.loads(line) for line in stream]
    translation_errors = [
        row["metrics"]["robot_pose"]["translation_3d_m"] for row in scored
    ]
    rotation_errors = [
        math.degrees(row["metrics"]["robot_pose"]["rotation_rad"]) for row in scored
    ]
    assert len(scored) == 6
    assert max(translation_errors) < 0.03
    # The added distortion case validates rendered/warped pixel geometry, not
    # a new acceptance gate for shallow two-tag PnP after interpolation.
    assert all(
        len(row["metrics"]["detection"]["corner_errors_px"]) >= 2 for row in scored
    )
    assert (
        max(
            error
            for row in scored
            for error in row["metrics"]["detection"]["corner_errors_px"]
        )
        < 1.0
    )
    assert max(rotation_errors) < 1.0
