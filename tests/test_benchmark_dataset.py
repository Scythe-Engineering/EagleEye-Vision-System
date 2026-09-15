"""Focused tests for the local benchmark dataset contract."""

import hashlib
import zipfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmarks.dataset import (
    Asset,
    DatasetManifest,
    cache_path,
    download_manifest,
    download_missing_videos,
    manifest_url_for_archive,
    verify_dataset,
)


def _manifest(data: bytes) -> DatasetManifest:
    """Build a minimal valid local manifest."""
    digest = hashlib.sha256(data).hexdigest()
    return DatasetManifest.model_validate(
        {
            "schema_version": 1,
            "dataset_id": "local",
            "release": "test",
            "description": "test fixture",
            "generation": {
                "git_revision": "abc",
                "blender_version": "4",
                "render_engine": "EEVEE",
                "render_settings": {},
                "random_seeds": {},
                "color_pipeline": "standard",
                "ffmpeg_version": "7",
                "encoding_settings": "lossless",
            },
            "sources": [],
            "calibration_profiles": {
                "camera": {
                    "camera_matrix": [
                        [1.0, 0.0, 0.5],
                        [0.0, 1.0, 0.5],
                        [0.0, 0.0, 1.0],
                    ],
                    "distortion": [],
                }
            },
            "mounting_transforms": {
                "camera": {
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "roll": 0.0,
                    "x_offset": 0.0,
                    "y_offset": 0.0,
                    "z_offset": 0.0,
                }
            },
            "assets": [
                {
                    "path": "tiny.bin",
                    "size": len(data),
                    "sha256": digest,
                    "roles": ["pilot"],
                }
            ],
            "clips": [
                {
                    "id": "tiny",
                    "scenario_id": "still",
                    "trajectory_id": "one",
                    "camera_id": "camera",
                    "variant_id": "clean",
                    "severity": {},
                    "roles": ["pilot"],
                    "width": 1,
                    "height": 1,
                    "pixel_format": "bgr",
                    "frame_rate_num": 30,
                    "frame_rate_den": 1,
                    "frame_count": 1,
                    "timestamp_convention": "exposure_midpoint_ns",
                    "video": "tiny.bin",
                    "calibration": "tiny.bin",
                    "ground_truth": "tiny.bin",
                    "events": "tiny.bin",
                }
            ],
            "detection_policy_version": "pilot-candidate-v1",
        }
    )


def test_asset_rejects_unsafe_path_and_network_fields() -> None:
    """Assets are local hash-pinned files with no publication URL."""
    with pytest.raises(ValidationError, match="safe relative"):
        Asset.model_validate(
            {"path": "../x", "size": 0, "sha256": "0" * 64, "roles": ["pilot"]}
        )
    with pytest.raises(ValidationError, match="url"):
        Asset.model_validate(
            {
                "path": "x",
                "url": "https://example.invalid/x",
                "size": 0,
                "sha256": "0" * 64,
                "roles": ["pilot"],
            }
        )


def test_verify_dataset_checks_local_size_and_hash(tmp_path: Path) -> None:
    """Verification accepts exact local bytes and identifies corruption."""
    data = b"verified"
    manifest = _manifest(data)
    target = cache_path(tmp_path, manifest.assets[0])
    target.parent.mkdir(parents=True)
    target.write_bytes(data)
    verify_dataset(manifest, tmp_path, "pilot")
    target.write_bytes(b"corrupt!")
    with pytest.raises(FileNotFoundError, match="checksum mismatch"):
        verify_dataset(manifest, tmp_path, "pilot")


def test_download_manifest_derives_url_and_reuses_download(tmp_path: Path) -> None:
    """The remote manifest is downloaded once into the requested directory."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    archive = source_dir / "benchmark-videos.zip"
    source = source_dir / "benchmark-videos_manifest.json"
    source.write_text(_manifest(b"video").model_dump_json())
    downloads = tmp_path / "downloads"

    assert manifest_url_for_archive(archive.as_uri()) == source.as_uri()
    downloaded = download_manifest(archive.as_uri(), downloads)
    source.unlink()

    assert downloaded == downloads / source.name
    assert download_manifest(archive.as_uri(), downloads) == downloaded
    assert DatasetManifest.model_validate_json(downloaded.read_bytes()).dataset_id == "local"


def test_download_missing_videos_keeps_archive_in_download_directory(tmp_path: Path) -> None:
    """The archive and verified video asset are retained in their local homes."""
    data = b"verified video"
    manifest = _manifest(data)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    archive = source_dir / "videos.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("tiny.bin", data)

    cache = tmp_path / "cache"
    downloads = tmp_path / "downloads"
    installed = download_missing_videos(
        manifest, cache, archive.as_uri(), "pilot", downloads
    )

    target = cache_path(cache, manifest.assets[0])
    assert installed == [target]
    assert target.read_bytes() == data
    assert (downloads / archive.name).is_file()
