"""Focused tests for the local benchmark dataset contract."""

import hashlib
import zipfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmarks.blender.package_dataset import package_dataset
from benchmarks.dataset import (
    Asset,
    DatasetManifest,
    cache_path,
    download_metadata,
    download_missing_assets,
    metadata_url_for_archive,
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


def test_download_metadata_keeps_archive_and_caches_assets(tmp_path: Path) -> None:
    """The metadata ZIP supplies and retains the manifest and non-video assets."""
    metadata = b"metadata"
    manifest = _manifest(metadata)
    manifest.clips[0].video = "video.bin"
    manifest.assets.append(
        Asset(
            path="video.bin",
            size=5,
            sha256=hashlib.sha256(b"video").hexdigest(),
            roles=["pilot"],
        )
    )
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    videos = source_dir / "benchmark-videos.zip"
    metadata_zip = source_dir / "benchmark-metadata.zip"
    videos.touch()
    with zipfile.ZipFile(metadata_zip, "w") as stream:
        stream.writestr("manifest.json", manifest.model_dump_json())
        stream.writestr("tiny.bin", metadata)
    downloads = tmp_path / "downloads"
    cache = tmp_path / "cache"

    assert metadata_url_for_archive(videos.as_uri()) == metadata_zip.as_uri()
    downloaded = download_metadata(videos.as_uri(), cache, downloads)
    metadata_zip.unlink()

    assert download_metadata(videos.as_uri(), cache, downloads) == downloaded
    assert (downloads / "benchmark-metadata.zip").is_file()
    assert DatasetManifest.model_validate_json(downloaded.read_bytes()).dataset_id == "local"


def test_download_missing_assets_keeps_both_archives(tmp_path: Path) -> None:
    """The downloader installs assets and retains both downloaded ZIPs."""
    metadata = b"metadata"
    video = b"video"
    manifest = _manifest(metadata)
    manifest.clips[0].video = "video.bin"
    manifest.assets.append(
        Asset(
            path="video.bin",
            size=len(video),
            sha256=hashlib.sha256(video).hexdigest(),
            roles=["pilot"],
        )
    )
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    videos_zip = source_dir / "benchmark-videos.zip"
    metadata_zip = source_dir / "benchmark-metadata.zip"
    with zipfile.ZipFile(videos_zip, "w") as stream:
        stream.writestr("video.bin", video)
    with zipfile.ZipFile(metadata_zip, "w") as stream:
        stream.writestr("tiny.bin", metadata)
    cache = tmp_path / "cache"
    downloads = tmp_path / "downloads"

    installed = download_missing_assets(
        manifest, cache, videos_zip.as_uri(), "pilot", downloads
    )

    assert len(installed) == 2
    assert all(path.is_file() for path in installed)
    assert (downloads / videos_zip.name).is_file()
    assert (downloads / metadata_zip.name).is_file()


def test_package_dataset_splits_video_and_metadata(tmp_path: Path) -> None:
    """The publication packager writes the two runtime archive layouts."""
    metadata = b"metadata"
    video = b"video"
    manifest = _manifest(metadata)
    manifest.clips[0].video = "video.bin"
    manifest.assets.append(
        Asset(
            path="video.bin",
            size=len(video),
            sha256=hashlib.sha256(video).hexdigest(),
            roles=["pilot"],
        )
    )
    manifest_path = tmp_path / "benchmark-videos_manifest.json"
    manifest_path.write_text(manifest.model_dump_json())
    cache = tmp_path / "cache"
    for asset, data in zip(manifest.assets, (metadata, video), strict=True):
        target = cache_path(cache, asset)
        target.parent.mkdir(parents=True)
        target.write_bytes(data)

    videos_zip, metadata_zip = package_dataset(manifest_path, cache, tmp_path)

    with zipfile.ZipFile(videos_zip) as archive:
        assert archive.namelist() == ["video.bin"]
    with zipfile.ZipFile(metadata_zip) as archive:
        assert set(archive.namelist()) == {"manifest.json", "tiny.bin"}
