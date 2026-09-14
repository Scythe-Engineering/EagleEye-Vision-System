"""Strict dataset manifests and content-addressed benchmark asset caching."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal
from urllib.error import HTTPError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, model_validator
from tqdm import tqdm

SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
DEFAULT_VIDEO_ARCHIVE_URL = (
    "https://benchmarks.scytheengineering.com/EagleEye-current-benchmark-videos.zip"
)


def manifest_url_for_archive(archive_url: str) -> str:
    """Derive the published manifest URL from a benchmark ZIP URL."""
    parsed = urlsplit(archive_url)
    if not parsed.path.lower().endswith(".zip"):
        raise ValueError("benchmark archive URL must end with .zip")
    return urlunsplit(
        parsed._replace(path=f"{parsed.path[:-4]}_manifest.json", fragment="")
    )


class StrictModel(BaseModel):
    """Base model that rejects coercion and unknown manifest fields."""

    model_config = ConfigDict(extra="forbid", strict=True)


class Transform(StrictModel):
    """A finite homogeneous row-major transform."""

    matrix: list[list[float]]

    @model_validator(mode="after")
    def validate_matrix(self) -> Transform:
        """Validate shape, finite values, and homogeneous final row."""
        import math

        if len(self.matrix) != 4 or any(len(row) != 4 for row in self.matrix):
            raise ValueError("transform must be 4x4")
        if not all(math.isfinite(value) for row in self.matrix for value in row):
            raise ValueError("transform contains nonfinite values")
        if self.matrix[3] != [0.0, 0.0, 0.0, 1.0]:
            raise ValueError("transform final row must be [0, 0, 0, 1]")
        import numpy as np

        rotation = np.asarray(self.matrix, dtype=float)[:3, :3]
        if not np.allclose(
            rotation.T @ rotation, np.eye(3), atol=1e-6
        ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
            raise ValueError("transform rotation must be a proper orthonormal matrix")
        return self


class Mounting(StrictModel):
    """Production camera mounting parameters in degrees and meters."""

    pitch: float
    yaw: float
    roll: float
    x_offset: float
    y_offset: float
    z_offset: float

    @model_validator(mode="after")
    def validate_values(self) -> Mounting:
        """Reject nonfinite mounting values."""
        import math

        if not all(math.isfinite(value) for value in self.model_dump().values()):
            raise ValueError("mounting transform contains nonfinite values")
        return self


class Calibration(StrictModel):
    """Pinned OpenCV calibration values."""

    camera_matrix: list[list[float]]
    distortion: list[float]

    @model_validator(mode="after")
    def validate_calibration(self) -> Calibration:
        """Reject malformed, singular, or nonfinite calibration data."""
        import math

        matrix = self.camera_matrix
        if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
            raise ValueError("camera_matrix must be 3x3")
        values = [v for row in matrix for v in row] + self.distortion
        if not all(math.isfinite(v) for v in values):
            raise ValueError("calibration contains nonfinite values")
        if matrix[0][0] <= 0 or matrix[1][1] <= 0 or matrix[2] != [0.0, 0.0, 1.0]:
            raise ValueError("invalid camera intrinsic matrix")
        return self


class Asset(StrictModel):
    """A single immutable local file."""

    path: str
    size: int = Field(ge=0)
    sha256: SHA256
    roles: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_asset(self) -> Asset:
        """Require a safe relative POSIX path."""
        path = PurePosixPath(self.path)
        if (
            path.is_absolute()
            or not self.path
            or ".." in path.parts
            or "\\" in self.path
        ):
            raise ValueError("asset path must be a safe relative POSIX path")
        return self


class Clip(StrictModel):
    """Media and truth references for one benchmark clip."""

    id: str
    scenario_id: str
    trajectory_id: str
    camera_id: str
    variant_id: str
    severity: dict[str, float]
    roles: list[str] = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    pixel_format: str
    frame_rate_num: int = Field(gt=0)
    frame_rate_den: int = Field(gt=0)
    frame_count: int = Field(gt=0)
    timestamp_convention: Literal["exposure_midpoint_ns"]
    video: str
    calibration: str
    ground_truth: str
    events: str


class Generation(StrictModel):
    """Pinned renderer and encoder provenance."""

    git_revision: str
    blender_version: str
    render_engine: str
    render_settings: dict[str, object]
    random_seeds: dict[str, int]
    color_pipeline: str
    ffmpeg_version: str
    encoding_settings: str


class SourceIdentity(StrictModel):
    """Identity of a source used to produce the dataset."""

    name: str
    sha256: SHA256


class DatasetManifest(StrictModel):
    """Top-level benchmark dataset contract."""

    schema_version: Literal[1]
    dataset_id: str
    release: str
    description: str
    generation: Generation
    sources: list[SourceIdentity]
    calibration_profiles: dict[str, Calibration]
    mounting_transforms: dict[str, Mounting]
    assets: list[Asset]
    clips: list[Clip]
    detection_policy_version: str

    @model_validator(mode="after")
    def validate_references(self) -> DatasetManifest:
        """Validate uniqueness and clip asset references."""
        paths = [asset.path for asset in self.assets]
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate asset path")
        ids = [clip.id for clip in self.clips]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate clip ID")
        by_path = {asset.path: asset for asset in self.assets}
        for clip in self.clips:
            for reference in (
                clip.video,
                clip.calibration,
                clip.ground_truth,
                clip.events,
            ):
                if reference not in by_path:
                    raise ValueError(
                        f"clip {clip.id} references unknown asset {reference}"
                    )
            if clip.camera_id not in self.calibration_profiles:
                raise ValueError(
                    f"clip {clip.id} references unknown calibration profile {clip.camera_id}"
                )
            if clip.camera_id not in self.mounting_transforms:
                raise ValueError(
                    f"clip {clip.id} references unknown mounting transform {clip.camera_id}"
                )
        return self

    def selected_assets(self, role: str | None = None) -> list[Asset]:
        """Return exactly the assets required by clips selected by role."""
        clips = (
            self.clips
            if role is None
            else [clip for clip in self.clips if role in clip.roles]
        )
        paths = {
            value
            for clip in clips
            for value in (clip.video, clip.calibration, clip.ground_truth, clip.events)
        }
        return [asset for asset in self.assets if asset.path in paths]


def cached_manifest_path(cache_dir: str | Path, manifest_url: str) -> Path:
    """Return the stable cache path for a remote manifest URL."""
    url_hash = hashlib.sha256(manifest_url.encode("utf-8")).hexdigest()
    filename = Path(urlsplit(manifest_url).path).name or "manifest.json"
    return Path(cache_dir) / "manifests" / url_hash / filename


def download_manifest(archive_url: str, cache_dir: str | Path) -> Path:
    """Download and validate an archive's manifest unless it is already cached."""
    manifest_url = manifest_url_for_archive(archive_url)
    target = cached_manifest_path(cache_dir, manifest_url)
    if target.is_file():
        return target

    request = Request(manifest_url, headers={"User-Agent": "EagleEye-Benchmark/1.0"})
    with urlopen(request) as response:
        data = response.read()
    DatasetManifest.model_validate_json(data)

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=target.parent, delete=False) as output:
        temporary = Path(output.name)
        output.write(data)
    os.replace(temporary, target)
    return target


def load_manifest(
    path: str | Path, pinned_sha256: str | None = None
) -> DatasetManifest:
    """Load a local manifest, optionally requiring its exact SHA-256 digest."""
    data = Path(path).read_bytes()
    if pinned_sha256 and hashlib.sha256(data).hexdigest() != pinned_sha256:
        raise ValueError("manifest checksum mismatch")
    return DatasetManifest.model_validate_json(data)


def cache_path(cache_dir: str | Path, asset: Asset) -> Path:
    """Return the content-addressed cache path for an asset."""
    return Path(cache_dir) / asset.sha256[:2] / asset.sha256 / asset.path


def _check_file(path: Path, asset: Asset) -> str | None:
    """Return a cache validation failure reason, or None when valid."""
    if not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    if size != asset.size:
        return f"size mismatch ({size} != {asset.size})"
    if digest.hexdigest() != asset.sha256:
        return "checksum mismatch"
    return None


def verify_dataset(
    manifest: DatasetManifest, cache_dir: str | Path, role: str | None = None
) -> None:
    """Rehash selected local assets and raise with detailed failures."""
    failures = [
        (asset.path, _check_file(cache_path(cache_dir, asset), asset))
        for asset in manifest.selected_assets(role)
    ]
    failures = [(path, reason) for path, reason in failures if reason]
    if failures:
        details = ", ".join(f"{path}: {reason}" for path, reason in failures)
        raise FileNotFoundError(f"dataset assets unavailable ({details})")


def _video_assets(manifest: DatasetManifest, role: str | None) -> list[Asset]:
    """Return the unique video assets required by the selected clips."""
    clips = (
        manifest.clips
        if role is None
        else [clip for clip in manifest.clips if role in clip.roles]
    )
    video_paths = {clip.video for clip in clips}
    return [asset for asset in manifest.assets if asset.path in video_paths]


def _download_archive(url: str, directory: Path, attempts: int = 3) -> Path:
    """Download an archive, resuming interrupted transfers up to three times."""
    directory.mkdir(parents=True, exist_ok=True)
    archive_path = directory / "video-download.partial"
    offset = archive_path.stat().st_size if archive_path.exists() else 0
    headers = {"User-Agent": "EagleEye-Benchmark/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(url, headers=headers)
    try:
        with urlopen(request) as response:
            resuming = offset > 0 and response.status == 206
            if not resuming:
                offset = 0
            length = response.headers.get("Content-Length")
            remaining = int(length) if length and length.isdigit() else None
            total = offset + remaining if remaining is not None else None
            with (
                archive_path.open("ab" if resuming else "wb") as handle,
                tqdm(
                    total=total,
                    initial=offset,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc="Downloading benchmark videos",
                ) as progress,
            ):
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
                    progress.update(len(chunk))
        return archive_path
    except HTTPError as error:
        if error.code == 416 and offset and attempts > 1:
            archive_path.unlink(missing_ok=True)
            return _download_archive(url, directory, attempts - 1)
        raise
    except OSError:
        if attempts <= 1:
            raise
        return _download_archive(url, directory, attempts - 1)


def download_missing_videos(
    manifest: DatasetManifest,
    cache_dir: str | Path,
    archive_url: str = DEFAULT_VIDEO_ARCHIVE_URL,
    role: str | None = None,
) -> list[Path]:
    """Download, verify, and cache missing selected videos from a ZIP archive."""
    cache = Path(cache_dir)
    missing = [
        asset
        for asset in _video_assets(manifest, role)
        if _check_file(cache_path(cache, asset), asset) is not None
    ]
    if not missing:
        return []

    archive_path = _download_archive(archive_url, cache)
    try:
        if not zipfile.is_zipfile(archive_path):
            archive_path.unlink(missing_ok=True)
            raise ValueError("downloaded video archive is not a ZIP file")
        with zipfile.ZipFile(archive_path) as archive:
            members = {member.filename: member for member in archive.infolist()}
            unavailable = [
                asset.path
                for asset in missing
                if asset.path not in members
                or members[asset.path].is_dir()
                or members[asset.path].file_size != asset.size
            ]
            if unavailable:
                raise ValueError(
                    "video archive does not contain the expected assets: "
                    + ", ".join(unavailable)
                )

            installed: list[Path] = []
            with tqdm(
                total=len(missing), unit="video", desc="Extracting benchmark videos"
            ) as progress:
                for asset in missing:
                    target = cache_path(cache, asset)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(
                        mode="wb", dir=target.parent, delete=False
                    ) as output:
                        temporary = Path(output.name)
                        try:
                            with archive.open(members[asset.path]) as source:
                                shutil.copyfileobj(source, output, 1024 * 1024)
                        except BaseException:
                            temporary.unlink(missing_ok=True)
                            raise
                    reason = _check_file(temporary, asset)
                    if reason is not None:
                        temporary.unlink(missing_ok=True)
                        raise ValueError(f"downloaded {asset.path}: {reason}")
                    os.replace(temporary, target)
                    installed.append(target)
                    progress.update(1)
            return installed
    finally:
        archive_path.unlink(missing_ok=True)
