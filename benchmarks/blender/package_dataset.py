"""Package a benchmark manifest's assets into video and metadata ZIP files."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from benchmarks.dataset import DatasetManifest, cache_path, verify_dataset


def package_dataset(
    manifest_path: Path, cache_dir: Path, output_dir: Path
) -> tuple[Path, Path]:
    """Write separate video and metadata archives for a cached dataset."""
    manifest = DatasetManifest.model_validate_json(manifest_path.read_bytes())
    verify_dataset(manifest, cache_dir)
    video_paths = {clip.video for clip in manifest.clips}
    stem = manifest_path.stem.removesuffix("_manifest").removesuffix("-manifest")
    stem = stem.removesuffix("-videos").removesuffix("_videos")
    videos_zip = output_dir / f"{stem}-videos.zip"
    metadata_zip = output_dir / f"{stem}-metadata.zip"
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = [(videos_zip, True), (metadata_zip, False)]

    try:
        for archive_path, include_videos in pending:
            with zipfile.ZipFile(
                archive_path.with_suffix(".zip.tmp"), "w", zipfile.ZIP_STORED
            ) as archive:
                if not include_videos:
                    archive.writestr(
                        "manifest.json",
                        json.dumps(manifest.model_dump(), indent=2) + "\n",
                    )
                for asset in manifest.assets:
                    if (asset.path in video_paths) == include_videos:
                        archive.write(cache_path(cache_dir, asset), asset.path)
        for archive_path, _ in pending:
            archive_path.with_suffix(".zip.tmp").replace(archive_path)
    finally:
        for archive_path, _ in pending:
            archive_path.with_suffix(".zip.tmp").unlink(missing_ok=True)
    return videos_zip, metadata_zip


def main() -> None:
    """Package command-line inputs into the two publication archives."""
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    videos, metadata = package_dataset(args.manifest, args.cache_dir, args.output_dir)
    print(json.dumps({"videos": str(videos), "metadata": str(metadata)}))


if __name__ == "__main__":
    main()
