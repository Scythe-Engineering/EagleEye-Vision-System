"""Package ordered PNG frames as pixel-exact FFV1/bgr0 Matroska."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


def digest_bytes(data: bytes) -> str:
    """Return a SHA-256 digest for bytes."""
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    """Return a SHA-256 digest for a file."""
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ordered_frames(directory: Path) -> list[Path]:
    """Return contiguous canonical frame paths in numeric order."""
    frames = sorted(directory.glob("frame_*.png"))
    expected = [f"frame_{i:06d}.png" for i in range(len(frames))]
    if [p.name for p in frames] != expected or not frames:
        raise ValueError("frames must be nonempty and contiguous frame_000000.png ...")
    return frames


def package(source: Path, output: Path, fps: int, ffmpeg: str) -> dict[str, Any]:
    """Apply declared display-domain effects, encode, and verify decoded hashes."""
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    frames = ordered_frames(source)
    provenance_path = source.parent / "provenance.json"
    provenance = (
        json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    )
    effect_settings = provenance.get("effects", {})
    variant = effect_settings.get("requested", "clean")
    severity = effect_settings.get("severity", "medium")
    severity_scale = {"low": 0.5, "medium": 1.0, "high": 2.0}[severity]
    seed = int(effect_settings.get("seed", 2026))

    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"cannot decode {frames[0]}")
    source_height, source_width = first.shape[:2]
    camera = provenance.get("camera", {})
    width, height = camera.get("resolution", [source_width, source_height])
    margin_x, margin_y = camera.get("overscan_margin_px", [0, 0])
    if (source_width, source_height) != (width + 2 * margin_x, height + 2 * margin_y):
        raise ValueError(
            "render dimensions disagree with declared output resolution/overscan"
        )
    lens_map = None
    if variant == "combined-realistic":
        fx = (width / 2) / math.tan(math.radians(40))
        matrix = np.array(
            [[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1]], np.float64
        )
        coeff = np.array(
            [
                -0.06 * severity_scale,
                0.012 * severity_scale,
                0.0005 * severity_scale,
                -0.0005 * severity_scale,
                0.0,
            ],
            np.float64,
        )
        yy, xx = np.indices((height, width), dtype=np.float32)
        distorted = np.stack((xx, yy), axis=-1).reshape(-1, 1, 2)
        lens_map = np.asarray(
            cv2.undistortPoints(distorted, matrix, coeff, P=matrix), dtype=np.float32
        ).reshape(height, width, 2)
        lens_map += np.array([margin_x, margin_y], dtype=np.float32)
        if (margin_x or margin_y) and (
            not np.isfinite(lens_map).all()
            or lens_map[..., 0].min() < 0
            or lens_map[..., 1].min() < 0
            or lens_map[..., 0].max() > source_width - 1
            or lens_map[..., 1].max() > source_height - 1
        ):
            raise ValueError(
                "lens warp exceeds rendered overscan; increase the generator margin"
            )

    def apply_effects(image: Any, frame_index: int) -> Any:
        """Apply pixel postprocessing after Blender's display encoding.

        This intentionally inexpensive display-domain approximation is not called a
        radiometric sensor model. Lens remapping precedes seeded read/shot noise.
        """
        if lens_map is not None:
            image = cv2.remap(
                image,
                lens_map[..., 0],
                lens_map[..., 1],
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
        else:
            image = image[margin_y : margin_y + height, margin_x : margin_x + width]
        if variant == "combined-realistic":
            rng = np.random.default_rng(seed + frame_index)
            signal = image.astype(np.float32)
            # Explicitly a display-encoded 8-bit approximation: signal-dependent plus read noise.
            sigma = severity_scale * (1.0 + np.sqrt(signal / 255.0) * 2.0)
            image = np.clip(
                np.rint(signal + rng.normal(0.0, sigma, signal.shape)), 0, 255
            ).astype(np.uint8)
        return image

    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    hashes: list[str] = []
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-pix_fmt",
        "bgr0",
        str(output),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        assert process.stdin is not None
        for path in frames:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or image.shape != first.shape:
                process.kill()
                raise RuntimeError(f"invalid or mismatched frame {path}")
            image = apply_effects(image, len(hashes))
            raw = image.tobytes()
            hashes.append(digest_bytes(raw))
            process.stdin.write(raw)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("FFmpeg encoding failed")
    capture = cv2.VideoCapture(str(output))
    decoded: list[str] = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        decoded.append(digest_bytes(image.tobytes()))
    capture.release()
    if decoded != hashes:
        raise RuntimeError(
            f"pixel round-trip mismatch: source={len(hashes)} decoded={len(decoded)}"
        )
    ffmpeg_version = subprocess.check_output(
        [ffmpeg, "-version"], text=True
    ).splitlines()[0]
    return {
        "schema_version": 1,
        "video": output.name,
        "video_sha256": file_digest(output),
        "byte_size": output.stat().st_size,
        "frame_count": len(frames),
        "resolution": [width, height],
        "frame_rate": f"{fps}/1",
        "pixel_format": "bgr0",
        "codec": "FFV1 level 3",
        "timestamp_convention": "exposure midpoint; frame_index/fps",
        "canonical_bgr_sha256": hashes,
        "decoded_bgr_sha256": decoded,
        "pixel_round_trip": True,
        "postprocessing": {
            "variant": variant,
            "severity": severity,
            "seed": seed,
            "overscan_margin_px": [margin_x, margin_y],
            "order": "Blender display encoding, OpenCV lens remap, seeded display-domain noise",
            "source_intermediates_retained": True,
        },
        "ffmpeg": ffmpeg_version,
        "opencv": cv2.__version__,
        "platform": platform.platform(),
        "encoding_command": command[:-1] + [output.name],
    }


def parse_args() -> argparse.Namespace:
    """Parse package and explicit-cleanup command line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument(
        "--cleanup-frames",
        action="store_true",
        help="delete source frames only after successful verification",
    )
    return parser.parse_args()


def main() -> None:
    """Package, write a manifest, and optionally perform explicit cleanup."""
    args = parse_args()
    if args.fps < 1:
        raise ValueError("fps must be positive")
    manifest = package(args.frames, args.output, args.fps, args.ffmpeg)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.cleanup_frames:
        shutil.rmtree(args.frames)
    print(
        json.dumps(
            {
                "video": str(args.output),
                "manifest": str(manifest_path),
                "frames": manifest["frame_count"],
                "pixel_round_trip": True,
            }
        )
    )


if __name__ == "__main__":
    main()
