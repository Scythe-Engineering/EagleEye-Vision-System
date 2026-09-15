#!/usr/bin/env python3
"""Resumably render the combined-realistic CAD match dataset, one route at a time.

The runner deliberately delegates rendering and pixel processing to ``generate.py``
and ``package.py``.  Its only state is per-job ``status.json`` and append-only
``runner.log`` files under the selected output root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VARIANTS = ("combined-realistic",)
FPS = 120
WIDTH = 1280
HEIGHT = 800
DEFAULT_SAMPLES = 256


class RunnerError(ValueError):
    """Raised for a configuration or resumability error safe to show to users."""


@dataclass(frozen=True)
class Job:
    """One route and its currently supported visual variant."""

    route: dict[str, Any]
    variant: str

    @property
    def name(self) -> str:
        """Return the stable machine-assignment identifier."""
        return f"{self.route['name']}:{self.variant}"


def _sha256(path: Path) -> str:
    """Return a file digest without loading potentially large assets at once."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    """Serialize configuration deterministically for recipes and signatures."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_scenes(path: Path) -> list[dict[str, Any]]:
    """Load and validate the small, portable match-scenes route schema."""
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"cannot read scenes file {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise RunnerError("scenes must be an object with schema_version equal to 1")
    routes = document.get("routes")
    if not isinstance(routes, list) or not routes:
        raise RunnerError("scenes.routes must be a nonempty list")
    names: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, route in enumerate(routes):
        if not isinstance(route, dict):
            raise RunnerError(f"routes[{index}] must be an object")
        name = route.get("name")
        duration = route.get("duration_s")
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) is None
        ):
            raise RunnerError(
                f"routes[{index}].name must be a nonempty path-safe identifier"
            )
        if name in names:
            raise RunnerError(f"duplicate route name: {name}")
        if route.get("severity", "medium") not in ("low", "medium", "high"):
            raise RunnerError(f"invalid severity for route {name!r}")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration <= 0
        ):
            raise RunnerError(f"routes[{index}].duration_s must be positive")
        frames = duration * FPS
        if abs(frames - round(frames)) > 1e-9:
            raise RunnerError(
                f"route {name!r} duration_s must produce an integral frame count at {FPS} fps"
            )
        names.add(name)
        validated.append(route)
    return validated


def select_jobs(routes: Sequence[dict[str, Any]], requested: str | None) -> list[Job]:
    """Expand routes or validate an explicit, duplicate-free assignment list."""
    available = {
        f"{route['name']}:{variant}": Job(route, variant)
        for route in routes
        for variant in VARIANTS
    }
    if requested is None:
        return list(available.values())
    identifiers = requested.split(",")
    if not identifiers or any(not identifier for identifier in identifiers):
        raise RunnerError(
            "--jobs must be a comma-separated list of route:variant identifiers"
        )
    if len(set(identifiers)) != len(identifiers):
        raise RunnerError("--jobs contains duplicate job identifiers")
    invalid = [identifier for identifier in identifiers if identifier not in available]
    if invalid:
        raise RunnerError(f"invalid job identifier(s): {', '.join(invalid)}")
    return [available[identifier] for identifier in identifiers]


def parse_frame_indices(value: str | None) -> list[int] | None:
    """Validate sparse zero-based preflight frame indices before Blender starts."""
    if value is None:
        return None
    try:
        indices = [int(part) for part in value.split(",")]
    except ValueError as exc:
        raise RunnerError(
            "--frame-indices must be comma-separated nonnegative integers"
        ) from exc
    if (
        not indices
        or any(index < 0 for index in indices)
        or len(set(indices)) != len(indices)
    ):
        raise RunnerError(
            "--frame-indices must be nonempty, unique, nonnegative indices"
        )
    return indices


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse runner options while keeping benchmark geometry and timing fixed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenes", type=Path, default=Path(__file__).with_name("match-scenes.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--device", choices=("METAL", "OPTIX", "CPU"), default="CPU")
    parser.add_argument("--jobs")
    parser.add_argument("--list", action="store_true", dest="list_jobs")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--frame-indices")
    parser.add_argument("--samples", type=_positive_int, default=DEFAULT_SAMPLES)
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--persistent-data", action="store_true")
    return parser.parse_args(argv)


def _positive_int(value: str) -> int:
    """Reject nonpositive sample counts at argument parsing time."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _signature(
    job: Job, args: argparse.Namespace, generator: Path, packager: Path
) -> str:
    """Hash inputs which must match before a completed job can be skipped."""
    root = generator.parents[2]
    assets = root / "src/webui/assets/fields/2026"
    inputs = [
        *generator.parent.glob("*.py"),
        assets / "field_files/FE-2026-_REBUILTTM_Playing_Field.glb",
        assets / "apriltag_maps/FE-2026-_REBUILTTM_Playing_Field.fmap",
    ]
    blender_version = subprocess.run(
        [args.blender, "--version"], capture_output=True, text=True, check=True
    ).stdout
    payload = {
        "sources": {str(path.relative_to(root)): _sha256(path) for path in inputs},
        "blender_version": blender_version,
        "route": job.route,
        "variant": job.variant,
        "quality": {
            "width": WIDTH,
            "height": HEIGHT,
            "fps": FPS,
            "samples": args.samples,
            "engine": "cycles",
            "denoising": args.denoise,
            "persistent_data": args.persistent_data,
        },
        "device": args.device,
        "generator_sha256": _sha256(generator),
        "packager_sha256": _sha256(packager),
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


def _write_status(path: Path, status: dict[str, Any]) -> None:
    """Atomically replace status while retaining an append-only human log."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _completed(job_dir: Path, signature: str, variant: str) -> bool:
    """Check marker, package essentials, and recorded generated-input digests."""
    status_path = job_dir / "status.json"
    video = job_dir / f"{variant}.mkv"
    manifest = video.with_suffix(video.suffix + ".manifest.json")
    essentials = (
        status_path,
        video,
        manifest,
        job_dir / "truth.jsonl",
        job_dir / "provenance.json",
        job_dir / "settings.json",
    )
    if not all(path.is_file() and path.stat().st_size > 0 for path in essentials):
        return False
    try:
        status = json.loads(status_path.read_text())
        packaged = json.loads(manifest.read_text())
    except json.JSONDecodeError:
        return False
    return (
        status.get("state") == "complete"
        and status.get("input_signature") == signature
        and status.get("generated_digests")
        == {
            name: _sha256(job_dir / name)
            for name in ("provenance.json", "settings.json", "truth.jsonl")
        }
        and packaged.get("video") == video.name
        and packaged.get("decoded_frame_count") == packaged.get("frame_count")
        and packaged.get("codec", "").startswith("H.264")
        and packaged.get("video_sha256") == _sha256(video)
        and packaged.get("byte_size") == video.stat().st_size
    )


def run_job(
    job: Job, args: argparse.Namespace, frame_indices: list[int] | None
) -> None:
    """Run one isolated job, resuming Blender and packaging only full renders."""
    here = Path(__file__).resolve().parent
    generator, packager = here / "generate.py", here / "package.py"
    if not generator.is_file() or not packager.is_file():
        raise RunnerError(
            "generate.py and package.py must exist beside render_match.py"
        )
    signature = _signature(job, args, generator, packager)
    mode_root = (
        "preflight"
        if frame_indices is not None
        else "validate"
        if args.validate_only
        else ""
    )
    job_dir = (
        args.output / mode_root / job.route["name"] / job.variant
        if mode_root
        else args.output / job.route["name"] / job.variant
    )
    if (
        not args.validate_only
        and frame_indices is None
        and _completed(job_dir, signature, job.variant)
    ):
        return
    job_dir.mkdir(parents=True, exist_ok=True)
    recipe = args.output / "recipes" / f"{job.route['name']}.json"
    recipe.parent.mkdir(parents=True, exist_ok=True)
    recipe.write_text(_canonical(job.route) + "\n")
    status_path = job_dir / "status.json"
    status: dict[str, Any] = {
        "schema_version": 1,
        "job": job.name,
        "state": "running",
        "input_signature": signature,
        "started_unix_s": time.time(),
    }
    _write_status(status_path, status)
    command = [
        args.blender,
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        str(generator),
        "--",
        "--output",
        str(job_dir),
        "--trajectory",
        str(recipe),
        "--width",
        str(WIDTH),
        "--height",
        str(HEIGHT),
        "--fps",
        str(FPS),
        "--frames",
        str(round(job.route["duration_s"] * FPS)),
        "--samples",
        str(args.samples),
        "--engine",
        "cycles",
        "--device",
        args.device,
        "--python-executable",
        str(args.python_executable),
        "--variant",
        job.variant,
        "--severity",
        job.route.get("severity", "medium"),
        "--resume",
    ]
    if args.denoise:
        command.append("--denoise")
    if args.persistent_data:
        command.append("--persistent-data")
    if args.validate_only:
        command.append("--validate-only")
    if frame_indices is not None:
        command.extend(("--frame-indices", ",".join(map(str, frame_indices))))
    try:
        with (job_dir / "runner.log").open("a", encoding="utf-8") as log:
            log.write("$ " + " ".join(command) + "\n")
            log.flush()
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            if not args.validate_only and frame_indices is None:
                # An interrupted encoder must not poison a multi-day resume.
                # Generator settings were checked before replacing task-owned media.
                temporary_video = job_dir / f"{job.variant}.partial.mkv"
                temporary_video.unlink(missing_ok=True)
                package_command = [
                    str(args.python_executable),
                    str(packager),
                    "--frames",
                    str(job_dir / "frames"),
                    "--output",
                    str(temporary_video),
                    "--fps",
                    str(FPS),
                ]
                log.write("$ " + " ".join(package_command) + "\n")
                log.flush()
                subprocess.run(
                    package_command, stdout=log, stderr=subprocess.STDOUT, check=True
                )
                temporary_manifest = temporary_video.with_suffix(".mkv.manifest.json")
                packaged = json.loads(temporary_manifest.read_text())
                video = job_dir / f"{job.variant}.mkv"
                packaged["video"] = video.name
                packaged["encoding_command"][-1] = video.name
                temporary_manifest.write_text(json.dumps(packaged, indent=2) + "\n")
                temporary_video.replace(video)
                temporary_manifest.replace(video.with_suffix(".mkv.manifest.json"))
        if args.validate_only or frame_indices is not None:
            status["state"] = "validated" if args.validate_only else "preflighted"
        else:
            status["state"] = "complete"
            status["generated_digests"] = {
                name: _sha256(job_dir / name)
                for name in ("provenance.json", "settings.json", "truth.jsonl")
            }
        status["finished_unix_s"] = time.time()
        _write_status(status_path, status)
    except (OSError, subprocess.CalledProcessError) as exc:
        status.update(
            {"state": "failed", "finished_unix_s": time.time(), "error": str(exc)}
        )
        _write_status(status_path, status)
        raise RunnerError(
            f"job {job.name} failed; see {job_dir / 'runner.log'}"
        ) from exc


def main(argv: Sequence[str] | None = None) -> int:
    """Validate all assignments before starting any expensive Blender process."""
    args = parse_args(argv)
    try:
        routes = load_scenes(args.scenes)
        jobs = select_jobs(routes, args.jobs)
        indices = parse_frame_indices(args.frame_indices)
        if indices is not None:
            for job in jobs:
                frame_count = round(job.route["duration_s"] * FPS)
                if any(index >= frame_count for index in indices):
                    raise RunnerError(
                        f"--frame-indices exceeds route {job.route['name']!r} frame count"
                    )
        if args.list_jobs:
            print("\n".join(job.name for job in jobs))
            return 0
        for job in jobs:
            run_job(job, args, indices)
    except RunnerError as exc:
        print(f"render_match: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
