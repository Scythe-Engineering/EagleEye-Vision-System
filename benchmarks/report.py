"""Persistence and static offline accuracy reporting utilities."""

from __future__ import annotations

import csv
import gzip
import hashlib
import html
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

BOUNDARY = (
    "Synthetic replay excludes physical camera transport, the application's outer "
    "threaded pipeline loop, and preview encoding/display work."
)


def json_value(value: Any) -> Any:
    """Convert common numerical and object values to strict JSON-compatible values."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_value(v) for v in value]
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if hasattr(value, "__dict__"):
        return json_value(vars(value))
    return repr(value)


def atomic_json(path: Path, value: Any) -> None:
    """Atomically replace *path* with deterministic UTF-8 JSON."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def file_sha256(path: str | Path) -> str:
    """Return the hexadecimal SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_provenance(
    dataset_path: str | Path, graphs: Mapping[str, Path]
) -> dict[str, Any]:
    """Collect reproducibility metadata for a run without modifying the repository."""
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
    except (OSError, subprocess.SubprocessError):
        revision, dirty = "unknown", None
    dependencies: dict[str, str | None] = {}
    for name in (
        "numpy",
        "opencv-python",
        "opencv-contrib-python",
        "pillow",
        "psutil",
        "pydantic",
        "pupil-apriltags",
    ):
        try:
            dependencies[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            dependencies[name] = None
    resolved = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in graphs.items()
    }
    return {
        "dataset_manifest_sha256": file_sha256(dataset_path),
        "graphs": resolved,
        "graph_sha256": {name: file_sha256(path) for name, path in graphs.items()},
        "revision": revision,
        "git_dirty": dirty,
        "dependencies": dependencies,
        "python": sys.version,
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu": platform.processor() or "unknown",
        "logical_cpus": os.cpu_count(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "mode": "accuracy",
        "boundary": BOUNDARY,
    }


@dataclass
class RunWriter:
    """Incrementally persist a run while keeping incomplete work unmistakable."""

    directory: Path
    metadata: dict[str, Any]
    _stream: Any = None

    @classmethod
    def create(
        cls,
        directory: str | Path,
        metadata: Mapping[str, Any],
        *,
        overwrite: bool = False,
    ) -> RunWriter:
        """Create a new output directory, rejecting an existing path by default."""
        target = Path(directory)
        if target.exists():
            if not overwrite:
                raise FileExistsError(
                    f"output directory already exists: {target}; use --overwrite"
                )
            shutil.rmtree(target)
        target.mkdir(parents=True)
        (target / "images").mkdir()
        initial = {**json_value(metadata), "status": "incomplete"}
        atomic_json(target / "run.json", initial)
        writer = cls(target, initial)
        # The writer owns this stream until finish_frames() or close().
        writer._stream = gzip.open(  # noqa: SIM115
            target / "frames.jsonl.gz", "wt", encoding="utf-8"
        )
        return writer

    def write_frame(self, record: Mapping[str, Any]) -> None:
        """Append and flush one frame record so interruptions retain diagnostics."""
        self._stream.write(
            json.dumps(json_value(record), sort_keys=True, allow_nan=False) + "\n"
        )
        self._stream.flush()

    def finish_frames(self) -> None:
        """Finalize the gzip frame stream after measured work and before reporting."""
        if self._stream and not self._stream.closed:
            self._stream.close()

    def finish(
        self, summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
    ) -> None:
        """Write final artifacts and atomically transition run status to complete."""
        self.finish_frames()
        atomic_json(self.directory / "summary.json", summary)
        write_summary_csv(self.directory / "summary.csv", rows)
        complete = {**self.metadata, "status": "complete"}
        render_report(self.directory, complete, summary, rows)
        atomic_json(self.directory / "run.json", complete)

    def close(self) -> None:
        """Close the frame stream without marking an unfinished run complete."""
        if self._stream and not self._stream.closed:
            self._stream.close()


def write_summary_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write one flattened CSV row per clip/configuration/variant result."""
    flattened = [
        {
            k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
            for k, v in row.items()
        }
        for row in rows
    ]
    fields = sorted({key for row in flattened for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)


_PLOT_COLORS = (
    "#1769aa",
    "#d95f02",
    "#1b9e77",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#a6761d",
    "#666666",
)


def _number(value: Any) -> float | None:
    """Return a finite float, keeping malformed saved data out of SVG attributes."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _axis_number(value: float) -> str:
    """Format an SVG tick compactly without hiding small timing/error values."""
    magnitude = abs(value)
    if magnitude and (magnitude >= 10_000 or magnitude < 0.01):
        return f"{value:.2g}"
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    return f"{value:.3g}"


def _line_chart(
    title: str,
    series: Mapping[str, Sequence[tuple[float, float]]],
    *,
    x_label: str,
    y_label: str,
    y_limits: tuple[float, float] | None = None,
) -> str:
    """Render labelled, independently-coloured series with shared, explicit axes."""
    usable = [
        (label, list(points)) for label, points in sorted(series.items()) if points
    ]
    if not usable:
        return ""
    all_points = [
        point
        for _label, points in usable
        for point in points
        if all(math.isfinite(v) for v in point)
    ]
    if not all_points:
        return ""
    xs, ys = zip(*all_points)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = y_limits if y_limits is not None else (min(ys), max(ys))
    # A little headroom makes a constant-valued line visible and avoids clipped labels.
    if xmin == xmax:
        xmin -= 0.5
        xmax += 0.5
    if ymin == ymax:
        padding = abs(ymin) * 0.05 or 0.5
        ymin -= padding
        ymax += padding
    width, height = 760, 300
    left, right, top, bottom = 64, 18, 16, 48
    plot_width, plot_height = width - left - right, height - top - bottom

    def position(x: float, y: float) -> str:
        """Map a data point to SVG coordinates within the labelled axes."""
        px = left + (x - xmin) / (xmax - xmin) * plot_width
        py = top + (ymax - y) / (ymax - ymin) * plot_height
        return f"{px:.1f},{py:.1f}"

    grid: list[str] = []
    for tick in range(5):
        ratio = tick / 4
        x = left + ratio * plot_width
        y = top + (1 - ratio) * plot_height
        x_value = xmin + ratio * (xmax - xmin)
        y_value = ymin + ratio * (ymax - ymin)
        grid.append(
            f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}"/>'
        )
        grid.append(
            f'<text class="tick" x="{x:.1f}" y="{height - 27}" text-anchor="middle">{html.escape(_axis_number(x_value))}</text>'
        )
        grid.append(
            f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}"/>'
        )
        grid.append(
            f'<text class="tick" x="{left - 7}" y="{y + 4:.1f}" text-anchor="end">{html.escape(_axis_number(y_value))}</text>'
        )
    lines: list[str] = []
    legend: list[str] = []
    for index, (label, points) in enumerate(usable):
        color = _PLOT_COLORS[index % len(_PLOT_COLORS)]
        segments: list[list[str]] = [[]]
        for x, y in points:
            if not math.isfinite(y):
                segments.append([])
            else:
                segments[-1].append(position(x, y))
        for segment in segments:
            if len(segment) == 1:
                x_position, y_position = segment[0].split(",")
                lines.append(
                    f'<circle cx="{x_position}" cy="{y_position}" r="2.5" fill="{color}"/>'
                )
            elif segment:
                lines.append(
                    f'<polyline class="series" points="{" ".join(segment)}" style="stroke:{color}"/>'
                )
        legend.append(
            f'<li><span class="swatch" style="background:{color}"></span>{html.escape(label)}</li>'
        )
    safe_title, safe_x, safe_y = map(html.escape, (title, x_label, y_label))
    return (
        f'<section class="chart"><h2>{safe_title}</h2>'
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{safe_title}; x axis: {safe_x}; y axis: {safe_y}">'
        f'<g>{"".join(grid)}</g><line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>'
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>{"".join(lines)}'
        f'<text class="axis-label" x="{left + plot_width / 2:.1f}" y="{height - 7}" text-anchor="middle">{safe_x}</text>'
        f'<text class="axis-label" transform="translate(15 {top + plot_height / 2:.1f}) rotate(-90)" text-anchor="middle">{safe_y}</text>'
        f'</svg><ul class="legend">{"".join(legend)}</ul></section>'
    )


def _detection_size_chart(
    bins: Sequence[tuple[str, int, int]],
    unavailable: int,
    excluded: int,
    *,
    stream: str = "",
    provisional: bool = False,
) -> str:
    """Render eligible-tag recall bins and state their actual scoring denominator."""
    total = sum(denominator for _label, _detected, denominator in bins)
    if not total:
        excluded_note = (
            f" {excluded} excluded/provisional observation(s) were deliberately omitted."
            if excluded
            else ""
        )
        return (
            '<section class="chart"><h2>Detection recall by projected tag size</h2>'
            "<p>No eligible truth tags with a finite projected size were recorded, so recall bins are not plotted."
            f"{excluded_note}</p></section>"
        )
    width, height, left, top, bottom = 760, 320, 64, 16, 82
    plot_width, plot_height = width - left - 18, height - top - bottom
    bar_width = plot_width / len(bins) * 0.68
    bars: list[str] = []
    labels: list[str] = []
    grid: list[str] = []
    for tick in range(5):
        ratio = tick / 4
        y = top + (1 - ratio) * plot_height
        grid.append(
            f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}"/>'
        )
        grid.append(
            f'<text class="tick" x="{left - 7}" y="{y + 4:.1f}" text-anchor="end">{ratio:.0%}</text>'
        )
    for index, (label, detected, denominator) in enumerate(bins):
        center = left + (index + 0.5) * plot_width / len(bins)
        bar_height = plot_height * detected / denominator if denominator else 0
        x, y = center - bar_width / 2, top + plot_height - bar_height
        color = _PLOT_COLORS[index % len(_PLOT_COLORS)]
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}"/>'
        )
        bars.append(
            f'<text class="bar-value" x="{center:.1f}" y="{max(top + 11, y - 5):.1f}" text-anchor="middle">{detected}/{denominator}</text>'
        )
        labels.append(
            f'<text class="tick" x="{center:.1f}" y="{height - 45}" text-anchor="middle">{html.escape(label)}</text>'
        )
        labels.append(
            f'<text class="tick" x="{center:.1f}" y="{height - 29}" text-anchor="middle">n={denominator}</text>'
        )
    omitted = (
        f" {unavailable} observation(s) had no valid projected size and are not binned."
        if unavailable
        else ""
    )
    title = (
        "Provisional detection match rate by projected tag size"
        if provisional
        else "Detection recall by projected tag size"
    )
    population = (
        "Provisional truth tags only, occlusion unknown. This is not eligible recall"
        if provisional
        else "Eligible truth tags only"
    )
    return (
        f'<section class="chart"><h2>{title}</h2><p>{html.escape(stream)}</p>'
        f"<p>{population}: matched observations / truth observations in each bin (total binned: {total}).{omitted}</p>"
        f'<svg viewBox="0 0 760 320" role="img" aria-label="{title}; x axis: projected minimum edge length in pixels; y axis: matched truth observations">'
        f'<g>{"".join(grid)}</g><line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>'
        f'{"".join(bars)}{"".join(labels)}<text class="axis-label" transform="translate(15 {top + plot_height / 2:.1f}) rotate(-90)" text-anchor="middle">Matched truth observations (%)</text>'
        '<text class="axis-label" x="406" y="312" text-anchor="middle">Projected minimum edge length (px)</text></svg></section>'
    )


def _table(value: Mapping[str, Any]) -> str:
    """Render a mapping as an escaped two-column HTML table."""
    rows = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td><code>{html.escape(json.dumps(json_value(v), sort_keys=True))}</code></td></tr>"
        for k, v in value.items()
    )
    return f"<table>{rows}</table>"


def _append_bounded(
    values: list[tuple[float, float]], point: tuple[float, float], limit: int = 2000
) -> None:
    """Retain a deterministic multiresolution plot sample with bounded memory."""
    values.append(point)
    if len(values) > limit:
        values[:] = values[::2]


def _series_label(record: Mapping[str, Any]) -> str:
    """Identify a measured stream without silently combining clips or configurations."""
    return " · ".join(
        (
            f"clip={record.get('clip', 'unknown')}",
            f"config={record.get('configuration', 'unknown')}",
        )
    )


def _frame_number(record: Mapping[str, Any], fallback: int) -> float:
    """Use the per-clip frame number, never a concatenated-run timestamp."""
    frame = _number(record.get("frame_index"))
    return frame if frame is not None else float(fallback)


def _truth_xy(truth: Mapping[str, Any]) -> tuple[float, float] | None:
    """Extract the field-frame robot X/Y translation from either saved matrix form."""
    matrix: Any = truth.get("T_field_from_robot")
    if isinstance(matrix, Mapping):
        matrix = matrix.get("matrix")
    try:
        if isinstance(matrix, list) and len(matrix) == 16:
            values = (_number(matrix[3]), _number(matrix[7]))
        elif isinstance(matrix, list) and len(matrix) >= 2:
            values = (_number(matrix[0][3]), _number(matrix[1][3]))
        else:
            return None
    except (IndexError, TypeError):
        return None
    x, y = values
    return (x, y) if x is not None and y is not None else None


def _add_series_point(
    series: dict[str, list[tuple[float, float]]], label: str, point: tuple[float, float]
) -> None:
    """Add a sampled point to one explicitly named series."""
    _append_bounded(series.setdefault(label, []), point)


def render_report(
    directory: str | Path,
    run: Mapping[str, Any],
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    title: str = "Synthetic video benchmark",
) -> Path:
    """Render an offline report with axes, legends, and non-merged measurement series."""
    trajectories: dict[str, list[tuple[float, float]]] = {}
    pose_errors: dict[str, list[tuple[float, float]]] = {}
    availability: dict[str, list[tuple[float, float]]] = {}
    availability_counts: dict[str, list[int]] = {}
    size_edges = (16.0, 32.0, 64.0, 128.0)
    size_counts: dict[tuple[str, bool], list[list[int]]] = {}
    size_unavailable: dict[tuple[str, bool], int] = {}
    excluded_size_observations = 0
    stream_clips: dict[str, str] = {}
    trajectory_last_frame: dict[str, float] = {}
    records = (
        read_frames(directory) if (Path(directory) / "frames.jsonl.gz").exists() else ()
    )
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            continue
        frame = _frame_number(record, index)
        label = _series_label(record)
        clip_label = f"clip={record.get('clip', 'unknown')}"
        stream_clips[label] = clip_label
        truth = record.get("truth")
        xy = _truth_xy(truth) if isinstance(truth, Mapping) else None
        # The truth path is invariant across configurations; draw it once per clip.
        if xy is not None and frame > trajectory_last_frame.get(
            clip_label, float("-inf")
        ):
            _add_series_point(trajectories, clip_label, xy)
            trajectory_last_frame[clip_label] = frame
        metric = (
            record.get("metrics") if isinstance(record.get("metrics"), Mapping) else {}
        )
        robot_pose = metric.get("robot_pose") if isinstance(metric, Mapping) else None
        error = (
            _number(robot_pose.get("translation_3d_m"))
            if isinstance(robot_pose, Mapping)
            else None
        )
        _add_series_point(
            pose_errors, label, (frame, error if error is not None else math.nan)
        )
        available = (
            metric.get(
                "pose_available",
                record.get("pose_available", record.get("completed", False)),
            )
            if isinstance(metric, Mapping)
            else record.get("completed", False)
        )
        _add_series_point(availability, label, (frame, 1.0 if available else 0.0))
        counts = availability_counts.setdefault(label, [0, 0])
        counts[0] += int(bool(available))
        counts[1] += 1
        detection = metric.get("detection") if isinstance(metric, Mapping) else None
        if isinstance(detection, Mapping):
            by_tag = detection.get("by_tag")
            if isinstance(by_tag, list):
                for tag in by_tag:
                    if not isinstance(tag, Mapping):
                        continue
                    provisional = (
                        not tag.get("eligible") and tag.get("category") == "provisional"
                    )
                    if not tag.get("eligible"):
                        excluded_size_observations += 1
                        if not provisional:
                            continue
                    key = (label, provisional)
                    counts_by_size = size_counts.setdefault(
                        key, [[0, 0] for _ in range(len(size_edges) + 1)]
                    )
                    size = _number(tag.get("projected_size_px"))
                    if size is None or size < 0:
                        size_unavailable[key] = size_unavailable.get(key, 0) + 1
                        continue
                    bin_index = next(
                        (i for i, edge in enumerate(size_edges) if size < edge),
                        len(size_edges),
                    )
                    counts_by_size[bin_index][0] += int(bool(tag.get("detected")))
                    counts_by_size[bin_index][1] += 1
    size_labels = ("<16", "16–32", "32–64", "64–128", "≥128")
    plots = ""
    if not any(
        not provisional and any(count[1] for count in bins)
        for (_, provisional), bins in size_counts.items()
    ):
        plots = _detection_size_chart([], 0, excluded_size_observations)
    for clip_index, clip in enumerate(sorted(set(stream_clips.values()))):
        labels = {label for label, owner in stream_clips.items() if owner == clip}
        clip_availability = {
            f"{label} ({availability_counts[label][0]}/{availability_counts[label][1]} available)": availability[
                label
            ]
            for label in labels
        }
        charts = (
            _line_chart(
                "Ground-truth robot trajectory",
                {clip: trajectories.get(clip, [])},
                x_label="Field X (m)",
                y_label="Field Y (m)",
            )
            + _line_chart(
                "Robot translation error",
                {label: pose_errors[label] for label in labels},
                x_label="Frame within clip",
                y_label="Translation error (m)",
            )
            + _line_chart(
                "Pose availability by frame",
                clip_availability,
                x_label="Frame within clip",
                y_label="Pose available (1=yes, 0=no)",
                y_limits=(0.0, 1.0),
            )
        )
        for (label, provisional), bins in sorted(size_counts.items()):
            if label in labels:
                size_bins = [
                    (name, detected, total)
                    for name, (detected, total) in zip(size_labels, bins)
                    if total
                ]
                charts += _detection_size_chart(
                    size_bins,
                    size_unavailable.get((label, provisional), 0),
                    0,
                    stream=label,
                    provisional=provisional,
                )
        plots += f'<details class="clip-plots"{" open" if clip_index == 0 else ""}><summary>{html.escape(clip)}</summary>{charts}</details>'
    images = sorted((Path(directory) / "images").glob("*.jpg"))
    gallery = "".join(
        f'<figure><figcaption>{html.escape(path.stem)}</figcaption><img loading="lazy" src="images/{html.escape(path.name)}" alt="{html.escape(path.stem)}"></figure>'
        for path in images
    )
    if gallery:
        gallery = f"<h2>Diagnostics</h2><section class=gallery>{gallery}</section>"
    body = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font:14px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{border:1px solid #bbb;padding:.4rem;text-align:left;vertical-align:top}}svg{{width:100%;border:1px solid #ccc;background:#fff}}.series{{fill:none;stroke-width:2}}.grid{{stroke:#e2e2e2;stroke-width:1}}.axis{{stroke:#555;stroke-width:1}}.tick{{fill:#444;font-size:11px}}.axis-label{{fill:#222;font-size:12px;font-weight:600}}.bar-value{{fill:#222;font-size:11px;font-weight:600}}.legend{{display:flex;flex-wrap:wrap;gap:.3rem 1rem;list-style:none;padding:0;margin:.4rem 0 1.4rem}}.legend li{{white-space:nowrap}}.swatch{{display:inline-block;width:.8rem;height:.8rem;margin-right:.3rem;vertical-align:-.05rem}}code{{white-space:pre-wrap;word-break:break-word}}.boundary{{padding:1rem;background:#fff4ce}}.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:1rem}}figure{{margin:0}}img{{max-width:100%}}details{{border:1px solid #ddd;border-radius:6px;padding:1rem;margin:1rem 0}}summary{{cursor:pointer;font-weight:600}}.chart{{margin:1.5rem 0}}</style></head><body>
<h1>{html.escape(title)}</h1><p class="boundary">{html.escape(str(run.get("boundary", BOUNDARY)))}</p>
<h2>Summary</h2>{_table({key: value for key, value in summary.items() if key not in ("by_clip", "by_configuration", "diagnostic_images", "diagnostic_frame_ids")})}
<h2>Plots by clip</h2><p>Each clip has its own plots. Colors distinguish configurations; missing poses break error curves. Expand a clip to inspect it.</p>{plots}{gallery}
<details><summary>Full summary data</summary>{_table(summary)}</details>
<details><summary>Provenance and resolved configurations</summary>{_table(run)}</details>
<details><summary>Per clip/configuration/variant data</summary>{_table({str(i): row for i, row in enumerate(rows)})}</details></body></html>"""
    destination = Path(directory) / "index.html"
    destination.write_text(body, encoding="utf-8")
    return destination


def read_frames(directory: str | Path) -> Iterable[dict[str, Any]]:
    """Yield saved frame records from a run directory."""
    with gzip.open(
        Path(directory) / "frames.jsonl.gz", "rt", encoding="utf-8"
    ) as stream:
        for line in stream:
            yield json.loads(line)


def select_diagnostic_frame_ids(
    records: Iterable[Mapping[str, Any]], limit: int = 24
) -> list[int]:
    """Select a deterministic bounded set of failures and largest-error frames."""
    if limit < 0:
        raise ValueError("diagnostic limit must be nonnegative")
    ranked: list[tuple[int, float, int]] = []
    for record in records:
        frame = record.get("frame_index")
        if not isinstance(frame, int):
            continue
        metric = (
            record.get("metrics") if isinstance(record.get("metrics"), Mapping) else {}
        )
        pose = metric.get("robot_pose") if isinstance(metric, Mapping) else None
        error = pose.get("translation_3d_m") if isinstance(pose, Mapping) else None
        detection = metric.get("detection") if isinstance(metric, Mapping) else None
        exceptional = bool(
            record.get("failure")
            or (
                isinstance(detection, Mapping)
                and (detection.get("fp") or detection.get("fn"))
            )
            or record.get("reacquisition")
        )
        ranked.append((0 if exceptional else 1, -float(error or 0.0), frame))
    return sorted({item[2] for item in sorted(ranked)[:limit]})


def write_diagnostic_images(
    directory: str | Path,
    records: Iterable[Mapping[str, Any]],
    videos: Mapping[str, Path],
    limit: int = 24,
) -> list[str]:
    """Draw the live detector overlay from saved outputs after measured work."""
    import cv2
    import numpy as np

    from src.main_operations.modules.apriltags.utils.visualization import (
        draw_apriltag_overlay,
    )

    candidates: list[tuple[tuple[int, float, str, str, int], Mapping[str, Any]]] = []
    for record in records:
        frame = record.get("frame_index")
        clip = record.get("clip")
        if (
            not isinstance(frame, int)
            or not isinstance(clip, str)
            or clip not in videos
        ):
            continue
        metric = (
            record.get("metrics") if isinstance(record.get("metrics"), Mapping) else {}
        )
        pose = metric.get("robot_pose") if isinstance(metric, Mapping) else None
        error = (
            float(pose.get("translation_3d_m", 0.0))
            if isinstance(pose, Mapping)
            else 0.0
        )
        detection = metric.get("detection") if isinstance(metric, Mapping) else None
        exceptional = bool(
            record.get("failure")
            or (
                isinstance(detection, Mapping)
                and (detection.get("fp") or detection.get("fn"))
            )
        )
        key = (
            0 if exceptional else 1,
            -error,
            clip,
            str(record.get("configuration", "unknown")),
            frame,
        )
        candidates.append((key, record))
        candidates.sort(key=lambda item: item[0])
        del candidates[limit:]

    image_dir = Path(directory) / "images"
    image_dir.mkdir(exist_ok=True)
    written: list[str] = []
    captures: dict[str, Any] = {}
    try:
        for _key, record in candidates:
            clip = str(record["clip"])
            frame = int(record["frame_index"])
            if clip not in captures:
                captures[clip] = cv2.VideoCapture(str(videos[clip]))
            capture = captures[clip]
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = capture.read()
            if not ok or image is None:
                continue
            output_value = record.get("output")
            output: Mapping[str, Any] = (
                output_value if isinstance(output_value, Mapping) else {}
            )
            detections = [
                (int(detection["tag_id"]), np.asarray(detection["corners"]))
                for detection in output.get("detections") or []
                if isinstance(detection, Mapping)
                and detection.get("corners") is not None
            ]
            regions = [
                np.asarray(region) for region in output.get("search_regions") or []
            ]
            image = draw_apriltag_overlay(image, detections, regions)
            cv2.rectangle(image, (0, 0), (image.shape[1], 64), (0, 0, 0), -1)
            cv2.putText(
                image,
                "GREEN: detected tag + ID | RED: actual searched ROI | Geometric truth not drawn",
                (12, 53),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            label = f"{clip} {record.get('configuration', '')} frame {frame}"
            cv2.putText(
                image,
                label,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            safe_clip = re.sub(r"[^A-Za-z0-9_.-]+", "_", clip)
            safe_config = re.sub(
                r"[^A-Za-z0-9_.-]+", "_", str(record.get("configuration", "unknown"))
            )
            relative = f"images/{safe_clip}-{safe_config}-{frame:06d}.jpg"
            if cv2.imwrite(str(Path(directory) / relative), image):
                written.append(relative)
    finally:
        for capture in captures.values():
            capture.release()
    return written
