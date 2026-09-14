"""Deterministic, smooth match-trajectory sampling and kinematic validation.

The limits in this module are conservative REV MAXSwerve controller settings,
not measurements of universal real-match robot performance.  Translation and
rotation share the four module wheel-speed budget.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_WHEEL_SPEED_M_S = 4.8
MAX_TRANSLATION_ACCELERATION_M_S2 = 3.0
MAX_ANGULAR_SPEED_RAD_S = 2.0 * math.pi
MAX_ANGULAR_ACCELERATION_RAD_S2 = math.pi
WHEELBASE_M = 0.6731
TRACKWIDTH_M = 0.6731
ROBOT_X_BOUNDS_M = (-0.60, 0.10)
ROBOT_Y_BOUNDS_M = (-0.35, 0.35)
ROBOT_Z_BOUNDS_M = (0.0, 0.5)
VARIANTS = ("combined-realistic",)
SEVERITIES = ("low", "medium", "high")


@dataclass(frozen=True)
class PoseSample:
    """One analytic planar pose and its first two derivatives."""

    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    yaw_rate: float
    ax: float
    ay: float
    yaw_acceleration: float
    labels: tuple[str, ...]


@dataclass(frozen=True)
class MotionValidation:
    """Observed frame-wise kinematic extrema for one moving body."""

    frame_count: int
    max_speed_m_s: float
    max_acceleration_m_s2: float
    max_yaw_rate_rad_s: float
    max_yaw_acceleration_rad_s2: float
    max_module_speed_m_s: float
    passed: bool


def _finite_number(value: Any, context: str) -> float:
    """Convert a finite JSON number to float or raise a useful error."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _pose(value: Any, context: str) -> tuple[float, float, float]:
    """Parse an ``[x, y, yaw_degrees]`` pose."""
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{context} must be [x, y, yaw_degrees]")
    return (
        _finite_number(value[0], f"{context}[0]"),
        _finite_number(value[1], f"{context}[1]"),
        math.radians(_finite_number(value[2], f"{context}[2]")),
    )


def load_trajectory(path: Path) -> dict[str, Any]:
    """Load and structurally validate one per-route trajectory JSON file."""
    route = json.loads(path.read_text())
    validate_route_schema(route)
    return route


def load_match_scenes(path: Path) -> list[dict[str, Any]]:
    """Load the central route catalog and validate every embedded route."""
    document = json.loads(path.read_text())
    if document.get("schema_version") != 1 or not isinstance(
        document.get("routes"), list
    ):
        raise ValueError(
            "match scene catalog must use schema_version 1 and contain routes"
        )
    routes = document["routes"]
    for route in routes:
        validate_route_schema(route)
    names = [route["name"] for route in routes]
    if len(names) != len(set(names)):
        raise ValueError("route names must be unique")
    return routes


def validate_route_schema(route: dict[str, Any]) -> None:
    """Validate the small route schema, continuity inputs, and durations."""
    if not isinstance(route, dict) or route.get("schema_version") != 1:
        raise ValueError("trajectory must use schema_version 1")
    if not isinstance(route.get("name"), str) or not route["name"]:
        raise ValueError("trajectory name must be a non-empty string")
    duration = _finite_number(route.get("duration_s"), "duration_s")
    if duration <= 0:
        raise ValueError("duration_s must be positive")
    _pose(route.get("start"), "start")
    segments = route.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("segments must be a non-empty list")
    segment_total = 0.0
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise TypeError(f"segments[{index}] must be an object")
        kind = segment.get("type")
        if kind not in ("hold", "move"):
            raise ValueError(f"segments[{index}].type must be hold or move")
        part_duration = _finite_number(
            segment.get("duration_s"), f"segments[{index}].duration_s"
        )
        if part_duration <= 0:
            raise ValueError(f"segments[{index}].duration_s must be positive")
        segment_total += part_duration
        if kind == "move":
            _pose(segment.get("end"), f"segments[{index}].end")
            controls = segment.get("control_points")
            if controls is not None:
                if not isinstance(controls, list) or len(controls) != 2:
                    raise ValueError(
                        f"segments[{index}].control_points must contain two XY points"
                    )
                for control_index, point in enumerate(controls):
                    if not isinstance(point, list) or len(point) != 2:
                        raise ValueError(
                            f"segments[{index}].control_points[{control_index}] must be XY"
                        )
                    _finite_number(point[0], "control x")
                    _finite_number(point[1], "control y")
        labels = segment.get("labels", [])
        if not isinstance(labels, list) or not all(
            isinstance(label, str) for label in labels
        ):
            raise ValueError(f"segments[{index}].labels must be strings")
    if not math.isclose(segment_total, duration, abs_tol=1e-9):
        raise ValueError(
            f"segment durations total {segment_total}, expected {duration}"
        )
    occluders = route.get("occluders", [])
    if not isinstance(occluders, list):
        raise TypeError("occluders must be a list")
    for index, occluder in enumerate(occluders):
        if not isinstance(occluder, dict) or not isinstance(occluder.get("name"), str):
            raise TypeError(f"occluders[{index}] requires a name")
        dimensions = occluder.get("dimensions_m")
        if not isinstance(dimensions, list) or len(dimensions) != 3:
            raise ValueError(f"occluders[{index}].dimensions_m must have three values")
        if any(_finite_number(v, "occluder dimension") <= 0 for v in dimensions):
            raise ValueError("occluder dimensions must be positive")
        pseudo_route = {
            "schema_version": 1,
            "name": occluder["name"],
            "duration_s": duration,
            "start": occluder.get("start"),
            "segments": occluder.get("segments"),
            "occluders": [],
        }
        validate_route_schema(pseudo_route)


def trajectory_sha256(route: dict[str, Any]) -> str:
    """Hash canonical route contents, including all occluder motion."""
    return hashlib.sha256(
        json.dumps(route, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _smootherstep(u: float) -> tuple[float, float, float]:
    """Return quintic progress and first two derivatives with respect to u."""
    q = u * u * u * (u * (u * 6.0 - 15.0) + 10.0)
    dq = 30.0 * u * u * (u - 1.0) * (u - 1.0)
    ddq = 60.0 * u * (2.0 * u * u - 3.0 * u + 1.0)
    return q, dq, ddq


def _shortest_yaw_delta(start: float, end: float) -> float:
    """Return the deterministic shortest signed angle from start to end."""
    return (end - start + math.pi) % (2.0 * math.pi) - math.pi


def _bezier_axis(
    start: float, control_a: float, control_b: float, end: float, q: float
) -> tuple[float, float, float]:
    """Evaluate a cubic Bezier axis and its first two q derivatives."""
    one = 1.0 - q
    value = (
        one**3 * start
        + 3 * one * one * q * control_a
        + 3 * one * q * q * control_b
        + q**3 * end
    )
    first = (
        3 * one * one * (control_a - start)
        + 6 * one * q * (control_b - control_a)
        + 3 * q * q * (end - control_b)
    )
    second = 6 * one * (control_b - 2 * control_a + start) + 6 * q * (
        end - 2 * control_b + control_a
    )
    return value, first, second


def sample_motion(spec: dict[str, Any], time_s: float) -> PoseSample:
    """Sample a validated robot or occluder motion at an arbitrary time."""
    start_x, start_y, start_yaw = _pose(spec["start"], "start")
    duration = float(spec["duration_s"])
    t = min(max(float(time_s), 0.0), duration)
    elapsed = 0.0
    current = (start_x, start_y, start_yaw)
    for segment in spec["segments"]:
        segment_duration = float(segment["duration_s"])
        is_last = segment is spec["segments"][-1]
        if t <= elapsed + segment_duration or is_last:
            labels = tuple(segment.get("labels", []))
            if segment["type"] == "hold":
                return PoseSample(*current, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, labels)
            end_x, end_y, end_yaw = _pose(segment["end"], "end")
            u = min(max((t - elapsed) / segment_duration, 0.0), 1.0)
            q, dq_du, ddq_du2 = _smootherstep(u)
            controls = segment.get("control_points")
            if controls is None:
                controls = [
                    [
                        current[0] + (end_x - current[0]) / 3.0,
                        current[1] + (end_y - current[1]) / 3.0,
                    ],
                    [
                        current[0] + 2.0 * (end_x - current[0]) / 3.0,
                        current[1] + 2.0 * (end_y - current[1]) / 3.0,
                    ],
                ]
            x, dx_dq, d2x_dq2 = _bezier_axis(
                current[0], float(controls[0][0]), float(controls[1][0]), end_x, q
            )
            y, dy_dq, d2y_dq2 = _bezier_axis(
                current[1], float(controls[0][1]), float(controls[1][1]), end_y, q
            )
            q_rate = dq_du / segment_duration
            q_acceleration = ddq_du2 / (segment_duration * segment_duration)
            yaw_delta = _shortest_yaw_delta(current[2], end_yaw)
            return PoseSample(
                x,
                y,
                current[2] + yaw_delta * q,
                dx_dq * q_rate,
                dy_dq * q_rate,
                yaw_delta * q_rate,
                d2x_dq2 * q_rate * q_rate + dx_dq * q_acceleration,
                d2y_dq2 * q_rate * q_rate + dy_dq * q_acceleration,
                yaw_delta * q_acceleration,
                labels,
            )
        elapsed += segment_duration
        if segment["type"] == "move":
            end_x, end_y, end_yaw = _pose(segment["end"], "end")
            # Keep Blender's Euler keys continuous across the ±pi boundary.
            current = (
                end_x,
                end_y,
                current[2] + _shortest_yaw_delta(current[2], end_yaw),
            )
    raise AssertionError("validated trajectory has no sampleable segment")


def module_speeds(sample: PoseSample) -> tuple[float, float, float, float]:
    """Calculate four MAXSwerve module speeds from robot-frame chassis motion."""
    cosine, sine = math.cos(sample.yaw), math.sin(sample.yaw)
    vx_robot = cosine * sample.vx + sine * sample.vy
    vy_robot = -sine * sample.vx + cosine * sample.vy
    half_wheelbase = WHEELBASE_M / 2.0
    half_trackwidth = TRACKWIDTH_M / 2.0
    modules = (
        (half_wheelbase, half_trackwidth),
        (half_wheelbase, -half_trackwidth),
        (-half_wheelbase, half_trackwidth),
        (-half_wheelbase, -half_trackwidth),
    )
    speeds = [
        math.hypot(
            vx_robot - sample.yaw_rate * module_y,
            vy_robot + sample.yaw_rate * module_x,
        )
        for module_x, module_y in modules
    ]
    return speeds[0], speeds[1], speeds[2], speeds[3]


def validate_motion(spec: dict[str, Any], fps: int) -> MotionValidation:
    """Validate analytic kinematics at every delivered frame and boundaries."""
    if fps < 1:
        raise ValueError("fps must be positive")
    frame_count_float = float(spec["duration_s"]) * fps
    frame_count = round(frame_count_float)
    if not math.isclose(frame_count_float, frame_count, abs_tol=1e-9):
        raise ValueError("duration_s * fps must be an integer")
    samples = [sample_motion(spec, frame / fps) for frame in range(frame_count)]
    # Segment endpoints are not all necessarily delivered frame midpoints.
    elapsed = 0.0
    for segment in spec["segments"]:
        elapsed += float(segment["duration_s"])
        samples.append(sample_motion(spec, elapsed))
    speeds = [math.hypot(sample.vx, sample.vy) for sample in samples]
    accelerations = [math.hypot(sample.ax, sample.ay) for sample in samples]
    yaw_rates = [abs(sample.yaw_rate) for sample in samples]
    yaw_accelerations = [abs(sample.yaw_acceleration) for sample in samples]
    wheel_speeds = [max(module_speeds(sample)) for sample in samples]
    maxima = MotionValidation(
        frame_count=frame_count,
        max_speed_m_s=max(speeds),
        max_acceleration_m_s2=max(accelerations),
        max_yaw_rate_rad_s=max(yaw_rates),
        max_yaw_acceleration_rad_s2=max(yaw_accelerations),
        max_module_speed_m_s=max(wheel_speeds),
        passed=False,
    )
    passed = (
        maxima.max_acceleration_m_s2 <= MAX_TRANSLATION_ACCELERATION_M_S2 + 1e-9
        and maxima.max_yaw_rate_rad_s <= MAX_ANGULAR_SPEED_RAD_S + 1e-9
        and maxima.max_yaw_acceleration_rad_s2 <= MAX_ANGULAR_ACCELERATION_RAD_S2 + 1e-9
        and maxima.max_module_speed_m_s <= MAX_WHEEL_SPEED_M_S + 1e-9
    )
    return MotionValidation(**{**maxima.__dict__, "passed": passed})


def catalog_duration_s(routes: list[dict[str, Any]]) -> float:
    """Return the unique duration represented by a route catalog."""
    return sum(float(route["duration_s"]) for route in routes)
