"""Small non-Blender checks for the production match trajectory catalog."""

from __future__ import annotations

import json
import math
from itertools import pairwise
from pathlib import Path

from benchmarks.blender.match_motion import (
    MAX_ANGULAR_ACCELERATION_RAD_S2,
    MAX_ANGULAR_SPEED_RAD_S,
    MAX_TRANSLATION_ACCELERATION_M_S2,
    MAX_WHEEL_SPEED_M_S,
    PoseSample,
    catalog_duration_s,
    load_match_scenes,
    module_speeds,
    sample_motion,
    trajectory_sha256,
    validate_motion,
)

CATALOG = Path(__file__).parents[1] / "benchmarks/blender/match-scenes.json"


def test_catalog_has_exactly_180_seconds() -> None:
    """The production budget is unique motion, not variant-expanded runtime."""
    document = json.loads(CATALOG.read_text())
    routes = load_match_scenes(CATALOG)
    assert document["variants"] == ["combined-realistic"]
    assert document["defaults"] == {
        "width": 1280,
        "height": 800,
        "fps": 120,
        "samples": 256,
        "engine": "cycles",
    }
    assert len(routes) == 6
    assert catalog_duration_s(routes) == 180.0
    assert {route["duration_s"] for route in routes} == {30}


def test_generator_accepts_explicit_fast_cycles_options() -> None:
    """Performance choices are explicit rather than silently changing old renders."""
    from benchmarks.blender.generate import parse_args

    args = parse_args(
        ["--output", "/tmp/unused", "--samples", "64", "--denoise", "--persistent-data"]
    )
    assert args.samples == 64
    assert args.denoise and args.persistent_data
    defaults = parse_args(["--output", "/tmp/unused"])
    assert not defaults.denoise and not defaults.persistent_data


def test_every_route_obeys_rev_controller_envelope_at_every_frame() -> None:
    """Analytic frame samples obey translational, angular, and wheel limits."""
    for route in load_match_scenes(CATALOG):
        validation = validate_motion(route, 120)
        assert validation.frame_count == 3600
        assert validation.passed, route["name"]
        assert validation.max_acceleration_m_s2 <= MAX_TRANSLATION_ACCELERATION_M_S2
        assert validation.max_yaw_rate_rad_s <= MAX_ANGULAR_SPEED_RAD_S
        assert validation.max_yaw_acceleration_rad_s2 <= MAX_ANGULAR_ACCELERATION_RAD_S2
        assert validation.max_module_speed_m_s <= MAX_WHEEL_SPEED_M_S
        for occluder in route["occluders"]:
            occluder_spec = {**occluder, "duration_s": route["duration_s"]}
            assert validate_motion(occluder_spec, 120).passed


def test_wheel_budget_couples_otherwise_legal_translation_and_rotation() -> None:
    """Independent chassis caps do not imply feasible simultaneous motion."""
    sample = PoseSample(
        x=0.0,
        y=0.0,
        yaw=0.0,
        vx=4.5,
        vy=0.0,
        yaw_rate=1.0,
        ax=0.0,
        ay=0.0,
        yaw_acceleration=0.0,
        labels=(),
    )
    assert math.hypot(sample.vx, sample.vy) < MAX_WHEEL_SPEED_M_S
    assert abs(sample.yaw_rate) < MAX_ANGULAR_SPEED_RAD_S
    assert max(module_speeds(sample)) > MAX_WHEEL_SPEED_M_S


def test_sampling_is_deterministic_and_routes_mix_motion_pauses_and_events() -> None:
    """The catalog includes repeatable holds, curves, turns, and recovery."""
    routes = load_match_scenes(CATALOG)
    first_hashes = [trajectory_sha256(route) for route in routes]
    second_hashes = [trajectory_sha256(route) for route in load_match_scenes(CATALOG)]
    assert first_hashes == second_hashes
    assert [sample_motion(routes[0], t) for t in (0.0, 3.0, 8.0)] == [
        sample_motion(routes[0], t) for t in (0.0, 3.0, 8.0)
    ]

    samples = [
        sample_motion(route, frame / 120)
        for route in routes
        for frame in range(int(route["duration_s"] * 120))
    ]
    labels = {label for sample in samples for label in sample.labels}
    assert any(math.hypot(sample.vx, sample.vy) == 0.0 for sample in samples)
    assert any(math.hypot(sample.vx, sample.vy) > 3.0 for sample in samples)
    assert any(abs(sample.yaw_rate) > math.radians(120) for sample in samples)
    for route in routes:
        for body in [
            route,
            *({**o, "duration_s": route["duration_s"]} for o in route["occluders"]),
        ]:
            yaws = [sample_motion(body, frame / 120).yaw for frame in range(3600)]
            assert (
                max(abs(b - a) for a, b in pairwise(yaws))
                <= MAX_ANGULAR_SPEED_RAD_S / 120
            )
    assert {
        "approach",
        "retreat",
        "lateral",
        "curve",
        "loss",
        "reacquisition",
    } <= labels
