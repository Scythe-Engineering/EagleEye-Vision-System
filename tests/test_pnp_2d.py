"""Constrained geometry, current mounting, and capture-clock regressions."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
from pnp_localization_2d import align_heading as native_align_heading

from src.config.utils.operation import Connection, Operation
from src.main_operations.definitions import pnp_camera_localization_2d as module
from src.utils.camera_utils.camera_config_manager import CameraExtrinsics
from src.utils.camera_utils.camera_coordinate_transforms import (
    build_robot_from_camera_transform,
)
from src.utils.timestamped_samples import align_heading
from src.utils.timing import TimedValue, TimingMetadata, get_timing, unwrap_timed


@dataclass
class Scene:
    """Calibration, mutable map fixture, and fresh immutable native solver factory."""

    solver: module.PnpCameraLocalization2DDefinition
    config: SimpleNamespace
    inputs: Callable[[], tuple[dict[str, Any], np.ndarray]]
    matrix: np.ndarray
    distortion: np.ndarray
    tags: dict[int, SimpleNamespace]
    fresh_solver: Callable[[], module.PnpCameraLocalization2DDefinition]


@pytest.fixture(params=[4, 5, 8, 12, 14])
def scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Scene:
    """Build distorted observations with a live six-parameter camera mount."""
    matrix = np.array([[700.0, 0, 640], [0, 710, 400], [0, 0, 1]])
    distortion = np.array(
        [
            -0.12,
            0.03,
            0.002,
            -0.001,
            0.005,
            0.01,
            -0.002,
            0.001,
            0.001,
            -0.0002,
            0.0005,
            -0.0001,
            0.02,
            -0.015,
        ][: request.param]
    )
    path = tmp_path / "intrinsics.json"
    path.write_text(
        json.dumps(
            {
                "camera_matrix": matrix.tolist(),
                "distortion_coefficients": distortion.tolist(),
            }
        )
    )
    config = SimpleNamespace(
        intrinsics_path=str(path),
        extrinsics=CameraExtrinsics(
            pitch=13, yaw=-17, roll=7, x_offset=0.3, y_offset=-0.2, z_offset=0.7
        ),
    )
    registry = SimpleNamespace(get_config=lambda bus: config)
    yaw = np.pi
    robot = np.eye(4)
    robot[:3, :3] = np.diag([-1.0, -1.0, 1.0])
    robot[:3, 3] = [3, 2, 0]
    pose = robot @ build_robot_from_camera_transform(config.extrinsics)
    local = np.array(
        [
            [-0.6, -0.4, 4],
            [-0.2, -0.4, 4],
            [-0.2, 0, 4],
            [-0.6, 0, 4],
            [0.3, 0.1, 6],
            [0.7, 0.1, 6],
            [0.7, 0.5, 6],
            [0.3, 0.5, 6],
        ]
    )
    points = local @ pose[:3, :3].T + pose[:3, 3]
    tags = {
        index + 1: SimpleNamespace(global_corners=point)
        for index, point in enumerate(points.reshape(-1, 4, 3))
    }
    monkeypatch.setattr(module, "load_fmap_file", lambda path: tags)

    def fresh_solver() -> module.PnpCameraLocalization2DDefinition:
        """Construct a solver after deliberate fixture-map edits."""
        return module.PnpCameraLocalization2DDefinition("camera", "map", registry)

    solver = fresh_solver()

    def inputs() -> tuple[dict[str, Any], np.ndarray]:
        """Project fresh detections using the current camera mounting."""
        current_pose = robot @ build_robot_from_camera_transform(config.extrinsics)
        rotation = current_pose[:3, :3].T
        translation = -rotation @ current_pose[:3, 3]
        pixels = cv2.projectPoints(
            np.vstack([tag.global_corners for tag in tags.values()]),
            cv2.Rodrigues(rotation)[0],
            translation,
            matrix,
            distortion,
        )[0].reshape(-1, 4, 2)
        detections = [
            SimpleNamespace(tag_id=index + 1, corners=corner)
            for index, corner in enumerate(pixels)
        ]
        return {
            "detections": TimedValue(
                detections, TimingMetadata(1_000_000, 55_000_000, frame_seq=9)
            ),
            "gyro_samples": [
                {"timestamp_us": 990_000, "value": yaw - 0.02},
                {"timestamp_us": 1_010_000, "value": -yaw + 0.02},
            ],
        }, current_pose

    return Scene(solver, config, inputs, matrix, distortion, tags, fresh_solver)


def test_exact_asymmetric_distorted_wrap_and_live_mount(scene: Scene) -> None:
    """Recover the exact pose across yaw wrap and live mounting edits."""
    solver, config, inputs = scene.solver, scene.config, scene.inputs
    for new_offset in (0.3, 0.45):
        config.extrinsics.x_offset = new_offset
        config.extrinsics.roll += 1
        data, expected = inputs()
        result = solver.run(data)
        assert result["diagnostics"]["reason"] == "ok"
        assert result["diagnostics"]["alignment"] == "interpolated"
        np.testing.assert_allclose(result["camera_pose"], expected, atol=1e-7)
        np.testing.assert_allclose(
            result["diagnostics"]["robot_xy_m"], [3, 2], atol=1e-7
        )
        assert set(result) == {"camera_pose", "pose_meta", "diagnostics"}
        assert result["camera_pose"].shape == (4, 4)
        assert result["camera_pose"].dtype == np.float64
        assert isinstance(result["pose_meta"], list)
        assert len(result["pose_meta"]) == 3
        assert result["pose_meta"][0] == 2
        centers = np.array(
            [tag.global_corners.mean(axis=0) for tag in scene.tags.values()]
        )
        assert result["pose_meta"][1] == pytest.approx(
            np.linalg.norm(centers - expected[:3, 3], axis=1).mean(), abs=1e-7
        )
        assert result["pose_meta"][2] < 1e-7


def test_wrapper_preserves_detection_capture_only(scene: Scene) -> None:
    """Propagate detection capture metadata to every solver output."""
    solver, inputs = scene.solver, scene.inputs
    data, _ = inputs()
    operation = Operation(
        solver,
        "2d",
        "2d",
        input_ports=["detections", "gyro_samples"],
        output_ports=["camera_pose", "pose_meta", "diagnostics"],
    )
    result = operation.run(data)
    assert set(result) == {"camera_pose", "pose_meta", "diagnostics"}
    for output in result.values():
        assert get_timing(output) == data["detections"].timing
    assert unwrap_timed(result["diagnostics"])["reason"] == "ok"


@pytest.mark.parametrize(
    "samples,reason",
    [
        (None, "missing_gyro"),
        ([], "missing_gyro"),
        ([{"timestamp_us": 2, "value": 0}], "stale_gyro"),
        ([{"timestamp_us": 1_000_000, "value": float("nan")}], "invalid_gyro"),
        ([{"timestamp_us": True, "value": 0}], "invalid_gyro"),
    ],
)
def test_missing_stale_invalid(scene: Scene, samples: object, reason: str) -> None:
    """Reject absent, stale, or invalid gyro measurements explicitly."""
    solver, inputs = scene.solver, scene.inputs
    data, _ = inputs()
    data["gyro_samples"] = samples
    result = solver.run(data)
    assert result["camera_pose"] is result["pose_meta"] is None
    assert result["diagnostics"]["reason"] == reason


def test_capture_and_geometry_rejection(scene: Scene) -> None:
    """Reject invalid capture clocks and degenerate image geometry."""
    solver, inputs = scene.solver, scene.inputs
    data, _ = inputs()
    data["detections"] = unwrap_timed(data["detections"])
    assert solver.run(data)["diagnostics"]["reason"] == "missing_capture"
    for capture in (0, 1, -1, True, 1.5, 2**63):
        data, _ = inputs()
        data["detections"] = TimedValue(
            data["detections"].value, TimingMetadata(capture, 55_000_000)
        )
        assert solver.run(data)["diagnostics"]["reason"] == "missing_capture"
        with pytest.raises(ValueError, match="missing_capture"):
            align_heading(data["gyro_samples"], capture, 100_000, 20_000)
    data, _ = inputs()
    for detection in data["detections"].value:
        detection.corners[:] = [640, 400]
    assert solver.run(data)["diagnostics"]["reason"] == "degenerate_geometry"
    data["detections"].value[0].corners[0, 0] = np.nan
    assert solver.run(data)["diagnostics"]["reason"] == "invalid_points"


def test_gap_nearest_and_no_extrapolation() -> None:
    """Bound interpolation gaps and nearest alignment without extrapolation."""
    samples = [{"timestamp_us": 2, "value": 1}, {"timestamp_us": 200_000, "value": 2}]
    with pytest.raises(ValueError, match="gyro_gap_too_large"):
        align_heading(samples, 100_000, 100_000, 20_000)
    # Even near an endpoint, an oversized bracket must not use nearest fallback.
    with pytest.raises(ValueError, match="gyro_gap_too_large"):
        align_heading(samples, 190_000, 100_000, 20_000)
    assert align_heading(samples, 210_000, 100_000, 20_000)[1]["alignment"] == "nearest"
    with pytest.raises(ValueError, match="stale_gyro"):
        align_heading(samples, 221_000, 100_000, 20_000)


def test_heading_recovers_with_unused_invalid_history() -> None:
    """Ignore unused bad yaw while validating every history envelope."""
    for invalid in (float("nan"), float("inf"), True, "bad", None, 10**400):
        samples = [
            {"timestamp_us": 10_000, "value": invalid},
            {"timestamp_us": 20_000, "value": 0.4},
            {"timestamp_us": 30_000, "value": 0.6},
            {"timestamp_us": 40_000, "value": invalid},
        ]
        for capture, expected, alignment in (
            (20_000, 0.4, "exact"),
            (25_000, 0.5, "interpolated"),
        ):
            yaw, metadata = align_heading(samples, capture, 100_000, 20_000)
            assert yaw == pytest.approx(expected)
            assert metadata["alignment"] == alignment
        for history, capture, expected in (
            (samples[:2], 35_000, 0.4),
            (samples[2:], 15_000, 0.6),
        ):
            yaw, metadata = align_heading(history, capture, 100_000, 20_000)
            assert yaw == pytest.approx(expected)
            assert metadata["alignment"] == "nearest"
        # Exact, nearest and either bracket endpoint must not skip bad yaw.
        for history, capture in (
            (samples, 10_000),
            (samples, 40_000),
            (samples, 5_000),
            (samples, 45_000),
            (samples, 15_000),
            (samples, 35_000),
        ):
            with pytest.raises(ValueError, match="invalid_gyro"):
                align_heading(history, capture, 100_000, 20_000)
    # Unused envelopes/timestamps still require validation at the public boundary.
    for malformed in (
        None,
        {"timestamp_us": 10_000},
        {"timestamp_us": 10_000, "value": 0, "extra": 0},
        *({"timestamp_us": t, "value": 0} for t in (0, 1, -1, True, 1.5, 2**63)),
    ):
        with pytest.raises(ValueError, match="invalid_gyro"):
            align_heading(
                [malformed, {"timestamp_us": 20_000, "value": 0.4}],
                20_000,
                100_000,
                20_000,
            )


def test_pixel_refinement_reduces_noisy_reprojection(scene: Scene) -> None:
    """Reduce squared pixel error with bounded refinement."""
    solver, inputs = scene.solver, scene.inputs
    data, _ = inputs()
    data["detections"].value[0].corners += np.array(
        [[2, -1], [-1, 2], [1, 1], [-2, -1]]
    )
    solver.update_config({"refinement_iterations": 0})
    linear = solver.run(data)
    solver.update_config({"refinement_iterations": 20})
    refined = solver.run(data)
    assert refined["diagnostics"]["reason"] == "ok"

    # Optimizer minimizes squared pixel error, not the metadata's mean norm.
    def squared(pose: np.ndarray) -> np.floating[Any]:
        """Measure total squared distorted-pixel reprojection error."""
        inverse = np.linalg.inv(pose)
        points = np.vstack([tag.global_corners for tag in scene.tags.values()])
        pixels = cv2.projectPoints(
            points,
            cv2.Rodrigues(inverse[:3, :3])[0],
            inverse[:3, 3],
            scene.matrix,
            scene.distortion,
        )[0].reshape(-1, 2)
        return np.sum(
            (pixels - np.vstack([det.corners for det in data["detections"].value])) ** 2
        )

    assert squared(refined["camera_pose"]) < squared(linear["camera_pose"])


def test_behind_camera_rejected(scene: Scene) -> None:
    """Reject mapped points behind the calibrated camera."""
    inputs = scene.inputs
    _, pose = inputs()
    for tag in scene.tags.values():
        tag.global_corners[:] = 2 * pose[:3, 3] - tag.global_corners
    data, _ = inputs()
    result = scene.fresh_solver().run(data)
    assert result["camera_pose"] is None
    assert result["diagnostics"]["reason"] == "behind_camera"


def test_near_horizontal_rays_ill_conditioned(scene: Scene) -> None:
    """Reject XY geometry with insufficient numerical conditioning."""
    config, inputs = scene.config, scene.inputs
    config.extrinsics.pitch = config.extrinsics.yaw = config.extrinsics.roll = 0
    _, pose = inputs()
    local = np.array(
        [[-1e-9, -1e-9, 4], [1e-9, -1e-9, 4], [1e-9, 1e-9, 4], [-1e-9, 1e-9, 4]]
    )
    for tag in scene.tags.values():
        tag.global_corners[:] = local @ pose[:3, :3].T + pose[:3, 3]
    data, _ = inputs()
    result = scene.fresh_solver().run(data)
    assert result["camera_pose"] is None
    assert result["diagnostics"]["reason"] == "ill_conditioned_geometry"


def test_unknown_ids_ignored_and_invalid_mount_rejected(scene: Scene) -> None:
    """Ignore unmapped detections and reject invalid live mounting."""
    solver, config, inputs = scene.solver, scene.config, scene.inputs
    data, expected = inputs()
    data["detections"].value.append(
        SimpleNamespace(tag_id=999, corners=np.full((4, 2), np.nan))
    )
    np.testing.assert_allclose(solver.run(data)["camera_pose"], expected, atol=1e-7)
    config.extrinsics.x_offset = None
    assert solver.run(data)["diagnostics"]["reason"] == "invalid_mounting"
    assert (
        module.PnpCameraLocalization2dDefinition
        is module.PnpCameraLocalization2DDefinition
    )


def test_wrapper_single_detection_connection_preserves_capture(scene: Scene) -> None:
    """Preserve bare routed detection timing on missing-gyro rejection."""
    solver, inputs = scene.solver, scene.inputs
    data, _ = inputs()
    operation = Operation(
        solver,
        "2d",
        "2d",
        input_ports=["detections", "gyro_samples"],
        output_ports=["camera_pose", "pose_meta", "diagnostics"],
    )
    source = Operation(SimpleNamespace(), "tags", "tags", output_ports=["detections"])
    Connection(source, "detections", operation, "detections", "data")
    assert operation.is_only_input_connection(source.uuid)
    # FlowManager passes the sole routed output bare, not a port dictionary.
    result = operation.run(data["detections"])
    diagnostics = unwrap_timed(result["diagnostics"])
    assert diagnostics == {"capture_nt_us": 1_000_000, "reason": "missing_gyro"}
    assert unwrap_timed(result["camera_pose"]) is None
    assert unwrap_timed(result["pose_meta"]) is None
    assert get_timing(result["diagnostics"]) == data["detections"].timing


def test_requires_registry_and_intrinsics(scene: Scene) -> None:
    """Require current registry calibration at construction."""
    config = scene.config
    with pytest.raises(ValueError, match="registry"):
        module.PnpCameraLocalization2DDefinition("camera", "map")
    config.intrinsics_path = None
    registry = SimpleNamespace(get_config=lambda bus: config)
    with pytest.raises(ValueError, match="intrinsics calibration"):
        module.PnpCameraLocalization2DDefinition("camera", "map", registry)


def test_native_alignment_matches_shared_helper(scene: Scene) -> None:
    """Compare public native diagnostics with exact/bracket/nearest IEEE semantics."""
    histories: list[tuple[Any, int]] = [
        ([{"timestamp_us": 1_000_000, "value": yaw}], 1_000_000)
        for yaw in (np.pi, -np.pi, 1e300, -1e300)
    ]
    histories += [
        (
            [
                {"timestamp_us": 990_000, "value": np.pi - 0.02},
                {"timestamp_us": 1_010_000, "value": -np.pi + 0.02},
            ],
            1_000_000,
        ),
        ([{"timestamp_us": 990_000, "value": 0.4}], 1_000_000),
        ([{"timestamp_us": 1_010_000, "value": -0.4}], 1_000_000),
        (
            [
                {"timestamp_us": 1_000_000, "value": None},
                {"timestamp_us": 1_000_000, "value": 0.4},
            ],
            1_000_000,
        ),
        (
            [
                {"timestamp_us": 900_000, "value": None},
                {"timestamp_us": 1_000_000, "value": 0.4},
                {"timestamp_us": 1_100_000, "value": float("nan")},
            ],
            1_000_000,
        ),
        (
            [
                {"timestamp_us": 900_000, "value": 0.4},
                {"timestamp_us": 1_100_000, "value": 0.5},
            ],
            1_000_000,
        ),
        ([{"timestamp_us": 900_000, "value": 0.4}], 1_000_000),
        ([{"timestamp_us": True, "value": 0.4}], 1_000_000),
        ([{"timestamp_us": 1_000_000, "value": 0.4, "extra": 1}], 1_000_000),
        (({"timestamp_us": 1_000_000, "value": 0.4},), 1_000_000),
    ]
    for samples, capture in histories:
        data, _ = scene.inputs()
        data["gyro_samples"] = samples
        data["detections"] = TimedValue([], TimingMetadata(capture, 55_000_000))
        result = scene.solver.run(data)
        try:
            yaw, expected = align_heading(samples, capture, 100_000, 20_000)
        except ValueError as exc:
            assert result["diagnostics"]["reason"] == str(exc)
        else:
            assert result["diagnostics"]["reason"] == "no_mapped_tags"
            assert result["diagnostics"]["yaw_rad"] == yaw
            for key, value in expected.items():
                assert result["diagnostics"][key] == value


def test_native_validation_precedence_and_detection_metadata(scene: Scene) -> None:
    """Validate detections before mounting and ignore duplicate/unknown bad corners."""
    data, expected = scene.inputs()
    data["detections"].value.extend(
        [
            SimpleNamespace(tag_id=1, corners=None),
            SimpleNamespace(tag_id=999, corners=None),
        ]
    )
    np.testing.assert_allclose(
        scene.solver.run(data)["camera_pose"], expected, atol=1e-7
    )
    scene.config.extrinsics.x_offset = None
    data["gyro_samples"] = None
    assert scene.solver.run(data)["diagnostics"]["reason"] == "missing_gyro"
    data["gyro_samples"] = [{"timestamp_us": 1_000_000, "value": np.pi}]
    data["detections"].value[0].corners = [[1, 2]]
    assert scene.solver.run(data)["diagnostics"]["reason"] == "invalid_points"
    data["detections"] = TimedValue(None, TimingMetadata(1_000_000, 55_000_000))
    assert scene.solver.run(data)["diagnostics"]["reason"] == "invalid_detections"


@pytest.mark.parametrize(
    "settings",
    [
        {"refinement_iterations": 1.5},
        {"refinement_iterations": 101},
        {"gyro_max_gap_ms": float("nan")},
        {"gyro_nearest_ms": -1},
    ],
)
def test_wrapper_validates_live_configuration(
    scene: Scene, settings: dict[str, float]
) -> None:
    """Keep bounded wrapper configuration independent of native argument coercion."""
    with pytest.raises(ValueError):
        scene.solver.update_config(settings)


@pytest.mark.parametrize("limits", [(-1, 20_000), (100_000, -1), (-1, -1)])
@pytest.mark.parametrize(
    "capture,samples",
    [
        (1_000_000, [{"timestamp_us": 1_000_000, "value": 0}]),
        (
            1_000_000,
            [
                {"timestamp_us": 990_000, "value": 0},
                {"timestamp_us": 1_010_000, "value": 0},
            ],
        ),
        (1_000_000, [{"timestamp_us": 990_000, "value": 0}]),
        (1_000_000, None),
        (1_000_000, [{"timestamp_us": True, "value": 0}]),
        (None, None),
    ],
)
def test_native_negative_limits_raise_before_input_validation(
    capture: Any, samples: Any, limits: tuple[int, int]
) -> None:
    """Reject either negative limit before exact/bracket/nearest or bad inputs."""
    solver = module.PnpLocalization2D(np.eye(3).reshape(-1).tolist(), [], [], [])
    with pytest.raises(ValueError, match="gyro limits must be nonnegative"):
        native_align_heading(samples, capture, *limits)
    with pytest.raises(ValueError, match="gyro limits must be nonnegative"):
        solver.solve(capture, samples, None, None, 10, *limits)


@pytest.mark.parametrize("capture", [None, 1_000_000])
def test_native_refinement_limit_raises_before_input_validation(capture: Any) -> None:
    """Reject 101 iterations even when capture, gyro, detections or mount are absent."""
    solver = module.PnpLocalization2D(np.eye(3).reshape(-1).tolist(), [], [], [])
    with pytest.raises(
        ValueError, match="refinement_iterations must be between 0 and 100"
    ):
        solver.solve(capture, None, None, None, 101, 100_000, 20_000)


@pytest.mark.parametrize("iterations,limit", [(0, 0), (100, 10_000_000)])
def test_native_valid_setting_boundaries(
    scene: Scene, iterations: int, limit: int
) -> None:
    """Accept zero gyro limits on exact samples and both refinement endpoints."""
    data, expected = scene.inputs()
    samples = [{"timestamp_us": 1_000_000, "value": np.pi}]
    assert native_align_heading(samples, 1_000_000, limit, limit)[1] == {
        "alignment": "exact",
        "gyro_delta_us": 0,
    }
    result = scene.solver.native_solver.solve(
        1_000_000,
        samples,
        unwrap_timed(data["detections"]),
        build_robot_from_camera_transform(scene.config.extrinsics).reshape(-1).tolist(),
        iterations,
        limit,
        limit,
    )
    assert result["diagnostics"]["reason"] == "ok"
    np.testing.assert_allclose(
        np.reshape(result["camera_pose"], (4, 4)), expected, atol=1e-7
    )


@pytest.mark.parametrize(
    "settings,error,message",
    [
        (
            {
                "refinement_iterations": 20,
                "gyro_max_gap_ms": 200,
                "gyro_nearest_ms": -1,
            },
            ValueError,
            "gyro_nearest_ms must be finite and between 0 and 10000",
        ),
        (
            {"refinement_iterations": 20, "gyro_max_gap_ms": "bad"},
            ValueError,
            "could not convert string to float",
        ),
        (
            {
                "refinement_iterations": 20,
                "gyro_max_gap_ms": 200,
                "gyro_nearest_ms": None,
            },
            TypeError,
            "float",
        ),
        (
            {"refinement_iterations": 1.5, "gyro_max_gap_ms": 200},
            ValueError,
            "refinement_iterations must be an integer",
        ),
    ],
)
def test_wrapper_failed_configuration_leaves_all_settings_unchanged(
    settings: dict[str, Any], error: type[Exception], message: str
) -> None:
    """Keep every setting unchanged if any conversion or validation fails."""
    solver = module.PnpCameraLocalization2DDefinition.__new__(
        module.PnpCameraLocalization2DDefinition
    )
    solver.update_config(
        {"refinement_iterations": 10, "gyro_max_gap_ms": 100, "gyro_nearest_ms": 20}
    )
    before = vars(solver).copy()
    with pytest.raises(error, match=message):
        solver.update_config(settings)
    assert vars(solver) == before


def test_wrapper_configuration_converts_partial_full_and_unknown_settings() -> None:
    """Preserve conversions, partial updates, ignored keys and valid boundaries."""
    solver = module.PnpCameraLocalization2DDefinition.__new__(
        module.PnpCameraLocalization2DDefinition
    )
    solver.update_config(
        {
            "refinement_iterations": "100",
            "gyro_max_gap_ms": "10000",
            "gyro_nearest_ms": 0,
        }
    )
    assert vars(solver) == {
        "refinement_iterations": 100,
        "gyro_max_gap_ms": 10000.0,
        "gyro_nearest_ms": 0.0,
    }
    assert type(solver.refinement_iterations) is int
    assert type(solver.gyro_nearest_ms) is float
    solver.update_config({"gyro_nearest_ms": "12.5", "unknown": object()})
    assert vars(solver) == {
        "refinement_iterations": 100,
        "gyro_max_gap_ms": 10000.0,
        "gyro_nearest_ms": 12.5,
    }
    solver.update_config({"refinement_iterations": 0, "gyro_max_gap_ms": 0})
    assert solver.refinement_iterations == solver.gyro_max_gap_ms == 0
    assert solver.gyro_nearest_ms == 12.5


def test_missing_native_module_has_actionable_error(
    scene: Scene, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never silently substitute a Python solver when the extension is unavailable."""
    monkeypatch.setattr(module, "PnpLocalization2D", None)
    with pytest.raises(ImportError, match="build.*Rust extension"):
        scene.fresh_solver()
