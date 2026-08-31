"""Tests for NetworkTables WPILib pose publishing helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.secondary_operations.publish_to_networktables import (
    PublishToNetworktables,
    _matrix_to_pose3d,
)
from src.utils.timing import TimedValue, TimingMetadata
from tests.utils.dummy_dependencies import FakeNetworkTable


def test_matrix_to_pose3d_converts_opencv_camera_axes_to_wpilib_body_axes() -> None:
    """A level robot pose should not publish with a 90 degree roll."""

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=float,
    )
    transform[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=float)

    pose = _matrix_to_pose3d(transform)

    assert pose.X() == pytest.approx(1.0)
    assert pose.Y() == pytest.approx(2.0)
    assert pose.Z() == pytest.approx(3.0)
    assert math.isclose(math.degrees(pose.rotation().X()), 0.0, abs_tol=1e-9)
    assert math.isclose(math.degrees(pose.rotation().Y()), 0.0, abs_tol=1e-9)
    assert math.isclose(math.degrees(pose.rotation().Z()), 0.0, abs_tol=1e-9)


def test_matrix_to_pose3d_preserves_wpilib_yaw_after_basis_conversion() -> None:
    """OpenCV camera-axis pose should publish robot yaw in WPILib NWU axes."""

    yaw = math.radians(45.0)
    field_from_robot = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    camera_from_robot_axes = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=float,
    )
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = field_from_robot @ camera_from_robot_axes

    pose = _matrix_to_pose3d(transform)

    assert math.isclose(math.degrees(pose.rotation().X()), 0.0, abs_tol=1e-9)
    assert math.isclose(math.degrees(pose.rotation().Y()), 0.0, abs_tol=1e-9)
    assert math.isclose(math.degrees(pose.rotation().Z()), 45.0, abs_tol=1e-9)


def test_publish_uses_capture_timestamp_when_value_is_timed() -> None:
    table = FakeNetworkTable()
    publisher = PublishToNetworktables(table, "RobotPose2D", schema="pose2d")
    transform = np.eye(4, dtype=float)
    timing = TimingMetadata(capture_nt_us=123456, capture_monotonic_ns=789)

    publisher.run(TimedValue(transform, timing))

    assert "RobotPose2D" in table.values
    assert table.values["RobotPose2D:timestamp"] == 123456


def test_publish_supports_detection_json_with_capture_timestamp() -> None:
    """Publish structured detections without inventing an incomplete WPILib struct."""
    table = FakeNetworkTable()
    publisher = PublishToNetworktables(table, "detections/front", schema="json")
    timing = TimingMetadata(capture_nt_us=222333, capture_monotonic_ns=444)
    detections = [
        {
            "class_name": "game-piece",
            "confidence": 0.9,
            "position_3d": [1.0, 2.0, 0.0],
        }
    ]

    publisher.run(TimedValue(detections, timing))

    assert table.values["detections/front"] == (
        '[{"class_name":"game-piece","confidence":0.9,'
        '"position_3d":[1.0,2.0,0.0]}]'
    )
    assert table.values["detections/front:timestamp"] == 222333


def test_publish_supports_primitive_double_arrays() -> None:
    table = FakeNetworkTable()
    publisher = PublishToNetworktables(table, "values", schema="double_array")
    timing = TimingMetadata(capture_nt_us=654321, capture_monotonic_ns=987)

    publisher.run(TimedValue([1, 2.5], timing))

    assert table.values["values"] == [1.0, 2.5]
    assert table.values["values:timestamp"] == 654321


def test_schema_change_recreates_the_typed_publisher() -> None:
    class _Publisher:
        def __init__(self, publisher_type: str) -> None:
            self.publisher_type = publisher_type
            self.values: list[float | str] = []

        def set(self, value: float | str) -> None:
            self.values.append(value)

    class _Topic:
        def __init__(self, publisher_type: str, publishers: list[_Publisher]) -> None:
            self.publisher_type = publisher_type
            self.publishers = publishers

        def publish(self) -> _Publisher:
            publisher = _Publisher(self.publisher_type)
            self.publishers.append(publisher)
            return publisher

    class _NetworkTable:
        def __init__(self) -> None:
            self.publishers: list[_Publisher] = []

        def getDoubleTopic(self, _key: str) -> _Topic:
            return _Topic("double", self.publishers)

        def getStringTopic(self, _key: str) -> _Topic:
            return _Topic("string", self.publishers)

    table = _NetworkTable()
    operation = PublishToNetworktables(table, "value", schema="double")

    operation.run(1)
    operation.update_config({"schema": "string"})
    operation.run("updated")

    assert [publisher.publisher_type for publisher in table.publishers] == [
        "double",
        "string",
    ]
    assert table.publishers[0].values == [1.0]
    assert table.publishers[1].values == ["updated"]
