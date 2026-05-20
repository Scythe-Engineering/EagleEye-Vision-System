from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.main_operations.definitions.base.base_class import OperationInstance
from src.utils.timing import TimedValue, get_timing, unwrap_timed


@dataclass(frozen=True)
class RoiCircle:
    """Circular robot-position region where camera frames are allowed."""

    x: float
    y: float
    radius: float


class RobotRoiFrameFilter(OperationInstance):
    """Filter dynamic camera frame streams by shared robot-space circular ROIs.

    Inputs:
        Dict with `robot_pose` and dynamic `camera_frame` / `camera_frame_N` ports.
    Outputs:
        Dict with mirrored `filtered_frame` / `filtered_frame_N` ports.
    """

    uses_timed_inputs = True

    def __init__(self, allowed_regions: list[dict[str, Any]] | None = None) -> None:
        """Initialize the robot ROI frame filter.

        Args:
            allowed_regions: Circular allowed regions shared by every camera stream.
                Each region must include `x`, `y`, and `radius` values in meters.
        """
        if allowed_regions is None:
            allowed_regions = [{"x": 0.0, "y": 0.0, "radius": 1.0}]

        self.allowed_regions = self._parse_allowed_regions(allowed_regions)
        self.robot_pose_port_name = "robot_pose"
        self.input_base_name = "camera_frame"
        self.output_base_name = "filtered_frame"

    def _parse_allowed_regions(
        self, allowed_regions: list[dict[str, Any]]
    ) -> list[RoiCircle]:
        """Parse configured ROI circles.

        Args:
            allowed_regions: Raw region dictionaries from operation configuration.

        Returns:
            Parsed ROI circles.

        Raises:
            ValueError: If the configuration does not contain valid circles.
        """
        parsed_regions: list[RoiCircle] = []
        for region in allowed_regions:
            if not isinstance(region, dict):
                raise ValueError("Each allowed region must be a dictionary.")

            try:
                x_position = float(region["x"])
                y_position = float(region["y"])
                radius = float(region["radius"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Each allowed region must include numeric x, y, and radius values."
                ) from exc

            if (
                not np.isfinite([x_position, y_position, radius]).all()
                or radius <= 0.0
            ):
                raise ValueError(
                    "Allowed region coordinates must be finite and radius must be positive."
                )

            parsed_regions.append(
                RoiCircle(x=x_position, y=y_position, radius=radius)
            )

        if not parsed_regions:
            raise ValueError("At least one allowed ROI circle must be configured.")

        return parsed_regions

    def update_config(self, json_config: dict[str, Any]) -> None:
        """Update operation parameters from JSON configuration.

        Args:
            json_config: Configuration dictionary with updated parameters.
        """
        if "allowed_regions" in json_config:
            self.allowed_regions = self._parse_allowed_regions(
                json_config["allowed_regions"]
            )

    def _contains_timed_value(self, value: Any) -> bool:
        """Check whether a value contains any timed pipeline value.

        Args:
            value: Pipeline value or nested container.

        Returns:
            True when at least one TimedValue wrapper is present.
        """
        if isinstance(value, TimedValue):
            return True
        if isinstance(value, dict):
            return any(self._contains_timed_value(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(self._contains_timed_value(item) for item in value)
        return False

    def _extract_robot_position(self, robot_pose: Any) -> tuple[float, float] | None:
        """Extract the robot x/y position from a pose value.

        Args:
            robot_pose: 4x4 robot pose matrix or mapping with `x` and `y` values.

        Returns:
            Robot x/y position when pose data is usable, otherwise None.
        """
        if isinstance(robot_pose, dict):
            try:
                x_position = float(robot_pose["x"])
                y_position = float(robot_pose["y"])
            except (KeyError, TypeError, ValueError):
                return None
            if np.isfinite([x_position, y_position]).all():
                return x_position, y_position
            return None

        try:
            robot_pose_matrix = np.asarray(robot_pose, dtype=float)
        except (TypeError, ValueError):
            return None

        if (
            robot_pose_matrix.shape != (4, 4)
            or not np.isfinite(robot_pose_matrix).all()
        ):
            return None

        return float(robot_pose_matrix[0, 3]), float(robot_pose_matrix[1, 3])

    def _extract_current_robot_position(
        self, input_data: dict[str, Any]
    ) -> tuple[float, float] | None:
        """Extract a non-stale robot position from operation input.

        Args:
            input_data: Operation input dictionary.

        Returns:
            Current robot x/y position when available, otherwise None.
        """
        robot_pose = input_data.get(self.robot_pose_port_name)
        if robot_pose is None:
            return None

        robot_pose_timing = get_timing(robot_pose)
        if robot_pose_timing is None and self._contains_timed_value(input_data):
            return None

        return self._extract_robot_position(unwrap_timed(robot_pose))

    def _is_position_allowed(self, robot_position: tuple[float, float]) -> bool:
        """Check whether a robot position is inside any configured ROI circle.

        Args:
            robot_position: Robot x/y position in meters.

        Returns:
            True when the position is inside at least one allowed region.
        """
        robot_x_position, robot_y_position = robot_position
        for region in self.allowed_regions:
            x_delta = robot_x_position - region.x
            y_delta = robot_y_position - region.y
            if x_delta * x_delta + y_delta * y_delta <= region.radius * region.radius:
                return True
        return False

    def _camera_frame_items(self, input_data: Any) -> list[tuple[str, Any]]:
        """Collect static template and dynamic camera frame inputs.

        Args:
            input_data: Operation input payload.

        Returns:
            Sorted frame input port names with their values.
        """
        if not isinstance(input_data, dict):
            if input_data is None:
                return []
            return [(self.input_base_name, input_data)]

        camera_frame_items = [
            (port_name, frame)
            for port_name, frame in input_data.items()
            if port_name == self.input_base_name
            or port_name.startswith(f"{self.input_base_name}_")
        ]
        return sorted(camera_frame_items, key=lambda item: item[0])

    def _output_port_name(self, input_port_name: str) -> str:
        """Map an input camera frame port to its mirrored output port.

        Args:
            input_port_name: Input port name.

        Returns:
            Mirrored output port name.
        """
        if input_port_name == self.input_base_name:
            return self.output_base_name

        index_token = input_port_name.split(f"{self.input_base_name}_", 1)[-1]
        return f"{self.output_base_name}_{index_token}"

    def run(self, input_data: Any) -> dict[str, Any] | None:
        """Filter camera frame streams by robot pose ROI eligibility.

        Args:
            input_data: Dict with `robot_pose` and dynamic camera frame inputs, or
                a single frame for direct pass-through compatibility.

        Returns:
            Mirrored output dict with frames or None values, or None without frames.
        """
        camera_frame_items = self._camera_frame_items(input_data)
        if not camera_frame_items:
            return None

        if not isinstance(input_data, dict):
            return {
                self._output_port_name(port_name): unwrap_timed(frame)
                for port_name, frame in camera_frame_items
            }

        robot_position = self._extract_current_robot_position(input_data)
        if robot_position is None:
            return {
                self._output_port_name(port_name): unwrap_timed(frame)
                for port_name, frame in camera_frame_items
            }

        frame_is_allowed = self._is_position_allowed(robot_position)
        return {
            self._output_port_name(port_name): (
                unwrap_timed(frame) if frame_is_allowed else None
            )
            for port_name, frame in camera_frame_items
        }
