"""Blender-only CAD and body-clearance validation for match trajectories."""

from __future__ import annotations

import math
from typing import Any

from match_motion import (  # type: ignore[import-not-found]
    ROBOT_X_BOUNDS_M,
    ROBOT_Y_BOUNDS_M,
    ROBOT_Z_BOUNDS_M,
    sample_motion,
)

# The imported CAD models a thin carpet/tape playing-surface stack. Contact
# with that stack is legal; every polygon rising above it remains collision CAD.
FLOOR_TRIANGLE_MAX_Z_M = 0.012
CLEARANCE_BOX_MIN_Z_M = 0.0


def _cad_bvh(imported: list[Any], depsgraph: Any) -> tuple[Any, int]:
    """Build a world-space BVH of actual CAD triangles above the legal floor."""
    from mathutils.bvhtree import BVHTree  # type: ignore

    vertices: list[tuple[float, float, float]] = []
    polygons: list[tuple[int, ...]] = []
    for obj in imported:
        if obj.type != "MESH":
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            world_vertices = [
                evaluated.matrix_world @ vertex.co for vertex in mesh.vertices
            ]
            for polygon in mesh.polygons:
                points = [world_vertices[index] for index in polygon.vertices]
                if max(point.z for point in points) <= FLOOR_TRIANGLE_MAX_Z_M:
                    continue
                base = len(vertices)
                vertices.extend(tuple(point) for point in points)
                polygons.append(tuple(range(base, base + len(points))))
        finally:
            evaluated.to_mesh_clear()
    if not polygons:
        raise RuntimeError("CAD clearance BVH contains no above-floor triangles")
    return BVHTree.FromPolygons(vertices, polygons, all_triangles=False), len(polygons)


def _box_bvh(
    x: float,
    y: float,
    yaw: float,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
) -> Any:
    """Create an oriented world-space BVH for one body envelope."""
    from mathutils.bvhtree import BVHTree  # type: ignore

    cosine, sine = math.cos(yaw), math.sin(yaw)
    vertices = []
    for z_value in z_bounds:
        for x_local, y_local in (
            (x_bounds[0], y_bounds[0]),
            (x_bounds[1], y_bounds[0]),
            (x_bounds[1], y_bounds[1]),
            (x_bounds[0], y_bounds[1]),
        ):
            vertices.append(
                (
                    x + cosine * x_local - sine * y_local,
                    y + sine * x_local + cosine * y_local,
                    z_value,
                )
            )
    polygons = [
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 4, 0),
    ]
    return BVHTree.FromPolygons(vertices, polygons, all_triangles=False)


def _world_corners(
    x: float,
    y: float,
    yaw: float,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> list[tuple[float, float]]:
    """Return the four planar corners of an oriented body."""
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return [
        (
            x + cosine * x_local - sine * y_local,
            y + sine * x_local + cosine * y_local,
        )
        for x_local, y_local in (
            (x_bounds[0], y_bounds[0]),
            (x_bounds[1], y_bounds[0]),
            (x_bounds[1], y_bounds[1]),
            (x_bounds[0], y_bounds[1]),
        )
    ]


def _overlap_2d(
    first: list[tuple[float, float]], second: list[tuple[float, float]]
) -> bool:
    """Use the separating-axis theorem for two oriented rectangles."""
    for rectangle in (first, second):
        for index in range(4):
            edge = (
                rectangle[(index + 1) % 4][0] - rectangle[index][0],
                rectangle[(index + 1) % 4][1] - rectangle[index][1],
            )
            axis = (-edge[1], edge[0])
            first_projection = [
                point[0] * axis[0] + point[1] * axis[1] for point in first
            ]
            second_projection = [
                point[0] * axis[0] + point[1] * axis[1] for point in second
            ]
            if max(first_projection) < min(second_projection) or max(
                second_projection
            ) < min(first_projection):
                return False
    return True


def validate_scene_clearance(
    route: dict[str, Any],
    fps: int,
    field_length_m: float,
    field_width_m: float,
    imported: list[Any],
    depsgraph: Any,
) -> dict[str, Any]:
    """Check actual CAD/body intersections and body pairs at every route frame.

    The playable floor is intentionally omitted from the CAD BVH.  Every other
    imported CAD polygon remains eligible to collide with the exact robot or
    occluder prism.  This validates flat routes; it does not invent traversal
    over bumps or elevated structures.
    """
    cad, triangle_count = _cad_bvh(imported, depsgraph)
    frame_count = round(float(route["duration_s"]) * fps)
    bodies: list[
        tuple[
            str,
            dict[str, Any],
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ]
    ] = [
        (
            "robot",
            route,
            ROBOT_X_BOUNDS_M,
            ROBOT_Y_BOUNDS_M,
            ROBOT_Z_BOUNDS_M,
        )
    ]
    for occluder in route.get("occluders", []):
        length, width, height = (float(value) for value in occluder["dimensions_m"])
        bodies.append(
            (
                occluder["name"],
                {**occluder, "duration_s": route["duration_s"]},
                (-length / 2, length / 2),
                (-width / 2, width / 2),
                (CLEARANCE_BOX_MIN_Z_M, height),
            )
        )

    for frame_index in range(frame_count):
        time_s = frame_index / fps
        rectangles: list[tuple[str, list[tuple[float, float]]]] = []
        for name, motion, x_bounds, y_bounds, z_bounds in bodies:
            sample = sample_motion(motion, time_s)
            corners = _world_corners(sample.x, sample.y, sample.yaw, x_bounds, y_bounds)
            if any(
                x < 0.0 or x > field_length_m or y < 0.0 or y > field_width_m
                for x, y in corners
            ):
                raise RuntimeError(
                    f"{route['name']} frame {frame_index}: {name} leaves the field"
                )
            if cad.overlap(
                _box_bvh(sample.x, sample.y, sample.yaw, x_bounds, y_bounds, z_bounds)
            ):
                raise RuntimeError(
                    f"{route['name']} frame {frame_index}: {name} intersects actual CAD"
                )
            rectangles.append((name, corners))
        for first_index, (first_name, first) in enumerate(rectangles):
            for second_name, second in rectangles[first_index + 1 :]:
                if _overlap_2d(first, second):
                    raise RuntimeError(
                        f"{route['name']} frame {frame_index}: {first_name} intersects {second_name}"
                    )
    return {
        "method": "world-space BVH triangle overlap with oriented body prisms at every delivered frame",
        "cad_source": "evaluated imported GLB meshes",
        "cad_polygons_checked": triangle_count,
        "floor_exclusion_max_z_m": FLOOR_TRIANGLE_MAX_Z_M,
        "body_min_z_m": CLEARANCE_BOX_MIN_Z_M,
        "frame_sample_rate_hz": fps,
        "frames_checked": frame_count,
        "bodies_checked": [name for name, *_rest in bodies],
        "field_boundary_checked": True,
        "body_pair_overlap_checked": True,
        "passed": True,
    }
