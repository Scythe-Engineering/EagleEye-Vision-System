"""Recipe-driven Blender renderer for the deterministic FRC 2026 renderer.

The GLB is a millimetre, centred, Y-up visual asset.  Field/map truth remains
independent and uses metres in NWU coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

SCRIPT_DIR = Path(__file__).resolve().parent
# Blender executes this file as a script rather than as a package module.
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from match_motion import (  # type: ignore[import-not-found]
    MAX_ANGULAR_ACCELERATION_RAD_S2,
    MAX_ANGULAR_SPEED_RAD_S,
    MAX_TRANSLATION_ACCELERATION_M_S2,
    MAX_WHEEL_SPEED_M_S,
    TRACKWIDTH_M,
    WHEELBASE_M,
    MotionValidation,
    load_trajectory,
    sample_motion,
    trajectory_sha256,
    validate_motion,
)
from match_motion import VARIANTS as MATCH_VARIANTS  # type: ignore[import-not-found]
from match_scene_validation import (  # type: ignore[import-not-found]
    validate_scene_clearance,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FMAP = (
    ROOT
    / "src/webui/assets/fields/2026/apriltag_maps/FE-2026-_REBUILTTM_Playing_Field.fmap"
)
DEFAULT_GLB = (
    ROOT
    / "src/webui/assets/fields/2026/field_files/FE-2026-_REBUILTTM_Playing_Field.glb"
)
# Clean is retained only for the tiny local smoke recipe.  The production
# interface exposes the composed effect stack, not its historical components.
VARIANTS = ("clean", "combined-realistic")
CV_TO_BLENDER = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, -1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
# OpenCV camera (right, down, forward) to robot NWU (forward, left, up).
ROBOT_NWU_FROM_CV = (
    (0.0, 0.0, 1.0, 0.0),
    (-1.0, 0.0, 0.0, 0.0),
    (0.0, -1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
# A sub-millimetre render-only depth bias prevents coplanar CAD placeholder
# faces from winning the z-buffer. Rendered corner truth uses this same pose;
# production localization intentionally continues to use the surveyed fmap.
TAG_DECAL_OFFSET_M = 0.0002

OFFICIAL_SOURCES = {
    "playing_field": "https://www.firstinspires.org/resources/library/frc/playing-field",
    "step": "https://firstfrc.blob.core.windows.net/frc2026/FieldAssets/FE-2026-rev-rebuilt-playing-field.step",
    "dimension_drawings": "https://firstfrc.blob.core.windows.net/frc2026/FieldAssets/2026-field-dimension-dwgs.pdf",
    "welded_map": "repository 2026 WPILib-style welded .fmap (hash recorded below)",
    "maxswerve_limits": "https://docs.revrobotics.com/ion-build/motion/maxswerve/programming-maxswerve",
    "maxswerve_template_constants": "https://github.com/REVrobotics/MAXSwerve-Java-Template/blob/main/src/main/java/frc/robot/Constants.java",
}


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse Blender arguments and merge an optional recipe."""
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--recipe", type=Path)
    p.add_argument("--fmap", type=Path, default=DEFAULT_FMAP)
    p.add_argument("--glb", type=Path, default=DEFAULT_GLB)
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--fps", type=int)
    p.add_argument("--frames", type=int)
    p.add_argument("--samples", type=int)
    p.add_argument(
        "--denoise",
        action="store_true",
        help="Use Cycles OpenImageDenoise with albedo/normal guides",
    )
    p.add_argument(
        "--persistent-data",
        action="store_true",
        help="Keep static render data between frames",
    )
    p.add_argument(
        "--trajectory",
        type=Path,
        help="Per-route JSON object using the match_motion schema",
    )
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="Export complete truth/provenance and validation without rendering PNGs",
    )
    p.add_argument(
        "--frame-indices",
        help="Comma-separated absolute zero-based preflight frames; truth remains complete",
    )
    p.add_argument("--variant", choices=VARIANTS)
    p.add_argument("--severity", choices=("low", "medium", "high"))
    p.add_argument("--seed", type=int)
    p.add_argument("--start-x", type=float)
    p.add_argument("--start-y", type=float)
    p.add_argument("--yaw-degrees", type=float)
    p.add_argument("--speed", type=float)
    p.add_argument("--engine", choices=("cycles", "eevee"))
    p.add_argument(
        "--device",
        default=None,
        help="Cycles compute device: METAL, OPTIX, CUDA, HIP, ONEAPI, or CPU",
    )
    default_python = (
        ROOT
        / ".venv"
        / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    )
    p.add_argument("--python-executable", type=Path, default=default_python)
    p.add_argument("--no-field-mesh", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--make-tags", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--tag-dir", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--tag-ids", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    explicitly_set = {
        option.lstrip("-").split("=", 1)[0].replace("-", "_")
        for option in argv
        if option.startswith("--")
    }
    defaults = {
        "width": 1280,
        "height": 800,
        "fps": 120,
        "frames": 120,
        "samples": 64,
        "variant": "clean",
        "engine": "cycles",
        "severity": "medium",
        "seed": 2026,
        "start_x": 2.0,
        "yaw_degrees": 0.0,
        "speed": 1.0,
    }
    if args.recipe:
        recipe = json.loads(args.recipe.read_text())
        if recipe.get("schema_version") != 1:
            raise ValueError("unsupported recipe schema")
        recipe_arguments = recipe.get("generator_arguments", {})
        defaults.update(recipe_arguments)
        explicitly_set.update(recipe_arguments)
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    args.explicitly_set = explicitly_set
    if args.frame_indices:
        try:
            args.selected_frame_indices = sorted(
                {int(value) for value in args.frame_indices.split(",")}
            )
        except ValueError as exc:
            raise ValueError(
                "--frame-indices must be comma-separated integers"
            ) from exc
    else:
        args.selected_frame_indices = None
    return args


def make_and_verify_tags(directory: Path, ids: list[int]) -> None:
    """Generate tag36h11 PNGs and verify their IDs independently."""
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    from pupil_apriltags import Detector  # type: ignore

    directory.mkdir(parents=True, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    detector = Detector(families="tag36h11")
    for tag_id in ids:
        marker = cv2.aruco.generateImageMarker(dictionary, tag_id, 512, borderBits=1)
        image: Any = np.full((640, 640), 255, np.uint8)
        image[64:576, 64:576] = marker
        decoded = [int(x.tag_id) for x in detector.detect(image)]
        if decoded != [tag_id] or not cv2.imwrite(
            str(directory / f"tag36h11-{tag_id}.png"), image
        ):
            raise RuntimeError(
                f"independent tag bitmap verification failed for {tag_id}: {decoded}"
            )


def verify_fmap_landmarks(fmap: dict[str, Any]) -> dict[str, Any]:
    """Guard the expected welded-map revision with two separated references."""
    # These values were transcribed from the same welded fmap. They catch map
    # revision/coordinate-convention drift; they are not a CAD landmark survey.
    # IDs 1 and 29 are on opposite field ends.
    expected = {1: (3.6075, 3.3903, 0.8890), 29: (-8.2628, -3.3685, 0.5524)}
    by_id = {int(item["id"]): item for item in fmap["fiducials"]}
    measured = {}
    for tag_id, target in expected.items():
        if tag_id not in by_id:
            raise RuntimeError(f"required landmark tag {tag_id} is absent")
        transform = by_id[tag_id]["transform"]
        actual = tuple(float(transform[i]) for i in (3, 7, 11))
        if max(abs(a - b) for a, b in zip(actual, target)) > 0.001:
            raise RuntimeError(f"fmap landmark {tag_id} changed: {actual} != {target}")
        measured[str(tag_id)] = list(actual)
    separation = math.dist(measured["1"], measured["29"])
    if separation < 10:
        raise RuntimeError("landmark check is not spatially separated")
    return {
        "source": "values transcribed from welded fmap",
        "purpose": "map revision and coordinate-convention guard only",
        "positions_centered_m": measured,
        "separation_m": separation,
        "tolerance_m": 0.001,
        "proves_cad_tag_alignment": False,
    }


def timestamp_ns(frame_index: int, fps: int) -> int:
    """Round an exact rational frame midpoint to integer nanoseconds."""
    numerator = frame_index * 1_000_000_000
    quotient, remainder = divmod(numerator, fps)
    twice = remainder * 2
    return quotient + int(twice > fps or (twice == fps and quotient % 2 == 1))


def flat(matrix: Any) -> list[float]:
    """Serialize one 4x4 matrix in row-major order."""
    return [float(matrix[r][c]) for r in range(4) for c in range(4)]


def valid_png(path: Path) -> bool:
    """Reject missing, truncated, and obviously corrupt PNG resume frames."""
    try:
        if path.stat().st_size < 100:
            return False
        with path.open("rb") as image:
            if image.read(8) != b"\x89PNG\r\n\x1a\n":
                return False
            image.seek(-12, 2)
            return image.read(12) == b"\x00\x00\x00\x00IEND\xaeB`\x82"
    except OSError:
        return False


def set_linear_interpolation(obj: Any) -> None:
    """Set every transform key on a Blender 5 layered action to linear."""
    action = getattr(getattr(obj, "animation_data", None), "action", None)
    if action is None:
        return
    for layer in action.layers:
        for strip in layer.strips:
            for channelbag in strip.channelbags:
                for curve in channelbag.fcurves:
                    for keyframe in curve.keyframe_points:
                        keyframe.interpolation = "LINEAR"


def distortion_point(
    x: float, y: float, k: list[float], fx: float, fy: float, cx: float, cy: float
) -> list[float]:
    """Apply OpenCV k1,k2,p1,p2,k3 forward distortion to an ideal pixel."""
    xn, yn = (x - cx) / fx, (y - cy) / fy
    k1, k2, p1, p2, k3 = k
    r2 = xn * xn + yn * yn
    radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
    yd = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
    return [fx * xd + cx, fy * yd + cy]


def run_blender(args: argparse.Namespace) -> None:
    """Build, validate, render, and export one deterministic clip."""
    import bpy  # type: ignore
    from bpy_extras.object_utils import world_to_camera_view  # type: ignore
    from mathutils import Matrix, Vector  # type: ignore

    route = load_trajectory(Path(args.trajectory)) if args.trajectory else None
    if route:
        expected_frames = round(float(route["duration_s"]) * args.fps)
        if "frames" not in args.explicitly_set:
            args.frames = expected_frames
        elif args.frames != expected_frames:
            raise ValueError(
                f"--frames {args.frames} disagrees with route duration/fps ({expected_frames})"
            )
        if "samples" not in args.explicitly_set:
            args.samples = 256
        required = {
            "width": (args.width, 1280),
            "height": (args.height, 800),
            "fps": (args.fps, 120),
            "engine": (args.engine, "cycles"),
        }
        wrong = [
            name for name, (actual, expected) in required.items() if actual != expected
        ]
        if wrong:
            raise ValueError(
                f"match trajectories require production settings: {', '.join(wrong)}"
            )
        if args.variant not in MATCH_VARIANTS:
            raise ValueError(
                "match trajectories support only the combined-realistic variant"
            )
        if args.no_field_mesh:
            raise ValueError(
                "match trajectory validation requires the actual CAD field mesh"
            )
    if (
        min(args.frames, args.fps, args.samples) < 1
        or min(args.width, args.height) < 16
    ):
        raise ValueError("invalid frame settings")
    if not route and (
        not all(
            math.isfinite(value)
            for value in (args.start_x, args.yaw_degrees, args.speed)
        )
        or abs(args.speed) > 4.0
    ):
        raise ValueError("trajectory values must be finite and speed within 4 m/s")
    if not route and args.start_y is not None and not math.isfinite(args.start_y):
        raise ValueError("trajectory start Y must be finite")
    if args.selected_frame_indices is not None and any(
        frame < 0 or frame >= args.frames for frame in args.selected_frame_indices
    ):
        raise ValueError("--frame-indices contains a frame outside the complete route")
    route_motion_validation = validate_motion(route, args.fps) if route else None
    occluder_motion_validations: dict[str, MotionValidation] = {}
    route_duration_s = float(route["duration_s"]) if route else 0.0
    if route:
        assert route_motion_validation is not None
        if not route_motion_validation.passed:
            raise RuntimeError(
                f"route {route['name']} violates the REV motion envelope"
            )
        for occluder in route.get("occluders", []):
            validation = validate_motion(
                {**occluder, "duration_s": route["duration_s"]}, args.fps
            )
            if not validation.passed:
                raise RuntimeError(
                    f"occluder {occluder['name']} violates the REV motion envelope"
                )
            occluder_motion_validations[occluder["name"]] = validation
    fmap = json.loads(args.fmap.read_text())
    length, width = float(fmap["fieldlength"]), float(fmap["fieldwidth"])
    landmark_validation = verify_fmap_landmarks(fmap)
    args.output.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output / "frames"
    tag_dir = args.output / "tags"
    frame_dir.mkdir(exist_ok=True)
    ids = [int(x["id"]) for x in fmap["fiducials"]]
    helper = [
        str(args.python_executable),
        str(Path(__file__)),
        "--output",
        str(args.output),
        "--make-tags",
        "--tag-dir",
        str(tag_dir),
        "--tag-ids",
        ",".join(map(str, ids)),
    ]
    subprocess.run(helper, check=True)
    settings = {
        "schema": 2,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "frames": args.frames,
        "samples": args.samples,
        "denoising": args.denoise,
        "persistent_data": args.persistent_data,
        "variant": args.variant,
        "severity": args.severity,
        "seed": args.seed,
        "engine": args.engine,
        "device": args.device,
        "trajectory": (
            {
                "mode": "match-route",
                "name": route["name"],
                "contents": route,
                "contents_sha256": trajectory_sha256(route),
            }
            if route
            else {
                "mode": "constant-speed-smoke",
                "start_x": args.start_x,
                "start_y": args.start_y,
                "yaw_degrees": args.yaw_degrees,
                "speed": args.speed,
            }
        ),
        "render_selection": {
            "validate_only": args.validate_only,
            "frame_indices": args.selected_frame_indices,
            "complete_truth_exported": True,
            "sparse_preflight_not_full_output": args.selected_frame_indices is not None,
        },
        # Debug renders without CAD must never share a resume hash with
        # the benchmark scene, even when every other option matches.
        "field_mesh": {"enabled": not args.no_field_mesh, "path": str(args.glb)},
        "fmap_sha256": sha256(args.fmap),
        "glb_sha256": sha256(args.glb),
        "generator_sha256": sha256(Path(__file__)),
        "motion_helper_sha256": sha256(SCRIPT_DIR / "match_motion.py"),
        "scene_validation_helper_sha256": sha256(
            SCRIPT_DIR / "match_scene_validation.py"
        ),
        "blender_version": bpy.app.version_string,
    }
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    state = args.output / "settings.json"
    if state.exists() and json.loads(state.read_text())["settings_hash"] != digest:
        raise RuntimeError("output settings changed; use a new/clean directory")
    state.write_text(
        json.dumps({"settings_hash": digest, "resolved": settings}, indent=2) + "\n"
    )

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    # Render real surrounding geometry for the inverse lens warp, rather than
    # inventing black pixels at the packaged image boundary. Packaging checks
    # that this margin covers every requested source pixel.
    distorted = args.variant == "combined-realistic"
    margin_x = math.ceil(args.width * 0.15) if distorted else 0
    margin_y = math.ceil(args.height * 0.15) if distorted else 0
    render_width, render_height = args.width + 2 * margin_x, args.height + 2 * margin_y
    scene.render.resolution_x = render_width
    scene.render.resolution_y = render_height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "16"
    scene.render.fps = args.fps
    scene.render.film_transparent = False
    if args.engine == "eevee":
        scene.render.engine = (
            "BLENDER_EEVEE" if bpy.app.version >= (5, 0, 0) else "BLENDER_EEVEE_NEXT"
        )
    else:
        scene.render.engine = "CYCLES"
    scene.render.use_persistent_data = args.persistent_data
    if args.engine == "cycles":
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = args.denoise
        if args.denoise:
            scene.cycles.denoiser = "OPENIMAGEDENOISE"
            scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
            scene.cycles.denoising_prefilter = "ACCURATE"
            scene.cycles.denoising_use_gpu = bool(
                args.device and args.device.upper() != "CPU"
            )
        scene.cycles.seed = args.seed
        if args.device and args.device.upper() != "CPU":
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.compute_device_type = args.device.upper()
            prefs.get_devices()
            for dev in prefs.devices:
                dev.use = dev.type == args.device.upper()
            scene.cycles.device = "GPU"
    scene.world = bpy.data.worlds.new("render_world")
    scene.world.color = (0.04, 0.04, 0.04)
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"

    shift = Matrix.Translation((length / 2, width / 2, 0))
    mesh_validation = None
    tag_surface_validation = None
    clearance_validation = None
    mesh_import = {
        "enabled": False,
        "reason": "--no-field-mesh debug render",
        "object_count": 0,
        "root_count": 0,
        "imported_image_texture_count": 0,
        "objects_hidden": [],
    }
    if not args.no_field_mesh:
        before = set(bpy.context.scene.objects)
        bpy.ops.import_scene.gltf(filepath=str(args.glb))
        imported = [obj for obj in bpy.context.scene.objects if obj not in before]
        roots = [obj for obj in imported if obj.parent not in imported]
        if not imported or not roots:
            raise RuntimeError("GLB import produced no rooted objects")
        # Inspecting the evaluated import (rather than assuming generic glTF
        # axes) shows this CAD's floor spans Blender X/Z and height is Blender
        # Y. Convert that effective basis once at the imported hierarchy roots.
        # Applying it to children too compounds transforms in this nested GLB.
        align = Matrix(
            (
                (0.001, 0, 0, length / 2),
                (0, 0, -0.001, width / 2),
                (0, 0.001, 0, 0),
                (0, 0, 0, 1),
            )
        )
        candidates = []
        for obj in imported:
            raw = obj.dimensions.copy()
            obj.name = "visual_field_" + obj.name
            if obj.type == "MESH":
                candidates.append(
                    (
                        abs(raw.x / 1000 - length) + abs(raw.z / 1000 - width),
                        obj.name,
                        raw.x / 1000,
                        raw.z / 1000,
                    )
                )
        for root in roots:
            root.matrix_world = align @ root.matrix_world
        _, name, measured_l, measured_w = min(candidates)
        errors = [abs(measured_l - length), abs(measured_w - width)]
        if max(errors) > 0.20:
            raise RuntimeError(
                f"no GLB field candidate agrees with fmap dimensions: best {name} {measured_l:.3f}x{measured_w:.3f} m"
            )
        # Never infer that an ordinary textured CAD object is a baked tag. The
        # checked-in asset has no image textures; replacements are rendered
        # intact and must be reviewed explicitly for duplicate tag artwork.
        imported_images = {
            node.image.name
            for obj in imported
            for material in list(getattr(getattr(obj, "data", None), "materials", []))
            if material and material.node_tree
            for node in material.node_tree.nodes
            if node.type == "TEX_IMAGE" and node.image is not None
        }
        mesh_validation = {
            "candidate": name,
            "measured_m": [measured_l, measured_w],
            "fmap_m": [length, width],
            "tolerance_m": 0.20,
            "errors_m": errors,
        }
        # Independently query the transformed CAD, before generated planes are
        # added. Every welded-map tag centre must coincide with a real CAD
        # mounting face. This validates geometry alignment rather than merely
        # copying positions back out of the fmap.
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        surface_errors = []
        surface_objects = set()
        for item in fmap["fiducials"]:
            raw = Matrix(
                tuple(
                    tuple(item["transform"][r * 4 + c] for c in range(4))
                    for r in range(4)
                )
            )
            centre = (shift @ raw).translation
            ray_axis = raw.to_3x3() @ Vector((1, 0, 0))
            ray_axis.normalize()
            hit, location, _normal, _face, hit_object, _matrix = scene.ray_cast(
                depsgraph, centre + ray_axis * 0.02, -ray_axis, distance=0.04
            )
            if not hit:
                raise RuntimeError(
                    f"CAD has no mounting surface at fmap tag {item['id']}"
                )
            error = (location - centre).length
            if error > 0.002:
                raise RuntimeError(
                    f"CAD mounting surface for tag {item['id']} is {error:.4f} m from fmap centre"
                )
            surface_errors.append(error)
            surface_objects.add(hit_object.name)
        tag_surface_validation = {
            "method": "independent CAD ray cast at every fmap tag centre before generated planes exist",
            "count": len(surface_errors),
            "tolerance_m": 0.002,
            "max_error_m": max(surface_errors),
            "hit_objects": sorted(surface_objects),
        }
        mesh_import = {
            "enabled": True,
            "reason": None,
            "object_count": len(imported),
            "root_count": len(roots),
            "imported_image_texture_count": len(imported_images),
            "objects_hidden": [],
            "evaluated_import_basis": "CAD floor X/Z, height Y; measured after Blender glTF import",
        }
        if route:
            clearance_validation = validate_scene_clearance(
                route, args.fps, length, width, imported, depsgraph
            )

    # Independently express image-right, image-up, outward in fmap's +X-out,
    # +Y-left, +Z-up tag frame. Do not derive truth from the production parser.
    fmap_from_image = Matrix(((0, 0, 1, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
    tags = []
    for item in fmap["fiducials"]:
        bpy.ops.mesh.primitive_plane_add(size=float(item["size"]) / 1000 * 10 / 8)
        obj = bpy.context.object
        obj.name = f"tag36h11_{item['id']}"
        raw = Matrix(
            tuple(
                tuple(item["transform"][r * 4 + c] for c in range(4)) for r in range(4)
            )
        )
        truth_world = shift @ raw @ fmap_from_image
        # The CAD includes coplanar blank mounting faces. Apply a documented
        # 0.2 mm depth bias, and use the biased transform for rendered corner
        # truth. Camera/robot truth remains the actual scene pose while the
        # production map deliberately remains the surveyed fmap.
        render_world = truth_world @ Matrix.Translation((0, 0, TAG_DECAL_OFFSET_M))
        obj.matrix_world = render_world
        mat = bpy.data.materials.new(obj.name + "_material")
        mat.use_nodes = True
        tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(
            str(tag_dir / f"tag36h11-{item['id']}.png"), check_existing=False
        )
        tex.interpolation = "Closest"
        mat.node_tree.links.new(
            tex.outputs["Color"],
            mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"],
        )
        obj.data.materials.append(mat)
        tags.append((item, obj, truth_world, render_world))

    bpy.ops.object.empty_add(location=(0, 0, 0))
    robot = bpy.context.object
    robot.name = "robot_truth_NWU_ground"
    # Keep opaque robot geometry behind the forward-mounted camera (the old
    # centred cube enclosed the optical centre and could blank the footage).
    bpy.ops.mesh.primitive_cube_add(location=(-0.25, 0, 0.25), scale=(0.35, 0.35, 0.25))
    visual = bpy.context.object
    visual.name = "robot_visual_offset"
    visual.parent = robot
    occluder_objects: list[tuple[dict[str, Any], Any]] = []
    for occluder_index, occluder in enumerate(
        route.get("occluders", []) if route else []
    ):
        dimensions = [float(value) for value in occluder["dimensions_m"]]
        bpy.ops.mesh.primitive_cube_add(
            location=(0, 0, dimensions[2] / 2),
            scale=(dimensions[0] / 2, dimensions[1] / 2, dimensions[2] / 2),
        )
        occluder_object = bpy.context.object
        occluder_object.name = f"moving_occluder_{occluder['name']}"
        material = bpy.data.materials.new(occluder_object.name + "_opaque")
        material.diffuse_color = (
            0.08 + 0.10 * (occluder_index % 2),
            0.12,
            0.18,
            1.0,
        )
        occluder_object.data.materials.append(material)
        occluder_objects.append((occluder, occluder_object))
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "camera_OpenCV_mount"
    camera.data.lens_unit = "FOV"
    fx = (args.width / 2) / math.tan(math.radians(40))
    camera.data.angle = 2 * math.atan(render_width / (2 * fx))
    camera.data.sensor_fit = "HORIZONTAL"
    scene.camera = camera
    severity_scale = {"low": 0.5, "medium": 1.0, "high": 2.0}[args.severity]
    uneven_lighting = args.variant == "combined-realistic"
    light_location = (
        (length * 0.30, width * 0.25, 7)
        if uneven_lighting
        else (length / 2, width / 2, 8)
    )
    bpy.ops.object.light_add(type="AREA", location=light_location)
    light = bpy.context.object
    light.data.energy = (1800 / severity_scale) if uneven_lighting else 2600
    light.data.shape = "RECTANGLE"
    light.data.size = 7 if uneven_lighting else 12
    distortion = (
        [
            -0.06 * severity_scale,
            0.012 * severity_scale,
            0.0005 * severity_scale,
            -0.0005 * severity_scale,
            0.0,
        ]
        if args.variant == "combined-realistic"
        else [0.0] * 5
    )
    fy = fx
    cx = args.width / 2
    cy = args.height / 2
    exposure_ns = (
        round(2_000_000 * severity_scale) if args.variant == "combined-realistic" else 0
    )
    scene.render.use_motion_blur = exposure_ns > 0
    if exposure_ns:
        scene.render.motion_blur_shutter = exposure_ns / 1e9 * args.fps
    camera.data.dof.use_dof = args.variant == "combined-realistic"
    camera.data.dof.focus_distance = 3.0
    camera.data.dof.aperture_fstop = {"low": 5.6, "medium": 2.8, "high": 1.4}[
        args.severity
    ]
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = (
        {"low": -0.35, "medium": -0.8, "high": -1.5}[args.severity]
        if args.variant == "combined-realistic"
        else 0.0
    )

    cv_to_robot = Matrix(ROBOT_NWU_FROM_CV)
    cv_to_blender = Matrix(CV_TO_BLENDER)
    mount = Matrix.Translation((0.25, 0, 0.5)) @ cv_to_robot
    camera.parent = robot
    camera.matrix_parent_inverse = Matrix.Identity(4)
    camera.matrix_basis = mount @ cv_to_blender

    # Keyframes provide evaluated subframe transforms for motion blur.
    yaw = math.radians(args.yaw_degrees)
    start_y = width / 2 if args.start_y is None else args.start_y
    # Include shutter endpoints outside the delivered frame range, otherwise
    # Blender clamps the first/last exposure to half of the intended motion.
    for i in range(-1, args.frames + 1):
        if route:
            robot_sample = sample_motion(route, i / args.fps)
            robot.location = (robot_sample.x, robot_sample.y, 0)
            robot.rotation_euler = (0, 0, robot_sample.yaw)
        else:
            distance = args.speed * i / args.fps
            robot.location = (
                args.start_x + distance * math.cos(yaw),
                start_y + distance * math.sin(yaw),
                0,
            )
            robot.rotation_euler = (0, 0, yaw)
        robot.keyframe_insert("location", frame=i + 1)
        robot.keyframe_insert("rotation_euler", frame=i + 1)
        for occluder, occluder_object in occluder_objects:
            occluder_sample = sample_motion(
                {**occluder, "duration_s": route_duration_s}, i / args.fps
            )
            occluder_object.location.x = occluder_sample.x
            occluder_object.location.y = occluder_sample.y
            occluder_object.rotation_euler = (0, 0, occluder_sample.yaw)
            occluder_object.keyframe_insert("location", frame=i + 1)
            occluder_object.keyframe_insert("rotation_euler", frame=i + 1)
    set_linear_interpolation(robot)
    for _occluder, occluder_object in occluder_objects:
        set_linear_interpolation(occluder_object)

    rows = []
    for i in range(args.frames):
        scene.frame_set(i + 1)
        camera_cv = robot.matrix_world @ mount
        bpy.context.view_layer.update()
        projected = []
        for item, obj, production_world, render_world in tags:
            half = float(item["size"]) / 2000
            pts = []
            depths = []
            # Match pupil-apriltags' canonical corner order after the fmap's
            # tag-local basis conversion. This is also the order used by PnP.
            for a, b in ((half, half), (-half, half), (-half, -half), (half, -half)):
                ndc = world_to_camera_view(
                    scene, camera, render_world @ Vector((a, b, 0))
                )
                p = [
                    ndc.x * render_width - margin_x,
                    (1 - ndc.y) * render_height - margin_y,
                ]
                pts.append(distortion_point(p[0], p[1], distortion, fx, fy, cx, cy))
                depths.append(ndc.z)
            inside = all(
                d > 0 and 0 <= p[0] < args.width and 0 <= p[1] < args.height
                for p, d in zip(pts, depths)
            )
            normal = render_world.to_3x3() @ Vector((0, 0, 1))
            front = (
                normal.dot(camera.matrix_world.translation - render_world.translation)
                > 0
            )
            center_hit = None
            if route and front and inside:
                # A center ray verifies blocker placement, not visible pixel area.
                direction = render_world.translation - camera.matrix_world.translation
                hit, _point, _normal, _face, hit_object, _matrix = scene.ray_cast(
                    bpy.context.evaluated_depsgraph_get(),
                    camera.matrix_world.translation,
                    direction.normalized(),
                    distance=direction.length + 0.001,
                )
                center_hit = hit_object.name if hit else None
            edges = [math.dist(pts[j], pts[(j + 1) % 4]) for j in range(4)]
            reasons = (
                ([] if front else ["back_facing"])
                + ([] if inside else ["not_fully_in_frame"])
                + ([] if min(edges) >= 16 else ["too_small"])
            )
            projected.append(
                {
                    "id": int(item["id"]),
                    "family": "tag36h11",
                    "corners_px": pts,
                    "projected_min_edge_px": min(edges),
                    "in_frame": inside,
                    "front_facing": bool(front),
                    "visible_fraction": None,
                    "center_ray_first_hit": center_hit,
                    "visibility_method": "geometric-only; visible area unknown; center ray is diagnostic only",
                    "eligibility": "provisional"
                    if front and inside and min(edges) >= 16
                    else "excluded",
                    "exclusion_reasons": reasons,
                    "T_field_from_tag_rendered": flat(render_world),
                    "T_field_from_tag_production": flat(production_world),
                    "render_to_production_offset_m": TAG_DECAL_OFFSET_M,
                }
            )
        if route:
            frame_motion = sample_motion(route, i / args.fps)
            scenario_labels = list(frame_motion.labels)
        else:
            scenario_labels = ["constant-speed-smoke"]
        occluder_truth = [
            {
                "name": occluder["name"],
                "label": "robot-sized simple opaque moving occluder",
                "dimensions_m": occluder["dimensions_m"],
                "T_field_from_occluder": flat(occluder_object.matrix_world),
                "pose_origin": "body center",
                "visibility_measurement": None,
            }
            for occluder, occluder_object in occluder_objects
        ]
        rows.append(
            json.dumps(
                {
                    "frame_index": i,
                    "timestamp_ns": timestamp_ns(i, args.fps),
                    "exposure_duration_ns": exposure_ns,
                    "variant": args.variant,
                    "severity": args.severity,
                    "T_field_from_robot": flat(robot.matrix_world),
                    "T_field_from_camera": flat(camera_cv),
                    "calibration": {
                        "camera_matrix": [fx, 0, cx, 0, fy, cy, 0, 0, 1],
                        "distortion_coefficients": distortion,
                    },
                    "tags": projected,
                    "occluders": occluder_truth,
                    "scenario_labels": scenario_labels,
                },
                separators=(",", ":"),
            )
        )
        out = frame_dir / f"frame_{i:06d}.png"
        selected = (
            args.selected_frame_indices is None or i in args.selected_frame_indices
        )
        if not args.validate_only and selected:
            if args.resume and out.exists() and valid_png(out):
                continue
            if out.exists():
                out.unlink()
            scene.render.filepath = str(out)
            render_started = time.perf_counter()
            bpy.ops.render.render(write_still=True)
            with (args.output / "render-timings.jsonl").open("a") as timing_log:
                timing_log.write(
                    json.dumps(
                        {
                            "frame_index": i,
                            "render_and_save_seconds": time.perf_counter()
                            - render_started,
                        }
                    )
                    + "\n"
                )
    (args.output / "truth.jsonl").write_text("\n".join(rows) + "\n")
    effects = {
        "clean": [],
        "combined-realistic": [
            "lighting",
            "exposure response",
            "depth of field",
            "motion blur",
            "OpenCV lens warp",
            "seeded sensor noise",
        ],
    }[args.variant]
    match_validation = cast(MotionValidation, route_motion_validation)
    provenance = {
        "schema_version": 2,
        "settings_hash": digest,
        "generator": sha256(Path(__file__)),
        "blender": bpy.app.version_string,
        "platform": platform.platform(),
        "render_engine": scene.render.engine,
        "render_device": args.device or "default",
        "samples": args.samples,
        "denoising": args.denoise,
        "denoiser": "OPENIMAGEDENOISE" if args.denoise else None,
        "denoising_device": (args.device or "CPU") if args.denoise else None,
        "persistent_data": args.persistent_data,
        "color_management": {
            "view": "AgX",
            "look": scene.view_settings.look,
            "output": "16-bit display-encoded PNG",
        },
        "sources": {
            **OFFICIAL_SOURCES,
            "fmap": str(args.fmap),
            "fmap_sha256": sha256(args.fmap),
            "glb": str(args.glb),
            "glb_sha256": sha256(args.glb),
        },
        "field_mesh": mesh_import,
        "mesh_alignment": {
            "mapping_after_blender_gltf_import": "x'=x/1000+L/2, y'=-z/1000+W/2, z'=y/1000",
            "applied_to": "import roots only",
            "perimeter_validation": mesh_validation,
            "tag_surface_validation": tag_surface_validation,
            "fmap_reference_validation": landmark_validation,
            "claim": "CAD perimeter dimensions and CAD mounting faces independently checked; fmap references separately guard map revision",
        },
        "tags": {
            "production_transform": "surveyed fmap after field-centre shift; image right=map Y, image up=map Z, outward=map X",
            "render_transform": "production transform plus local +Z depth bias; serialized per tag in truth",
            "render_to_production_offset_m": TAG_DECAL_OFFSET_M,
            "localization_map": "unmodified production fmap",
            "bitmap_verification": "generated tag36h11 images independently decoded with pupil-apriltags",
        },
        "render_selection": settings["render_selection"],
        "match_route": (
            {
                "name": route["name"],
                "duration_s": route["duration_s"],
                "geography": route.get("geography"),
                "variants": [args.variant],
                "occluders": [
                    {
                        "name": occluder["name"],
                        "dimensions_m": occluder["dimensions_m"],
                        "truth_exported": True,
                    }
                    for occluder in route.get("occluders", [])
                ],
            }
            if route
            else None
        ),
        "camera": {
            "resolution": [args.width, args.height],
            "render_resolution": [render_width, render_height],
            "overscan_margin_px": [margin_x, margin_y],
            "opencv_camera_to_robot_nwu": ROBOT_NWU_FROM_CV,
            "mount_translation_robot_m": [0.25, 0, 0.5],
        },
        "trajectory_validation": (
            {
                "route_name": route["name"],
                "route_sha256": trajectory_sha256(route),
                "frame_count": match_validation.frame_count,
                "max_speed_m_s": match_validation.max_speed_m_s,
                "max_acceleration_m_s2": match_validation.max_acceleration_m_s2,
                "max_yaw_rate_rad_s": match_validation.max_yaw_rate_rad_s,
                "max_yaw_acceleration_rad_s2": match_validation.max_yaw_acceleration_rad_s2,
                "max_module_speed_m_s": match_validation.max_module_speed_m_s,
                "limits": {
                    "wheel_speed_m_s": MAX_WHEEL_SPEED_M_S,
                    "translation_acceleration_m_s2": MAX_TRANSLATION_ACCELERATION_M_S2,
                    "angular_speed_rad_s": MAX_ANGULAR_SPEED_RAD_S,
                    "angular_acceleration_rad_s2": MAX_ANGULAR_ACCELERATION_RAD_S2,
                    "wheelbase_m": WHEELBASE_M,
                    "trackwidth_m": TRACKWIDTH_M,
                    "source_note": "REV MAXSwerve control settings, not universal measured match limits",
                },
                "module_budget_method": "four module velocity vectors from simultaneous robot-frame translation and yaw",
                "occluders": {
                    name: {
                        "frame_count": validation.frame_count,
                        "max_speed_m_s": validation.max_speed_m_s,
                        "max_acceleration_m_s2": validation.max_acceleration_m_s2,
                        "max_yaw_rate_rad_s": validation.max_yaw_rate_rad_s,
                        "max_yaw_acceleration_rad_s2": validation.max_yaw_acceleration_rad_s2,
                        "max_module_speed_m_s": validation.max_module_speed_m_s,
                        "passed": validation.passed,
                    }
                    for name, validation in occluder_motion_validations.items()
                },
                "cad_clearance": clearance_validation,
                "passed": match_validation.passed
                and bool(clearance_validation and clearance_validation["passed"]),
            }
            if route
            else {
                "max_speed_m_s": abs(args.speed),
                "max_acceleration_m_s2": 0.0,
                "max_yaw_rate_deg_s": 0.0,
                "limits": [4.0, 3.0, 180.0],
                "passed": abs(args.speed) <= 4.0,
            }
        ),
        "effects": {
            "requested": args.variant,
            "severity": args.severity,
            "seed": args.seed,
            "implemented_or_limit": effects,
            "order": "scene lighting/materials; exposure integration and depth of field; lens mapping; sensor noise/response; display encoding",
        },
        "visibility_limit": "No object-ID/depth masks: tag eligibility is provisional; occluder poses/labels do not claim measured visibility or recall",
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    if args.validate_only:
        scene_validation = {
            "schema_version": 1,
            "settings_hash": digest,
            "route_name": route["name"] if route else None,
            "field_perimeter": mesh_validation,
            "tag_mounting_surfaces": tag_surface_validation,
            "fmap_references": landmark_validation,
            "body_and_cad_clearance": clearance_validation,
            "kinematics": provenance["trajectory_validation"],
            "passed": bool(
                mesh_validation
                and tag_surface_validation
                and landmark_validation
                and (not route or clearance_validation)
                and provenance["trajectory_validation"]["passed"]
            ),
        }
        (args.output / "scene-validation.json").write_text(
            json.dumps(scene_validation, indent=2) + "\n"
        )


def main() -> None:
    """Generate tags when called by project Python, otherwise render in Blender."""
    args = parse_args(sys.argv[1:])
    if args.make_tags:
        if not args.tag_dir or args.tag_ids is None:
            raise ValueError("internal tag helper arguments missing")
        make_and_verify_tags(args.tag_dir, [int(x) for x in args.tag_ids.split(",")])
    else:
        try:
            import bpy  # type: ignore # noqa: F401
        except ImportError as exc:
            raise SystemExit("run through Blender --background --python") from exc
        run_blender(args)


if __name__ == "__main__":
    main()
