from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import bmesh
import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Euler, Matrix, Vector

# ============================================================
# Configuration
# ============================================================


@dataclass
class DatasetConfig:
    """Configuration for synthetic AprilTag dataset generation.

    Attributes:
        dataset_root: Output root directory.
        tag_image_dir: Directory containing AprilTag images.
        construct_scene_only: If True, only builds one scene and does not render/save.
        num_sequences: Number of sequences to generate.
        frames_per_sequence: Number of frames in each sequence.
        fps: Frames per second.
        image_width: Render width.
        image_height: Render height.
        render_engine: Blender render engine.
        cycles_samples: Cycles sample count if Cycles is used.
        seed: Random seed.
        room_width: Room width in meters.
        room_depth: Room depth in meters.
        room_height: Room height in meters.
        min_tags: Minimum number of tags per scene.
        max_tags: Maximum number of tags per scene.
        tag_size_m: Fixed physical size of every tag in meters.
        max_tag_tilt_deg: Maximum tag tilt away from mounting surface.
        tag_surface_gap_m: Minimum gap between tag plane center and its mounting surface.
        min_clutter: Minimum clutter objects.
        max_clutter: Maximum clutter objects.
        camera_keyframes: Number of camera keyframes.
        enable_motion_blur: Whether to enable motion blur.
        enable_dof: Whether to enable depth of field.
        render_batch_size: Number of frames to render per batch. 1 is most responsive;
            larger values reduce render-operator overhead. Values >= frames_per_sequence
            render each sequence in one batch.
        write_blend_copy: Whether to save a .blend copy per rendered sequence.
    """

    dataset_root: str = r"E:\Ceph-Mirror\Python-Files\Projects\FIRST-Note-Detection\apriltag_benchmark_data"
    tag_image_dir: str = r"E:\Ceph-Mirror\Python-Files\Projects\FIRST-Note-Detection\src\webui\assets\apriltags"

    construct_scene_only: bool = False

    num_sequences: int = 10
    frames_per_sequence: int = 240
    fps: int = 30

    image_width: int = 1920
    image_height: int = 1080

    render_engine: str = "BLENDER_EEVEE"
    cycles_samples: int = 48
    seed: int = 42

    room_width: float = 7.0
    room_depth: float = 14.0
    room_height: float = 3.0

    min_tags: int = 18
    max_tags: int = 32
    tag_size_m: float = 0.24
    max_tag_tilt_deg: float = 6.0
    tag_surface_gap_m: float = 0.03

    min_clutter: int = 18
    max_clutter: int = 45

    camera_keyframes: int = 6
    enable_motion_blur: bool = False
    enable_dof: bool = False
    render_batch_size: int = 50
    write_blend_copy: bool = False


@dataclass
class AprilTagAsset:
    """Metadata for one AprilTag source image.

    Attributes:
        path: Path to the tag image.
        family: Tag family string, for example tag36h11.
        tag_id: Numeric tag id.
        source_width_px: Source image width in pixels.
        source_height_px: Source image height in pixels.
    """

    path: Path
    family: str
    tag_id: int
    source_width_px: int
    source_height_px: int


CONFIG = DatasetConfig()

# ============================================================
# Basic helpers
# ============================================================


def ensure_dir(path: Path) -> None:
    """Creates a directory if it does not exist.

    Args:
        path: Directory path.
    """
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    """Sets the random seed.

    Args:
        seed: Random seed.
    """
    random.seed(seed)


def rand_color(
    min_v: float = 0.05, max_v: float = 0.9
) -> tuple[float, float, float, float]:
    """Generates a random RGBA color.

    Args:
        min_v: Minimum channel value.
        max_v: Maximum channel value.

    Returns:
        RGBA color.
    """
    return (
        random.uniform(min_v, max_v),
        random.uniform(min_v, max_v),
        random.uniform(min_v, max_v),
        1.0,
    )


def pastel_color(
    min_v: float = 0.58,
    max_v: float = 0.95,
    mix_with_white: float = 0.45,
) -> tuple[float, float, float, float]:
    """Generates a soft pastel RGBA color.

    Args:
        min_v: Minimum random channel value before pastel mixing.
        max_v: Maximum random channel value before pastel mixing.
        mix_with_white: Amount to blend toward white.

    Returns:
        Pastel RGBA color.
    """
    color = rand_color(min_v, max_v)
    return (
        color[0] * (1.0 - mix_with_white) + mix_with_white,
        color[1] * (1.0 - mix_with_white) + mix_with_white,
        color[2] * (1.0 - mix_with_white) + mix_with_white,
        1.0,
    )


def safe_set_input(
    node: bpy.types.Node,
    input_names: list[str],
    value: float | tuple[float, float, float, float],
) -> None:
    """Sets the first matching node input.

    Args:
        node: Blender node.
        input_names: Candidate input names.
        value: Value to assign.
    """
    for input_name in input_names:
        if input_name in node.inputs:
            node.inputs[input_name].default_value = value
            return


def clear_scene() -> None:
    """Clears the current scene and purges unused datablocks."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for collection in list(bpy.data.collections):
        if collection.users == 0:
            bpy.data.collections.remove(collection)

    datablock_groups = (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.lights,
        bpy.data.cameras,
        bpy.data.curves,
        bpy.data.textures,
    )

    for datablock_group in datablock_groups:
        for datablock in list(datablock_group):
            if datablock.users == 0:
                datablock_group.remove(datablock)

    try:
        bpy.ops.outliner.orphans_purge(do_recursive=True)
    except Exception:
        pass


def object_collection_link(
    obj: bpy.types.Object,
    collection: bpy.types.Collection | None = None,
) -> None:
    """Links an object into a collection if needed.

    Args:
        obj: Object to link.
        collection: Target collection. Defaults to the scene collection.
    """
    target = collection or bpy.context.scene.collection
    if obj.name not in target.objects:
        target.objects.link(obj)


# ============================================================
# AprilTag asset handling
# ============================================================


def parse_apriltag_filename(path: Path) -> tuple[str, int] | None:
    """Parses an AprilTag filename like tag36_11_00004.webp.

    Args:
        path: File path.

    Returns:
        Tuple of (family, tag_id), or None if invalid.
    """
    parts = path.stem.split("_")
    if len(parts) != 3:
        return None
    if not parts[0].startswith("tag"):
        return None

    try:
        family = f"{parts[0]}h{parts[1]}"
        tag_id = int(parts[2])
    except ValueError:
        return None

    return family, tag_id


def load_apriltag_assets(tag_dir: Path) -> list[AprilTagAsset]:
    """Loads AprilTag assets from disk.

    Expects names like:
        tag36_11_00000.webp

    Args:
        tag_dir: Directory containing the tag images.

    Returns:
        Loaded AprilTag assets.

    Raises:
        RuntimeError: If no valid tag files are found.
    """
    if not tag_dir.exists():
        raise RuntimeError(f"Tag directory does not exist: {tag_dir}")

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    assets: list[AprilTagAsset] = []

    for path in sorted(tag_dir.iterdir()):
        if path.suffix.lower() not in exts:
            continue

        parsed = parse_apriltag_filename(path)
        if parsed is None:
            continue

        family, tag_id = parsed
        image = bpy.data.images.load(str(path), check_existing=True)
        width_px, height_px = image.size

        assets.append(
            AprilTagAsset(
                path=path,
                family=family,
                tag_id=tag_id,
                source_width_px=int(width_px),
                source_height_px=int(height_px),
            )
        )

    if not assets:
        raise RuntimeError(f"No valid AprilTag images found in: {tag_dir}")

    return assets


# ============================================================
# Render and compositor
# ============================================================


def set_render_engine(scene: bpy.types.Scene, engine_name: str) -> None:
    """Sets the render engine with a fallback for Eevee naming differences.

    Args:
        scene: Active scene.
        engine_name: Preferred engine name.
    """
    if engine_name == "BLENDER_EEVEE":
        try:
            scene.render.engine = "BLENDER_EEVEE"
            return
        except Exception:
            scene.render.engine = "BLENDER_EEVEE_NEXT"
            return

    scene.render.engine = engine_name


def setup_compositor(scene: bpy.types.Scene) -> None:
    """Sets up a small compositor chain for realism.

    Args:
        scene: Active Blender scene.
    """
    if not hasattr(scene, "use_nodes"):
        print(
            "Warning: this Blender context does not support scene compositor nodes; skipping compositor setup."
        )
        return

    scene.use_nodes = True
    node_tree = getattr(scene, "node_tree", None)
    if node_tree is None:
        print(
            "Warning: scene compositor node tree is unavailable; skipping compositor setup."
        )
        return

    nodes = node_tree.nodes
    links = node_tree.links
    nodes.clear()

    render_layers = nodes.new(type="CompositorNodeRLayers")
    composite = nodes.new(type="CompositorNodeComposite")
    current_output = render_layers.outputs[0]

    lens = nodes.new(type="CompositorNodeLensdist")
    safe_set_input(lens, ["Distort", "Distortion"], random.uniform(-0.01, 0.015))
    safe_set_input(lens, ["Dispersion"], random.uniform(0.0, 0.01))
    links.new(current_output, lens.inputs[0])
    current_output = lens.outputs[0]

    blur = nodes.new(type="CompositorNodeBlur")
    blur.filter_type = "GAUSS"
    blur.size_x = random.randint(0, 1)
    blur.size_y = random.randint(0, 1)
    links.new(current_output, blur.inputs[0])
    current_output = blur.outputs[0]

    texture_node = nodes.new(type="CompositorNodeTexture")
    grain_texture = bpy.data.textures.new(name="DatasetGrain", type="CLOUDS")
    grain_texture.noise_scale = random.uniform(0.05, 0.25)
    grain_texture.noise_depth = 2
    texture_node.texture = grain_texture

    mix = nodes.new(type="CompositorNodeMixRGB")
    mix.blend_type = "ADD"
    mix.inputs[0].default_value = random.uniform(0.004, 0.02)

    links.new(current_output, mix.inputs[1])
    links.new(texture_node.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], composite.inputs[0])


def set_image_output_settings(
    scene: bpy.types.Scene, preferred_format: str = "PNG"
) -> str:
    """Sets render image output settings when the active Blender build supports them.

    Some Blender contexts expose only movie output formats (for example FFMPEG) on
    ``scene.render.image_settings.file_format``. Assigning PNG in that case raises
    a TypeError before a scene can even be constructed. Keep scene construction and
    rendering usable by selecting PNG when available and otherwise leaving the
    current supported format in place.
    """
    image_settings = scene.render.image_settings
    file_format_property = image_settings.bl_rna.properties["file_format"]
    available_formats = {item.identifier for item in file_format_property.enum_items}
    static_formats = {item.identifier for item in file_format_property.enum_items_static}

    if preferred_format in available_formats:
        # Work around Blender contexts where the dynamic enum lists PNG but the
        # current file-format enum accepts only FFMPEG until the filepath suffix
        # makes Blender refresh the image-output enum.
        scene.render.filepath = str(Path(scene.render.filepath or "render.png").with_suffix(".png"))
        image_settings.file_format = preferred_format
    else:
        raise RuntimeError(
            f"Unable to set render output to {preferred_format!r}. "
            f"Dynamic formats: {sorted(available_formats)}. Static formats: {sorted(static_formats)}. "
            "Refusing to fall back to FFMPEG because this script must write PNG image sequences."
        )

    if "color_mode" in image_settings.bl_rna.properties:
        available_color_modes = {
            item.identifier
            for item in image_settings.bl_rna.properties["color_mode"].enum_items
        }
        if "RGB" in available_color_modes:
            image_settings.color_mode = "RGB"

    return image_settings.file_format


def configure_render(scene: bpy.types.Scene, cfg: DatasetConfig) -> None:
    """Configures render settings.

    Args:
        scene: Active scene.
        cfg: Dataset configuration.
    """
    set_render_engine(scene, cfg.render_engine)

    scene.render.resolution_x = cfg.image_width
    scene.render.resolution_y = cfg.image_height
    scene.render.resolution_percentage = 100
    scene.render.fps = cfg.fps
    set_image_output_settings(scene, "PNG")
    scene.render.film_transparent = False

    scene.frame_start = 1
    scene.frame_end = cfg.frames_per_sequence

    scene.view_settings.exposure = random.uniform(-0.5, 0.8)
    scene.view_settings.gamma = random.uniform(0.95, 1.05)

    setup_compositor(scene)

    if scene.render.engine == "CYCLES":
        scene.cycles.samples = cfg.cycles_samples
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = 0.03
        scene.cycles.max_bounces = 4
        scene.cycles.use_denoising = True
    else:
        eevee = getattr(scene, "eevee", None)
        if eevee is not None:
            if hasattr(eevee, "taa_render_samples"):
                eevee.taa_render_samples = 64
            if hasattr(eevee, "use_gtao"):
                eevee.use_gtao = True
            if hasattr(eevee, "use_bloom"):
                eevee.use_bloom = False

            ray_tracing = getattr(eevee, "ray_tracing", None)
            if ray_tracing is not None:
                if hasattr(ray_tracing, "trace_max_roughness"):
                    ray_tracing.trace_max_roughness = 1.0
                if hasattr(ray_tracing, "resolution_scale"):
                    ray_tracing.resolution_scale = "1"
                if hasattr(ray_tracing, "screen_trace_quality"):
                    ray_tracing.screen_trace_quality = 1.0
                if hasattr(ray_tracing, "screen_trace_thickness"):
                    ray_tracing.screen_trace_thickness = 1.0
                if hasattr(ray_tracing, "use_denoise"):
                    ray_tracing.use_denoise = False

    scene.render.use_motion_blur = False


# ============================================================
# Materials
# ============================================================


def create_principled_material(
    name: str,
    base_color: tuple[float, float, float, float],
    roughness: float = 0.5,
    metallic: float = 0.0,
    add_noise: bool = True,
) -> bpy.types.Material:
    """Creates a procedural Principled material.

    Args:
        name: Material name.
        base_color: Base RGBA color.
        roughness: Roughness.
        metallic: Metallic value.
        add_noise: Whether to add slight procedural variation.

    Returns:
        Material.
    """
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new(type="ShaderNodeOutputMaterial")
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")

    safe_set_input(bsdf, ["Base Color"], base_color)
    safe_set_input(bsdf, ["Roughness"], roughness)
    safe_set_input(bsdf, ["Metallic"], metallic)

    if add_noise:
        texcoord = nodes.new(type="ShaderNodeTexCoord")
        mapping = nodes.new(type="ShaderNodeMapping")
        noise = nodes.new(type="ShaderNodeTexNoise")
        ramp = nodes.new(type="ShaderNodeValToRGB")
        mix = nodes.new(type="ShaderNodeMixRGB")

        noise.inputs["Scale"].default_value = random.uniform(2.0, 12.0)
        noise.inputs["Detail"].default_value = random.uniform(3.0, 10.0)
        mix.blend_type = "MIX"
        mix.inputs[0].default_value = random.uniform(0.18, 0.38)

        darker = (
            max(0.0, base_color[0] * random.uniform(0.72, 0.88)),
            max(0.0, base_color[1] * random.uniform(0.72, 0.88)),
            max(0.0, base_color[2] * random.uniform(0.72, 0.88)),
            1.0,
        )
        lighter = (
            min(
                1.0, base_color[0] + (1.0 - base_color[0]) * random.uniform(0.25, 0.55)
            ),
            min(
                1.0, base_color[1] + (1.0 - base_color[1]) * random.uniform(0.25, 0.55)
            ),
            min(
                1.0, base_color[2] + (1.0 - base_color[2]) * random.uniform(0.25, 0.55)
            ),
            1.0,
        )
        ramp.color_ramp.elements[0].color = darker
        ramp.color_ramp.elements[1].color = lighter

        links.new(texcoord.outputs["Object"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
        links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
        mix.inputs[1].default_value = base_color
        links.new(ramp.outputs["Color"], mix.inputs[2])
        links.new(mix.outputs[0], bsdf.inputs["Base Color"])

    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def create_tag_material(asset: AprilTagAsset, name: str) -> bpy.types.Material:
    """Creates a pure black/white tag material.

    This uses the original tiny source image directly, with nearest-neighbor
    sampling and a constant color ramp so each source pixel stays a hard black
    or white square.

    Args:
        asset: AprilTag asset.
        name: Material name.

    Returns:
        Material.
    """
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    material.use_backface_culling = False

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new(type="ShaderNodeOutputMaterial")
    texcoord = nodes.new(type="ShaderNodeTexCoord")
    image_texture = nodes.new(type="ShaderNodeTexImage")
    rgb_to_bw = nodes.new(type="ShaderNodeRGBToBW")
    ramp = nodes.new(type="ShaderNodeValToRGB")
    emission = nodes.new(type="ShaderNodeEmission")

    image = bpy.data.images.load(str(asset.path), check_existing=True)
    try:
        image.colorspace_settings.name = "Non-Color"
    except Exception:
        pass

    image_texture.image = image
    image_texture.interpolation = "Closest"
    image_texture.extension = "CLIP"

    ramp.color_ramp.interpolation = "CONSTANT"
    ramp.color_ramp.elements[0].position = 0.5
    ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    ramp.color_ramp.elements[1].position = 0.5
    ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    emission.inputs["Strength"].default_value = 1.0

    links.new(texcoord.outputs["UV"], image_texture.inputs["Vector"])
    links.new(image_texture.outputs["Color"], rgb_to_bw.inputs["Color"])
    links.new(rgb_to_bw.outputs["Val"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])

    return material


# ============================================================
# Scene geometry
# ============================================================


def add_plane(
    name: str,
    size_x: float,
    size_y: float,
    location: Vector,
    rotation: Euler,
    material: bpy.types.Material,
) -> bpy.types.Object:
    """Creates a rectangular plane.

    Args:
        name: Object name.
        size_x: Plane X size.
        size_y: Plane Y size.
        location: Plane location.
        rotation: Plane rotation.
        material: Material to assign.

    Returns:
        Created object.
    """
    mesh = bpy.data.meshes.new(name=f"{name}_Mesh")
    obj = bpy.data.objects.new(name, mesh)
    object_collection_link(obj)

    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5)
    bm.to_mesh(mesh)
    bm.free()

    obj.location = location
    obj.rotation_euler = rotation
    obj.scale = Vector((size_x, size_y, 1.0))
    obj.data.materials.append(material)

    return obj


def build_room(cfg: DatasetConfig) -> dict[str, bpy.types.Object]:
    """Builds a simple room.

    Args:
        cfg: Dataset configuration.

    Returns:
        Mapping of room surface names to objects.
    """
    floor_mat = create_principled_material(
        "FloorMat", pastel_color(0.50, 0.80), roughness=0.8
    )
    wall_mats = {
        side: create_principled_material(
            f"WallMat_{side.capitalize()}",
            pastel_color(0.58, 0.90),
            roughness=0.9,
        )
        for side in ("north", "south", "east", "west")
    }
    ceil_mat = create_principled_material(
        "CeilMat", pastel_color(0.70, 0.98), roughness=0.95
    )

    surfaces = {
        "floor": add_plane(
            "Floor",
            cfg.room_width,
            cfg.room_depth,
            Vector((0.0, 0.0, 0.0)),
            Euler((0.0, 0.0, 0.0)),
            floor_mat,
        ),
        "ceiling": add_plane(
            "Ceiling",
            cfg.room_width,
            cfg.room_depth,
            Vector((0.0, 0.0, cfg.room_height)),
            Euler((math.pi, 0.0, 0.0)),
            ceil_mat,
        ),
        "north": add_plane(
            "Wall_North",
            cfg.room_width,
            cfg.room_height,
            Vector((0.0, cfg.room_depth / 2.0, cfg.room_height / 2.0)),
            Euler((math.pi / 2.0, 0.0, 0.0)),
            wall_mats["north"],
        ),
        "south": add_plane(
            "Wall_South",
            cfg.room_width,
            cfg.room_height,
            Vector((0.0, -cfg.room_depth / 2.0, cfg.room_height / 2.0)),
            Euler((-math.pi / 2.0, 0.0, 0.0)),
            wall_mats["south"],
        ),
        "east": add_plane(
            "Wall_East",
            cfg.room_height,
            cfg.room_depth,
            Vector((cfg.room_width / 2.0, 0.0, cfg.room_height / 2.0)),
            Euler((0.0, -math.pi / 2.0, 0.0)),
            wall_mats["east"],
        ),
        "west": add_plane(
            "Wall_West",
            cfg.room_height,
            cfg.room_depth,
            Vector((-cfg.room_width / 2.0, 0.0, cfg.room_height / 2.0)),
            Euler((0.0, math.pi / 2.0, 0.0)),
            wall_mats["west"],
        ),
    }

    bpy.context.scene.world.use_nodes = True
    bg = bpy.context.scene.world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs["Color"].default_value = pastel_color(0.45, 0.75, mix_with_white=0.25)
        bg.inputs["Strength"].default_value = random.uniform(0.2, 0.7)

    return surfaces


def add_light_rig(cfg: DatasetConfig) -> None:
    """Adds randomized lights.

    Args:
        cfg: Dataset configuration.
    """
    area_data = bpy.data.lights.new(name="AreaLight", type="AREA")
    area_data.energy = 1.0
    area_data.shape = "RECTANGLE"
    area_data.size = random.uniform(1.0, 2.5)
    area_data.size_y = random.uniform(1.0, 3.0)

    area_obj = bpy.data.objects.new("AreaLight", area_data)
    object_collection_link(area_obj)
    area_obj.location = Vector(
        (
            random.uniform(-cfg.room_width * 0.25, cfg.room_width * 0.25),
            random.uniform(-cfg.room_depth * 0.25, cfg.room_depth * 0.25),
            random.uniform(cfg.room_height * 0.75, cfg.room_height * 0.98),
        )
    )
    area_obj.rotation_euler = Euler(
        (
            random.uniform(math.radians(5), math.radians(35)),
            0.0,
            random.uniform(-math.pi, math.pi),
        )
    )

    sun_data = bpy.data.lights.new(name="SunLight", type="SUN")
    sun_data.energy = 1.0

    sun_obj = bpy.data.objects.new("SunLight", sun_data)
    object_collection_link(sun_obj)
    sun_obj.rotation_euler = Euler(
        (
            random.uniform(math.radians(20), math.radians(70)),
            random.uniform(math.radians(-20), math.radians(20)),
            random.uniform(-math.pi, math.pi),
        )
    )

    for idx in range(random.randint(1, 3)):
        point_data = bpy.data.lights.new(name=f"PointLight_{idx}", type="POINT")
        point_data.energy = 1.0
        point_data.shadow_soft_size = random.uniform(0.05, 0.5)

        point_obj = bpy.data.objects.new(f"PointLight_{idx}", point_data)
        object_collection_link(point_obj)
        point_obj.location = Vector(
            (
                random.uniform(-cfg.room_width * 0.45, cfg.room_width * 0.45),
                random.uniform(-cfg.room_depth * 0.45, cfg.room_depth * 0.45),
                random.uniform(0.2, cfg.room_height * 0.9),
            )
        )


def add_clutter(cfg: DatasetConfig) -> list[bpy.types.Object]:
    """Adds random clutter objects.

    Args:
        cfg: Dataset configuration.

    Returns:
        Created clutter objects.
    """
    clutter: list[bpy.types.Object] = []
    count = random.randint(cfg.min_clutter, cfg.max_clutter)

    for idx in range(count):
        shape = random.choice(["CUBE", "UVSPHERE", "CYLINDER", "CONE", "ICO"])

        if shape == "CUBE":
            bpy.ops.mesh.primitive_cube_add(size=1.0)
        elif shape == "UVSPHERE":
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, segments=24, ring_count=12)
        elif shape == "CYLINDER":
            bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.5, depth=1.0)
        elif shape == "CONE":
            bpy.ops.mesh.primitive_cone_add(vertices=24, radius1=0.5, depth=1.0)
        else:
            bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=0.5)

        obj = bpy.context.active_object
        obj.name = f"Clutter_{idx}"

        sx = random.uniform(0.1, 0.8)
        sy = random.uniform(0.1, 0.8)
        sz = random.uniform(0.1, 1.4)

        obj.scale = Vector((sx, sy, sz))
        obj.location = Vector(
            (
                random.uniform(-cfg.room_width * 0.42, cfg.room_width * 0.42),
                random.uniform(-cfg.room_depth * 0.42, cfg.room_depth * 0.42),
                max(0.01, sz * 0.5),
            )
        )
        obj.rotation_euler = Euler(
            (
                random.uniform(-0.5, 0.5),
                random.uniform(-0.5, 0.5),
                random.uniform(-math.pi, math.pi),
            )
        )

        mat = create_principled_material(
            f"ClutterMat_{idx}",
            pastel_color(0.45, 0.92),
            roughness=random.uniform(0.2, 0.95),
            metallic=random.uniform(0.0, 0.25),
        )
        obj.data.materials.append(mat)
        clutter.append(obj)

    return clutter


def surface_anchor_position(
    surface: str,
    cfg: DatasetConfig,
    tag_size_m: float,
) -> tuple[Vector, Vector]:
    """Chooses an anchor point and inward normal for a mounting surface.

    Args:
        surface: Surface name.
        cfg: Dataset configuration.
        tag_size_m: Physical tag size.

    Returns:
        Tuple of (anchor position, inward normal).
    """
    margin = tag_size_m * 1.35

    if surface == "north":
        return (
            Vector(
                (
                    random.uniform(
                        -cfg.room_width / 2.0 + margin, cfg.room_width / 2.0 - margin
                    ),
                    cfg.room_depth / 2.0,
                    random.uniform(margin, cfg.room_height - margin),
                )
            ),
            Vector((0.0, -1.0, 0.0)),
        )

    if surface == "south":
        return (
            Vector(
                (
                    random.uniform(
                        -cfg.room_width / 2.0 + margin, cfg.room_width / 2.0 - margin
                    ),
                    -cfg.room_depth / 2.0,
                    random.uniform(margin, cfg.room_height - margin),
                )
            ),
            Vector((0.0, 1.0, 0.0)),
        )

    if surface == "east":
        return (
            Vector(
                (
                    cfg.room_width / 2.0,
                    random.uniform(
                        -cfg.room_depth / 2.0 + margin, cfg.room_depth / 2.0 - margin
                    ),
                    random.uniform(margin, cfg.room_height - margin),
                )
            ),
            Vector((-1.0, 0.0, 0.0)),
        )

    if surface == "west":
        return (
            Vector(
                (
                    -cfg.room_width / 2.0,
                    random.uniform(
                        -cfg.room_depth / 2.0 + margin, cfg.room_depth / 2.0 - margin
                    ),
                    random.uniform(margin, cfg.room_height - margin),
                )
            ),
            Vector((1.0, 0.0, 0.0)),
        )

    return (
        Vector(
            (
                random.uniform(
                    -cfg.room_width / 2.0 + margin, cfg.room_width / 2.0 - margin
                ),
                random.uniform(
                    -cfg.room_depth / 2.0 + margin, cfg.room_depth / 2.0 - margin
                ),
                0.0,
            )
        ),
        Vector((0.0, 0.0, 1.0)),
    )


def spawn_tag(
    tag_index: int,
    asset: AprilTagAsset,
    cfg: DatasetConfig,
) -> bpy.types.Object:
    """Spawns a fixed-size AprilTag plane outside the wall/floor.

    Fixes:
    - All tags are the same size.
    - Tags are offset from walls/floor using the mounting surface normal.
    - Tags use the original tag image with hard nearest-neighbor black/white.

    Args:
        tag_index: Scene-local tag index.
        asset: AprilTag asset.
        cfg: Dataset configuration.

    Returns:
        Created tag object.
    """
    del tag_index

    bpy.ops.mesh.primitive_plane_add(size=1.0)
    obj = bpy.context.active_object
    obj.name = f"{asset.family}_{asset.tag_id:05d}"
    obj.scale = Vector((cfg.tag_size_m, cfg.tag_size_m, 1.0))

    surface = random.choices(
        ["north", "south", "east", "west", "floor"],
        weights=[3.0, 3.0, 2.5, 2.5, 0.6],
        k=1,
    )[0]
    anchor_pos, inward_normal = surface_anchor_position(surface, cfg, cfg.tag_size_m)

    base_quat = inward_normal.to_track_quat("Z", "Y")
    obj.rotation_euler = base_quat.to_euler()

    tilt_x = math.radians(random.uniform(-cfg.max_tag_tilt_deg, cfg.max_tag_tilt_deg))
    tilt_y = math.radians(random.uniform(-cfg.max_tag_tilt_deg, cfg.max_tag_tilt_deg))
    yaw_z = math.radians(random.uniform(-20.0, 20.0))

    obj.rotation_euler.rotate_axis("X", tilt_x)
    obj.rotation_euler.rotate_axis("Y", tilt_y)
    obj.rotation_euler.rotate_axis("Z", yaw_z)

    # Conservative clearance to keep corners out of the wall/floor.
    max_tilt_rad = math.radians(cfg.max_tag_tilt_deg)
    tilt_clearance = 0.5 * cfg.tag_size_m * math.sin(max_tilt_rad)
    total_gap = cfg.tag_surface_gap_m + tilt_clearance + 0.002

    # Push strictly along the mounting surface normal into the room.
    obj.location = anchor_pos + inward_normal * total_gap

    material = create_tag_material(
        asset, f"TagMaterial_{asset.family}_{asset.tag_id:05d}"
    )
    obj.data.materials.append(material)

    obj["tag_family"] = asset.family
    obj["tag_id"] = asset.tag_id
    obj["tag_image"] = asset.path.name
    obj["tag_size_m"] = cfg.tag_size_m
    obj["source_width_px"] = asset.source_width_px
    obj["source_height_px"] = asset.source_height_px
    obj["surface_name"] = surface

    return obj


# ============================================================
# Camera
# ============================================================


def create_camera(cfg: DatasetConfig) -> tuple[bpy.types.Object, bpy.types.Object]:
    """Creates a camera and a look target.

    Args:
        cfg: Dataset configuration.

    Returns:
        Tuple of (camera object, target object).
    """
    camera_data = bpy.data.cameras.new(name="DatasetCamera")
    camera = bpy.data.objects.new("DatasetCamera", camera_data)
    object_collection_link(camera)

    target = bpy.data.objects.new("CameraTarget", None)
    target.empty_display_type = "PLAIN_AXES"
    object_collection_link(target)

    camera_data.lens = random.uniform(18.0, 30.0)
    camera_data.sensor_width = random.uniform(32.0, 36.0)
    camera_data.clip_start = 0.01
    camera_data.clip_end = 100.0

    constraint = camera.constraints.new(type="TRACK_TO")
    constraint.target = target
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"

    bpy.context.scene.camera = camera
    animate_camera_and_target(camera, target, cfg)

    return camera, target


def animate_camera_and_target(
    camera: bpy.types.Object,
    target: bpy.types.Object,
    cfg: DatasetConfig,
) -> None:
    """Animates the camera and its target.

    Uses a long, mostly linear dolly path down the room with a smoothly moving
    look target. This avoids rapid orbiting/panning while keeping wall-mounted
    tags in view for most frames.

    Args:
        camera: Camera object.
        target: Target object.
        cfg: Dataset configuration.
    """
    keyframes = max(2, cfg.camera_keyframes)
    frame_numbers = [
        int(1 + index * (cfg.frames_per_sequence - 1) / (keyframes - 1))
        for index in range(keyframes)
    ]

    travel_y = cfg.room_depth * random.uniform(0.66, 0.82)
    start_y = -travel_y * 0.5
    end_y = travel_y * 0.5
    lane_x = random.uniform(-cfg.room_width * 0.16, cfg.room_width * 0.16)
    base_z = random.uniform(0.9, cfg.room_height * 0.62)
    look_ahead = cfg.room_depth * random.uniform(0.18, 0.28)
    target_x_bias = random.uniform(-cfg.room_width * 0.08, cfg.room_width * 0.08)
    target_z = random.uniform(0.9, cfg.room_height * 0.68)

    for index, frame in enumerate(frame_numbers):
        t = index / max(1, keyframes - 1)
        y = start_y + (end_y - start_y) * t
        gentle_sway = math.sin(t * math.pi * 1.5 + random.uniform(-0.25, 0.25))

        camera.location = Vector(
            (
                lane_x + gentle_sway * cfg.room_width * random.uniform(0.025, 0.06),
                y,
                base_z + math.sin(t * math.pi) * random.uniform(-0.08, 0.12),
            )
        )

        target.location = Vector(
            (
                target_x_bias
                + gentle_sway * cfg.room_width * random.uniform(0.04, 0.09),
                max(-cfg.room_depth * 0.44, min(cfg.room_depth * 0.44, y + look_ahead)),
                target_z + math.sin(t * math.pi * 1.2) * random.uniform(-0.08, 0.12),
            )
        )

        camera.keyframe_insert(data_path="location", frame=frame)
        target.keyframe_insert(data_path="location", frame=frame)


# ============================================================
# Metadata
# ============================================================


def get_camera_intrinsics(
    scene: bpy.types.Scene,
    camera_obj: bpy.types.Object,
) -> dict[str, float]:
    """Computes approximate pinhole intrinsics.

    Args:
        scene: Active scene.
        camera_obj: Camera object.

    Returns:
        Intrinsics dictionary.
    """
    cam = camera_obj.data
    width = scene.render.resolution_x * scene.render.resolution_percentage / 100.0
    height = scene.render.resolution_y * scene.render.resolution_percentage / 100.0

    sensor_width = cam.sensor_width
    if cam.sensor_fit == "VERTICAL":
        sensor_height = cam.sensor_height
    else:
        sensor_height = sensor_width * height / width

    fx = cam.lens / sensor_width * width
    fy = cam.lens / sensor_height * height

    cx = width * 0.5 - cam.shift_x * width
    cy = height * 0.5 + cam.shift_y * height

    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "width": float(width),
        "height": float(height),
    }


def matrix_to_list(matrix: Matrix) -> list[list[float]]:
    """Converts a Blender matrix to nested Python lists.

    Args:
        matrix: Matrix.

    Returns:
        Nested list representation.
    """
    return [[float(value) for value in row] for row in matrix]


def tag_world_corners(tag_obj: bpy.types.Object) -> list[Vector]:
    """Gets world-space corners of a tag plane.

    Args:
        tag_obj: Tag object.

    Returns:
        Four world-space corners.
    """
    half = 0.5
    local_corners = [
        Vector((-half, -half, 0.0)),
        Vector((half, -half, 0.0)),
        Vector((half, half, 0.0)),
        Vector((-half, half, 0.0)),
    ]
    return [tag_obj.matrix_world @ point for point in local_corners]


def project_points(
    scene: bpy.types.Scene,
    camera_obj: bpy.types.Object,
    points_world: list[Vector],
) -> tuple[list[list[float]], bool]:
    """Projects world points into image pixels.

    Args:
        scene: Active scene.
        camera_obj: Camera object.
        points_world: World-space points.

    Returns:
        Tuple of (pixel coordinates, all points are in front).
    """
    width = scene.render.resolution_x
    height = scene.render.resolution_y

    points_2d: list[list[float]] = []
    in_front = True

    for point in points_world:
        ndc = world_to_camera_view(scene, camera_obj, point)
        x_px = ndc.x * width
        y_px = (1.0 - ndc.y) * height

        points_2d.append([float(x_px), float(y_px)])

        if ndc.z < 0.0:
            in_front = False

    return points_2d, in_front


def is_point_occluded(
    scene: bpy.types.Scene,
    camera_obj: bpy.types.Object,
    world_point: Vector,
    expected_obj: bpy.types.Object,
) -> bool:
    """Checks whether a point is occluded from the camera.

    Args:
        scene: Active scene.
        camera_obj: Camera object.
        world_point: Target point.
        expected_obj: Object expected at the point.

    Returns:
        True if something else blocks the point.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    origin = camera_obj.matrix_world.translation
    direction = world_point - origin
    distance = direction.length

    if distance <= 1e-6:
        return False

    direction.normalize()

    hit, _, _, _, hit_object, _ = scene.ray_cast(
        depsgraph,
        origin,
        direction,
        distance=distance - 1e-4,
    )

    if not hit:
        return False

    return hit_object != expected_obj


def collect_frame_metadata(
    scene: bpy.types.Scene,
    camera_obj: bpy.types.Object,
    tags: list[bpy.types.Object],
    frame_index: int,
) -> dict[str, Any]:
    """Collects per-frame metadata.

    Args:
        scene: Active scene.
        camera_obj: Camera object.
        tags: Tag objects.
        frame_index: Frame number.

    Returns:
        Frame metadata.
    """
    scene.frame_set(frame_index)

    frame_data: dict[str, Any] = {
        "frame": int(frame_index),
        "camera_matrix_world": matrix_to_list(camera_obj.matrix_world.copy()),
        "tags": [],
    }

    for tag in tags:
        corners_world = tag_world_corners(tag)
        corners_px, in_front = project_points(scene, camera_obj, corners_world)

        center_world = tag.matrix_world.translation.copy()
        center_ndc = world_to_camera_view(scene, camera_obj, center_world)

        in_image = (
            0.0 <= center_ndc.x <= 1.0
            and 0.0 <= center_ndc.y <= 1.0
            and center_ndc.z >= 0.0
        )

        occluded = is_point_occluded(scene, camera_obj, center_world, tag)

        frame_data["tags"].append(
            {
                "object_name": tag.name,
                "tag_family": str(tag.get("tag_family", "")),
                "tag_id": int(tag.get("tag_id", -1)),
                "tag_image": str(tag.get("tag_image", "")),
                "tag_size_m": float(tag.get("tag_size_m", 0.0)),
                "source_width_px": int(tag.get("source_width_px", 0)),
                "source_height_px": int(tag.get("source_height_px", 0)),
                "surface_name": str(tag.get("surface_name", "")),
                "matrix_world": matrix_to_list(tag.matrix_world.copy()),
                "corners_world": [
                    [float(v.x), float(v.y), float(v.z)] for v in corners_world
                ],
                "corners_image_px": corners_px,
                "center_world": [
                    float(center_world.x),
                    float(center_world.y),
                    float(center_world.z),
                ],
                "center_ndc": [
                    float(center_ndc.x),
                    float(center_ndc.y),
                    float(center_ndc.z),
                ],
                "visible": bool(in_front and in_image and not occluded),
                "occluded_center_ray": bool(occluded),
            }
        )

    return frame_data


# ============================================================
# Dataset generation
# ============================================================


def build_random_scene(
    cfg: DatasetConfig,
    tag_assets: list[AprilTagAsset],
) -> tuple[bpy.types.Object, list[bpy.types.Object]]:
    """Builds one randomized scene.

    Args:
        cfg: Dataset configuration.
        tag_assets: Loaded AprilTag assets.

    Returns:
        Tuple of (camera object, tag objects).
    """
    clear_scene()

    scene = bpy.context.scene
    configure_render(scene, cfg)

    build_room(cfg)
    add_light_rig(cfg)
    add_clutter(cfg)

    camera, _target = create_camera(cfg)

    if cfg.min_tags < 0 or cfg.max_tags < 0:
        raise ValueError("min_tags and max_tags must be non-negative")
    if cfg.min_tags > cfg.max_tags:
        raise ValueError("min_tags cannot be greater than max_tags")
    if not tag_assets and cfg.max_tags > 0:
        raise RuntimeError("Cannot place AprilTags because no tag assets were loaded")

    # Each synthesized scene gets its own random number of AprilTags. This means
    # rendered sequences naturally vary in tag density between cfg.min_tags and
    # cfg.max_tags instead of always using a fixed scene-wide count.
    requested_tag_count = random.randint(cfg.min_tags, cfg.max_tags)
    actual_tag_count = min(requested_tag_count, len(tag_assets))
    chosen_assets = random.sample(tag_assets, k=actual_tag_count)
    tags = [spawn_tag(index, asset, cfg) for index, asset in enumerate(chosen_assets)]

    scene["requested_apriltag_count"] = requested_tag_count
    scene["apriltag_count"] = len(tags)

    scene.frame_set(1)
    return camera, tags


def print_progress_bar(
    completed: int,
    total: int,
    prefix: str = "Rendering",
    width: int = 40,
) -> None:
    """Prints an in-place text progress bar.

    Args:
        completed: Number of completed frames.
        total: Total frames to process.
        prefix: Label shown before the bar.
        width: Character width of the progress bar.
    """
    if total <= 0:
        return

    completed = max(0, min(completed, total))
    fraction = completed / total
    filled = int(width * fraction)
    bar = "#" * filled + "-" * (width - filled)
    percent = fraction * 100.0
    end = "\n" if completed >= total else "\r"
    print(
        f"{prefix}: [{bar}] {completed}/{total} frames ({percent:5.1f}%)",
        end=end,
        flush=True,
    )


def collect_batch_metadata(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    tags: list[bpy.types.Object],
    images_dir: Path,
    start_frame: int,
    end_frame: int,
) -> list[dict[str, Any]]:
    """Collects metadata for a contiguous frame batch before rendering it."""
    frames_metadata: list[dict[str, Any]] = []
    for frame in range(start_frame, end_frame + 1):
        scene.frame_set(frame)
        frame_meta = collect_frame_metadata(scene, camera, tags, frame)
        frame_meta["image_path"] = str(images_dir / f"frame_{frame:05d}.png")
        frames_metadata.append(frame_meta)
    return frames_metadata


def render_frame_batch(
    scene: bpy.types.Scene,
    images_dir: Path,
    start_frame: int,
    end_frame: int,
) -> None:
    """Renders a contiguous batch as individual PNG still images."""
    original_frame = scene.frame_current
    original_filepath = scene.render.filepath
    original_file_format = scene.render.image_settings.file_format

    try:
        set_image_output_settings(scene, "PNG")
        for frame in range(start_frame, end_frame + 1):
            scene.frame_set(frame)
            scene.render.filepath = str(images_dir / f"frame_{frame:05d}.png")
            bpy.ops.render.render(write_still=True)
    finally:
        scene.frame_set(original_frame)
        scene.render.filepath = original_filepath
        try:
            scene.render.image_settings.file_format = original_file_format
        except TypeError:
            pass


def iter_frame_batches(start_frame: int, end_frame: int, batch_size: int):
    """Yields inclusive frame ranges split by batch_size."""
    batch_size = max(1, batch_size)
    frame = start_frame
    while frame <= end_frame:
        batch_end = min(frame + batch_size - 1, end_frame)
        yield frame, batch_end
        frame = batch_end + 1


def generate_sequence(
    sequence_index: int,
    cfg: DatasetConfig,
    tag_assets: list[AprilTagAsset],
    dataset_root: Path,
    completed_frames: int = 0,
    total_frames: int | None = None,
) -> int:
    """Generates one rendered sequence and metadata.

    Args:
        sequence_index: Sequence index.
        cfg: Dataset configuration.
        tag_assets: Loaded AprilTag assets.
        dataset_root: Output root.
        completed_frames: Number of frames already rendered before this sequence.
        total_frames: Total frames to render across the whole dataset.

    Returns:
        Updated number of completed rendered frames.
    """
    sequence_dir = dataset_root / f"seq_{sequence_index:04d}"
    images_dir = sequence_dir / "images"

    ensure_dir(sequence_dir)
    ensure_dir(images_dir)

    camera, tags = build_random_scene(cfg, tag_assets)
    scene = bpy.context.scene

    sequence_metadata: dict[str, Any] = {
        "sequence_index": sequence_index,
        "config": asdict(cfg),
        "requested_apriltag_count": int(
            scene.get("requested_apriltag_count", len(tags))
        ),
        "apriltag_count": len(tags),
        "intrinsics": get_camera_intrinsics(scene, camera),
        "frames": [],
    }

    if total_frames is not None:
        print_progress_bar(completed_frames, total_frames)

    batch_size = max(1, int(cfg.render_batch_size))
    for batch_start, batch_end in iter_frame_batches(
        scene.frame_start, scene.frame_end, batch_size
    ):
        sequence_metadata["frames"].extend(
            collect_batch_metadata(
                scene, camera, tags, images_dir, batch_start, batch_end
            )
        )
        render_frame_batch(scene, images_dir, batch_start, batch_end)

        completed_frames += batch_end - batch_start + 1
        if total_frames is not None:
            print_progress_bar(completed_frames, total_frames)
        else:
            print(
                f"[Sequence {sequence_index:04d}] Rendered frames {batch_start}-{batch_end}/{cfg.frames_per_sequence}"
            )

    with open(sequence_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(sequence_metadata, file, indent=2)

    if cfg.write_blend_copy:
        bpy.ops.wm.save_as_mainfile(
            filepath=str(sequence_dir / f"scene_{sequence_index:04d}.blend"),
        )

    print(f"Finished sequence {sequence_index:04d}")
    return completed_frames


def main() -> None:
    """Runs the generator."""
    cfg = CONFIG
    bpy.app.debug_value = 256
    set_seed(cfg.seed)

    tag_assets = load_apriltag_assets(Path(cfg.tag_image_dir))

    if cfg.construct_scene_only:
        print("construct_scene_only=True: building one scene only.", flush=True)
        camera, tags = build_random_scene(cfg, tag_assets)
        bpy.context.view_layer.update()
        print(
            f"Scene constructed with camera {camera.name!r} and {len(tags)} AprilTags. "
            "No frames were rendered because construct_scene_only=True.",
            flush=True,
        )
        return

    dataset_root = Path(cfg.dataset_root)
    ensure_dir(dataset_root)

    total_frames = cfg.num_sequences * cfg.frames_per_sequence
    completed_frames = 0
    print_progress_bar(completed_frames, total_frames)

    for seq_idx in range(cfg.num_sequences):
        set_seed(cfg.seed + seq_idx * 10007)
        completed_frames = generate_sequence(
            seq_idx,
            cfg,
            tag_assets,
            dataset_root,
            completed_frames=completed_frames,
            total_frames=total_frames,
        )

    print("Dataset generation complete.")


if __name__ == "__main__":
    main()
