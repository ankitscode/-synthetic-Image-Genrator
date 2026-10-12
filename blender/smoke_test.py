"""Build step 1 smoke test: one MPFB human with skin, eyes and eyebrows -> one 256x256 PNG.

Run headless:
    /snap/bin/blender --background --python blender/smoke_test.py

Asset names come from output/smoke/mpfb_inspect.txt (blender/inspect_mpfb.py).
Writes output/smoke/face.png.
"""

import sys
from pathlib import Path

import bpy
from mathutils import Vector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "blender"))
sys.path.insert(0, str(PROJECT_ROOT))

from human import load_mpfb, service  # noqa: E402

OUT_FILE = PROJECT_ROOT / "output" / "smoke" / "face.png"

# Asset names exactly as printed by inspect_mpfb.py (makehuman_system_assets, CC0)
SKIN = "young_african_male"
EYES = "high-poly"  # default eye material in its .mhclo is brown.mhmat
EYEBROWS = "eyebrow001"

# Macro overrides on top of TargetService.get_default_macro_info_dict()
GENDER = 1.0  # male, to match the skin

RESOLUTION = 256
SAMPLES = 48
FOCAL_MM = 35.0
CAMERA_DISTANCE_M = 0.35  # from the midpoint between the eyes
KEY_LIGHT_POWER_W = 30.0
KEY_LIGHT_OFFSET_M = Vector((0.6, -1.0, 0.4))  # to the side, in front, above the eyes
WORLD_STRENGTH = 0.4


def find_asset(list_fn, subdir: str, stem: str) -> str:
    for path in list_fn(subdir):
        if path.stem == stem:
            return str(path)
    raise FileNotFoundError(f"MPFB asset '{stem}' not found in '{subdir}'")


def clear_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.cameras, bpy.data.lights):
        for block in list(coll):
            coll.remove(block)


def world_centroid(obj: bpy.types.Object) -> Vector:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    total = Vector((0.0, 0.0, 0.0))
    for vert in mesh.vertices:
        total += evaluated.matrix_world @ vert.co
    centroid = total / len(mesh.vertices)
    evaluated.to_mesh_clear()
    return centroid


def aim(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def build_human():
    AssetService = service("assetservice", "AssetService")
    HumanService = service("humanservice", "HumanService")
    TargetService = service("targetservice", "TargetService")

    macros = TargetService.get_default_macro_info_dict()
    macros["gender"] = GENDER
    basemesh = HumanService.create_human(macro_detail_dict=macros)

    skin = find_asset(AssetService.list_mhmat_assets, "skins", SKIN)
    HumanService.set_character_skin(skin, basemesh)

    eyes_file = find_asset(AssetService.list_mhclo_assets, "eyes", EYES)
    eyes = HumanService.add_mhclo_asset(eyes_file, basemesh, asset_type="Eyes", set_up_rigging=False)

    brows_file = find_asset(AssetService.list_mhclo_assets, "eyebrows", EYEBROWS)
    HumanService.add_mhclo_asset(brows_file, basemesh, asset_type="Eyebrows", set_up_rigging=False)

    return basemesh, eyes


def setup_scene(eye_midpoint: Vector) -> None:
    scene = bpy.context.scene

    # MPFB humans face -Y, so the camera sits on -Y looking back at the face.
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = FOCAL_MM
    camera = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(camera)
    camera.location = eye_midpoint + Vector((0.0, -CAMERA_DISTANCE_M, 0.0))
    aim(camera, eye_midpoint)
    scene.camera = camera

    light_data = bpy.data.lights.new("Key", type="AREA")
    light_data.energy = KEY_LIGHT_POWER_W
    light = bpy.data.objects.new("Key", light_data)
    scene.collection.objects.link(light)
    light.location = eye_midpoint + KEY_LIGHT_OFFSET_M
    aim(light, eye_midpoint)

    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    background = scene.world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    background.inputs["Strength"].default_value = WORLD_STRENGTH

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = True
    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(OUT_FILE)


def main() -> None:
    load_mpfb()
    clear_scene()
    _, eyes = build_human()
    eye_midpoint = world_centroid(eyes)
    setup_scene(eye_midpoint)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.render.render(write_still=True)
    print(f"Rendered {OUT_FILE}")


main()
