"""Camera, key light, background and render settings. Pose comes from the camera, not the head."""

import math

import bpy
import numpy as np
from mathutils import Quaternion, Vector

from shared import constants as C

CAMERA_NAME = "SynthFaceCamera"
KEY_LIGHT_NAME = "SynthFaceKeyLight"
WORLD_NAME = "SynthFaceWorld"
LUMA_709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
FULL_FRAME_WIDTH_MM = 36.0
EYE_RAY_MAX_STEPS = 8
EYE_RAY_STEP_M = 1e-4  # nudge past a see-through hit  # only for reporting 35 mm-equivalent focal lengths


def enable_gpu(device_type: str) -> None:
    """Use every Cycles device of this type (CUDA / OPTIX / HIP...). Session only: in
    --background mode Blender does not save preferences."""
    prefs = bpy.context.preferences.addons["cycles"].preferences
    if prefs.compute_device_type == device_type and any(d.use and d.type == device_type for d in prefs.devices):
        return
    prefs.compute_device_type = device_type
    prefs.get_devices()
    found = [d for d in prefs.devices if d.type == device_type]
    if not found:
        raise RuntimeError(f"No Cycles {device_type} device found")
    for d in prefs.devices:
        d.use = d.type == device_type


def configure_render(samples: int, device: str) -> None:
    """device: "CPU" or a Cycles GPU backend ("CUDA", "OPTIX", ...)."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    if device == "CPU":
        scene.cycles.device = "CPU"
    else:
        enable_gpu(device)
        scene.cycles.device = "GPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = C.RENDER_RESOLUTION
    scene.render.resolution_y = C.RENDER_RESOLUTION
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = C.VIEW_TRANSFORM
    scene.view_settings.look = C.VIEW_LOOK
    scene.view_settings.exposure = C.VIEW_EXPOSURE


def create_rig() -> tuple[bpy.types.Object, bpy.types.Object]:
    """Camera + key light for the current identity (call after clear_scene)."""
    scene = bpy.context.scene

    cam_data = bpy.data.cameras.new(CAMERA_NAME)
    cam_data.sensor_fit = "HORIZONTAL"
    camera = bpy.data.objects.new(CAMERA_NAME, cam_data)
    camera.rotation_mode = "QUATERNION"
    scene.collection.objects.link(camera)
    scene.camera = camera

    light_data = bpy.data.lights.new(KEY_LIGHT_NAME, type="AREA")
    light_data.use_temperature = True
    light = bpy.data.objects.new(KEY_LIGHT_NAME, light_data)
    light.rotation_mode = "QUATERNION"
    scene.collection.objects.link(light)
    return camera, light


def _orbit_offset(azimuth_deg: float, elevation_deg: float, distance: float) -> Vector:
    """Offset from the face on a sphere. The face looks toward -Y; +azimuth moves toward +X
    (the subject's left), +elevation moves up."""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    return Vector((
        distance * math.sin(az) * math.cos(el),
        -distance * math.cos(az) * math.cos(el),
        distance * math.sin(el),
    ))


def _look_at(obj: bpy.types.Object, target: Vector, roll_deg: float = 0.0) -> None:
    base = (target - obj.location).to_track_quat("-Z", "Y")
    obj.rotation_quaternion = base @ Quaternion((0.0, 0.0, 1.0), math.radians(roll_deg))


def face_centre(eye_mid: Vector, iod: float) -> Vector:
    """Centre of the chin-to-hairline face box (the eyes sit above it)."""
    return eye_mid - Vector((0.0, 0.0, C.FACE_CENTER_BELOW_EYES_PER_IOD * iod))


def place_camera(camera: bpy.types.Object, eye_mid: Vector, iod: float, cam: dict) -> dict:
    """Camera on a sphere around the eye midpoint, optical axis through the eye midpoint.

    Perspective depends only on the distance. The face box (FACE_HEIGHT_PER_IOD x IOD high)
    should fill `face_fill` of the frame at the sampled focal length, so the sensor width is
    derived from it, and a vertical lens shift centres the face box (more chin margin) without
    moving the optical axis off the eyes.

    Returns derived values for params.json.
    """
    distance = cam["distance_m"]
    camera.location = eye_mid + _orbit_offset(cam["yaw"], cam["pitch"], distance)
    _look_at(camera, eye_mid, cam["roll"])

    face_height = C.FACE_HEIGHT_PER_IOD * iod
    frame_m = face_height / cam["face_fill"]  # square frame, metres at the eye distance
    sensor_mm = frame_m * cam["focal_mm"] / distance
    shift_y = -(C.FACE_CENTER_BELOW_EYES_PER_IOD * iod) / frame_m
    camera.data.lens = cam["focal_mm"]
    camera.data.sensor_width = sensor_mm
    camera.data.shift_y = shift_y
    bpy.context.view_layer.update()  # matrix_world must be current for ray casts
    return {
        "sensor_width_mm": sensor_mm,
        "focal_35mm_equiv": cam["focal_mm"] * FULL_FRAME_WIDTH_MM / sensor_mm,
        "shift_y": shift_y,
        "iod_m": iod,
        "face_height_m": face_height,
    }


def white_balanced_kelvin(kelvin: float) -> float:
    """Phone-like auto white balance: pull the colour temperature toward neutral (C.WHITE_BALANCE_*)."""
    return C.WHITE_BALANCE_NEUTRAL_K + (kelvin - C.WHITE_BALANCE_NEUTRAL_K) * (1.0 - C.WHITE_BALANCE_STRENGTH)


def place_key_light(light: bpy.types.Object, eye_mid: Vector, params: dict) -> float:
    """Returns the rendered colour temperature (after white balance)."""
    light.location = eye_mid + _orbit_offset(params["azimuth"], params["elevation"], C.KEY_LIGHT_DISTANCE_M)
    _look_at(light, eye_mid)
    light.data.energy = C.KEY_LIGHT_BASE_POWER_W * params["intensity"]
    light.data.temperature = white_balanced_kelvin(params["kelvin"])
    light.data.size = params["size_m"]
    return light.data.temperature


def set_background(params: dict) -> None:
    """World = what the camera sees (solid or vertical gradient) + separate ambient light.

    Light Path "Is Camera Ray" picks the visible background for camera rays; every other ray
    gets a near-neutral ambient (background colour mixed in by `ambient_tint`) at
    `ambient_strength`. This keeps bright backgrounds from washing out the face.
    No external HDRIs.
    """
    scene = bpy.context.scene
    world = bpy.data.worlds.get(WORLD_NAME) or bpy.data.worlds.new(WORLD_NAME)
    scene.world = world
    if world.node_tree is None:
        world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()

    out = tree.nodes.new("ShaderNodeOutputWorld")
    visible = tree.nodes.new("ShaderNodeBackground")
    ambient = tree.nodes.new("ShaderNodeBackground")
    light_path = tree.nodes.new("ShaderNodeLightPath")
    select = tree.nodes.new("ShaderNodeMixShader")
    tree.links.new(light_path.outputs["Is Camera Ray"], select.inputs["Fac"])
    tree.links.new(ambient.outputs["Background"], select.inputs[1])
    tree.links.new(visible.outputs["Background"], select.inputs[2])
    tree.links.new(select.outputs["Shader"], out.inputs["Surface"])

    bottom = params["bottom_rgb"]
    top = params["top_rgb"]
    tint = params["ambient_tint"]
    mean_bg = [(b + t) / 2 for b, t in zip(bottom, top)]
    ambient.inputs["Color"].default_value = (*[(1.0 - tint) + tint * c for c in mean_bg], 1.0)
    ambient.inputs["Strength"].default_value = params["ambient_strength"]
    visible.inputs["Strength"].default_value = 1.0

    if params["kind"] == "solid":
        visible.inputs["Color"].default_value = (*bottom, 1.0)
        return

    # Generated coordinates in a world shader = view direction; map z from [-1, 1] to [0, 1].
    coords = tree.nodes.new("ShaderNodeTexCoord")
    split = tree.nodes.new("ShaderNodeSeparateXYZ")
    to_fac = tree.nodes.new("ShaderNodeMath")
    to_fac.operation = "MULTIPLY_ADD"
    to_fac.inputs[1].default_value = 0.5
    to_fac.inputs[2].default_value = 0.5
    to_fac.use_clamp = True
    mix = tree.nodes.new("ShaderNodeMixRGB")
    mix.inputs["Color1"].default_value = (*bottom, 1.0)
    mix.inputs["Color2"].default_value = (*top, 1.0)
    tree.links.new(coords.outputs["Generated"], split.inputs["Vector"])
    tree.links.new(split.outputs["Z"], to_fac.inputs[0])
    tree.links.new(to_fac.outputs["Value"], mix.inputs["Fac"])
    tree.links.new(mix.outputs["Color"], visible.inputs["Color"])


def face_exposure_qa(camera: bpy.types.Object, basemesh: bpy.types.Object, eye_mid: Vector, iod: float,
                     png_path: str, step: int = C.QA_SAMPLE_STEP_PX) -> dict:
    """Fraction of pure-white face pixels in a saved render.

    Face pixels: camera rays (every QA_SAMPLE_STEP_PX pixel) whose first hit is the skin
    mesh within QA_FACE_RADIUS_PER_IOD x IOD of the face centre (hair, eyes, brows excluded).
    """
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    pct = scene.render.resolution_percentage
    width, height = scene.render.resolution_x * pct // 100, scene.render.resolution_y * pct // 100
    top_right, bottom_right, bottom_left, top_left = camera.data.view_frame(scene=scene)
    rot = camera.matrix_world.to_3x3()
    origin = camera.matrix_world.translation
    centre = face_centre(eye_mid, iod)
    radius = C.QA_FACE_RADIUS_PER_IOD * iod

    img = bpy.data.images.load(png_path)
    pixels = np.empty(width * height * 4, dtype=np.float32)
    img.pixels.foreach_get(pixels)
    bpy.data.images.remove(img)
    pixels = pixels.reshape(height, width, 4)[::-1]  # Blender rows are bottom-up

    face = white = near_white = 0
    luma_sum = 0.0
    for py in range(step // 2, height, step):
        v = (py + 0.5) / height
        for px in range(step // 2, width, step):
            u = (px + 0.5) / width
            local = top_left + (top_right - top_left) * u + (bottom_left - top_left) * v
            hit, location, _, _, obj, _ = scene.ray_cast(depsgraph, origin, (rot @ local).normalized())
            if not hit or obj.original != basemesh or (location - centre).length > radius:
                continue
            rgb = pixels[py, px, :3]
            face += 1
            luma = float(rgb @ LUMA_709)
            white += bool((rgb >= C.QA_WHITE_LEVEL).all())
            near_white += luma >= C.QA_NEAR_WHITE_LUMA
            luma_sum += luma
    return {
        "face_samples": face,
        "face_white_fraction": white / face if face else 0.0,
        "face_near_white_fraction": near_white / face if face else 0.0,
        "face_mean_luma": luma_sum / face if face else 0.0,
    }


def preview_face_luma(camera: bpy.types.Object, basemesh: bpy.types.Object, eye_mid: Vector, iod: float,
                      preview_path: str) -> float:
    """Mean face display luma of a small, quick render at exposure 0 (current scene setup)."""
    scene = bpy.context.scene
    render = scene.render
    saved = (render.resolution_percentage, scene.cycles.samples, render.filepath, scene.view_settings.exposure)
    render.resolution_percentage = C.AE_PREVIEW_PERCENT
    scene.cycles.samples = C.AE_PREVIEW_SAMPLES
    scene.view_settings.exposure = 0.0
    render.filepath = preview_path
    try:
        bpy.ops.render.render(write_still=True)
        luma = face_exposure_qa(camera, basemesh, eye_mid, iod, preview_path, step=1)["face_mean_luma"]
    finally:
        render.resolution_percentage, scene.cycles.samples, render.filepath, scene.view_settings.exposure = saved
    return luma


def auto_exposure(measured_luma: float, reference_luma: float, jitter_ev: float) -> float:
    """Exposure (EV) that brings the face from measured to reference brightness, + jitter. Sets it."""
    if measured_luma <= 0.0 or reference_luma <= 0.0:
        ev = jitter_ev
    else:
        ev = C.AE_DISPLAY_GAMMA * math.log2(reference_luma / measured_luma) + jitter_ev
    ev = max(-C.AE_MAX_EV, min(C.AE_MAX_EV, ev))
    bpy.context.scene.view_settings.exposure = ev
    return ev


def set_environment(env: dict, background: dict, light: bpy.types.Object, camera: bpy.types.Object,
                    eye_mid: Vector, hdri_files: dict[str, str]) -> dict:
    """HDRI world (+ key light scaled down, depth of field) or the flat backdrop. Call after
    place_key_light. Returns derived values for params.json."""
    if env["kind"] != "hdri":
        set_background(background)
        set_depth_of_field(camera, eye_mid, None)
        return {}
    derived = set_hdri_world(hdri_files[env["hdri"]], env["rotation_deg"], env["mean_luminance"])
    light.data.energy *= env["key_factor"]
    set_depth_of_field(camera, eye_mid, env["fstop"])
    return derived


def eye_visibility(camera: bpy.types.Object, eyes: bpy.types.Object, centres: dict[str, Vector],
                   see_through: set[str]) -> dict:
    """Ray cast from the camera to each eye centre.

    Visible when the first hit is the eye mesh; objects in `see_through` (eyelash / eyebrow
    alpha cards) are stepped through. Anything else first (hair, eyelid, nose) = hidden.
    """
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    result = {}
    for side, centre in centres.items():
        origin = camera.matrix_world.translation.copy()
        direction = (centre - origin).normalized()
        blocked_by = None
        for _ in range(EYE_RAY_MAX_STEPS):
            remaining = (centre - origin).length
            hit, location, _, _, obj, _ = scene.ray_cast(depsgraph, origin, direction, distance=remaining)
            if not hit or obj.original == eyes:
                break
            if obj.original.name in see_through:
                origin = location + direction * EYE_RAY_STEP_M
                continue
            blocked_by = obj.original.name
            break
        result[side] = {"visible": blocked_by is None, "blocked_by": blocked_by}
    return result


def _mean_luminance(image) -> float:
    width, height = image.size
    pixels = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    return float(pixels.reshape(-1, 4)[:, :3].mean(axis=0) @ LUMA_709)


def set_hdri_world(hdri_path: str, rotation_deg: float, mean_luminance: float) -> dict:
    """World = HDRI environment (lights the face and is seen behind it), rotated about Z.

    HDRIs differ a lot in brightness, so the strength is set from the image's mean
    luminance: strength = mean_luminance / measured mean. Returns derived values.
    """
    scene = bpy.context.scene
    world = bpy.data.worlds.get(WORLD_NAME) or bpy.data.worlds.new(WORLD_NAME)
    scene.world = world
    if world.node_tree is None:
        world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()

    image = bpy.data.images.load(hdri_path, check_existing=True)
    measured = _mean_luminance(image)
    strength = mean_luminance / max(measured, 1e-6)

    out = tree.nodes.new("ShaderNodeOutputWorld")
    bg = tree.nodes.new("ShaderNodeBackground")
    env = tree.nodes.new("ShaderNodeTexEnvironment")
    mapping = tree.nodes.new("ShaderNodeMapping")
    coords = tree.nodes.new("ShaderNodeTexCoord")
    env.image = image
    mapping.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(rotation_deg))
    bg.inputs["Strength"].default_value = strength
    tree.links.new(coords.outputs["Generated"], mapping.inputs["Vector"])
    tree.links.new(mapping.outputs["Vector"], env.inputs["Vector"])
    tree.links.new(env.outputs["Color"], bg.inputs["Color"])
    tree.links.new(bg.outputs["Background"], out.inputs["Surface"])
    return {"hdri_mean_luminance": measured, "strength": strength}


def set_depth_of_field(camera: bpy.types.Object, focus_point: Vector, fstop: float | None) -> None:
    """Background blur like a phone camera: focus on `focus_point` (the eyes); None = off."""
    camera.data.dof.use_dof = fstop is not None
    if fstop is not None:
        camera.data.dof.focus_distance = (focus_point - camera.matrix_world.translation).length
        camera.data.dof.aperture_fstop = fstop
