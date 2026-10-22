"""MPFB access for the Blender side. The MPFB import path lives here and nowhere else."""

import importlib
import logging
import os
from types import ModuleType

import addon_utils
import bpy
import numpy as np
from mathutils import Matrix, Vector

from shared import constants as C

log = logging.getLogger(__name__)

MPFB_MODULE = "bl_ext.blender_org.mpfb"

SKIN_SHIFT_NODE = "SynthFaceSkinShift"
SKIN_DARK_TEXTURE_NODE = "SynthFaceDarkSkinTexture"
HAIR_COLOR_PREFIX = "SynthFaceHairColor"
HAIR_UNDERLAYER_SUFFIX = "underlayer"
HAIR_UNDERLAYER_SCALE = 0.985
HAIR_UNDERLAYER_ROUGHNESS = 0.8
IRIS_SATURATION_GAIN = 3.0  # saturation 0.33+ counts as fully iris
STRAND_SPECULAR = 0.25  # MPFB default 0.5 makes dark hair look grey under bright key lights
LUMA = (0.2126, 0.7152, 0.0722)  # Rec.709, linear RGB


def load_mpfb() -> ModuleType:
    """Enable MPFB for this Blender session and return its root package.

    In --background mode the add-on is not registered automatically, and MPFB's services
    only work after register() has run. default_set=True is required because register()
    reads MPFB's own preferences entry; in --background mode Blender does not save the
    preferences on exit, so userpref.blend stays unchanged (verified).
    """
    loaded_default, loaded_state = addon_utils.check(MPFB_MODULE)
    if not loaded_state:
        mod = addon_utils.enable(MPFB_MODULE, default_set=True, handle_error=None)
        if mod is None:
            raise RuntimeError(f"Could not enable {MPFB_MODULE}")
    return importlib.import_module(MPFB_MODULE)


def service(name: str, cls: str):
    """Return an MPFB service class, e.g. service("humanservice", "HumanService")."""
    module = importlib.import_module(f"{MPFB_MODULE}.services.{name}")
    return getattr(module, cls)


# ---------------------------------------------------------------------------
# Scene housekeeping
# ---------------------------------------------------------------------------
def clear_scene() -> None:
    """Remove all objects and every data block they leave behind (meshes, materials, images...)."""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)


def _find_asset(list_fn, subdir: str, stem: str) -> str:
    for path in list_fn(subdir):
        if path.stem == stem:
            return str(path)
    raise FileNotFoundError(f"MPFB asset '{stem}' not found in '{subdir}'")


# ---------------------------------------------------------------------------
# Face shape targets
# ---------------------------------------------------------------------------
def apply_face_targets(basemesh, face_targets: list[dict], categories: dict[str, dict]) -> None:
    """value < 0 -> negative targets, value > 0 -> positive targets, weight = |value|.

    Single-target categories (no negative side) use the positive target with |value|.
    """
    TargetService = service("targetservice", "TargetService")
    for item in face_targets:
        cat = categories[item["category"]]
        value = item["value"]
        names = cat["negative"] if value < 0 and cat["negative"] else cat["positive"]
        for name in names:
            path = TargetService.target_full_path(name)
            TargetService.load_target(basemesh, path, weight=abs(value), name=name)


# ---------------------------------------------------------------------------
# Material tweaks
# ---------------------------------------------------------------------------
def _skin_group_nodes(basemesh) -> list:
    """The ShaderNodeGroup instance in every skin material slot (they share one node tree)."""
    nodes = [n for mat in basemesh.data.materials for n in mat.node_tree.nodes
             if n.bl_idname == "ShaderNodeGroup" and n.node_tree is not None]
    if not nodes:
        raise RuntimeError("Skin node group not found on basemesh")
    return nodes


def apply_skin_shading(basemesh, roughness: float, specular: float, clearcoat: float) -> None:
    """Less shiny skin: group inputs Roughness / Clearcoat on every skin slot (MPFB sets the
    same 0.45 / 0.1 on all of them), specular once inside the shared group."""
    for node in _skin_group_nodes(basemesh):
        node.inputs["Roughness"].default_value = roughness
        node.inputs["Clearcoat"].default_value = clearcoat
    group = _skin_group_nodes(basemesh)[0].node_tree
    principled = next(n for n in group.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    principled.inputs["Specular IOR Level"].default_value = specular


def read_applied_skin(basemesh) -> dict:
    """Skin values as they are in the Blender material right now (for params.json checks)."""
    body_mat = next(m for m in basemesh.data.materials if m.name.endswith(".body"))
    body = next(n for n in body_mat.node_tree.nodes if n.bl_idname == "ShaderNodeGroup")
    group = body.node_tree
    hsv = group.nodes[SKIN_SHIFT_NODE]
    principled = next(n for n in group.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    return {
        "hue_shift": hsv.inputs["Hue"].default_value - 0.5,
        "saturation": hsv.inputs["Saturation"].default_value,
        "value": hsv.inputs["Value"].default_value,
        "roughness": body.inputs["Roughness"].default_value,
        "specular": principled.inputs["Specular IOR Level"].default_value,
        "clearcoat": body.inputs["Clearcoat"].default_value,
        "tone": group.nodes["Mix"].inputs[0].default_value,
    }


def _mhmat_diffuse_path(mhmat_file: str) -> str:
    with open(mhmat_file, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("diffuseTexture "):
                return os.path.join(os.path.dirname(mhmat_file), line.split(None, 1)[1].strip())
    raise ValueError(f"No diffuseTexture in {mhmat_file}")


def apply_skin_tone_blend(basemesh, dark_mhmat_file: str, tone: float) -> None:
    """Blend the dark base skin's diffuse texture into the current (light) skin.

    Inside MPFB's skin group the light texture enters "Mix" as Color1; Color2 / Factor come
    from the group inputs colorMixIn / colorMixInStrength (a flat colour at 0.05). We feed
    Color2 from a second image texture (same UV layout) and set Factor = tone instead.
    """
    tree = _skin_group_nodes(basemesh)[0].node_tree
    mix = tree.nodes["Mix"]
    light_tex = tree.nodes["DiffuseTexture"]
    dark_tex = tree.nodes.new("ShaderNodeTexImage")
    dark_tex.name = SKIN_DARK_TEXTURE_NODE
    dark_tex.image = bpy.data.images.load(_mhmat_diffuse_path(dark_mhmat_file), check_existing=True)
    dark_tex.image.colorspace_settings.name = light_tex.image.colorspace_settings.name
    uv_link = next(link for link in tree.links if link.to_socket == light_tex.inputs["Vector"])
    tree.links.new(uv_link.from_socket, dark_tex.inputs["Vector"])
    for link in [link for link in tree.links if link.to_node == mix and link.to_socket != mix.inputs[1]]:
        tree.links.remove(link)
    tree.links.new(dark_tex.outputs["Color"], mix.inputs[2])
    mix.inputs[0].default_value = tone


def correct_skin_texture_saturation(basemesh, light: str, dark: str) -> None:
    """Desaturate over-saturated base textures (C.SKIN_TEXTURE_SATURATION_FIX) before the tone blend.

    DiffuseTexture -> [HSV sat x fix] -> Mix.Color1 ; dark texture -> [HSV sat x fix] -> Mix.Color2
    """
    tree = _skin_group_nodes(basemesh)[0].node_tree
    mix = tree.nodes["Mix"]
    for skin_name, socket in ((light, mix.inputs[1]), (dark, mix.inputs[2])):
        fix = C.SKIN_TEXTURE_SATURATION_FIX.get(skin_name)
        if fix is None or not socket.links:
            continue
        source = socket.links[0].from_socket
        hsv = tree.nodes.new("ShaderNodeHueSaturation")
        hsv.name = f"{SKIN_SHIFT_NODE}.texture_fix.{skin_name}"
        hsv.inputs["Saturation"].default_value = fix
        tree.links.new(source, hsv.inputs["Color"])
        tree.links.new(hsv.outputs["Color"], socket)


def apply_skin_shift(basemesh, hue_shift: float, saturation: float, value: float) -> None:
    """Insert a Hue/Saturation/Value node into the skin node group.

    MPFB's ENHANCED_SSS skin is one node group shared by every body material slot
    (body, lips, ears, nails...). Inside it the diffuse texture goes
    DiffuseTexture -> Mix -> Bright/Contrast -> Principled Base Color.
    The HSV node goes between Mix and Bright/Contrast, so the whole skin shifts together
    and SSS/roughness/bump are untouched.
    """
    group = None
    for mat in basemesh.data.materials:
        for node in mat.node_tree.nodes:
            if node.bl_idname == "ShaderNodeGroup" and node.node_tree is not None:
                group = node.node_tree
                break
        if group:
            break
    if group is None:
        raise RuntimeError("Skin node group not found on basemesh")

    tree = group
    mix = tree.nodes.get("Mix")
    bright = tree.nodes.get("Bright/Contrast")
    if mix is None or bright is None:
        raise RuntimeError("Unexpected skin node group layout (Mix / Bright/Contrast missing)")

    hsv = tree.nodes.get(SKIN_SHIFT_NODE) or tree.nodes.new("ShaderNodeHueSaturation")
    hsv.name = SKIN_SHIFT_NODE
    hsv.inputs["Hue"].default_value = 0.5 + hue_shift
    hsv.inputs["Saturation"].default_value = saturation
    hsv.inputs["Value"].default_value = value
    hsv.inputs["Fac"].default_value = 1.0
    tree.links.new(mix.outputs["Color"], hsv.inputs["Color"])
    tree.links.new(hsv.outputs["Color"], bright.inputs["Color"])


def _mean_texture_luminance(image) -> float:
    """Mean linear luminance of the opaque pixels of an image."""
    width, height = image.size
    pixels = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    pixels = pixels.reshape(-1, 4)
    rgb = pixels[pixels[:, 3] > 0.5][:, :3]
    if image.colorspace_settings.name == "sRGB":
        rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    if rgb.size == 0:
        return 1.0
    return float(np.mean(rgb @ np.array(LUMA, dtype=np.float32)))


def apply_strand_color(obj, rgb: list[float], detail: float = 1.0) -> None:
    """Recolour a MAKESKIN hair/eyebrow material while keeping its strand detail.

    diffuseTexture -> RGBtoBW (luminance) -> x (1 / mean luminance) -> [detail: 1 + detail x (v - 1)]
    -> scale target RGB -> diffuseIntensity.Color2 (which feeds Principled Base Color). Alpha is untouched.
    detail < 1 flattens the texture contrast toward the plain colour (C.HAIR_TEXTURE_DETAIL).
    """
    mat = obj.data.materials[0]
    tree = mat.node_tree
    tex = tree.nodes["diffuseTexture"]
    mix = tree.nodes["diffuseIntensity"]
    tree.nodes["Principled BSDF"].inputs["Specular IOR Level"].default_value = STRAND_SPECULAR

    bw = tree.nodes.new("ShaderNodeRGBToBW")
    norm = tree.nodes.new("ShaderNodeMath")
    norm.operation = "MULTIPLY"
    norm.inputs[1].default_value = 1.0 / max(_mean_texture_luminance(tex.image), 1e-4)
    scale = tree.nodes.new("ShaderNodeVectorMath")
    scale.operation = "SCALE"
    scale.inputs["Vector"].default_value = rgb
    for node in (bw, norm, scale):
        node.name = f"{HAIR_COLOR_PREFIX}.{node.bl_idname}"

    tree.links.new(tex.outputs["Color"], bw.inputs["Color"])
    tree.links.new(bw.outputs["Val"], norm.inputs[0])
    value = norm.outputs["Value"]
    if detail < 1.0:
        soften = tree.nodes.new("ShaderNodeMath")
        soften.operation = "MULTIPLY_ADD"  # v x detail + (1 - detail)
        soften.use_clamp = False
        soften.inputs[1].default_value = detail
        soften.inputs[2].default_value = 1.0 - detail
        soften.name = f"{HAIR_COLOR_PREFIX}.detail"
        tree.links.new(value, soften.inputs[0])
        value = soften.outputs["Value"]
    tree.links.new(value, scale.inputs["Scale"])
    tree.links.new(scale.outputs["Vector"], mix.inputs["Color2"])


def darken_iris(eyes_obj, factor: float) -> None:
    """Darken the iris only: multiply by `factor`, weighted by the texture's HSV saturation.

    The eye texture holds iris and sclera together; the sclera is unsaturated, so it stays white.
    diffuseTexture -> MixRGB(MULTIPLY, Fac = saturation x gain) -> diffuseIntensity.Color2
    """
    tree = eyes_obj.data.materials[0].node_tree
    tex = tree.nodes["diffuseTexture"]
    mix = tree.nodes["diffuseIntensity"]
    hsv = tree.nodes.new("ShaderNodeSeparateColor")
    hsv.mode = "HSV"
    gain = tree.nodes.new("ShaderNodeMath")
    gain.operation = "MULTIPLY"
    gain.use_clamp = True
    gain.inputs[1].default_value = IRIS_SATURATION_GAIN
    darken = tree.nodes.new("ShaderNodeMixRGB")
    darken.blend_type = "MULTIPLY"
    darken.inputs["Color2"].default_value = (factor, factor, factor, 1.0)
    tree.links.new(tex.outputs["Color"], hsv.inputs["Color"])
    tree.links.new(hsv.outputs["Green"], gain.inputs[0])  # HSV mode: Green output = saturation
    tree.links.new(gain.outputs["Value"], darken.inputs["Fac"])
    tree.links.new(tex.outputs["Color"], darken.inputs["Color1"])
    tree.links.new(darken.outputs["Color"], mix.inputs["Color2"])


def add_hair_underlayer(hair_obj, rgb: list[float]) -> None:
    """Opaque copy of the hair mesh, shrunk toward its centre, in the hair colour.

    MakeHuman hair is alpha-cut cards with ~50% fully transparent texels, so the scalp skin
    shows between strands and turns any hair colour into skin-brown. This inner shell hides
    the scalp while the outer cards keep the strand detail and wispy silhouette.
    """
    under = hair_obj.copy()
    under.data = hair_obj.data.copy()
    under.name = f"{hair_obj.name}.{HAIR_UNDERLAYER_SUFFIX}"
    for collection in hair_obj.users_collection:
        collection.objects.link(under)

    # Centre from the fitted vertices: obj.bound_box can still describe the unfitted mesh here.
    coords = [hair_obj.matrix_world @ v.co for v in hair_obj.data.vertices]
    centre = Vector([(min(c[i] for c in coords) + max(c[i] for c in coords)) / 2 for i in range(3)])
    shrink = Matrix.Translation(centre) @ Matrix.Scale(HAIR_UNDERLAYER_SCALE, 4) @ Matrix.Translation(-centre)
    under.matrix_world = shrink @ hair_obj.matrix_world

    mat = bpy.data.materials.new(under.name)
    mat.use_nodes = True
    principled = next(n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (*rgb, 1.0)
    principled.inputs["Roughness"].default_value = HAIR_UNDERLAYER_ROUGHNESS
    principled.inputs["Specular IOR Level"].default_value = STRAND_SPECULAR
    under.data.materials.clear()
    under.data.materials.append(mat)


# ---------------------------------------------------------------------------
# Build one identity
# ---------------------------------------------------------------------------
def build_human(identity: dict, face_categories: dict[str, dict]):
    """Create the human described by `identity` (from shared.sampling.sample_identity).

    Returns (basemesh, eyes object).
    """
    AssetService = service("assetservice", "AssetService")
    HumanService = service("humanservice", "HumanService")

    basemesh = HumanService.create_human(macro_detail_dict=identity["macros"])
    # Shape targets before any mhclo asset, so eyes/brows/hair are fitted to the final face.
    apply_face_targets(basemesh, identity["face_targets"], face_categories)

    skin = identity["skin"]
    HumanService.set_character_skin(_find_asset(AssetService.list_mhmat_assets, "skins", skin["light"]), basemesh)
    apply_skin_tone_blend(basemesh, _find_asset(AssetService.list_mhmat_assets, "skins", skin["dark"]), skin["tone"])
    correct_skin_texture_saturation(basemesh, skin["light"], skin["dark"])
    apply_skin_shift(basemesh, skin["hue_shift"], skin["saturation"], skin["value"])
    apply_skin_shading(basemesh, skin["roughness"], skin["specular"], skin["clearcoat"])

    eyes_cfg = identity["eyes"]
    eyes_file = _find_asset(AssetService.list_mhclo_assets, "eyes", eyes_cfg["mesh"])
    eyes_uuid = _mhclo_uuid(eyes_file)
    eyes = HumanService.add_mhclo_asset(
        eyes_file, basemesh, asset_type="Eyes", set_up_rigging=False,
        alternative_materials={eyes_uuid: f"materials/{eyes_cfg['material']}.mhmat"},
    )
    if eyes_cfg["iris_darken"] is not None:
        darken_iris(eyes, eyes_cfg["iris_darken"])

    brows_file = _find_asset(AssetService.list_mhclo_assets, "eyebrows", identity["eyebrows"])
    brows = HumanService.add_mhclo_asset(brows_file, basemesh, asset_type="Eyebrows", set_up_rigging=False)
    apply_strand_color(brows, identity["eyebrow_rgb"])

    lashes_file = _find_asset(AssetService.list_mhclo_assets, "eyelashes", identity["eyelashes"])
    HumanService.add_mhclo_asset(lashes_file, basemesh, asset_type="Eyelashes", set_up_rigging=False)

    hair = identity["hair"]
    if hair["style"] != C.HAIR_NONE:
        add_hair(basemesh, hair["style"], hair["rgb"])

    beard = identity.get("beard")
    if beard and beard["style"] != C.BEARD_NONE:
        eye_mid, iod = eye_geometry(eyes)
        add_beard(basemesh, beard, eye_mid, iod)

    return basemesh, eyes


def add_hair(basemesh, style: str, rgb: list[float]) -> list:
    """Add a hairstyle (recoloured, with its opaque underlayer). Returns the new objects."""
    AssetService = service("assetservice", "AssetService")
    HumanService = service("humanservice", "HumanService")
    hair_file = _find_asset(AssetService.list_mhclo_assets, "hair", style)
    hair_obj = HumanService.add_mhclo_asset(hair_file, basemesh, asset_type="Hair", set_up_rigging=False)
    apply_strand_color(hair_obj, rgb, C.HAIR_TEXTURE_DETAIL.get(style, 1.0))
    add_hair_underlayer(hair_obj, rgb)
    return [hair_obj, bpy.data.objects[f"{hair_obj.name}.{HAIR_UNDERLAYER_SUFFIX}"]]


# ---------------------------------------------------------------------------
# Beard (in-house asset: hair particles on a density map from the face geometry)
# ---------------------------------------------------------------------------
BEARD_GROUP = "SynthFaceBeard"
BEARD_MATERIAL = "SynthFaceBeard"
BEARD_ATTRIBUTE = "SynthFaceBeard"  # same weights as a point attribute, read by the skin shader
BEARD_SKIN_NODE = "SynthFaceBeardSkin"
BEARD_STRAND_ROUGHNESS = 0.55
BEARD_STRAND_SPECULAR = 0.3


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = min(max((x - edge0) / (edge1 - edge0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _world_coords(basemesh) -> np.ndarray:
    """World-space vertex positions with shape targets applied, in original vertex order
    (the helper-geometry Mask modifier is switched off while evaluating)."""
    masks = [m for m in basemesh.modifiers if m.type == "MASK"]
    states = [m.show_viewport for m in masks]
    for m in masks:
        m.show_viewport = False
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated = basemesh.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    coords = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", coords)
    evaluated.to_mesh_clear()
    for m, state in zip(masks, states):
        m.show_viewport = state
    coords = coords.reshape(-1, 3)
    mw = np.array(basemesh.matrix_world)
    return coords @ mw[:3, :3].T + mw[:3, 3]


def _group_members(basemesh, name: str) -> np.ndarray:
    index = basemesh.vertex_groups[name].index
    return np.array([v.index for v in basemesh.data.vertices if any(g.group == index for g in v.groups)])


def beard_weights(basemesh, eye_mid: Vector, iod: float, region: str, density: float,
                  noise_amount: float, noise_offset: tuple[float, float, float]) -> np.ndarray:
    """Per-vertex beard density 0..1 from the face geometry (all sizes x IOD, see C.BEARD_GEOMETRY)."""
    from mathutils import noise

    g = C.BEARD_GEOMETRY
    co = _world_coords(basemesh)
    body = _group_members(basemesh, "body")
    lips = co[_group_members(basemesh, "lips")]
    ears = co[_group_members(basemesh, "ears")]
    lips_top, lips_bottom = lips[:, 2].max(), lips[:, 2].min()
    lips_half_width = np.abs(lips[:, 0] - eye_mid.x).max()
    ear_y = ears[:, 1].mean()
    chin_area = body[(np.abs(co[body, 0] - eye_mid.x) < 0.25 * iod)
                     & (co[body, 1] < lips[:, 1].mean() + g["chin_front_depth"] * iod)
                     & (co[body, 2] < lips_bottom) & (co[body, 2] > lips_bottom - g["chin_search_below_lips"] * iod)]
    chin_z = co[chin_area, 2].min()
    edge = g["edge"] * iod
    neck = {"full_neck": g["neck_below_chin"] + g["full_neck_extra"],
            "goatee": g["goatee_below_chin"]}.get(region, g["neck_below_chin"])
    sideburn_top = eye_mid.z - g["sideburn_top_below_eyes"] * iod

    weights = np.zeros(len(co))
    for i in body:
        x, y, z = co[i]
        ax = abs(x - eye_mid.x)
        if z > sideburn_top + edge or z < chin_z - neck * iod - edge or y > ear_y - g["front_of_ears_margin"] * iod:
            continue
        # upper edge: mustache band under the nose, cheek line beside it, sideburns near the ears
        if ax < g["nose_half_width"] * iod:
            top = lips_top + g["mustache_top_above_lips"] * iod
        else:
            top = lips_top + (g["cheek_line_above_lips"] + g["cheek_line_rise"] * (ax / iod - g["nose_half_width"])) * iod
        near_ear = _smoothstep(ear_y - (g["sideburn_depth"] + 0.3) * iod, ear_y - g["sideburn_depth"] * iod, y)
        top = top + near_ear * max(sideburn_top - top, 0.0)
        w = 1.0 - _smoothstep(top - edge, top, z)
        w *= _smoothstep(chin_z - neck * iod - edge, chin_z - neck * iod, z)
        w *= 1.0 - _smoothstep(ear_y - (g["front_of_ears_margin"] + 0.1) * iod, ear_y - g["front_of_ears_margin"] * iod, y)
        # keep the lips clear
        if lips_bottom - g["lip_margin"] * iod < z < lips_top + g["lip_margin"] * iod \
                and ax < lips_half_width + g["lip_margin"] * iod:
            w = 0.0
        if region == "goatee":
            w *= 1.0 - _smoothstep((g["goatee_half_width"] - g["edge"]) * iod, g["goatee_half_width"] * iod, ax)
        if w <= 0.0:
            continue
        p = Vector((x, y, z)) * (C.BEARD_NOISE_SCALE_PER_IOD / iod) + Vector(noise_offset)
        patch = 0.5 + 0.5 * noise.noise(p)  # 0..1
        weights[i] = w * density * (1.0 - noise_amount * (1.0 - patch))
    return weights


def apply_beard_skin(basemesh, rgb: list[float], cover: float) -> None:
    """Skin under the beard takes the beard colour (factor = beard weight x cover): inserted
    between the HSV shift and Bright/Contrast in the shared skin group, so lips/ears are untouched
    (their weight is 0)."""
    tree = _skin_group_nodes(basemesh)[0].node_tree
    hsv = tree.nodes.get(SKIN_SHIFT_NODE)
    bright = tree.nodes.get("Bright/Contrast")
    if hsv is None or bright is None:
        raise RuntimeError("apply_skin_shift must run before apply_beard_skin")
    attr = tree.nodes.new("ShaderNodeAttribute")
    attr.attribute_type = "GEOMETRY"
    attr.attribute_name = BEARD_ATTRIBUTE
    amount = tree.nodes.new("ShaderNodeMath")
    amount.operation = "MULTIPLY"
    amount.use_clamp = True
    amount.inputs[1].default_value = cover
    mix = tree.nodes.new("ShaderNodeMixRGB")
    mix.name = BEARD_SKIN_NODE
    mix.inputs["Color2"].default_value = (*rgb, 1.0)
    tree.links.new(attr.outputs["Fac"], amount.inputs[0])
    tree.links.new(amount.outputs["Value"], mix.inputs["Fac"])
    tree.links.new(hsv.outputs["Color"], mix.inputs["Color1"])
    tree.links.new(mix.outputs["Color"], bright.inputs["Color"])


def _mean_strand_length(basemesh, sample: int = 200) -> float:
    """Mean root-to-tip distance of the first `sample` parent strands (world units)."""
    evaluated = basemesh.evaluated_get(bpy.context.evaluated_depsgraph_get())
    psys = evaluated.particle_systems[0]
    last = 2 ** psys.settings.display_step
    n = min(sample, len(psys.particles))
    lengths = [(psys.co_hair(evaluated, particle_no=i, step=last) - psys.co_hair(evaluated, particle_no=i, step=0)).length
               for i in range(n)]
    return float(np.mean(lengths)) if lengths else 0.0


def add_beard(basemesh, beard: dict, eye_mid: Vector, iod: float) -> None:
    """Grow the identity's beard (shared.sampling.sample_beard) as hair particles."""
    style = C.BEARD_STYLES[beard["style"]]
    weights = beard_weights(basemesh, eye_mid, iod, style["region"], beard["density"], beard["noise_amount"],
                            tuple(beard["noise_offset"]))
    group = basemesh.vertex_groups.new(name=BEARD_GROUP)
    for i in np.nonzero(weights)[0]:
        group.add([int(i)], float(weights[i]), "REPLACE")
    attribute = basemesh.data.attributes.new(BEARD_ATTRIBUTE, "FLOAT", "POINT")
    attribute.data.foreach_set("value", weights.astype(np.float32))
    apply_beard_skin(basemesh, beard["rgb"], style["skin_cover"])

    # Matte strands: a glossy hair BSDF on sub-pixel strands reads as a light sheen, not a beard.
    mat = bpy.data.materials.new(BEARD_MATERIAL)
    mat.use_nodes = True
    principled = next(n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (*beard["rgb"], 1.0)
    principled.inputs["Roughness"].default_value = BEARD_STRAND_ROUGHNESS
    principled.inputs["Specular IOR Level"].default_value = BEARD_STRAND_SPECULAR
    basemesh.data.materials.append(mat)

    mod = basemesh.modifiers.new("SynthFaceBeard", "PARTICLE_SYSTEM")
    # Hair breaks after a topology-changing modifier (the helper-geometry Mask): put it first.
    basemesh.modifiers.move(len(basemesh.modifiers) - 1, 0)
    psys = mod.particle_system
    psys.seed = beard["particle_seed"]
    psys.vertex_group_density = BEARD_GROUP
    ps = psys.settings
    ps.type = "HAIR"
    ps.count = style["parents"]
    ps.hair_length = beard["length_mm"] / 1000.0
    ps.emit_from = "FACE"
    ps.use_modifier_stack = False
    # Direction = surface normal + a downward pull (droop). In "advanced" mode Blender takes the
    # strand length from that velocity, so set the direction, measure, then rescale to length_mm.
    ps.use_advanced_hair = True
    ps.normal_factor = 1.0
    ps.object_align_factor = (0.0, 0.0, -style["droop"])
    ps.display_step = ps.render_step = 3
    measured = _mean_strand_length(basemesh)
    scale = beard["length_mm"] / 1000.0 / max(measured, 1e-9)
    ps.normal_factor *= scale
    ps.object_align_factor = tuple(v * scale for v in ps.object_align_factor)
    ps.child_type = "INTERPOLATED"
    ps.child_percent = style["children"]
    ps.rendered_child_count = style["children"]
    ps.clump_factor = style["clump"]
    ps.roughness_1 = style["roughness_mm"] / 1000.0
    ps.roughness_endpoint = style["roughness_mm"] / 2000.0
    ps.root_radius = 1.0
    ps.tip_radius = 0.2
    ps.radius_scale = style["root_mm"] / 1000.0 / 2.0
    ps.material_slot = BEARD_MATERIAL


def _mhclo_uuid(mhclo_file: str) -> str:
    with open(mhclo_file, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("uuid "):
                return line.split(None, 1)[1].strip()
    raise ValueError(f"No uuid in {mhclo_file}")


def eye_centres(eyes) -> tuple[Vector, Vector]:
    """World-space centres of the subject's left (+X) and right (-X) eyeballs.

    The human is centred on x = 0, so the eye mesh splits into the two eyes by the sign of x.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = eyes.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    left, right = Vector(), Vector()
    n_left = n_right = 0
    for vert in mesh.vertices:
        co = evaluated.matrix_world @ vert.co
        if co.x >= 0:
            left += co
            n_left += 1
        else:
            right += co
            n_right += 1
    evaluated.to_mesh_clear()
    if not n_left or not n_right:
        raise RuntimeError("Eye mesh does not straddle x = 0")
    return left / n_left, right / n_right


def eye_geometry(eyes) -> tuple[Vector, float]:
    """(midpoint between eye centres, inter-ocular distance) in world space."""
    left, right = eye_centres(eyes)
    return (left + right) / 2, (left - right).length
