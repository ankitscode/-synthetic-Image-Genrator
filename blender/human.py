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


def apply_strand_color(obj, rgb: list[float]) -> None:
    """Recolour a MAKESKIN hair/eyebrow material while keeping its strand detail.

    diffuseTexture -> RGBtoBW (luminance) -> x (1 / mean luminance) -> scale target RGB
    -> diffuseIntensity.Color2 (which feeds Principled Base Color). Alpha is untouched.
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
    tree.links.new(norm.outputs["Value"], scale.inputs["Scale"])
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

    return basemesh, eyes


def add_hair(basemesh, style: str, rgb: list[float]) -> list:
    """Add a hairstyle (recoloured, with its opaque underlayer). Returns the new objects."""
    AssetService = service("assetservice", "AssetService")
    HumanService = service("humanservice", "HumanService")
    hair_file = _find_asset(AssetService.list_mhclo_assets, "hair", style)
    hair_obj = HumanService.add_mhclo_asset(hair_file, basemesh, asset_type="Hair", set_up_rigging=False)
    apply_strand_color(hair_obj, rgb)
    add_hair_underlayer(hair_obj, rgb)
    return [hair_obj, bpy.data.objects[f"{hair_obj.name}.{HAIR_UNDERLAYER_SUFFIX}"]]


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
