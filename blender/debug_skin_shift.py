"""Debug sheet for the per-identity skin tone shift (human.apply_skin_shift).

Run headless:
    /snap/bin/blender --background --python blender/debug_skin_shift.py

Rows: one base skin per race (young male). Columns: HSV settings in SHIFTS.
Same face, frontal camera, neutral 5500 K light, grey background for every tile.
Writes output/smoke/skin_shift.png (row-major, top-left = first skin, first shift).
"""

import sys
from pathlib import Path

import bpy
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "blender"))
sys.path.insert(0, str(PROJECT_ROOT))

import human  # noqa: E402
import scene  # noqa: E402
from shared import constants as C  # noqa: E402

OUT_FILE = PROJECT_ROOT / "output" / "smoke" / "skin_shift.png"
SKINS = ["young_african_male", "young_asian_male", "young_caucasian_male"]
# (hue_shift, saturation, value): min value, mode value, unshifted, max value, hue/sat extremes
SHIFTS = [
    (0.0, 1.0, 0.75),
    (0.0, 1.0, 0.92),
    (0.0, 1.0, 1.00),
    (0.0, 1.0, 1.15),
    (-0.02, 0.85, 1.0),
    (0.02, 1.15, 1.0),
]
CAMERA = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "distance_m": 0.55, "focal_mm": 50.0, "face_fill": 0.58}
LIGHT = {"azimuth": 30.0, "elevation": 20.0, "kelvin": 5500.0, "intensity": 1.0, "size_m": 0.6}
BACKGROUND = {"kind": "solid", "top_rgb": [0.3, 0.3, 0.3], "bottom_rgb": [0.3, 0.3, 0.3], "ambient_strength": 0.25, "ambient_tint": 0.0}


def read_png(path: Path) -> np.ndarray:
    img = bpy.data.images.load(str(path))
    width, height = img.size
    pixels = np.empty(width * height * 4, dtype=np.float32)
    img.pixels.foreach_get(pixels)
    bpy.data.images.remove(img)
    return pixels.reshape(height, width, 4)


def main() -> None:
    human.load_mpfb()
    HumanService = human.service("humanservice", "HumanService")
    AssetService = human.service("assetservice", "AssetService")
    tmp_dir = OUT_FILE.parent / "skin_shift_tiles"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for skin in SKINS:
        human.clear_scene()
        macros = dict(C.MPFB_MACRO_DEFAULTS, gender=1.0)
        basemesh = HumanService.create_human(macro_detail_dict=macros)
        skin_file = next(str(p) for p in AssetService.list_mhmat_assets("skins") if p.stem == skin)
        HumanService.set_character_skin(skin_file, basemesh)
        eyes_file = next(str(p) for p in AssetService.list_mhclo_assets("eyes") if p.stem == C.EYES_MESH)
        eyes = HumanService.add_mhclo_asset(eyes_file, basemesh, asset_type="Eyes", set_up_rigging=False)
        eye_mid, iod = human.eye_geometry(eyes)
        camera, light = scene.create_rig()
        scene.configure_render(C.DEFAULT_RENDER_SAMPLES, C.DEFAULT_RENDER_DEVICE)
        scene.place_camera(camera, eye_mid, iod, CAMERA)
        scene.place_key_light(light, eye_mid, LIGHT)
        scene.set_background(BACKGROUND)

        tiles = []
        for i, (hue, sat, val) in enumerate(SHIFTS):
            human.apply_skin_shift(basemesh, hue, sat, val)
            path = tmp_dir / f"{skin}_{i}.png"
            bpy.context.scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            tiles.append(read_png(path))
        rows.append(np.concatenate(tiles, axis=1))

    sheet = np.concatenate(rows[::-1], axis=0)  # Blender pixel rows are bottom-up
    height, width, _ = sheet.shape
    img = bpy.data.images.new("skin_shift", width, height)
    img.pixels.foreach_set(sheet.ravel())
    img.filepath_raw = str(OUT_FILE)
    img.file_format = "PNG"
    img.save()
    print(f"Wrote {OUT_FILE}")


main()
