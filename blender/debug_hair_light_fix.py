"""Before / after check of the afro01 texture softening and the key-light white balance.

    blender --background --python blender/debug_hair_light_fix.py -- --seed 20261013 \
        --ids 193,205,210,250,286,307,244,255,346 [--device OPTIX] [--out output/smoke/hair_light_fix]
    python tools/audit_sheet.py output/smoke/hair_light_fix

Each person (rebuilt exactly as rendered in a job with that seed) is rendered frontally under the
warmest key light (3000 K) and a neutral one (5500 K): "before" = old look (texture detail 1.0,
no white balance), "after" = current constants. Writes PNGs + manifest.json for tools/audit_sheet.py.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "blender"))
sys.path.insert(0, str(PROJECT_ROOT))

import bpy  # noqa: E402

import audit_common  # noqa: E402
import human  # noqa: E402
import scene  # noqa: E402
from shared import constants as C  # noqa: E402
from shared import sampling  # noqa: E402

FACE_TARGETS_FILE = PROJECT_ROOT / "config" / "face_targets.json"
KELVINS = (3000.0, 5500.0)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ids", required=True, help="identity indices, e.g. 193,205,286")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "output" / "smoke" / "hair_light_fix")
    args = parser.parse_args(argv)

    categories = {c["name"]: c for c in json.loads(FACE_TARGETS_FILE.read_text())["categories"]}
    regions = {n: c["region"] for n, c in categories.items()}
    params = sampling.resolve_params()
    after = {"detail": dict(C.HAIR_TEXTURE_DETAIL), "wb": C.WHITE_BALANCE_STRENGTH}
    variants = {"before": {"detail": {}, "wb": 0.0}, "after": after}
    human.load_mpfb()

    tiles = []
    for index in (int(i) for i in args.ids.split(",")):
        identity = sampling.sample_identity(args.seed, index, params, regions)
        for name, variant in variants.items():
            C.HAIR_TEXTURE_DETAIL.clear()
            C.HAIR_TEXTURE_DETAIL.update(variant["detail"])
            C.WHITE_BALANCE_STRENGTH = variant["wb"]
            human.clear_scene()
            basemesh, eyes = human.build_human(identity, categories)
            eye_mid, iod = human.eye_geometry(eyes)
            camera = audit_common.frontal_neutral(eye_mid, iod)
            scene.configure_render(audit_common.AUDIT_SAMPLES, args.device)
            light = next(o for o in bpy.data.objects if o.type == "LIGHT")
            for kelvin in KELVINS:
                rendered = scene.place_key_light(light, eye_mid, {**audit_common.NEUTRAL_LIGHT, "kelvin": kelvin})
                file = f"id_{index:05d}_{name}_{int(kelvin)}K.png"
                audit_common.render_tile(args.out / file)
                beard = identity.get("beard", {}).get("style", "none")
                tiles.append({"file": file, "label": f"id_{index:05d} {name} {int(kelvin)}K (rendered {rendered:.0f}K)\n"
                                                     f"{identity['hair']['style']} {identity['hair']['color_name']}, "
                                                     f"tone {identity['skin']['tone']:.2f}, beard {beard}"})
            print(f"FIX_PREVIEW id_{index:05d} {name}", flush=True)
    audit_common.write_manifest(args.out, "Hair texture + white balance: before | after, 3000 K and 5500 K", 4, tiles)


main()
