"""Beard audit: every beard style on a few men, frontal and turned (in-house particle beards).

    /snap/bin/blender --background --python blender/debug_beard.py -- [--seed 20261013] [--people 11,19]
    venv/bin/python tools/audit_sheet.py output/smoke/beard_audit

Each person keeps their own sampled beard colour / density / patchiness; only the style changes.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "blender"))
sys.path.insert(0, str(PROJECT_ROOT))

import audit_common  # noqa: E402
import bpy  # noqa: E402
import human  # noqa: E402
import scene  # noqa: E402
from shared import constants as C  # noqa: E402
from shared import sampling  # noqa: E402

OUT_DIR = PROJECT_ROOT / "output" / "smoke" / "beard_audit"
VIEWS = {"front": 0.0, "yaw35": 35.0}
SAMPLES = 32
FALLBACK_BEARD = {"color_name": "black", "rgb": list(C.HAIR_COLORS["black"]), "length_mm": 0.0, "density": 0.85,
                  "noise_amount": 0.3, "noise_offset": [1.0, 2.0, 3.0], "particle_seed": 7}


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--people", default="11,19")
    args = parser.parse_args(argv)

    categories = {c["name"]: c for c in json.loads((PROJECT_ROOT / "config" / "face_targets.json").read_text())["categories"]}
    params = sampling.resolve_params()
    human.load_mpfb()
    tiles = []
    for index in (int(i) for i in args.people.split(",")):
        ident = sampling.sample_identity(args.seed, index, params, {n: c["region"] for n, c in categories.items()})
        own = ident["beard"] if ident["beard"]["style"] != C.BEARD_NONE else dict(FALLBACK_BEARD)
        for style in [C.BEARD_NONE, *C.BEARD_STYLES]:
            beard = {"style": C.BEARD_NONE} if style == C.BEARD_NONE else \
                {**own, "style": style, "length_mm": sum(C.BEARD_STYLES[style]["length_mm"]) / 2}
            human.clear_scene()
            basemesh, eyes = human.build_human({**ident, "beard": beard}, categories)
            eye_mid, iod = human.eye_geometry(eyes)
            camera = audit_common.frontal_neutral(eye_mid, iod)
            bpy.context.scene.cycles.samples = SAMPLES
            for view, yaw in VIEWS.items():
                scene.place_camera(camera, eye_mid, iod, {**audit_common.NEUTRAL_CAMERA, "yaw": yaw})
                name = f"id_{index:05d}_{style}_{view}.png"
                audit_common.render_tile(OUT_DIR / name)
                tiles.append({"file": name, "label": f"id_{index:05d} {style} {view}\n"
                              f"{beard.get('color_name', '')} {beard.get('length_mm', 0):.0f}mm"})
                print(f"BEARD {name}", flush=True)
    audit_common.write_manifest(OUT_DIR, f"Beard audit (seed {args.seed})", len(VIEWS) * 2, tiles)


main()
