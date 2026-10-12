"""Approach A sample: 3D pipeline with all realism upgrades (preview only).

Run headless:
    /snap/bin/blender --background --python blender/debug_approach_a.py

Natural CC0 skin + CC0 HDRI environment + key light + background blur. Two identities from
the test_12x6 seed, one image each. Writes output/smoke/approaches/A_<n>.png.
"""

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "blender"))
sys.path.insert(0, str(PROJECT_ROOT))

import audit_common  # noqa: E402
import human  # noqa: E402
import scene  # noqa: E402
from shared import sampling  # noqa: E402

OUT_DIR = PROJECT_ROOT / "output" / "smoke" / "approaches"
FACE_TARGETS_FILE = PROJECT_ROOT / "config" / "face_targets.json"
HDRI_DIR = PROJECT_ROOT / "assets" / "hdri"
JOB_SEED = 20261010
SAMPLES = [(4, 1, "callharvey3d_midtoned_female", "lebombo"), (11, 1, "toigo_light_skin_male_bronze", "wide_street_01")]
HDRI_MEAN_LUMINANCE = 0.3
KEY_LIGHT_FACTOR = 0.5
FSTOP = 2.0
RENDER_SAMPLES = 64


def main() -> None:
    categories = {c["name"]: c for c in json.loads(FACE_TARGETS_FILE.read_text())["categories"]}
    params = sampling.resolve_params()
    hdris = {h["id"]: h for h in json.loads((HDRI_DIR / "hdris.json").read_text())}
    human.load_mpfb()
    for n, (index, image_index, skin_name, hdri_id) in enumerate(SAMPLES, start=1):
        ident = sampling.sample_identity(JOB_SEED, index, params, {k: c["region"] for k, c in categories.items()})
        ident["skin"]["light"] = skin_name
        human.clear_scene()
        _, eyes = human.build_human(ident, categories)
        eye_mid, iod = human.eye_geometry(eyes)
        camera, light = scene.create_rig()
        scene.configure_render(RENDER_SAMPLES, "CPU")
        image = sampling.sample_image(JOB_SEED, index, image_index, params)
        scene.place_camera(camera, eye_mid, iod, image["camera"])
        scene.place_key_light(light, eye_mid, image["light"])
        light.data.energy *= KEY_LIGHT_FACTOR
        scene.set_hdri_world(str(HDRI_DIR / hdris[hdri_id]["file"]), random.Random(image["seed"]).uniform(0, 360),
                             HDRI_MEAN_LUMINANCE)
        scene.set_depth_of_field(camera, eye_mid, FSTOP)
        audit_common.render_tile(OUT_DIR / f"A_{n}.png")
        print(f"APPROACH_A A_{n}.png id_{index:05d} skin={skin_name} hdri={hdri_id}")


main()
