"""Before/after realism preview: same identities and poses, three looks.

Run headless:
    /snap/bin/blender --background --python blender/debug_realism_preview.py
    venv/bin/python tools/audit_sheet.py output/smoke/realism_preview --tile 192

  A current   : system skin blend, flat/gradient background (as test_12x6)
  B +skin     : natural CC0 skin (Skins 01/02 packs) as the light base, same tone blend
  C +skin+HDRI: B lit by a CC0 Poly Haven HDRI (seen behind the face) + the key light at 0.5x

Identities/poses come from the test_12x6 job seed, so A matches what was reviewed there.
Writes output/smoke/realism_preview/{*.png, manifest.json}.
"""

import copy
import json
import random
import sys
from pathlib import Path

import bpy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "blender"))
sys.path.insert(0, str(PROJECT_ROOT))

import audit_common  # noqa: E402
import human  # noqa: E402
import scene  # noqa: E402
from shared import sampling  # noqa: E402

OUT_DIR = PROJECT_ROOT / "output" / "smoke" / "realism_preview"
FACE_TARGETS_FILE = PROJECT_ROOT / "config" / "face_targets.json"
HDRI_DIR = PROJECT_ROOT / "assets" / "hdri"
JOB_SEED = 20261010  # output/test_12x6
IDENTITIES = [4, 5, 11]
IMAGES = [1, 2]

# Natural-looking skins from the CC0 Skins 01/02 packs (no tattoos, no heavy makeup).
NATURAL_SKINS = {
    "female": [
        "callharvey3d_midtoned_female", "cutoff3d_indian_female_skin", "cutoff3d_indian_female_enhanced",
        "skalldyrssuppe_creamy_female", "onlytheghosts_young_eurasian_female",
        "onlytheghosts_middle_aged_eurasian_female", "toigo_light_skin_female_bronze",
        "darthfurby_caucasian_female", "toigo_light_skin_female_freckles",
    ],
    "male": [
        "toigo_light_skin_male_bronze", "onlytheghosts_old_eurasian_male", "mindfront_aksel_skin",
        "toigo_light_skin_male_freckles",
    ],
}
DARK_SKIN_MALE = "mindfront_skin_male_african_middleage"  # darker than the system african skins
HDRI_MEAN_LUMINANCE = 0.3
HDRI_KEY_LIGHT_FACTOR = 0.5


def natural_skin(identity: dict) -> dict:
    skin = copy.deepcopy(identity["skin"])
    rng = random.Random(identity["seed"])
    skin["light"] = rng.choice(NATURAL_SKINS[identity["sex"]])
    if identity["sex"] == "male":
        skin["dark"] = DARK_SKIN_MALE
    return skin


def main() -> None:
    categories = {c["name"]: c for c in json.loads(FACE_TARGETS_FILE.read_text())["categories"]}
    regions = {n: c["region"] for n, c in categories.items()}
    params = sampling.resolve_params()
    hdris = json.loads((HDRI_DIR / "hdris.json").read_text())
    human.load_mpfb()

    tiles = []
    for index in IDENTITIES:
        base = sampling.sample_identity(JOB_SEED, index, params, regions)
        looks = [("A current", base, False), ("B +skin", dict(base, skin=natural_skin(base)), False),
                 ("C +skin+HDRI", dict(base, skin=natural_skin(base)), True)]
        for look, ident, use_hdri in looks:
            human.clear_scene()
            _, eyes = human.build_human(ident, categories)
            eye_mid, iod = human.eye_geometry(eyes)
            camera, light = scene.create_rig()
            scene.configure_render(audit_common.AUDIT_SAMPLES, "CPU")
            for image_index in IMAGES:
                image = sampling.sample_image(JOB_SEED, index, image_index, params)
                scene.place_camera(camera, eye_mid, iod, image["camera"])
                scene.place_key_light(light, eye_mid, image["light"])
                extra = ""
                if use_hdri:
                    rng = random.Random(image["seed"])
                    hdri = rng.choice(hdris)
                    world = scene.set_hdri_world(str(HDRI_DIR / hdri["file"]), rng.uniform(0.0, 360.0),
                                                 HDRI_MEAN_LUMINANCE)
                    light.data.energy *= HDRI_KEY_LIGHT_FACTOR
                    extra = f"\n{hdri['id']}"
                else:
                    scene.set_background(image["background"])
                name = f"id{index:02d}_{look.split()[0]}_{image_index}.png"
                audit_common.render_tile(OUT_DIR / name)
                qa = scene.face_exposure_qa(camera, human_basemesh(), eye_mid, iod, str(OUT_DIR / name))
                skin = ident["skin"]
                tiles.append({"file": name, "label": f"id_{index:05d} {look} img{image_index}\n"
                              f"{skin['light'][:26]}\ntone {skin['tone']:.2f}  face luma {qa['face_mean_luma']:.2f}"
                              f"  white {100 * qa['face_white_fraction']:.1f}%{extra}"})
                print(f"REALISM {name} luma={qa['face_mean_luma']:.2f} white={qa['face_white_fraction']:.3%}")
    audit_common.write_manifest(OUT_DIR, "Realism preview: A current | B + natural CC0 skin | C + skin + CC0 HDRI",
                                6, tiles)


def human_basemesh():
    return next(o for o in bpy.data.objects if o.name == "Human")


main()
