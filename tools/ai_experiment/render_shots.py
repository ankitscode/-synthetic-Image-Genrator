"""Consistency test, step 1: fixed shots of one main person + control people (Blender).

Run headless (experiment only, outputs stay in output/ai_experiment/):
    /snap/bin/blender --background --python tools/ai_experiment/render_shots.py

For every person and shot it writes, under output/ai_experiment/consistency/shots/<person>/:
    <shot>.png         512x512 render (the B input)
    <shot>_depth.png   inverse depth, near = bright, background black (ControlNet-depth input)
    <shot>_normal.png  camera-space normals as RGB
    <shot>_mask.png    face mask: skin mesh within QA radius of the face centre
    <shot>.json        shot settings + eye centres in pixels
    <shot>_hair.png    hair mask: pixels whose first hit is the hair or its underlayer
Depth, normals and masks come from ray casts against the 3D scene (no face detector).

Hair masks only (no render; the camera placement is deterministic, so they match the PNGs):
    /snap/bin/blender --background --python tools/ai_experiment/render_shots.py -- --hair-masks

Many people (a named set, e.g. on a Colab GPU), output/ai_experiment/sets/<set>/shots/<person>/:
    blender --background --python tools/ai_experiment/render_shots.py -- \
        --set colab_10 --seed 20261013 --people 1-10 --shots s1,s2,s3,s6 --device OPTIX
Shots already written (<shot>.json exists) are skipped, so a re-run resumes.
"""

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "blender"))
sys.path.insert(0, str(PROJECT_ROOT))

import human  # noqa: E402
import scene  # noqa: E402
from shared import constants as C  # noqa: E402
from shared import sampling  # noqa: E402

OUT_DIR = PROJECT_ROOT / "output" / "ai_experiment" / "consistency" / "shots"
SETS_DIR = PROJECT_ROOT / "output" / "ai_experiment" / "sets"
FACE_TARGETS_FILE = PROJECT_ROOT / "config" / "face_targets.json"
HDRI_DIR = PROJECT_ROOT / "assets" / "hdri"
JOB_SEED = 20261010  # people from output/test_12x6
MAIN_PERSON = 11
CONTROL_PEOPLE = [2, 5, 8, 12]  # other men from the same job, so the test is not too easy
CONTROL_SHOTS = ["s1", "s2", "s3", "s6"]
SIZE = 512
SAMPLES = 48
HDRI_MEAN_LUMINANCE = 0.3
HDRI_KEY_FACTOR = 0.5
FSTOP = 2.8

CAMERA_BASE = {"roll": 0.0, "distance_m": 0.55, "focal_mm": 50.0, "face_fill": 0.58}
KEY = {"azimuth": 30.0, "elevation": 25.0, "kelvin": 5500.0, "intensity": 1.0, "size_m": 0.6}
WARM_KEY = {"azimuth": -35.0, "elevation": 20.0, "kelvin": 3200.0, "intensity": 1.5, "size_m": 0.5}
FLAT_BG = {"kind": "solid", "top_rgb": [0.32, 0.31, 0.30], "bottom_rgb": [0.32, 0.31, 0.30],
           "ambient_strength": 0.2, "ambient_tint": 0.0}
# shot id -> (yaw, pitch, lighting); lighting = HDRI id or "warm_key"
SHOTS = {
    "s1": (0.0, 0.0, "lebombo"),
    "s2": (15.0, 0.0, "cayley_interior"),
    "s3": (-15.0, 0.0, "wide_street_01"),
    "s4": (30.0, 0.0, "canary_wharf"),
    "s5": (-30.0, 0.0, "paul_lobe_haus"),
    "s6": (0.0, 10.0, "warm_key"),
    "s7": (0.0, -10.0, "brown_photostudio_02"),
    "s8": (15.0, 10.0, "warm_key"),
}
HDRI_ROTATION_DEG = 120.0
HAIR_MASK_SHOTS = {MAIN_PERSON: ["s1", "s2", "s3", "s6"], 2: ["s1", "s2", "s3", "s6"]}  # hair-fix + blind test


def hair_objects() -> set:
    """The hairstyle mesh and its opaque underlayer (see human.add_hair)."""
    under = [o for o in bpy.data.objects if o.name.endswith(f".{human.HAIR_UNDERLAYER_SUFFIX}")]
    return {o for u in under for o in (u, bpy.data.objects[u.name.rsplit(".", 1)[0]])}


def ray_maps(camera, basemesh, eye_mid, iod) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-pixel inverse depth, camera-space normal, face mask and hair mask by ray casting."""
    scn = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    top_right, _, bottom_left, top_left = camera.data.view_frame(scene=scn)
    rot = camera.matrix_world.to_3x3()
    rot_inv = rot.inverted()
    origin = camera.matrix_world.translation
    centre = scene.face_centre(eye_mid, iod)
    radius = C.QA_FACE_RADIUS_PER_IOD * iod
    depth = np.zeros((SIZE, SIZE), np.float32)
    normal = np.zeros((SIZE, SIZE, 3), np.float32)
    mask = np.zeros((SIZE, SIZE), np.uint8)
    hair = np.zeros((SIZE, SIZE), np.uint8)
    hair_objs = hair_objects()
    for py in range(SIZE):
        v = (py + 0.5) / SIZE
        for px in range(SIZE):
            u = (px + 0.5) / SIZE
            local = top_left + (top_right - top_left) * u + (bottom_left - top_left) * v
            direction = (rot @ local).normalized()
            hit, location, n, _, obj, _ = scn.ray_cast(depsgraph, origin, direction)
            if not hit:
                continue
            depth[py, px] = 1.0 / (location - origin).length
            normal[py, px] = rot_inv @ n
            if obj.original == basemesh and (location - centre).length <= radius:
                mask[py, px] = 255
            if obj.original in hair_objs:
                hair[py, px] = 255
    hits = depth > 0
    lo, hi = depth[hits].min(), depth[hits].max()
    depth[hits] = 0.1 + 0.9 * (depth[hits] - lo) / max(hi - lo, 1e-6)  # near = 1.0, far hit = 0.1
    return depth, normal, mask, hair


def save_gray(path: Path, arr: np.ndarray) -> None:
    rgba = np.ones((SIZE, SIZE, 4), np.float32)
    rgba[..., :3] = arr[..., None]
    _save(path, rgba)


def _save(path: Path, rgba: np.ndarray) -> None:
    img = bpy.data.images.new(path.stem, SIZE, SIZE)
    img.pixels.foreach_set(rgba[::-1].ravel())  # Blender rows are bottom-up
    img.filepath_raw = str(path)
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)


def render_person(index: int, shot_ids: list[str], categories: dict, params: dict, hdris: dict,
                  hair_masks_only: bool = False, job_seed: int = JOB_SEED, out_dir: Path = OUT_DIR,
                  device: str = "CPU") -> None:
    person_dir = out_dir / f"id_{index:05d}"
    if not hair_masks_only:
        shot_ids = [s for s in shot_ids if not (person_dir / f"{s}.json").exists()]
        if not shot_ids:
            return
    ident = sampling.sample_identity(job_seed, index, params, {n: c["region"] for n, c in categories.items()})
    human.clear_scene()
    basemesh, eyes = human.build_human(ident, categories)
    eye_mid, iod = human.eye_geometry(eyes)
    eye_left, eye_right = human.eye_centres(eyes)
    camera, light = scene.create_rig()
    scene.configure_render(SAMPLES, device)
    scn = bpy.context.scene
    scn.render.resolution_x = scn.render.resolution_y = SIZE
    person_dir.mkdir(parents=True, exist_ok=True)

    for shot in shot_ids:
        yaw, pitch, lighting = SHOTS[shot]
        cam = dict(CAMERA_BASE, yaw=yaw, pitch=pitch)
        derived = scene.place_camera(camera, eye_mid, iod, cam)
        if hair_masks_only:
            if (person_dir / f"{shot}_hair.png").exists():
                continue
            hair = ray_maps(camera, basemesh, eye_mid, iod)[3]
            save_gray(person_dir / f"{shot}_hair.png", hair.astype(np.float32) / 255.0)
            print(f"HAIR id_{index:05d} {shot} pixels={int((hair > 0).sum())}")
            continue
        if lighting == "warm_key":
            scene.place_key_light(light, eye_mid, WARM_KEY)
            scene.set_background(FLAT_BG)
            scene.set_depth_of_field(camera, eye_mid, None)
        else:
            scene.place_key_light(light, eye_mid, KEY)
            light.data.energy *= HDRI_KEY_FACTOR
            scene.set_hdri_world(str(HDRI_DIR / hdris[lighting]["file"]), HDRI_ROTATION_DEG, HDRI_MEAN_LUMINANCE)
            scene.set_depth_of_field(camera, eye_mid, FSTOP)
        scn.render.filepath = str(person_dir / f"{shot}.png")
        bpy.ops.render.render(write_still=True)

        depth, normal, mask, hair = ray_maps(camera, basemesh, eye_mid, iod)
        save_gray(person_dir / f"{shot}_depth.png", depth)
        save_gray(person_dir / f"{shot}_hair.png", hair.astype(np.float32) / 255.0)
        save_gray(person_dir / f"{shot}_mask.png", mask.astype(np.float32) / 255.0)
        rgba = np.ones((SIZE, SIZE, 4), np.float32)
        rgba[..., :3] = normal * 0.5 + 0.5
        _save(person_dir / f"{shot}_normal.png", rgba)

        eyes_px = {}
        for side, co in (("subject_left", eye_left), ("subject_right", eye_right)):
            ndc = world_to_camera_view(scn, camera, co)
            eyes_px[side] = [ndc.x * SIZE, (1.0 - ndc.y) * SIZE]
        meta = {"person": f"id_{index:05d}", "shot": shot, "yaw": yaw, "pitch": pitch, "lighting": lighting,
                "camera": cam, "derived": derived, "eyes_px": eyes_px, "face_mask_pixels": int((mask > 0).sum()),
                "job_seed": job_seed, "identity_seed": ident["seed"], "sex": ident["sex"],
                "age_years": ident["age_years"], "hair": ident["hair"], "skin": ident["skin"]}
        (person_dir / f"{shot}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"SHOT id_{index:05d} {shot} yaw={yaw:+.0f} pitch={pitch:+.0f} {lighting} mask={meta['face_mask_pixels']}")


def parse_people(spec: str) -> list[int]:
    people: list[int] = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        people += list(range(int(lo), int(hi or lo) + 1))
    return people


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--hair-masks", action="store_true", help="only write <shot>_hair.png (no render)")
    parser.add_argument("--set", help="named set under output/ai_experiment/sets/ (many people)")
    parser.add_argument("--seed", type=int, default=JOB_SEED, help="job seed of the identities")
    parser.add_argument("--people", default="1-4", help="identity indexes, e.g. 1-10 or 1,3,7")
    parser.add_argument("--shots", default="s1,s2,s3,s6", help=f"comma list from {','.join(SHOTS)}")
    parser.add_argument("--device", default="CPU", help="CPU, CUDA or OPTIX")
    args = parser.parse_args(argv)
    categories = {c["name"]: c for c in json.loads(FACE_TARGETS_FILE.read_text())["categories"]}
    params = sampling.resolve_params()
    hdris = {h["id"]: h for h in json.loads((HDRI_DIR / "hdris.json").read_text())}
    human.load_mpfb()
    if args.set:
        out_dir = SETS_DIR / args.set / "shots"
        shots = args.shots.split(",")
        for index in parse_people(args.people):
            render_person(index, shots, categories, params, hdris, job_seed=args.seed, out_dir=out_dir,
                          device=args.device)
            print(f"PERSON_DONE id_{index:05d}", flush=True)
        return
    if args.hair_masks:
        for index, shots in HAIR_MASK_SHOTS.items():
            render_person(index, shots, categories, params, hdris, hair_masks_only=True)
        return
    render_person(MAIN_PERSON, list(SHOTS), categories, params, hdris)
    for index in CONTROL_PEOPLE:
        render_person(index, CONTROL_SHOTS, categories, params, hdris)


main()
