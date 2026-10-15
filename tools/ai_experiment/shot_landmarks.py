"""5 landmarks for every shot of an approach-B set, from the 3D model (no detector, no render).

    /snap/bin/blender --background --python tools/ai_experiment/shot_landmarks.py -- --set colab_b_mix20_v4

The B images keep the 3D geometry (shape drift ~1-2 px of 512), so the projected 3D landmarks
of each shot are also the landmarks of its photoreal versions. Writes
output/ai_experiment/sets/<set>/shots/<person>/landmarks.json (same format as a 3D job).
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "blender"))
sys.path.insert(0, str(PROJECT_ROOT))

import bpy  # noqa: E402
import human  # noqa: E402
import landmarks  # noqa: E402
import scene  # noqa: E402
from shared import sampling  # noqa: E402

SETS_DIR = PROJECT_ROOT / "output" / "ai_experiment" / "sets"
FACE_TARGETS_FILE = PROJECT_ROOT / "config" / "face_targets.json"
SIZE = 512  # render_shots.py resolution


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", required=True)
    args = parser.parse_args(argv)
    categories = {c["name"]: c for c in json.loads(FACE_TARGETS_FILE.read_text())["categories"]}
    params = sampling.resolve_params()
    vertices = landmarks.load_vertices()
    human.load_mpfb()
    for person_dir in sorted((SETS_DIR / args.set / "shots").glob("id_*")):
        shots = {p.stem: json.loads(p.read_text()) for p in sorted(person_dir.glob("s*.json"))}
        if not shots:
            continue
        first = next(iter(shots.values()))
        ident = sampling.sample_identity(first["job_seed"], int(person_dir.name.split("_")[1]), params,
                                         {n: c["region"] for n, c in categories.items()})
        human.clear_scene()
        basemesh, eyes = human.build_human(ident, categories)
        eye_mid, iod = human.eye_geometry(eyes)
        camera, _ = scene.create_rig()
        scene.configure_render(1, "CPU")
        bpy.context.scene.render.resolution_x = bpy.context.scene.render.resolution_y = SIZE
        points = landmarks.world_points(basemesh, eyes, vertices)
        see_through = {o.name for o in bpy.data.objects
                       if o.name.endswith((f".{ident['eyebrows']}", f".{ident['eyelashes']}"))}
        images = {}
        for shot, meta in shots.items():
            scene.place_camera(camera, eye_mid, iod, meta["camera"])
            images[shot] = landmarks.project(camera, points, basemesh, eyes, see_through)
            # sanity: the eye centres must match the ones render_shots stored for this shot
            stored = [meta["eyes_px"]["subject_right"], meta["eyes_px"]["subject_left"]]
            err = max(abs(a - b) for p, q in zip(images[shot]["points"][:2], stored) for a, b in zip(p, q))
            if err > 0.5:
                raise RuntimeError(f"{person_dir.name} {shot}: eye centres differ from render_shots by {err:.2f} px")
        out = {"order": landmarks.LANDMARK_NAMES, "vertices": vertices, "resolution": SIZE, "images": images}
        (person_dir / "landmarks.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        hidden = sum(not v for img in images.values() for v in img["visible"])
        print(f"LANDMARKS {person_dir.name} shots={len(images)} hidden_points={hidden}", flush=True)


main()
