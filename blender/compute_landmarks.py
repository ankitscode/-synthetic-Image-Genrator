"""Write landmarks.json for an already rendered job, from its params.json (no rendering).

    /snap/bin/blender --background --python blender/compute_landmarks.py -- --job output/<job_id> [--ids 1,2]

Each identity is rebuilt from its stored parameters, the camera is placed exactly as for each
image, and the 5 landmarks are projected (blender/landmarks.py). Output per identity:
    landmarks.json {"order": [...], "vertices": {...}, "images": {"img_001": {"points", "visible", "blocked_by"}}}
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "blender"))
sys.path.insert(0, str(PROJECT_ROOT))

import bpy  # noqa: E402
import human  # noqa: E402
import landmarks  # noqa: E402
import scene  # noqa: E402
from shared import constants as C  # noqa: E402

FACE_TARGETS_FILE = PROJECT_ROOT / "config" / "face_targets.json"


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--ids", help="identity indexes, e.g. 1,2 (default: all rendered)")
    args = parser.parse_args(argv)

    categories = {c["name"]: c for c in json.loads(FACE_TARGETS_FILE.read_text())["categories"]}
    vertices = landmarks.load_vertices()
    human.load_mpfb()
    wanted = {int(i) for i in args.ids.split(",")} if args.ids else None
    for params_file in sorted(args.job.glob("id_*/params.json")):
        record = json.loads(params_file.read_text(encoding="utf-8"))
        identity = record["identity"]
        if record.get("status") != "done" or (wanted and identity["identity_index"] not in wanted):
            continue
        human.clear_scene()
        basemesh, eyes = human.build_human(identity, categories)
        eye_mid, iod = human.eye_geometry(eyes)
        camera, _ = scene.create_rig()
        scene.configure_render(1, "CPU")  # sets the output resolution used by the projection
        points = landmarks.world_points(basemesh, eyes, vertices)
        see_through = {o.name for o in bpy.data.objects
                       if o.name.endswith((f".{identity['eyebrows']}", f".{identity['eyelashes']}"))}
        images = {}
        for image in record["images"]:
            scene.place_camera(camera, eye_mid, iod, image["camera"])
            images[Path(image["file"]).stem] = landmarks.project(camera, points, basemesh, eyes, see_through)
        out = {"order": landmarks.LANDMARK_NAMES, "vertices": vertices, "resolution": C.RENDER_RESOLUTION,
               "images": images}
        (params_file.parent / "landmarks.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        hidden = sum(not v for img in images.values() for v in img["visible"])
        print(f"LANDMARKS {params_file.parent.name} images={len(images)} hidden_points={hidden}", flush=True)


main()
