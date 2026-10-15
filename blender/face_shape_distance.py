"""3D face-shape distance between every pair of identities of a rendered job (no rendering).

    /snap/bin/blender --background --python blender/face_shape_distance.py -- --job output/<job_id>

Each identity is rebuilt from params.json. Face vertices (front of the head, chosen once on the
default human, same indices for everyone) are moved so the eye midpoint is the origin and scaled
so the inter-ocular distance (IOD) is 1. Distance = RMS vertex distance in IOD units (x 63 mm
gives a rough millimetre figure). Writes <job>/face_shape.json.
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "blender"))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

import human  # noqa: E402

FACE_TARGETS_FILE = PROJECT_ROOT / "config" / "face_targets.json"
FACE_RADIUS_PER_IOD = 1.7  # face region around the eye midpoint ...
FRONT_OF_EARS_PER_IOD = 0.6  # ... and in front of the ear centres by this much
TYPICAL_IOD_MM = 63.0


def face_vertices() -> np.ndarray:
    """Front-of-face vertex indices on the default human."""
    basemesh = human.service("humanservice", "HumanService").create_human()
    co = human._world_coords(basemesh)
    body = human._group_members(basemesh, "body")
    left = co[human._group_members(basemesh, "helper-l-eye")].mean(axis=0)
    right = co[human._group_members(basemesh, "helper-r-eye")].mean(axis=0)
    mid, iod = (left + right) / 2, float(np.linalg.norm(left - right))
    ear_y = co[human._group_members(basemesh, "ears"), 1].mean()
    near = np.linalg.norm(co[body] - mid, axis=1) < FACE_RADIUS_PER_IOD * iod
    front = co[body, 1] < ear_y - FRONT_OF_EARS_PER_IOD * iod
    human.clear_scene()
    return body[near & front]


def normalised_face(identity: dict, categories: dict, vertices: np.ndarray) -> np.ndarray:
    human.clear_scene()
    basemesh, eyes = human.build_human({**identity, "beard": {"style": "none"}}, categories)
    mid, iod = human.eye_geometry(eyes)
    co = human._world_coords(basemesh)[vertices]
    return (co - np.array(mid)) / iod


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, type=Path)
    args = parser.parse_args(argv)
    categories = {c["name"]: c for c in json.loads(FACE_TARGETS_FILE.read_text())["categories"]}
    human.load_mpfb()
    vertices = face_vertices()
    names, faces = [], []
    for params_file in sorted(args.job.glob("id_*/params.json")):
        record = json.loads(params_file.read_text(encoding="utf-8"))
        if record.get("status") != "done":
            continue
        names.append(params_file.parent.name)
        faces.append(normalised_face(record["identity"], categories, vertices))
        print(f"SHAPE {names[-1]}", flush=True)
    n = len(names)
    matrix = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        matrix[i, j] = matrix[j, i] = float(np.sqrt(((faces[i] - faces[j]) ** 2).sum(axis=1).mean()))
    out = {"identities": names, "face_vertices": int(len(vertices)), "units": "IOD (x ~63 mm)",
           "typical_iod_mm": TYPICAL_IOD_MM, "rms_distance": matrix.round(5).tolist()}
    (args.job / "face_shape.json").write_text(json.dumps(out) + "\n", encoding="utf-8")
    off = matrix[np.triu_indices(n, 1)]
    print(f"FACE_SHAPE pairs={off.size} min={off.min():.4f} median={np.median(off):.4f} IOD", flush=True)


main()
