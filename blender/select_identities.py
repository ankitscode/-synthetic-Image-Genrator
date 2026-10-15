"""Choose identity indices of a seed with no near-duplicate faces (no rendering, CPU only).

    /snap/bin/blender --background --python blender/select_identities.py -- --seed 20261013 --count 200 \
        [--mix female=100,male_clean=40,male_beard=60] [--out output/selections/<name>.json]

Walks identity indices 1, 2, 3, ... For each one it builds only the face shape (macros + face
targets, ~1 s), measures the RMS face-shape distance to every identity accepted so far (eye
midpoint origin, IOD = 1) and rejects it when it is a near-duplicate of one of them
(shared/similarity.py). Look-alikes (similar shape + shared hair / skin...) are only counted.
Deterministic: same seed and settings give the same selection.

The output's "accepted" list goes into job.json "identity_indices" (or the Colab notebooks).
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "blender"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

import human  # noqa: E402
from pick_identities import groups_of, parse_mix  # noqa: E402
from shared import constants as C  # noqa: E402
from shared import sampling, similarity  # noqa: E402

FACE_TARGETS_FILE = PROJECT_ROOT / "config" / "face_targets.json"
FACE_RADIUS_PER_IOD = 1.7  # same face region as blender/face_shape_distance.py
FRONT_OF_EARS_PER_IOD = 0.6
MAX_CANDIDATES = 100_000


def eye_frame(basemesh, co: np.ndarray) -> tuple[np.ndarray, float]:
    left = co[human._group_members(basemesh, "helper-l-eye")].mean(axis=0)
    right = co[human._group_members(basemesh, "helper-r-eye")].mean(axis=0)
    return (left + right) / 2, float(np.linalg.norm(left - right))


def face_region(basemesh) -> np.ndarray:
    co = human._world_coords(basemesh)
    body = human._group_members(basemesh, "body")
    mid, iod = eye_frame(basemesh, co)
    ear_y = co[human._group_members(basemesh, "ears"), 1].mean()
    near = np.linalg.norm(co[body] - mid, axis=1) < FACE_RADIUS_PER_IOD * iod
    front = co[body, 1] < ear_y - FRONT_OF_EARS_PER_IOD * iod
    return body[near & front]


def face_shape(identity: dict, categories: dict, region: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Normalised face vertices of an identity (shape only: no skin, eyes, hair)."""
    human.clear_scene()
    basemesh = human.service("humanservice", "HumanService").create_human(macro_detail_dict=identity["macros"])
    human.apply_face_targets(basemesh, identity["face_targets"], categories)
    if region is None:
        region = face_region(basemesh)
    co = human._world_coords(basemesh)
    mid, iod = eye_frame(basemesh, co)
    return (co[region] - mid) / iod, region


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--count", type=int, help="identities to accept (without --mix)")
    parser.add_argument("--mix", help="group quotas, e.g. female=100,male_clean=40,male_beard=60")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    need = parse_mix(args.mix) if args.mix else None
    target = sum(need.values()) if need else args.count
    out = args.out or PROJECT_ROOT / "output" / "selections" / f"seed{args.seed}_{target}.json"

    categories = {c["name"]: c for c in json.loads(FACE_TARGETS_FILE.read_text())["categories"]}
    params = sampling.resolve_params()
    human.load_mpfb()
    region = None
    accepted, idents, shapes, rejected, skipped_group = [], [], [], [], 0
    nearest, lookalike_pairs = [], 0
    t0 = time.perf_counter()
    for index in range(1, MAX_CANDIDATES + 1):
        if len(accepted) >= target:
            break
        identity = sampling.sample_identity(args.seed, index, params, {n: c["region"] for n, c in categories.items()})
        group = None
        if need is not None:
            group = next((g for g in groups_of(identity) if need.get(g, 0) > 0), None)
            if group is None:
                skipped_group += 1
                continue
        shape, region = face_shape(identity, categories, region)
        if shapes:
            dist = np.sqrt(((np.stack(shapes) - shape) ** 2).sum(axis=2).mean(axis=1))
            dupes = [(accepted[k], float(dist[k])) for k in np.argsort(dist)[:20]
                     if similarity.near_duplicate(float(dist[k]), identity, idents[k])]
            if dupes:
                rejected.append({"index": index, "near_duplicate_of": dupes[0][0], "shape_iod": round(dupes[0][1], 4)})
                print(f"REJECT id_{index:05d} near-duplicate of id_{dupes[0][0]:05d} ({dupes[0][1]:.3f} IOD)", flush=True)
                continue
            nearest.append(float(dist.min()))
            lookalike_pairs += sum(similarity.too_similar(float(d), similarity.shared_traits(identity, idents[k]))
                                   for k, d in enumerate(dist))
        accepted.append(index)
        idents.append(identity)
        shapes.append(shape)
        if need is not None:
            need[group] -= 1
        if len(accepted) % 25 == 0:
            print(f"ACCEPTED {len(accepted)}/{target}  candidates {index}  {time.perf_counter() - t0:.0f}s", flush=True)

    pairs = len(accepted) * (len(accepted) - 1) // 2
    result = {
        "seed": args.seed, "sampler_version": C.SAMPLER_VERSION, "mix": args.mix, "count": len(accepted),
        "accepted": accepted, "rejected_near_duplicates": rejected, "skipped_for_mix": skipped_group,
        "rules": {"near_duplicate_shape_iod": C.NEAR_DUPLICATE_SHAPE_IOD, "near_duplicate_tone_diff": C.NEAR_DUPLICATE_TONE_DIFF,
                  "lookalike_shape_iod": C.LOOKALIKE_SHAPE_IOD, "lookalike_min_shared": C.LOOKALIKE_MIN_SHARED},
        "nearest_accepted_shape_iod": {"min": min(nearest, default=None), "median": float(np.median(nearest)) if nearest else None},
        "lookalike_pairs": lookalike_pairs, "pairs": pairs, "seconds": round(time.perf_counter() - t0),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"SELECTION {len(accepted)} accepted, {len(rejected)} near-duplicates rejected, "
          f"look-alike pairs {lookalike_pairs}/{pairs} ({100 * lookalike_pairs / max(pairs, 1):.1f}%), "
          f"nearest shape min {result['nearest_accepted_shape_iod']['min']} -> {out}", flush=True)


main()
