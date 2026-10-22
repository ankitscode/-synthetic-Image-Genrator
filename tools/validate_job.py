"""Data-cleanliness check for a rendered 3D job (approach A). No model, no training.

    venv/bin/python tools/validate_job.py output/<job_id> [--shapes output/selections/<job_id>.shapes.npz]

Needs: params.json + landmarks.json (render; older jobs: blender/compute_landmarks.py), alignment.json +
aligned/ (app/postprocess.py), and face shapes: face_shape.json (blender/face_shape_distance.py) or,
with --shapes, the cache saved by blender/select_identities.py (no Blender needed).

1. Landmarks: all 5 visible, inside the frame, aligned residual small.
2. Same person in every image: identity values and the applied skin are identical across images.
3. Different people: 3D face-shape distance (IOD units) between every pair, with the shared rules
   (shared/similarity.py): near-duplicates (practically the same face) fail the check;
   look-alikes (similar shape + shared hair / skin / sex...) are listed for a visual look.
Writes <job>/validation.json and <job>/validation_pairs.png (closest pairs side by side).
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared import similarity  # noqa: E402

RESIDUAL_FLAG_PX = 6.0  # aligned residual above this (template pixels, 112 crop) = review
SKIN_MATCH_TOL = 1e-4  # applied skin values may differ only by float noise between images
PAIRS_SHOWN = 6
LOOKALIKES_SHOWN = 24  # sheet rows for look-alikes (all are listed in validation.json)
TYPICAL_IOD_MM = 63.0  # same as blender/face_shape_distance.py
TILE = 160


def appearance(identity: dict) -> dict:
    beard = identity.get("beard", {}).get("style", "none")
    return {"sex": identity["sex"], "tone": identity["skin"]["tone"], "eyes": identity["eyes"]["color"],
            "hair": f"{identity['hair']['style']} {identity['hair']['color_name']}", "beard": beard,
            "age": round(identity["age_years"])}


def frontal_aligned(identity_dir: Path, record: dict) -> Image.Image:
    best = min(record["images"], key=lambda im: abs(im["camera"]["yaw"]) + abs(im["camera"]["pitch"]))
    path = identity_dir / "aligned" / f"{Path(best['file']).stem}.jpg"
    if not path.exists():  # broken person (listed in validation.json "broken")
        return Image.new("RGB", (TILE, TILE), (90, 0, 0))
    return Image.open(path).convert("RGB").resize((TILE, TILE))


def shapes_from_selection(path: Path, names: list[str]) -> dict:
    """face_shape.json-like dict from a select_identities.py cache (<selection>.shapes.npz)."""
    import numpy as np
    cache = np.load(path)
    by_name = {f"id_{int(i):05d}": k for k, i in enumerate(cache["indices"])}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise SystemExit(f"{path}: no face shape for {missing[:5]}")
    shapes = cache["shapes"][[by_name[n] for n in names]].astype(np.float64)
    matrix = np.zeros((len(names), len(names)))
    for i in range(len(names)):
        matrix[i] = np.sqrt(((shapes - shapes[i]) ** 2).sum(axis=2).mean(axis=1))
    return {"identities": names, "typical_iod_mm": TYPICAL_IOD_MM, "rms_distance": matrix.tolist()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("job", type=Path)
    parser.add_argument("--shapes", type=Path, help="select_identities.py shape cache instead of face_shape.json")
    args = parser.parse_args()
    job = args.job
    records = {d.name: json.loads((d / "params.json").read_text()) for d in sorted(job.glob("id_*"))}
    records = {n: r for n, r in records.items() if r.get("status") == "done"}
    shape = (shapes_from_selection(args.shapes, list(records)) if args.shapes
             else json.loads((job / "face_shape.json").read_text()))
    report: dict = {"job": job.name, "identities": len(records), "landmarks": {}, "consistency": {}, "pairs": {}}

    # 1. landmarks (+ images that are missing or unreadable: those people must be re-rendered)
    bad = []
    broken: dict[str, list[str]] = {}
    residuals = []
    for name in records:
        lm_file, quality_file = job / name / "landmarks.json", job / name / "alignment.json"
        if not (lm_file.exists() and quality_file.exists()):
            broken[name] = ["landmarks.json" if not lm_file.exists() else "alignment.json"]
            continue
        lm = json.loads(lm_file.read_text())
        quality = json.loads(quality_file.read_text())
        size = lm["resolution"]
        for stem, entry in lm["images"].items():
            if "error" in quality.get(stem, {"error": "not aligned"}):
                broken.setdefault(name, []).append(stem)
                continue
            inside = all(0 <= x < size and 0 <= y < size for x, y in entry["points"])
            r = quality[stem]["residual_px"]
            residuals.append(r)
            if not (all(entry["visible"]) and inside and r <= RESIDUAL_FLAG_PX):
                bad.append({"image": f"{name}/{stem}", "visible": entry["visible"], "inside": inside, "residual_px": r})
    residuals.sort()
    report["landmarks"] = {"images": len(residuals), "flagged": bad,
                           "residual_px_median": residuals[len(residuals) // 2] if residuals else None,
                           "residual_px_max": residuals[-1] if residuals else None}
    report["broken"] = broken

    # 2. same person in every image
    inconsistent = []
    for name, record in records.items():
        skins = [im["applied_skin"] for im in record["images"]]
        worst = max(abs(s[k] - skins[0][k]) for s in skins for k in skins[0])
        if worst > SKIN_MATCH_TOL:
            inconsistent.append({"identity": name, "max_skin_difference": worst})
    report["consistency"] = {"identities_checked": len(records), "inconsistent": inconsistent,
                             "note": "identity values are stored once per identity; every image is rendered from them"}

    # 3. different people
    ids = shape["identities"]
    dist = shape["rms_distance"]
    looks = {n: appearance(records[n]["identity"]) for n in ids}
    pairs = []
    for i, j in combinations(range(len(ids)), 2):
        ia, ib = records[ids[i]]["identity"], records[ids[j]]["identity"]
        shared = similarity.shared_traits(ia, ib)
        d = dist[i][j]
        pairs.append({"a": ids[i], "b": ids[j], "shape_iod": d, "shape_mm": round(d * shape["typical_iod_mm"], 2),
                      "shared": shared, "near_duplicate": similarity.near_duplicate(d, ia, ib),
                      "review": similarity.too_similar(d, shared)})
    pairs.sort(key=lambda p: p["shape_iod"])
    nearest = {n: min((p for p in pairs if n in (p["a"], p["b"])), key=lambda p: p["shape_iod"]) for n in ids}
    report["pairs"] = {"count": len(pairs), "shape_iod_min": pairs[0]["shape_iod"],
                       "shape_iod_median": pairs[len(pairs) // 2]["shape_iod"],
                       "near_duplicates": [p for p in pairs if p["near_duplicate"]],
                       "review": [p for p in pairs if p["review"]], "closest": pairs[:PAIRS_SHOWN],
                       "nearest_per_identity": {n: {"other": p["b"] if p["a"] == n else p["a"], "shape_iod": p["shape_iod"]}
                                                for n, p in nearest.items()}}
    (job / "validation.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    # sheet: closest pairs, frontal aligned crops side by side
    font = ImageFont.load_default()
    rows = pairs[:PAIRS_SHOWN] + [p for p in pairs[PAIRS_SHOWN:] if p["review"]][:LOOKALIKES_SHOWN]
    sheet = Image.new("RGB", (2 * TILE + 260, 22 + len(rows) * (TILE + 6)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 5), f"{job.name}: the {PAIRS_SHOWN} most similar face shapes + look-alikes (max {LOOKALIKES_SHOWN})", fill=(230, 230, 230), font=font)
    for r, p in enumerate(rows):
        y = 22 + r * (TILE + 6)
        sheet.paste(frontal_aligned(job / p["a"], records[p["a"]]), (6, y))
        sheet.paste(frontal_aligned(job / p["b"], records[p["b"]]), (6 + TILE + 4, y))
        a, b = looks[p["a"]], looks[p["b"]]
        text = (f"{p['a']} vs {p['b']}\nshape {p['shape_iod']:.3f} IOD (~{p['shape_mm']} mm)\n"
                f"shared: {', '.join(p['shared']) or 'nothing'}\n{a['sex']} {a['age']}y / {b['sex']} {b['age']}y\n"
                f"tone {a['tone']:.2f} / {b['tone']:.2f}\n{a['hair']} /\n{b['hair']}\nbeard {a['beard']} / {b['beard']}\n"
                f"{'NEAR-DUPLICATE' if p['near_duplicate'] else 'look-alike' if p['review'] else 'ok'}")
        draw.multiline_text((2 * TILE + 16, y + 4), text, fill=(230, 230, 230), font=font, spacing=2)
    sheet.save(job / "validation_pairs.png")

    lm = report["landmarks"]
    if lm["images"]:
        print(f"LANDMARKS  {lm['images']} images, flagged {len(lm['flagged'])}, residual median {lm['residual_px_median']:.2f} px, max {lm['residual_px_max']:.2f} px")
    if broken:
        print(f"BROKEN     {len(broken)} people with missing/unreadable images (re-render them): "
              + ", ".join(f"{n} ({len(v)})" for n, v in broken.items()))
    print(f"SAME PERSON  {len(records)} identities, inconsistent {len(inconsistent)}")
    pr = report["pairs"]
    print(f"DIFFERENT   {pr['count']} pairs, face shape min {pr['shape_iod_min']:.3f} / median {pr['shape_iod_median']:.3f} IOD, "
          f"near-duplicates {len(pr['near_duplicates'])}, look-alikes {len(pr['review'])}")
    for p in pr["closest"]:
        print(f"  {p['a']} vs {p['b']}: {p['shape_iod']:.3f} IOD (~{p['shape_mm']} mm), shared: {', '.join(p['shared']) or 'nothing'}{'  NEAR-DUPLICATE' if p['near_duplicate'] else '  look-alike' if p['review'] else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
