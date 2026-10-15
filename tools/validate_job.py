"""Data-cleanliness check for a rendered 3D job (approach A). No model, no training.

    venv/bin/python tools/validate_job.py output/<job_id>

Needs: params.json (render), landmarks.json (blender/compute_landmarks.py), alignment.json +
aligned/ (app/postprocess.py), face_shape.json (blender/face_shape_distance.py).

1. Landmarks: all 5 visible, inside the frame, aligned residual small.
2. Same person in every image: identity values and the applied skin are identical across images.
3. Different people: 3D face-shape distance (IOD units) between every pair, with the shared rules
   (shared/similarity.py): near-duplicates (practically the same face) fail the check;
   look-alikes (similar shape + shared hair / skin / sex...) are listed for a visual look.
Writes <job>/validation.json and <job>/validation_pairs.png (closest pairs side by side).
"""

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
TILE = 160


def appearance(identity: dict) -> dict:
    beard = identity.get("beard", {}).get("style", "none")
    return {"sex": identity["sex"], "tone": identity["skin"]["tone"], "eyes": identity["eyes"]["color"],
            "hair": f"{identity['hair']['style']} {identity['hair']['color_name']}", "beard": beard,
            "age": round(identity["age_years"])}


def frontal_aligned(identity_dir: Path, record: dict) -> Image.Image:
    best = min(record["images"], key=lambda im: abs(im["camera"]["yaw"]) + abs(im["camera"]["pitch"]))
    return Image.open(identity_dir / "aligned" / f"{Path(best['file']).stem}.jpg").convert("RGB").resize((TILE, TILE))


def main() -> int:
    job = Path(sys.argv[1])
    records = {d.name: json.loads((d / "params.json").read_text()) for d in sorted(job.glob("id_*"))}
    shape = json.loads((job / "face_shape.json").read_text())
    report: dict = {"job": job.name, "identities": len(records), "landmarks": {}, "consistency": {}, "pairs": {}}

    # 1. landmarks
    bad = []
    residuals = []
    for name in records:
        lm = json.loads((job / name / "landmarks.json").read_text())
        quality = json.loads((job / name / "alignment.json").read_text())
        size = lm["resolution"]
        for stem, entry in lm["images"].items():
            inside = all(0 <= x < size and 0 <= y < size for x, y in entry["points"])
            r = quality[stem]["residual_px"]
            residuals.append(r)
            if not (all(entry["visible"]) and inside and r <= RESIDUAL_FLAG_PX):
                bad.append({"image": f"{name}/{stem}", "visible": entry["visible"], "inside": inside, "residual_px": r})
    residuals.sort()
    report["landmarks"] = {"images": len(residuals), "flagged": bad,
                           "residual_px_median": residuals[len(residuals) // 2], "residual_px_max": residuals[-1]}

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
    rows = pairs[:PAIRS_SHOWN] + [p for p in pairs[PAIRS_SHOWN:] if p["review"]]
    sheet = Image.new("RGB", (2 * TILE + 260, 22 + len(rows) * (TILE + 6)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 5), f"{job.name}: the {PAIRS_SHOWN} most similar face shapes + look-alikes", fill=(230, 230, 230), font=font)
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
    print(f"LANDMARKS  {lm['images']} images, flagged {len(lm['flagged'])}, residual median {lm['residual_px_median']:.2f} px, max {lm['residual_px_max']:.2f} px")
    print(f"SAME PERSON  {len(records)} identities, inconsistent {len(inconsistent)}")
    pr = report["pairs"]
    print(f"DIFFERENT   {pr['count']} pairs, face shape min {pr['shape_iod_min']:.3f} / median {pr['shape_iod_median']:.3f} IOD, "
          f"near-duplicates {len(pr['near_duplicates'])}, look-alikes {len(pr['review'])}")
    for p in pr["closest"]:
        print(f"  {p['a']} vs {p['b']}: {p['shape_iod']:.3f} IOD (~{p['shape_mm']} mm), shared: {', '.join(p['shared']) or 'nothing'}{'  NEAR-DUPLICATE' if p['near_duplicate'] else '  look-alike' if p['review'] else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
