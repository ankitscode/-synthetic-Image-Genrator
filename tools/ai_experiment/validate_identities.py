"""Same person within an identity, different people across identities: image-level check of a
3D job (approach A) and an approach-B set, on ArcFace-aligned 112x112 crops. No training.

    venv_ai/bin/python tools/ai_experiment/validate_identities.py --job output/colab_mix20_v4 --set colab_b_mix20_v4

Embedding: the CLIP ViT-H image encoder (already used for IP-Adapter; LAION-2B, experiment env).
It is a general image model, NOT a face recognizer: it also reacts to hair, skin colour and
lighting, so treat the numbers as a consistency check, not as face-recognition accuracy.

Per dataset: same-person / different-person cosine similarity, verification AUC and EER over
all image pairs, leave-one-out rank-1 (is the nearest identity centroid the image's own?),
nearest other identity per identity. Cross check: each B image against the A centroids
(does the photoreal version still look most like its own 3D person?).
Writes output/ai_experiment/sets/<set>/identity_check.json and identity_check.png (failures).
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import postprocess  # noqa: E402
import reference_test as rt  # noqa: E402

SETS_DIR = PROJECT_ROOT / "output" / "ai_experiment" / "sets"
TILE = 112


def align_b(set_dir: Path) -> dict[str, list[Path]]:
    """Align every B image (reference s1 + variations) with its shot's 3D landmarks."""
    out_dir = set_dir / "b_aligned"
    out_dir.mkdir(exist_ok=True)
    crops: dict[str, list[Path]] = {}
    for person_dir in sorted((set_dir / "shots").glob("id_*")):
        lm = json.loads((person_dir / "landmarks.json").read_text())
        for shot, entry in lm["images"].items():
            src = set_dir / "b" / f"{person_dir.name}_{shot}.png"
            if not src.exists():
                continue
            image = cv2.imread(str(src), cv2.IMREAD_COLOR)
            aligned, _ = postprocess.align(image, np.array(entry["points"], dtype=np.float32))
            dst = out_dir / f"{person_dir.name}_{shot}.jpg"
            cv2.imwrite(str(dst), aligned, [cv2.IMWRITE_JPEG_QUALITY, postprocess.ALIGNED_JPEG_QUALITY])
            crops.setdefault(person_dir.name, []).append(dst)
    return crops


def a_crops(job: Path) -> dict[str, list[Path]]:
    return {d.name: sorted((d / "aligned").glob("*.jpg")) for d in sorted(job.glob("id_*")) if (d / "aligned").exists()}


def embed_all(crops: dict[str, list[Path]], embed) -> dict[str, np.ndarray]:
    return {person: np.stack([embed(cv2.imread(str(p))) for p in paths]) for person, paths in crops.items()}


def auc_eer(same: np.ndarray, diff: np.ndarray) -> tuple[float, float]:
    scores = np.concatenate([same, diff])
    labels = np.concatenate([np.ones_like(same), np.zeros_like(diff)])
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    auc = (ranks[labels == 1].sum() - len(same) * (len(same) + 1) / 2) / (len(same) * len(diff))
    thresholds = np.unique(scores)
    far = np.array([(diff >= t).mean() for t in thresholds])
    frr = np.array([(same < t).mean() for t in thresholds])
    i = int(np.argmin(np.abs(far - frr)))
    return float(auc), float((far[i] + frr[i]) / 2)


def analyse(vectors: dict[str, np.ndarray], paths: dict[str, list[Path]]) -> tuple[dict, list]:
    people = list(vectors)
    same = np.array([float(a @ b) for p in people for a, b in combinations(vectors[p], 2)])
    diff = np.array([float(a @ b) for p, q in combinations(people, 2) for a in vectors[p] for b in vectors[q]])
    centroids = {p: v.mean(axis=0) for p, v in vectors.items()}
    failures, correct, total = [], 0, 0
    for p in people:
        for k, vec in enumerate(vectors[p]):
            others = np.delete(vectors[p], k, axis=0)
            own = others.mean(axis=0) if len(others) else centroids[p]
            scores = {q: float(vec @ (own if q == p else centroids[q]) / np.linalg.norm(own if q == p else centroids[q]))
                      for q in people}
            best = max(scores, key=scores.get)
            total += 1
            if best == p:
                correct += 1
            else:
                failures.append({"image": paths[p][k].name, "person": p, "matched": best,
                                 "own": round(scores[p], 3), "other": round(scores[best], 3)})
    nearest = {}
    for p in people:
        c = centroids[p] / np.linalg.norm(centroids[p])
        sims = {q: float(c @ centroids[q] / np.linalg.norm(centroids[q])) for q in people if q != p}
        q = max(sims, key=sims.get)
        nearest[p] = {"other": q, "centroid_cosine": round(sims[q], 3)}
    auc, eer = auc_eer(same, diff)
    stats = {"people": len(people), "images": int(sum(len(v) for v in vectors.values())),
             "same_person_cosine_mean": float(same.mean()), "same_person_cosine_p5": float(np.percentile(same, 5)),
             "different_person_cosine_mean": float(diff.mean()), "different_person_cosine_p95": float(np.percentile(diff, 95)),
             "verification_auc": auc, "verification_eer": eer, "rank1": correct / total,
             "rank1_failures": failures, "nearest_other_identity": nearest}
    return stats, failures


def cross(b: dict[str, np.ndarray], a: dict[str, np.ndarray], b_paths: dict[str, list[Path]]) -> dict:
    centroids = {p: v.mean(axis=0) / np.linalg.norm(v.mean(axis=0)) for p, v in a.items()}
    correct, total, failures = 0, 0, []
    for p, vecs in b.items():
        for k, vec in enumerate(vecs):
            scores = {q: float(vec @ c) for q, c in centroids.items()}
            best = max(scores, key=scores.get)
            total += 1
            correct += best == p
            if best != p:
                failures.append({"image": b_paths[p][k].name, "person": p, "matched_3d": best})
    return {"b_images_matched_to_own_3d_person": correct / total, "failures": failures}


def failure_sheet(out: Path, rows: list[tuple[str, Path, Path]]) -> None:
    font = ImageFont.load_default()
    sheet = Image.new("RGB", (3 * (TILE + 4) + 200, 20 + max(len(rows), 1) * (TILE + 4)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 4), "Rank-1 failures: image | its own person | the person it matched", fill=(230, 230, 230), font=font)
    for r, (label, img, own, other) in enumerate(rows):
        y = 20 + r * (TILE + 4)
        for c, path in enumerate((img, own, other)):
            sheet.paste(Image.open(path).convert("RGB").resize((TILE, TILE)), (4 + c * (TILE + 4), y))
        draw.multiline_text((3 * (TILE + 4) + 8, y + 4), label, fill=(230, 230, 230), font=font)
    sheet.save(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", type=Path, required=True, help="3D job folder (aligned/ from app/postprocess.py)")
    parser.add_argument("--set", required=True, help="approach-B set (shots/<person>/landmarks.json needed)")
    args = parser.parse_args()
    set_dir = SETS_DIR / args.set
    embed = rt.clip_embedder()
    a_paths = a_crops(args.job)
    b_paths = align_b(set_dir)
    a_vec, b_vec = embed_all(a_paths, embed), embed_all(b_paths, embed)
    a_stats, a_fail = analyse(a_vec, a_paths)
    b_stats, b_fail = analyse(b_vec, b_paths)
    report = {"embedding": "CLIP ViT-H/14 image encoder on aligned 112 crops (general image model, not a face recognizer)",
              "A_3d": a_stats, "B_photoreal": b_stats, "B_vs_A": cross(b_vec, {p: a_vec[p] for p in b_vec if p in a_vec}, b_paths)}
    (set_dir / "identity_check.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    rows = []
    for name, fails, paths, folder in (("A", a_fail, a_paths, None), ("B", b_fail, b_paths, None)):
        for f in fails:
            img = next(p for p in paths[f["person"]] if p.name == f["image"])
            rows.append((f"{name} {f['image']}\nown {f['person']} {f['own']}\nmatched {f['matched']} {f['other']}",
                         img, paths[f["person"]][0] if paths[f["person"]][0] != img else paths[f["person"]][-1],
                         paths[f["matched"]][0]))
    failure_sheet(set_dir / "identity_check.png", rows)

    for name, s in (("A 3D", a_stats), ("B photoreal", b_stats)):
        print(f"{name}: {s['people']} people, {s['images']} images | same {s['same_person_cosine_mean']:.3f} "
              f"(p5 {s['same_person_cosine_p5']:.3f}) vs different {s['different_person_cosine_mean']:.3f} "
              f"(p95 {s['different_person_cosine_p95']:.3f}) | AUC {s['verification_auc']:.3f} EER "
              f"{100 * s['verification_eer']:.1f}% | rank-1 {100 * s['rank1']:.1f}% ({len(s['rank1_failures'])} failures)")
    c = report["B_vs_A"]
    print(f"B vs A: {100 * c['b_images_matched_to_own_3d_person']:.1f}% of B images look most like their own 3D person "
          f"({len(c['failures'])} do not)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
