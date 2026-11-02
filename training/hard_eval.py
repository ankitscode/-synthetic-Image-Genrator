"""Harder 1:1 test of a trained run on its held-out people (no retraining).

    python -m training.hard_eval --root <synthface_output> --run <synthface_output>/models/<run> \
        [--cache-dir <synthface_output>/training_cache]

The test during training (training/metrics.py) uses random different-person pairs: most of them differ in sex,
skin tone, hair or beard, which a model can tell apart without looking at the face. Here every image pair between
the held-out people is scored (90 people x 20 images -> ~1.6 M different-person pairs, so FAR 1e-4 is measurable)
and split into harder groups:
- different-person groups (by person pair): same sex; + similar skin tone; + same hairstyle; "same look" (sex, tone,
  hairstyle, beard); similar face shape (< LOOKALIKE_SHAPE_IOD, shapes from selections/<chunk>.shapes.npz);
  look-alikes (shared.similarity.too_similar).
- same-person groups (by image pair): large yaw difference, different background kind (HDRI vs flat).

Per group: TAR at the FAR set on that group's own pairs, and - closer to real use, where one threshold is fixed -
the FAR of that group at the thresholds set on all pairs. Writes <run>/hard_eval.json and <run>/hard_pairs.png
(the most-confused different people, side by side). Image pairs of one person pair are correlated: a group with
few person pairs is noisy even with many image pairs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from shared import constants as C
from shared import similarity
from training import data

FARS = (1e-3, 1e-4)
MIN_FALSE_ACCEPTS = 10      # TAR@FAR is reported only when the group has >= far * pairs >= this
POSE_GAP_DEG = 25.0         # same-person pairs with a yaw difference of at least this
CONFUSED_SHOWN = 16         # rows in hard_pairs.png
IMAGES_PER_SIDE = 3


def rates(genuine: np.ndarray, impostor: np.ndarray) -> dict:
    """EER and TAR at each FAR for one set of same-person and different-person scores."""
    out: dict = {"genuine_pairs": int(len(genuine)), "impostor_pairs": int(len(impostor))}
    if not len(genuine) or not len(impostor):
        return out
    imp = np.sort(impostor)[::-1]
    for far in FARS:
        k = int(far * len(imp))
        if k < MIN_FALSE_ACCEPTS:
            out[f"tar@far={far:g}"] = out[f"threshold@far={far:g}"] = None
            continue
        out[f"threshold@far={far:g}"] = float(imp[k])
        out[f"tar@far={far:g}"] = float((genuine > imp[k]).mean())
    scores = np.concatenate([genuine, impostor])
    same = np.concatenate([np.ones(len(genuine), bool), np.zeros(len(impostor), bool)])
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(same[order]) / len(genuine)
    fp = np.cumsum(~same[order]) / len(impostor)
    i = int(np.argmin(np.abs((1 - tp) - fp)))
    out["eer"] = float(((1 - tp[i]) + fp[i]) / 2)
    out["impostor_score_mean"] = float(impostor.mean())
    return out


def impostor_groups(identities: list[dict], shape_rms: np.ndarray | None) -> dict[str, np.ndarray]:
    """name -> (P, P) bool matrix of person pairs (diagonal False)."""
    sex = np.array([i["sex"] for i in identities])
    tone = np.array([i["skin"]["tone"] for i in identities])
    hair = np.array([i["hair"]["style"] for i in identities])
    beard = np.array([i.get("beard", {}).get("style", "none") if i["sex"] == "male" else "-" for i in identities])
    other = ~np.eye(len(identities), dtype=bool)
    same_sex = (sex[:, None] == sex[None]) & other
    close_tone = np.abs(tone[:, None] - tone[None]) < C.LOOKALIKE_TONE_DIFF
    same_hair = hair[:, None] == hair[None]
    groups = {"all different people": other, "same sex": same_sex, "same sex + skin tone": same_sex & close_tone,
              "same sex + hairstyle": same_sex & same_hair,
              "same look (sex, tone, hair, beard)": same_sex & close_tone & same_hair & (beard[:, None] == beard[None])}
    if shape_rms is not None:
        groups[f"similar face shape (< {C.LOOKALIKE_SHAPE_IOD} IOD)"] = (shape_rms < C.LOOKALIKE_SHAPE_IOD) & other
        look = np.zeros_like(other)
        for a, b in zip(*np.nonzero(np.triu(other))):
            look[a, b] = look[b, a] = similarity.too_similar(
                float(shape_rms[a, b]), similarity.shared_traits(identities[a], identities[b]))
        groups["look-alikes (shape + 3 traits)"] = look
    return groups


def face_shapes(root: Path, people: list[data.Person]) -> np.ndarray | None:
    """(P, P) face-shape RMS distance in IOD units, or None when a selection shape cache is missing."""
    shapes, caches = [], {}
    for p in people:
        path = root / "selections" / f"{p.chunk}.shapes.npz"
        if not path.exists():
            print(f"NOTE no {path}: face-shape groups skipped", flush=True)
            return None
        if p.chunk not in caches:
            npz = np.load(path)
            caches[p.chunk] = ({int(i): k for k, i in enumerate(npz["indices"])}, npz["shapes"])
        by_index, arr = caches[p.chunk]
        shapes.append(arr[by_index[int(p.key.split(":")[1])]].astype(np.float64))
    s = np.stack(shapes)
    return np.stack([np.sqrt(((s - s[i]) ** 2).sum(axis=2).mean(axis=1)) for i in range(len(s))])


def evaluate(embeddings: np.ndarray, labels: np.ndarray, identities: list[dict], image_params: list[dict],
             shape_rms: np.ndarray | None) -> dict:
    """embeddings (N, D) of the held-out images, labels (N,) person index into identities, image_params (N,)."""
    emb = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(1e-12)
    sim = emb @ emb.T
    i, j = np.triu_indices(len(emb), 1)
    scores, la, lb = sim[i, j], labels[i], labels[j]
    same = la == lb
    genuine = scores[same]
    overall = rates(genuine, scores[~same])
    thresholds = {far: overall.get(f"threshold@far={far:g}") for far in FARS}

    report: dict = {"overall": overall, "impostor_groups": {}, "genuine_groups": {}}
    pa, pb = la[~same], lb[~same]
    for name, matrix in impostor_groups(identities, shape_rms).items():
        imp = scores[~same][matrix[pa, pb]]
        r = rates(genuine, imp) | {"person_pairs": int(np.triu(matrix).sum())}
        for far, t in thresholds.items():   # one fixed threshold (set on all pairs): how many of these get in
            r[f"far_at_global@{far:g}"] = float((imp > t).mean()) if t is not None and len(imp) else None
        report["impostor_groups"][name] = r

    yaw = np.array([p["camera"]["yaw"] for p in image_params])
    background = np.array([p["background"]["kind"] for p in image_params])
    gi, gj = i[same], j[same]
    for name, mask in {"all same-person pairs": np.ones(len(genuine), bool),
                       f"yaw difference >= {POSE_GAP_DEG:g} deg": np.abs(yaw[gi] - yaw[gj]) >= POSE_GAP_DEG,
                       "different background kind": background[gi] != background[gj]}.items():
        g = genuine[mask]
        r = {"pairs": int(len(g)), "score_mean": float(g.mean()) if len(g) else None}
        for far, t in thresholds.items():
            r[f"tar_at_global@{far:g}"] = float((g > t).mean()) if t is not None and len(g) else None
        report["genuine_groups"][name] = r

    # most-confused different people: highest mean score between their images
    onehot = np.eye(len(identities))[labels]
    counts = onehot.sum(0)
    person_sim = onehot.T @ sim @ onehot / np.outer(counts, counts)
    a_idx, b_idx = np.triu_indices(len(identities), 1)
    order = np.argsort(-person_sim[a_idx, b_idx])[:CONFUSED_SHOWN]
    t3 = thresholds[FARS[0]]
    confused = []
    for k in order:
        a, b = int(a_idx[k]), int(b_idx[k])
        block = sim[np.ix_(labels == a, labels == b)]
        confused.append({"a": a, "b": b, "mean_score": float(person_sim[a, b]), "max_score": float(block.max()),
                         "accepted_share_at_far_1e-3": float((block > t3).mean()) if t3 is not None else None,
                         "shape_iod": float(shape_rms[a, b]) if shape_rms is not None else None,
                         "shared_traits": similarity.shared_traits(identities[a], identities[b])})
    report["most_confused"] = confused
    return report


def confused_sheet(images: np.ndarray, labels: np.ndarray, names: list[str], confused: list[dict], path: Path) -> None:
    tile, gap, text_h = 112, 16, 14
    width = 2 * IMAGES_PER_SIDE * tile + gap
    sheet = Image.new("RGB", (width, max(1, len(confused)) * (tile + text_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, c in enumerate(confused):
        y = row * (tile + text_h)
        shape = f"  shape {c['shape_iod']:.3f} IOD" if c["shape_iod"] is not None else ""
        draw.text((2, y + 1), f"{names[c['a']]}  |  {names[c['b']]}   mean {c['mean_score']:.2f}  "
                              f"max {c['max_score']:.2f}{shape}", fill="black")
        for side, person in enumerate((c["a"], c["b"])):
            for k, idx in enumerate(np.flatnonzero(labels == person)[:IMAGES_PER_SIDE]):
                x = side * (IMAGES_PER_SIDE * tile + gap) + k * tile
                sheet.paste(Image.fromarray(np.asarray(images[idx])), (x, y + text_h))
    sheet.save(path)


def load_holdout(root: Path, run: Path, cache_dir: Path) -> tuple[list[data.Person], np.ndarray, np.ndarray]:
    """Held-out people of the run (from its index.json), their images (cache if present) and labels."""
    people = [data.Person(**p) for p in json.loads((run / "index.json").read_text())["people"]]
    holdout = [p for p in people if p.split == "holdout"]
    labels = np.concatenate([np.full(len(p.images), k) for k, p in enumerate(holdout)])
    stem = cache_dir / f"images_{data.fingerprint(people)}"
    cached, split = stem.with_name(stem.name + "_images.npy"), stem.with_name(stem.name + "_split.npy")
    if cached.exists() and split.exists():
        print(f"CACHE using {stem.name}", flush=True)
        images = np.asarray(np.load(cached, mmap_mode="r")[np.flatnonzero(np.load(split) == 1)])
    else:
        images = data.load_arrays(root, holdout)["images"]
    if len(images) != len(labels):
        raise SystemExit(f"cache has {len(images)} held-out images, index.json lists {len(labels)}")
    return holdout, images, labels


def embed_best(run: Path, images: np.ndarray, device: str) -> tuple[np.ndarray, int]:
    import torch

    from training.model import MobileFaceNet
    from training.train import embed

    ckpt = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    model = MobileFaceNet()
    model.load_state_dict(ckpt["model"])
    return embed(model.to(device), images, device), int(ckpt["epoch"])


def print_report(report: dict) -> None:
    def num(v: float | None, fmt: str = ".4f") -> str:
        return "  n/a " if v is None else format(v, fmt)

    o = report["overall"]
    print(f"\nALL PAIRS  same {o['genuine_pairs']}  different {o['impostor_pairs']}  EER {num(o.get('eer'))}  "
          f"TAR@1e-3 {num(o.get('tar@far=0.001'))}  TAR@1e-4 {num(o.get('tar@far=0.0001'))}")
    print("\nDIFFERENT PEOPLE                      person   image  | own threshold         | one fixed threshold")
    print("group                                  pairs    pairs  | EER    TAR@1e-3 1e-4  | FAR@1e-3  FAR@1e-4")
    for name, r in report["impostor_groups"].items():
        print(f"{name:36s} {r['person_pairs']:6d} {r['impostor_pairs']:9d}  | {num(r.get('eer'))} "
              f"{num(r.get('tar@far=0.001'))} {num(r.get('tar@far=0.0001'))} | "
              f"{num(r.get('far_at_global@0.001'))}    {num(r.get('far_at_global@0.0001'))}")
    print("\nSAME PERSON (TAR at the fixed thresholds)      pairs   score  TAR@1e-3  TAR@1e-4")
    for name, r in report["genuine_groups"].items():
        print(f"{name:44s} {r['pairs']:7d}   {num(r['score_mean'], '.3f')}  {num(r.get('tar_at_global@0.001'))}    "
              f"{num(r.get('tar_at_global@0.0001'))}")
    print("\nMOST CONFUSED DIFFERENT PEOPLE (mean / max score, share of image pairs accepted at FAR 1e-3)")
    for c in report["most_confused"]:
        shape = f"shape {c['shape_iod']:.3f}" if c["shape_iod"] is not None else ""
        print(f"  {c['a_name']:22s} {c['b_name']:22s} {c['mean_score']:.3f} / {c['max_score']:.3f}  "
              f"{num(c['accepted_share_at_far_1e-3'], '.2f')}  {shape}  {', '.join(c['shared_traits'])}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True, help="folder with the chunk folders (synthface_output)")
    p.add_argument("--run", type=Path, required=True, help="trained run folder (best.pt, index.json)")
    p.add_argument("--cache-dir", type=Path, help="default <root>/training_cache")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    holdout, images, labels = load_holdout(args.root, args.run, args.cache_dir or args.root / "training_cache")
    records = [json.loads((args.root / h.chunk / h.identity_id / "params.json").read_text()) for h in holdout]
    image_params = []
    for h, record in zip(holdout, records):
        by_stem = {Path(im["file"]).stem: im for im in record["images"]}
        image_params += [by_stem[Path(rel).stem] for rel in h.images]
    names = [f"{h.chunk}/{h.identity_id}" for h in holdout]
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings, epoch = embed_best(args.run, images, args.device)
    report = evaluate(embeddings, labels, [r["identity"] for r in records], image_params,
                      face_shapes(args.root, holdout))
    for c in report["most_confused"]:
        c["a_name"], c["b_name"] = names[c["a"]], names[c["b"]]
    report = {"run": str(args.run), "best_epoch": epoch, "people": len(holdout), "images": int(len(labels))} | report
    (args.run / "hard_eval.json").write_text(json.dumps(report, indent=1) + "\n")
    confused_sheet(images, labels, names, report["most_confused"], args.run / "hard_pairs.png")
    print(f"HARD EVAL best epoch {epoch}, {len(holdout)} held-out people, {len(labels)} images")
    print_report(report)
    print(f"\nwrote {args.run / 'hard_eval.json'} and {args.run / 'hard_pairs.png'}")


if __name__ == "__main__":
    main()
