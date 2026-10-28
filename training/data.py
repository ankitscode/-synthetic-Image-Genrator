"""Training data from rendered chunks (approach A only): which people, which images, which label.

A chunk folder (output of colab/synthface_3d_colab.ipynb) is used when it is fully rendered and its check
passed (progress.json "rendered", validation.json without "broken"). Per person: params.json "done" and
every aligned/img_XXX.jpg present. Labels are (seed, identity index): unique across chunks, because every
chunk was selected against all earlier ones.

Left out (reasons are written to the index):
- shared.similarity.excluded(identity), e.g. bald women (chunks before the selection rule existed),
- near-duplicates by the current rule among the pairs in validation.json (the later index is dropped),
- training/exclude.json (manual list, e.g. after a visual review).

No torch here: numpy + Pillow only, so it is testable without a GPU.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from shared import similarity

ALIGNED_SIZE = 112
EXCLUDE_FILE = Path(__file__).resolve().parent / "exclude.json"
DEFAULT_HOLDOUT_FRACTION = 0.10


@dataclass
class Person:
    key: str                      # "<seed>:<identity index>", the label identity
    chunk: str
    identity_id: str              # id_00193
    images: list[str] = field(default_factory=list)   # paths relative to the data root
    split: str = "train"          # "train" | "holdout"


def chunk_ready(chunk_dir: Path) -> bool:
    progress, validation = chunk_dir / "progress.json", chunk_dir / "validation.json"
    if not (progress.exists() and validation.exists()):
        return False
    if json.loads(progress.read_text()).get("status") != "rendered":
        return False
    return not json.loads(validation.read_text()).get("broken")


def find_chunks(root: Path, names: list[str] | None = None) -> list[Path]:
    """Ready chunk folders under root (all `*_chunk*` folders, or only `names`)."""
    candidates = [root / n for n in names] if names else sorted(p for p in root.glob("*_chunk*") if p.is_dir())
    return [c for c in candidates if chunk_ready(c)]


def _near_duplicate_drops(chunk_dir: Path, identities: dict[str, dict]) -> dict[str, str]:
    """identity_id -> reason, for pairs in validation.json that are near-duplicates by the current rule."""
    pairs = json.loads((chunk_dir / "validation.json").read_text())["pairs"]
    seen, drops = set(), {}
    for p in pairs.get("near_duplicates", []) + pairs.get("closest", []) + pairs.get("review", []):
        a, b = sorted((p["a"], p["b"]))
        if (a, b) in seen or a not in identities or b not in identities:
            continue
        seen.add((a, b))
        if similarity.near_duplicate(p["shape_iod"], identities[a], identities[b]):
            drops[b] = f"near-duplicate of {a} ({p['shape_iod']:.3f} IOD)"
    return drops


def load_manual_excludes(path: Path = EXCLUDE_FILE) -> dict[str, dict[str, str]]:
    """{"p1_chunk02": {"id_00236": "reason"}}"""
    if not path.exists():
        return {}
    return json.loads(path.read_text())["exclude"]


def holdout_split(key: str, fraction: float) -> str:
    """Stable per-person split: the same person is always in the same split."""
    h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "holdout" if h < fraction else "train"


def collect(root: Path, chunk_names: list[str] | None = None, holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
            manual: dict[str, dict[str, str]] | None = None) -> tuple[list[Person], dict]:
    """People to train / test on, plus a report of what was left out and why."""
    manual = load_manual_excludes() if manual is None else manual
    people: dict[str, Person] = {}
    report: dict = {"chunks": [], "excluded": [], "skipped_incomplete": []}
    for chunk_dir in find_chunks(root, chunk_names):
        seed = json.loads((chunk_dir / "job.json").read_text())["seed"]
        records = {}
        for d in sorted(chunk_dir.glob("id_*")):
            params = d / "params.json"
            if params.exists():
                record = json.loads(params.read_text())
                if record.get("status") == "done":
                    records[d.name] = record
        identities = {n: r["identity"] for n, r in records.items()}
        drops = {n: r for n, r in manual.get(chunk_dir.name, {}).items() if n in records}
        drops.update(_near_duplicate_drops(chunk_dir, identities))
        for name, identity in identities.items():
            reason = similarity.excluded(identity)
            if reason:
                drops.setdefault(name, reason)
        report["chunks"].append({"chunk": chunk_dir.name, "seed": seed, "people": len(records), "excluded": len(drops)})
        for name, record in records.items():
            if name in drops:
                report["excluded"].append({"chunk": chunk_dir.name, "identity": name, "reason": drops[name]})
                continue
            stems = [Path(im["file"]).stem for im in record["images"]]
            files = [chunk_dir / name / "aligned" / f"{s}.jpg" for s in stems]
            if not stems or not all(f.exists() for f in files):
                report["skipped_incomplete"].append(f"{chunk_dir.name}/{name}")
                continue
            key = f"{seed}:{int(name.split('_')[1])}"
            if key in people:
                raise ValueError(f"{key} appears in {people[key].chunk} and {chunk_dir.name}: chunks overlap")
            people[key] = Person(key, chunk_dir.name, name, [str(f.relative_to(root)) for f in files],
                                 holdout_split(key, holdout_fraction))
    result = sorted(people.values(), key=lambda p: (p.chunk, p.identity_id))
    report["people"] = {"train": sum(p.split == "train" for p in result),
                        "holdout": sum(p.split == "holdout" for p in result)}
    report["images"] = {s: sum(len(p.images) for p in result if p.split == s) for s in ("train", "holdout")}
    return result, report


def fingerprint(people: list[Person]) -> str:
    """Short hash of the exact image list + split (names the image cache)."""
    h = hashlib.sha256()
    for p in people:
        h.update(f"{p.key}|{p.split}|{'|'.join(p.images)}\n".encode())
    return h.hexdigest()[:12]


def load_arrays(root: Path, people: list[Person], log_every: int = 2000) -> dict[str, np.ndarray]:
    """All aligned crops as one uint8 array (N, 112, 112, 3) + labels.

    labels: index of the person in `people` (train people first get 0..T-1, holdout people T..);
    split: 0 train / 1 holdout per image.
    """
    order = [p for p in people if p.split == "train"] + [p for p in people if p.split == "holdout"]
    n = sum(len(p.images) for p in order)
    images = np.empty((n, ALIGNED_SIZE, ALIGNED_SIZE, 3), np.uint8)
    labels = np.empty(n, np.int64)
    split = np.empty(n, np.uint8)
    i = 0
    for label, person in enumerate(order):
        for rel in person.images:
            with Image.open(root / rel) as im:
                arr = np.asarray(im.convert("RGB"))
            if arr.shape != (ALIGNED_SIZE, ALIGNED_SIZE, 3):
                raise ValueError(f"{rel}: shape {arr.shape}")
            images[i], labels[i], split[i] = arr, label, person.split == "holdout"
            i += 1
            if log_every and i % log_every == 0:
                print(f"LOADED {i}/{n} images", flush=True)
    keys = np.array([p.key for p in order])
    return {"images": images, "labels": labels, "split": split, "keys": keys}


def save_index(path: Path, people: list[Person], report: dict) -> None:
    path.write_text(json.dumps({"report": report, "people": [asdict(p) for p in people]}, indent=1) + "\n")
