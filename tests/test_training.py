"""Training data selection, augmentation and verification metrics (no torch needed); model smoke test with torch."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from training import augment, data, metrics


def identity(sex: str = "male", tone: float = 0.5, hair: str = "short01") -> dict:
    return {"sex": sex, "skin": {"tone": tone}, "hair": {"style": hair, "color_name": "black"},
            "beard": {"style": "none"}}


def make_chunk(root: Path, name: str, seed: int, people: dict[int, dict], pairs: list[dict] | None = None,
               status: str = "rendered", broken: dict | None = None, images: int = 3) -> Path:
    chunk = root / name
    chunk.mkdir(parents=True)
    (chunk / "job.json").write_text(json.dumps({"seed": seed}))
    (chunk / "progress.json").write_text(json.dumps({"status": status}))
    (chunk / "validation.json").write_text(json.dumps({"broken": broken or {}, "pairs": {
        "near_duplicates": [], "closest": pairs or [], "review": []}}))
    for index, ident in people.items():
        d = chunk / f"id_{index:05d}"
        (d / "aligned").mkdir(parents=True)
        record = {"status": "done", "identity": ident, "images": []}
        for k in range(1, images + 1):
            Image.fromarray(np.full((112, 112, 3), index % 255, np.uint8)).save(d / "aligned" / f"img_{k:03d}.jpg")
            record["images"].append({"file": f"raw/img_{k:03d}.png"})
        (d / "params.json").write_text(json.dumps(record))
    return chunk


def test_collect_excludes_and_labels(tmp_path: Path) -> None:
    make_chunk(tmp_path, "p1_chunk01", 7, {1: identity(), 2: identity("female", hair="none"), 3: identity()},
               pairs=[{"a": "id_00001", "b": "id_00003", "shape_iod": 0.015}])
    make_chunk(tmp_path, "p1_chunk02", 7, {10: identity(), 11: identity("female")})
    make_chunk(tmp_path, "p1_chunk03", 7, {20: identity()}, status="partial")          # not finished: ignored
    make_chunk(tmp_path, "p1_chunk04", 7, {30: identity()}, broken={"id_00030": ["img_001"]})  # check failed
    people, report = data.collect(tmp_path, holdout_fraction=0.0,
                                  manual={"p1_chunk02": {"id_00011": "visual review"}})
    assert [p.key for p in people] == ["7:1", "7:10"]
    reasons = {(e["chunk"], e["identity"]): e["reason"] for e in report["excluded"]}
    assert reasons[("p1_chunk01", "id_00002")] == "bald woman"
    assert reasons[("p1_chunk01", "id_00003")].startswith("near-duplicate of id_00001")
    assert reasons[("p1_chunk02", "id_00011")] == "visual review"
    assert [c["chunk"] for c in report["chunks"]] == ["p1_chunk01", "p1_chunk02"]


def test_collect_refuses_overlapping_chunks(tmp_path: Path) -> None:
    make_chunk(tmp_path, "p1_chunk01", 7, {1: identity()})
    make_chunk(tmp_path, "p1_chunk02", 7, {1: identity()})
    with pytest.raises(ValueError, match="overlap"):
        data.collect(tmp_path, manual={})


def test_holdout_split_is_stable_and_about_the_fraction() -> None:
    keys = [f"7:{i}" for i in range(4000)]
    first = [data.holdout_split(k, 0.1) for k in keys]
    assert first == [data.holdout_split(k, 0.1) for k in keys]
    assert 0.08 < first.count("holdout") / len(keys) < 0.12


def test_load_arrays_orders_train_then_holdout(tmp_path: Path) -> None:
    make_chunk(tmp_path, "p1_chunk01", 7, {1: identity(), 2: identity(), 3: identity()}, images=2)
    people, _ = data.collect(tmp_path, holdout_fraction=0.0, manual={})
    people[1].split = "holdout"
    arrays = data.load_arrays(tmp_path, people, log_every=0)
    assert arrays["images"].shape == (6, 112, 112, 3)
    assert arrays["labels"].tolist() == [0, 0, 1, 1, 2, 2] and arrays["split"].tolist() == [0, 0, 0, 0, 1, 1]
    assert arrays["keys"].tolist() == ["7:1", "7:3", "7:2"]


def test_augment_is_deterministic_and_keeps_shape() -> None:
    image = np.random.default_rng(0).integers(0, 255, (112, 112, 3), dtype=np.uint8)
    a = augment.augment(image, np.random.default_rng([1, 2, 3]))
    b = augment.augment(image, np.random.default_rng([1, 2, 3]))
    assert a.shape == (112, 112, 3) and a.dtype == np.uint8 and np.array_equal(a, b)


def test_verification_metrics_perfect_and_random() -> None:
    labels = np.repeat(np.arange(20), 5)
    perfect = np.eye(20)[labels] + 0.01 * np.random.default_rng(0).normal(size=(100, 20))
    m = metrics.verification(perfect, labels)
    assert m["auc"] > 0.999 and m["eer"] < 0.01 and m["tar@far=0.001"] > 0.99
    rand = metrics.verification(np.random.default_rng(1).normal(size=(100, 32)), labels)
    assert 0.35 < rand["auc"] < 0.65


def test_model_shapes_and_arcface() -> None:
    torch = pytest.importorskip("torch")
    from training.model import ArcFace, MobileFaceNet, normalise
    model = MobileFaceNet().eval()
    x = normalise(torch.zeros(2, 112, 112, 3, dtype=torch.uint8))
    emb = model(x)
    assert emb.shape == (2, 128)
    head = ArcFace(128, 5)
    logits = head(emb, torch.tensor([0, 3]), 0.5)
    plain = head(emb, torch.tensor([0, 3]), 0.0)
    assert logits.shape == (2, 5) and (logits[0, 0] < plain[0, 0]) and torch.allclose(logits[0, 1:], plain[0, 1:])
