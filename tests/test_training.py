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


def test_hard_eval_groups_and_rates() -> None:
    from training import hard_eval

    rng = np.random.default_rng(0)
    people = [identity("male", 0.5, "short01"), identity("male", 0.52, "short01"), identity("female", 0.2, "long01"),
              identity("female", 0.9, "afro01"), identity("male", 0.9, "none"), identity("female", 0.5, "long01")]
    centres = rng.normal(size=(len(people), 64))
    centres[1] = centres[0] + 0.05 * rng.normal(size=64)          # same look and almost the same face
    labels = np.repeat(np.arange(len(people)), 40)
    emb = centres[labels] + 0.3 * rng.normal(size=(len(labels), 64))
    params = [{"camera": {"yaw": float(30 * (k % 2))}, "background": {"kind": "hdri" if k % 3 else "solid"}}
              for k in range(len(labels))]
    shape = np.full((6, 6), 0.2)
    shape[0, 1] = shape[1, 0] = 0.03
    np.fill_diagonal(shape, 0)
    report = hard_eval.evaluate(emb, labels, people, params, shape)

    groups = report["impostor_groups"]
    assert groups["all different people"]["person_pairs"] == 15
    assert groups["same sex"]["person_pairs"] == 3 + 3
    assert groups["same look (sex, tone, hair, beard)"]["person_pairs"] == 1
    assert groups["look-alikes (shape + 3 traits)"]["person_pairs"] == 1
    # with one threshold set on all pairs, the look-alike pair gets through far more often than random pairs
    assert groups["same look (sex, tone, hair, beard)"]["far_at_global@0.001"] >= 10 * 0.001
    assert report["most_confused"][0]["a"] == 0 and report["most_confused"][0]["b"] == 1
    assert report["genuine_groups"]["yaw difference >= 25 deg"]["pairs"] > 0

    r = hard_eval.rates(np.array([0.9, 0.8]), np.linspace(-1, 0, 100))
    assert r["tar@far=0.001"] is None and r["eer"] == 0.0     # too few pairs for 1e-3: not reported


def test_hard_eval_loads_holdout_from_cache(tmp_path: Path) -> None:
    from training import hard_eval

    make_chunk(tmp_path, "p1_chunk01", 7, {1: identity(), 2: identity(), 3: identity()}, images=2)
    people, report = data.collect(tmp_path, holdout_fraction=0.0, manual={})
    people[1].split = "holdout"
    run = tmp_path / "run"
    run.mkdir()
    data.save_index(run / "index.json", people, report)
    holdout, images, labels = hard_eval.load_holdout(tmp_path, run, tmp_path / "cache")     # no cache: from files
    assert [h.identity_id for h in holdout] == ["id_00002"] and images.shape[0] == 2 and labels.tolist() == [0, 0]
    arrays = data.load_arrays(tmp_path, people, log_every=0)
    stem = tmp_path / "cache" / f"images_{data.fingerprint(people)}"
    stem.parent.mkdir()
    for k in ("images", "split"):
        np.save(stem.with_name(f"{stem.name}_{k}.npy"), arrays[k])
    _, cached, _ = hard_eval.load_holdout(tmp_path, run, tmp_path / "cache")
    assert np.array_equal(cached, images)
    hard_eval.confused_sheet(images, labels, ["a"], [{"a": 0, "b": 0, "mean_score": 0.5, "max_score": 0.6,
                                                    "shape_iod": None}], run / "sheet.png")
    assert Image.open(run / "sheet.png").size == (2 * 3 * 112 + 16, 126)


def test_report_one_to_one_and_one_to_n() -> None:
    from training import report

    rng = np.random.default_rng(0)
    people = [identity("male" if p % 2 else "female", p / 12, "short01") for p in range(12)]
    labels = np.repeat(np.arange(12), 6)
    emb = rng.normal(size=(12, 64))[labels] + 0.2 * rng.normal(size=(72, 64))
    params = [{"camera": {"yaw": float(k % 6) * 5, "pitch": 0.0}, "background": {"kind": "solid"}} for k in range(72)]
    images = np.zeros((72, 112, 112, 3), np.uint8)
    names, files = [f"c/id_{p:05d}" for p in range(12)], [f"img_{k % 6 + 1:03d}" for k in range(72)]
    page, summary = report.build(emb, labels, images, names, files, people, params, None,
                                 {"run": "t", "epoch": 1, "people": 12, "images": 72}, unknown_fraction=0.25)
    c, cl, op = summary["one_to_one"], summary["one_to_n_closed"], summary["one_to_n_open"]
    assert c["same_matched"] + c["same_rejected"] == 12 * 15 and c["same_matched"] == 180   # well separated people
    assert c["different_matched"] + c["different_rejected"] == 72 * 71 // 2 - 180
    assert cl["searches"] == 72 - 12 and cl["rank1"] == 1.0          # one enrolled (most frontal) image per person
    assert op["gallery"] == 9 and op["unknown_people"] == 3 and op["unknown_searches"] == 15
    assert page.startswith("<!doctype html>") and "1:N identification" in page and page.count("data:image/jpeg") > 50
    assert summary["threshold"] == report.build(emb, labels, images, names, files, people, params, None,
                                                {"run": "t", "epoch": 1, "people": 12, "images": 72})[1]["threshold"]

    t = report.build(emb, labels, images, names, files, people, params, None,
                     {"run": "t", "epoch": 1, "people": 12, "images": 72}, threshold=0.99)[1]
    assert t["threshold_how"] == "fixed" and t["one_to_one"]["same_matched"] == 0
