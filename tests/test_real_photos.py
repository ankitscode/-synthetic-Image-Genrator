"""Real test photos: hand-clicked landmarks, alignment and the report section (stand-in images, no real faces)."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from shared import constants as C
from training import real_photos

POINTS = np.array([[1200, 1500], [1650, 1490], [1430, 1760], [1260, 1980], [1610, 1975]], float)


def photo_with_dots(points: np.ndarray, size: tuple[int, int] = (3000, 4000)) -> Image.Image:
    arr = np.full((size[1], size[0], 3), 90, np.uint8)
    for k, (x, y) in enumerate(points.astype(int)):
        arr[y - 12:y + 13, x - 12:x + 13] = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)][k]
    return Image.fromarray(arr)


def make_real_dir(root: Path, people: dict[str, int]) -> Path:
    for person, n in people.items():
        (root / person).mkdir(parents=True)
        for k in range(n):
            photo_with_dots(POINTS + 40 * k).save(root / person / f"IMG_{k:03d}.jpg", quality=95)
    return root


def test_alignment_puts_clicked_points_on_the_template() -> None:
    crop, info = real_photos.align_photo(np.asarray(photo_with_dots(POINTS)), POINTS)
    assert crop.shape == (112, 112, 3) and info["residual_px"] < 3.0
    red = np.argwhere((crop[..., 0] > 180) & (crop[..., 1] < 80) & (crop[..., 2] < 80)).mean(0)
    assert np.allclose(red[::-1], C.ARCFACE_112[0], atol=2.5)        # left eye dot lands on the template's left eye


def test_exif_rotation_is_applied(tmp_path: Path) -> None:
    im = Image.new("RGB", (40, 20))
    exif = im.getexif()
    exif[0x0112] = 6                                                # "rotate 90 CW to display"
    im.save(tmp_path / "p.jpg", exif=exif)
    assert real_photos.load_photo(tmp_path / "p.jpg").shape == (40, 20, 3)


def test_load_aligned_skips_missing_and_too_few(tmp_path: Path) -> None:
    real_dir = make_real_dir(tmp_path / "real", {"emp_a": 3, "emp_b": 2, "emp_c": 2})
    for k in range(3):
        real_photos.save_landmark(real_dir, f"emp_a/IMG_{k:03d}.jpg", (POINTS + 40 * k).tolist())
    real_photos.save_landmark(real_dir, "emp_b/IMG_000.jpg", (POINTS).tolist())
    real_photos.save_landmark(real_dir, "emp_b/IMG_001.jpg", real_photos.SKIP)       # emp_b keeps only 1 photo
    real_photos.save_landmark(real_dir, "emp_c/IMG_000.jpg", POINTS.tolist())         # emp_c/IMG_001 not clicked
    out = real_photos.load_aligned(real_dir, detect=False)
    assert out["names"] == ["real/emp_a"] and out["labels"].tolist() == [0, 0, 0]
    assert out["images"].shape == (3, 112, 112, 3) and max(out["residual_px"]) < 3.0
    assert out["skipped"] == ["emp_b/IMG_001.jpg"] and out["missing"] == ["emp_c/IMG_001.jpg"]
    assert out["too_few"] == ["emp_b", "emp_c"]
    real_photos.aligned_sheet(out, tmp_path / "sheet.png")
    assert Image.open(tmp_path / "sheet.png").size == (3 * 112, 112)


def test_click_page_resumes_at_first_unclicked_photo(tmp_path: Path) -> None:
    real_dir = make_real_dir(tmp_path / "real", {"emp_a": 2, "emp_b": 1})
    real_photos.save_landmark(real_dir, "emp_a/IMG_000.jpg", POINTS.tolist())
    page = real_photos.click_page(real_dir)
    assert '["emp_a/IMG_000.jpg", "emp_a/IMG_001.jpg", "emp_b/IMG_000.jpg"]' in page and "i = 1," in page
    shown = real_photos.photo_for_page(real_dir, "emp_a/IMG_000.jpg")
    assert shown["data"].startswith("data:image/jpeg;base64,") and shown["scale"] == 4000 / real_photos.CLICK_DISPLAY_MAX
    marks = json.loads((real_dir / "landmarks.json").read_text())
    assert marks["order"] == real_photos.POINT_NAMES


def test_report_real_section() -> None:
    from tests.test_training import identity
    from training import report

    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(12), 6)
    centres = rng.normal(size=(12, 64))
    emb = centres[labels] + 0.2 * rng.normal(size=(72, 64))
    params = [{"camera": {"yaw": 0.0, "pitch": 0.0}, "background": {"kind": "solid"}}] * 72
    images = np.zeros((72, 112, 112, 3), np.uint8)
    real_labels = np.repeat(np.arange(3), 4)
    real = {"embeddings": rng.normal(size=(3, 64))[real_labels] + 0.3 * rng.normal(size=(12, 64)),
            "images": np.zeros((12, 112, 112, 3), np.uint8), "labels": real_labels,
            "names": ["real/a", "real/b", "real/c"], "files": [f"IMG_{k}" for k in range(12)],
            "residual_px": [1.0] * 12, "skipped": [], "missing": [], "too_few": []}
    page, summary = report.build(emb, labels, images, [f"s/{p}" for p in range(12)], ["img"] * 72,
                                 [identity()] * 12, params, None, {"run": "t", "epoch": 1, "people": 12, "images": 72},
                                 real=real)
    r = summary["real"]
    assert r["people"] == 3 and r["photos"] == 12
    assert r["one_to_one"]["same_matched"] + r["one_to_one"]["same_rejected"] == 3 * 6
    assert r["one_to_n_real_gallery"]["gallery"] == 3 and r["one_to_n_real_gallery"]["searches"] == 9
    assert r["one_to_n_real_plus_synthetic"]["gallery"] == 15 and r["one_to_n_real_plus_synthetic"]["searches"] == 9
    assert summary["one_to_one"]["same_matched"] + summary["one_to_one"]["same_rejected"] == 12 * 15   # synthetic unchanged
    assert "Real people" in page and "do not share this report outside the team" in page


def test_bundle_has_what_the_report_needs(tmp_path: Path) -> None:
    import zipfile

    from tests.test_training import identity, make_chunk
    from training import bundle, data, hard_eval

    root = tmp_path / "synthface_output"
    make_chunk(root, "p1_chunk01", 7, {1: identity(), 2: identity(), 3: identity()}, images=2)
    (root / "selections").mkdir()
    np.savez_compressed(root / "selections" / "p1_chunk01.shapes.npz", shapes=np.arange(3 * 4 * 3, dtype=np.float32).reshape(3, 4, 3),
                        indices=np.array([1, 2, 3]))
    people, report = data.collect(root, holdout_fraction=0.0, manual={})
    people[1].split = people[2].split = "holdout"
    run = root / "models" / "r"
    run.mkdir(parents=True)
    data.save_index(run / "index.json", people, report)
    (run / "best.pt").write_bytes(b"weights")
    real = make_real_dir(tmp_path / "real", {"emp_a": 1})
    real_photos.save_landmark(real, "emp_a/IMG_000.jpg", POINTS.tolist())

    counts = bundle.export(root, run, tmp_path / "b.zip", real)
    assert counts["people"] == 2 and counts["images"] == 4 and counts["shapes"] == 2 and counts["real_landmarks"]
    out = tmp_path / "unzipped"
    zipfile.ZipFile(tmp_path / "b.zip").extractall(out)
    assert not (out / "p1_chunk01" / "id_00001").exists()                     # training people are not packed
    holdout, images, labels = hard_eval.load_holdout(out, out / "models" / "r", out / "no_cache")
    assert [h.identity_id for h in holdout] == ["id_00002", "id_00003"] and images.shape[0] == 4
    shapes = hard_eval.face_shapes(out, holdout)
    assert shapes.shape == (2, 2) and shapes[0, 1] > 0
    assert (out / "real_landmarks.json").exists() and (out / "models" / "r" / "best.pt").read_bytes() == b"weights"


def test_local_click_server_serves_and_saves(tmp_path: Path) -> None:
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    real_dir = make_real_dir(tmp_path / "real", {"emp_a": 1})
    captured = {}
    original = ThreadingHTTPServer.serve_forever

    def capture(self) -> None:          # run the real handler on a free port in a thread
        captured["server"] = self
        original(self)

    ThreadingHTTPServer.serve_forever = capture
    try:
        t = threading.Thread(target=real_photos.serve_local, args=(real_dir, 0), daemon=True)
        t.start()
        while "server" not in captured:
            pass
        port = captured["server"].server_address[1]
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode()
        assert "invokeFunction" in page and "emp_a/IMG_000.jpg" in page

        def call(name: str, args: list) -> dict:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/call", json.dumps({"name": name, "args": args}).encode())
            return json.loads(urllib.request.urlopen(req).read())

        assert call("synthface.photo", ["emp_a/IMG_000.jpg"])["data"].startswith("data:image/jpeg")
        assert call("synthface.save", ["emp_a/IMG_000.jpg", POINTS.tolist()]) == {"ok": True}
        assert real_photos.load_landmarks(real_dir)["points"]["emp_a/IMG_000.jpg"] == POINTS.tolist()
        try:
            call("synthface.save", ["../../etc/passwd", POINTS.tolist()])
            raise AssertionError("unknown photo accepted")
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
        captured["server"].shutdown()
    finally:
        ThreadingHTTPServer.serve_forever = original


def test_detector_points_are_ordered_and_scaled(monkeypatch) -> None:
    class FakeYuNet:      # YuNet row: box x, y, w, h, then right eye, left eye, nose, right / left mouth corner, score
        def setInputSize(self, size) -> None:
            self.size = size

        def detect(self, image):
            small = np.array([10, 10, 20, 20, 60, 50, 40, 50, 50, 60, 58, 70, 42, 70, 0.7], np.float32)
            big = np.array([100, 100, 300, 300, 300, 200, 200, 200, 250, 250, 290, 300, 210, 300, 0.9], np.float32)
            return 2, np.stack([small, big])

    monkeypatch.setattr(real_photos, "_detector", FakeYuNet())
    image = np.zeros((2560, 1920, 3), np.uint8)                      # shrunk by 0.5 to 1280 for detection
    pts, score = real_photos.detect_points(image)
    assert score == pytest.approx(0.9)                               # the largest face
    assert pts.tolist() == [[400, 400], [600, 400], [500, 500], [420, 600], [580, 600]]   # viewer's left first


def test_photos_without_clicks_use_the_detector(tmp_path: Path, monkeypatch) -> None:
    real_dir = make_real_dir(tmp_path / "real", {"emp_a": 2, "emp_b": 2})
    real_photos.save_landmark(real_dir, "emp_a/IMG_000.jpg", POINTS.tolist())        # clicked points win
    found = {"emp_a/IMG_001.jpg": POINTS + 40, "emp_b/IMG_000.jpg": POINTS, "emp_b/IMG_001.jpg": None}
    calls = []

    def fake_detect(photo):
        key = list(found)[len(calls)]
        calls.append(key)
        return None if found[key] is None else (found[key], 0.9)

    monkeypatch.setattr(real_photos, "detect_points", fake_detect)
    out = real_photos.load_aligned(real_dir)
    assert calls == list(found)                                      # the clicked photo was not detected
    assert out["source"] == {"clicked": 1, "detected": 2} and out["no_face"] == ["emp_b/IMG_001.jpg"]
    assert out["names"] == ["real/emp_a"] and out["too_few"] == ["emp_b"]


def test_yunet_finds_no_face_on_a_blank_photo() -> None:
    pytest.importorskip("cv2")
    assert real_photos.YUNET_MODEL.exists()
    assert real_photos.detect_points(np.full((800, 600, 3), 128, np.uint8)) is None
