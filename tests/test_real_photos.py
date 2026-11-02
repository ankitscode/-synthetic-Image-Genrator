"""Real test photos: hand-clicked landmarks, alignment and the report section (stand-in images, no real faces)."""

import json
from pathlib import Path

import numpy as np
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
    out = real_photos.load_aligned(real_dir)
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
