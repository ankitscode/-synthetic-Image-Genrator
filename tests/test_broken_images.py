"""Missing or damaged raw images (e.g. an interrupted Drive copy) are reported and re-rendered, not a crash."""

import ast
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from app import postprocess
from shared import constants as C

ROOT = Path(__file__).resolve().parent.parent


def load_complete_file():
    """complete_file() from blender/render_batch.py (that module imports bpy, so exec only the function)."""
    tree = ast.parse((ROOT / "blender" / "render_batch.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "complete_file")
    end = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "PNG_END")
    namespace = {"Path": Path}
    exec(compile(ast.Module([end, fn], []), "render_batch", "exec"), namespace)
    return namespace["complete_file"]


def make_identity(job: Path, name: str, broken_stem: str | None) -> None:
    d = job / name
    (d / "raw").mkdir(parents=True)
    images, record = {}, {"status": "done", "identity": {}, "images": []}
    points = np.array(C.ARCFACE_112, dtype=np.float32) * 2 + 10
    for k in (1, 2):
        stem = f"img_{k:03d}"
        img = np.full((256, 256, 3), 120, np.uint8)
        cv2.imwrite(str(d / "raw" / f"{stem}.png"), img)
        if stem == broken_stem:
            data = (d / "raw" / f"{stem}.png").read_bytes()
            (d / "raw" / f"{stem}.png").write_bytes(data[: len(data) // 2])  # cut off mid-file
        images[stem] = {"points": points.tolist(), "visible": [True] * 5}
    (d / "landmarks.json").write_text(json.dumps({"resolution": 256, "images": images}))
    (d / "params.json").write_text(json.dumps(record))


def test_complete_file_detects_cut_off_png(tmp_path: Path) -> None:
    complete_file = load_complete_file()
    good = tmp_path / "a.png"
    cv2.imwrite(str(good), np.zeros((8, 8, 3), np.uint8))
    cut = tmp_path / "b.png"
    cut.write_bytes(good.read_bytes()[:-5])
    empty = tmp_path / "c.png"
    empty.write_bytes(b"")
    assert complete_file(good) and not complete_file(cut) and not complete_file(empty)
    assert not complete_file(tmp_path / "missing.png")


def test_postprocess_skips_unreadable_image(tmp_path: Path) -> None:
    make_identity(tmp_path, "id_00001", broken_stem="img_002")
    report = postprocess.align_identity(tmp_path / "id_00001")
    assert "error" in report["img_002"] and report["img_001"]["residual_px"] < 1.0
    assert (tmp_path / "id_00001" / "aligned" / "img_001.jpg").exists()


def test_validate_job_lists_broken_people(tmp_path: Path) -> None:
    make_identity(tmp_path, "id_00001", broken_stem=None)
    make_identity(tmp_path, "id_00002", broken_stem="img_001")
    for name in ("id_00001", "id_00002"):
        postprocess.align_identity(tmp_path / name)
    np.savez_compressed(tmp_path / "sel.shapes.npz", shapes=np.random.default_rng(0).normal(size=(2, 5, 3)),
                        indices=np.array([1, 2]), region=np.arange(5))
    # minimal records for the consistency / pair parts
    for name in ("id_00001", "id_00002"):
        rec = {"status": "done", "identity": {"sex": "female", "skin": {"tone": 0.5}, "eyes": {"color": "brown"},
                                              "hair": {"style": "s", "color_name": "black"}, "age_years": 30},
               "images": [{"file": "raw/img_001.png", "camera": {"yaw": 0, "pitch": 0}, "applied_skin": {"x": 1.0}}]}
        (tmp_path / name / "params.json").write_text(json.dumps(rec))
    r = subprocess.run([sys.executable, "tools/validate_job.py", str(tmp_path), "--shapes", str(tmp_path / "sel.shapes.npz")],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    report = json.loads((tmp_path / "validation.json").read_text())
    assert report["broken"] == {"id_00002": ["img_001"]} and "BROKEN" in r.stdout
