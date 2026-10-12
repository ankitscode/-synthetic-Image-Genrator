"""Checks on rendered jobs' params.json files (output/<job>/id_*/params.json).

Runs on every job found under output/ (skips when nothing has been rendered yet).
Set SYNTHFACE_JOB_DIR to check one job only.
"""

import json
import os
from pathlib import Path

import pytest

from shared import constants as C
from shared import sampling

PROJECT_ROOT = Path(__file__).parent.parent
FACE_TARGETS = json.loads((PROJECT_ROOT / "config" / "face_targets.json").read_text())
CATEGORIES = {c["name"]: c["region"] for c in FACE_TARGETS["categories"]}
SKIN_KEYS = ("hue_shift", "saturation", "value", "roughness", "specular", "clearcoat", "tone")
PER_IMAGE_RENDER_KEYS = ("derived", "eye_visibility", "rejected_attempts", "applied_skin", "render_seconds", "file", "qa")
FLOAT32_TOL = 1e-5  # Blender stores node values as float32


def _job_dirs() -> list[Path]:
    if os.environ.get("SYNTHFACE_JOB_DIR"):
        return [Path(os.environ["SYNTHFACE_JOB_DIR"])]
    return sorted(p.parent for p in (PROJECT_ROOT / "output").glob("*/job.json"))


def _params_files() -> list[Path]:
    return [f for job in _job_dirs() for f in sorted(job.glob("id_*/params.json"))]


PARAMS_FILES = _params_files()
needs_output = pytest.mark.skipif(not PARAMS_FILES, reason="no rendered jobs under output/")


def _new_format(record: dict) -> bool:
    return record.get("status") == "done" and record.get("sampler_version") == C.SAMPLER_VERSION


def check_skin_consistency(record: dict) -> None:
    """Every image of an identity used exactly the identity's sampled skin values."""
    sampled = record["identity"]["skin"]
    for img in record["images"]:
        applied = img["applied_skin"]
        for key in SKIN_KEYS:
            assert applied[key] == pytest.approx(sampled[key], abs=FLOAT32_TOL), (img["file"], key)
    first = record["images"][0]["applied_skin"]
    assert all(img["applied_skin"] == first for img in record["images"])


def test_check_skin_consistency_catches_drift() -> None:
    skin = {k: 0.5 for k in SKIN_KEYS}
    good = {"identity": {"skin": skin}, "images": [{"file": f"raw/{i}.png", "applied_skin": dict(skin)} for i in range(3)]}
    check_skin_consistency(good)
    bad = json.loads(json.dumps(good))
    bad["images"][2]["applied_skin"]["value"] = 0.7
    with pytest.raises(AssertionError):
        check_skin_consistency(bad)


@needs_output
@pytest.mark.parametrize("path", PARAMS_FILES, ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}")
def test_skin_identical_across_identity_images(path: Path) -> None:
    record = json.loads(path.read_text())
    if not _new_format(record):
        pytest.skip("rendered with an older sampler version")
    check_skin_consistency(record)


@needs_output
@pytest.mark.parametrize("path", PARAMS_FILES, ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}")
def test_params_reproducible_from_seed(path: Path) -> None:
    """The stored identity equals a fresh sample from (job seed, identity index)."""
    record = json.loads(path.read_text())
    if not _new_format(record):
        pytest.skip("rendered with an older sampler version")
    job = json.loads((path.parent.parent / "job.json").read_text())
    params = sampling.resolve_params(job.get("params"))
    index = record["identity"]["identity_index"]
    assert json.loads(json.dumps(sampling.sample_identity(job["seed"], index, params, CATEGORIES))) == record["identity"]


@needs_output
@pytest.mark.parametrize("path", PARAMS_FILES, ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}")
def test_face_not_clipped(path: Path) -> None:
    record = json.loads(path.read_text())
    if not _new_format(record):
        pytest.skip("rendered with an older sampler version")
    for img in record["images"]:
        assert img["qa"]["face_samples"] > 0, img["file"]
        assert img["qa"]["face_white_fraction"] <= C.QA_MAX_WHITE_FRACTION, img["file"]


@needs_output
@pytest.mark.parametrize("path", PARAMS_FILES, ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}")
def test_image_params_reproducible_from_seed(path: Path) -> None:
    """Each stored image equals a fresh sample from (job seed, identity, image, attempt)."""
    record = json.loads(path.read_text())
    if not _new_format(record):
        pytest.skip("rendered with an older sampler version")
    job = json.loads((path.parent.parent / "job.json").read_text())
    params = sampling.resolve_params(job.get("params"))
    index = record["identity"]["identity_index"]
    for img in record["images"]:
        stored = {k: v for k, v in img.items() if k not in PER_IMAGE_RENDER_KEYS}
        fresh = sampling.sample_image(job["seed"], index, img["image_index"], params, img["attempt"])
        assert json.loads(json.dumps(fresh)) == stored, img["file"]


@needs_output
@pytest.mark.parametrize("path", PARAMS_FILES, ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}")
def test_both_eyes_visible(path: Path) -> None:
    record = json.loads(path.read_text())
    if not _new_format(record):
        pytest.skip("rendered with an older sampler version")
    for img in record["images"]:
        assert all(v["visible"] for v in img["eye_visibility"].values()), (img["file"], img["eye_visibility"])
