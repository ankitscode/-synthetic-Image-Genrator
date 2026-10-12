"""Render a batch of identities inside Blender.

    $BLENDER_PATH --background --python blender/render_batch.py -- --job output/<job_id>/job.json

job.json (written by the app; output goes next to it):
    {
      "job_id": "...", "seed": 123, "num_identities": 3, "images_per_identity": 4,
      "render": {"samples": 48, "device": "CPU"},   # optional; else env, else defaults
      "params": {...}                                 # optional overrides of DEFAULT_PARAMS
    }

stdout: one JSON line per finished identity ({"event": "identity_done", "index": i}).
Everything else goes to stderr via logging.
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import bpy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "blender"))
sys.path.insert(0, str(PROJECT_ROOT))

import human  # noqa: E402
import scene  # noqa: E402
from shared import constants as C  # noqa: E402
from shared import sampling  # noqa: E402

FACE_TARGETS_FILE = PROJECT_ROOT / "config" / "face_targets.json"

log = logging.getLogger("render_batch")


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(prog="render_batch.py")
    parser.add_argument("--job", required=True, type=Path)
    return parser.parse_args(argv)


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def emit(event: dict) -> None:
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def render_settings(job: dict) -> tuple[int, str]:
    render = job.get("render", {})
    samples = int(render.get("samples") or os.environ.get("RENDER_SAMPLES") or C.DEFAULT_RENDER_SAMPLES)
    device = str(render.get("device") or os.environ.get("RENDER_DEVICE") or C.DEFAULT_RENDER_DEVICE)
    return samples, device


def render_identity(job: dict, params: dict, index: int, categories: dict[str, dict], out_dir: Path) -> dict:
    identity = sampling.sample_identity(job["seed"], index, params, {n: c["region"] for n, c in categories.items()})
    record: dict = {"identity_id": C.IDENTITY_DIR_FMT.format(index), "status": "running",
                    "sampler_version": C.SAMPLER_VERSION, "identity": identity, "images": []}
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    human.clear_scene()
    basemesh, eyes = human.build_human(identity, categories)
    eye_mid, iod = human.eye_geometry(eyes)
    left_eye, right_eye = human.eye_centres(eyes)
    eye_points = {"subject_left": left_eye, "subject_right": right_eye}
    see_through = {o.name for o in bpy.data.objects
                   if o.name.endswith((f".{identity['eyebrows']}", f".{identity['eyelashes']}"))}
    camera, light = scene.create_rig()
    record["build_seconds"] = time.perf_counter() - t0

    for image_index in range(1, job["images_per_identity"] + 1):
        rejected = []
        for attempt in range(C.MAX_IMAGE_ATTEMPTS):
            image = sampling.sample_image(job["seed"], index, image_index, params, attempt)
            image["derived"] = scene.place_camera(camera, eye_mid, iod, image["camera"])
            image["eye_visibility"] = scene.eye_visibility(camera, eyes, eye_points, see_through)
            if all(v["visible"] for v in image["eye_visibility"].values()):
                break
            rejected.append({"attempt": attempt, "camera": image["camera"], "eye_visibility": image["eye_visibility"]})
        else:
            log.warning("%s image %d: eyes hidden after %d attempts, keeping the last one",
                        record["identity_id"], image_index, C.MAX_IMAGE_ATTEMPTS)
        image["rejected_attempts"] = rejected
        scene.place_key_light(light, eye_mid, image["light"])
        scene.set_background(image["background"])
        image["applied_skin"] = human.read_applied_skin(basemesh)  # proof: same for every image

        rel = Path("raw") / (C.IMAGE_STEM_FMT.format(image_index) + ".png")
        bpy.context.scene.render.filepath = str(out_dir / rel)
        t_img = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        image["render_seconds"] = time.perf_counter() - t_img
        image["file"] = str(rel)
        image["qa"] = scene.face_exposure_qa(camera, basemesh, eye_mid, iod, str(out_dir / rel))
        record["images"].append(image)
        log.info("%s %s rendered in %.1fs, face white %.2f%%", record["identity_id"], rel,
                 image["render_seconds"], 100 * image["qa"]["face_white_fraction"])

    record["status"] = "done"
    return record


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = parse_args()
    job_path = args.job.resolve()
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job_dir = job_path.parent
    params = sampling.resolve_params(job.get("params"))
    categories = {c["name"]: c for c in json.loads(FACE_TARGETS_FILE.read_text(encoding="utf-8"))["categories"]}

    samples, device = render_settings(job)
    human.load_mpfb()
    total = job["num_identities"]
    progress_file = job_dir / "progress.json"
    write_json(progress_file, {"done": 0, "total": total, "status": "running"})
    log.info("Job %s: %d identities x %d images, %d samples on %s",
             job.get("job_id"), total, job["images_per_identity"], samples, device)

    for index in range(1, total + 1):
        out_dir = job_dir / C.IDENTITY_DIR_FMT.format(index)
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        try:
            scene.configure_render(samples, device)
            record = render_identity(job, params, index, categories, out_dir)
        except Exception:  # one bad identity must not kill the batch
            log.exception("Identity %d failed", index)
            record = {"identity_id": C.IDENTITY_DIR_FMT.format(index), "status": "failed",
                      "error": traceback.format_exc()}
        record["total_seconds"] = time.perf_counter() - t0
        record["render"] = {"samples": samples, "device": device, "resolution": C.RENDER_RESOLUTION}
        write_json(out_dir / "params.json", record)
        write_json(progress_file, {"done": index, "total": total, "status": "running"})
        emit({"event": "identity_done", "index": index, "status": record["status"]})

    write_json(progress_file, {"done": total, "total": total, "status": "rendered"})


main()
