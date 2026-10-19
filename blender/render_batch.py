"""Render a batch of identities inside Blender.

    $BLENDER_PATH --background --python blender/render_batch.py -- --job output/<job_id>/job.json [--resume]

--resume skips identities whose params.json already says "done" (e.g. after a Colab disconnect).
--max-identities N renders at most N new identities, then exits (a fresh Blender process per group
keeps memory flat on long Colab runs). --stop-at T (unix time) starts no identity that would not
finish before T (estimate = slowest identity so far). Both leave progress.json "status": "partial".

job.json (written by the app; output goes next to it):
    {
      "job_id": "...", "seed": 123, "num_identities": 3, "images_per_identity": 4,
      "render": {"samples": 48, "device": "CPU"},   # optional; else env, else defaults
      "params": {...},                                # optional overrides of DEFAULT_PARAMS
      "identity_indices": [3, 7]                      # optional: only these identities (default 1..N)
    }

Writes landmarks.json per identity too (5 projected points per image, blender/landmarks.py).

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
import landmarks  # noqa: E402
import scene  # noqa: E402
from shared import constants as C  # noqa: E402
from shared import sampling  # noqa: E402

FACE_TARGETS_FILE = PROJECT_ROOT / "config" / "face_targets.json"
HDRI_DIR = PROJECT_ROOT / "assets" / "hdri"


def hdri_files(ids: list[str]) -> dict[str, str]:
    """{hdri id: file path} from assets/hdri/hdris.json; fails early if a file is missing."""
    known = {h["id"]: HDRI_DIR / h["file"] for h in json.loads((HDRI_DIR / "hdris.json").read_text(encoding="utf-8"))}
    missing = [i for i in ids if i not in known or not known[i].exists()]
    if missing:
        raise FileNotFoundError(f"HDRIs missing: {missing} (download them with colab/setup_colab.sh)")
    return {i: str(known[i]) for i in ids}

log = logging.getLogger("render_batch")


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(prog="render_batch.py")
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--resume", action="store_true", help="skip identities already rendered")
    parser.add_argument("--max-identities", type=int, help="render at most this many new identities, then exit")
    parser.add_argument("--stop-at", type=float, help="unix time: start no identity that would end after it")
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


def render_identity(job: dict, params: dict, index: int, categories: dict[str, dict], out_dir: Path,
                    hdris: dict[str, str]) -> dict:
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
    vertices = landmarks.load_vertices()
    points = landmarks.world_points(basemesh, eyes, vertices)
    projected = {}
    record["build_seconds"] = time.perf_counter() - t0

    # Auto-exposure reference: this person's face brightness under neutral light at exposure 0.
    preview = str(out_dir / "ae_preview.png")
    scene.place_camera(camera, eye_mid, iod, C.AE_REFERENCE_CAMERA)
    scene.place_key_light(light, eye_mid, C.AE_REFERENCE_LIGHT)
    scene.set_environment({"kind": "backdrop"}, C.AE_REFERENCE_BACKGROUND, light, camera, eye_mid, hdris)
    reference_luma = scene.preview_face_luma(camera, basemesh, eye_mid, iod, preview)
    record["exposure_reference_luma"] = reference_luma

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
        projected[C.IMAGE_STEM_FMT.format(image_index)] = landmarks.project(camera, points, basemesh, eyes, see_through)
        scene.place_key_light(light, eye_mid, image["light"])
        image["environment_derived"] = scene.set_environment(image["environment"], image["background"], light,
                                                             camera, eye_mid, hdris)
        measured = scene.preview_face_luma(camera, basemesh, eye_mid, iod, preview)
        image["exposure"] = {"preview_face_luma": measured,
                             "ev": scene.auto_exposure(measured, reference_luma, image["exposure_jitter_ev"])}
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

    Path(preview).unlink(missing_ok=True)
    write_json(out_dir / "landmarks.json", {"order": landmarks.LANDMARK_NAMES, "vertices": vertices,
                                            "resolution": C.RENDER_RESOLUTION, "images": projected})
    record["status"] = "done"
    return record


def already_done(out_dir: Path) -> bool:
    """True when this identity was rendered by the current sampler. An identity from an older
    sampler is an error: resuming would silently mix two code versions in one job."""
    params_file = out_dir / "params.json"
    try:
        record = json.loads(params_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if record.get("status") != "done":
        return False
    if record.get("sampler_version") != C.SAMPLER_VERSION:
        raise SystemExit(f"{params_file} was made with sampler {record.get('sampler_version')}, this code is "
                         f"sampler {C.SAMPLER_VERSION}: use a new job id instead of --resume")
    return True


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
    hdris = hdri_files(params["hdri_ids"])
    indices = job.get("identity_indices") or list(range(1, job["num_identities"] + 1))
    total = len(indices)
    progress_file = job_dir / "progress.json"
    write_json(progress_file, {"done": 0, "total": total, "status": "running"})
    log.info("Job %s: %d identities x %d images, %d samples on %s",
             job.get("job_id"), total, job["images_per_identity"], samples, device)

    rendered, slowest = 0, 0.0
    for done, index in enumerate(indices, start=1):
        out_dir = job_dir / C.IDENTITY_DIR_FMT.format(index)
        if args.resume and already_done(out_dir):
            emit({"event": "identity_done", "index": index, "status": "done", "skipped": True})
            continue
        if args.max_identities is not None and rendered >= args.max_identities:
            write_json(progress_file, {"done": done - 1, "total": total, "status": "partial"})
            emit({"event": "stopped", "reason": "max_identities", "next_index": index})
            return
        if args.stop_at is not None and time.time() + slowest > args.stop_at:
            write_json(progress_file, {"done": done - 1, "total": total, "status": "partial"})
            emit({"event": "stopped", "reason": "time_limit", "next_index": index})
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        try:
            scene.configure_render(samples, device)
            record = render_identity(job, params, index, categories, out_dir, hdris)
        except Exception:  # one bad identity must not kill the batch
            log.exception("Identity %d failed", index)
            record = {"identity_id": C.IDENTITY_DIR_FMT.format(index), "status": "failed",
                      "error": traceback.format_exc()}
        record["total_seconds"] = time.perf_counter() - t0
        rendered += 1
        slowest = max(slowest, record["total_seconds"])
        record["render"] = {"samples": samples, "device": device, "resolution": C.RENDER_RESOLUTION}
        write_json(out_dir / "params.json", record)
        write_json(progress_file, {"done": done, "total": total, "status": "running"})
        emit({"event": "identity_done", "index": index, "status": record["status"]})

    write_json(progress_file, {"done": total, "total": total, "status": "rendered"})


main()
