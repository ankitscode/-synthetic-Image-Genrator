"""Consistency test, step 2: run approach-B variants on the rendered shots (venv_ai, CPU).

    venv_ai/bin/python tools/ai_experiment/run_variants.py

Reads output/ai_experiment/consistency/shots/<person>/<shot>{.png,_depth.png}, writes
output/ai_experiment/consistency/b/<variant>/<person>/<shot>.png and appends one JSON line
per image to .../b/runs.jsonl. Resumable: existing outputs are skipped.

Variants (strength = how much SD may change the render):
  B0      img2img 0.45                       (current approach B)
  B1_15 / B1_25 / B1_35  img2img 0.15 / 0.25 / 0.35
  B2_35 / B2_50          img2img + ControlNet-depth (Blender depth map), 0.35 / 0.50
Order: the two variants used for the blind test (B1_35, B2_50, main person + controls) first.
Same prompt and same seed per person for every shot. Deterministic.
"""

import json
import logging
import sys
import time
from pathlib import Path

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, StableDiffusionImg2ImgPipeline
from huggingface_hub import snapshot_download
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import GUIDANCE, MODEL_FILES, MODEL_ID, MODEL_REVISION, NEGATIVE, STEPS  # noqa: E402

log = logging.getLogger("run_variants")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE = PROJECT_ROOT / "output" / "ai_experiment" / "consistency"
SHOTS_DIR = BASE / "shots"
OUT_DIR = BASE / "b"
RUNS_FILE = OUT_DIR / "runs.jsonl"
CONTROLNET_ID = "lllyasviel/control_v11f1p_sd15_depth"
CONTROLNET_REVISION = "539f99181d33db39cf1af2e517cd8056785f0a87"
CONTROLNET_FILES = ["config.json", "diffusion_pytorch_model.fp16.safetensors"]
CONTROLNET_SCALE = 1.0
SIZE = 512
PROMPT = ("close-up photo of a man, natural skin texture, neutral expression, natural light, "
          "shot on a phone camera, photorealistic")
SEED_BASE = 1000  # seed = SEED_BASE + person index, same for all shots of a person

MAIN = "id_00011"
CONTROLS = ["id_00002", "id_00005", "id_00008", "id_00012"]
MAIN_SHOTS = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
CONTROL_SHOTS = ["s1", "s2", "s3", "s6"]
VARIANTS = {
    "B0": {"strength": 0.45, "controlnet": False},
    "B1_15": {"strength": 0.15, "controlnet": False},
    "B1_25": {"strength": 0.25, "controlnet": False},
    "B1_35": {"strength": 0.35, "controlnet": False},
    "B2_35": {"strength": 0.35, "controlnet": True},
    "B2_50": {"strength": 0.50, "controlnet": True},
}
BLIND_TEST_VARIANTS = ["B1_35", "B2_50"]


def plan() -> list[tuple[str, str, str]]:
    """(variant, person, shot) in run order: blind-test variants with controls first."""
    jobs = []
    for variant in BLIND_TEST_VARIANTS:
        jobs += [(variant, MAIN, s) for s in MAIN_SHOTS]
        jobs += [(variant, p, s) for p in CONTROLS for s in CONTROL_SHOTS]
    for variant in VARIANTS:
        if variant not in BLIND_TEST_VARIANTS:
            jobs += [(variant, MAIN, s) for s in MAIN_SHOTS]
    return jobs


def load(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.Resampling.LANCZOS)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    common = {"torch_dtype": torch.float32, "variant": "fp16", "safety_checker": None,
              "feature_extractor": None, "requires_safety_checker": False}
    img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
        snapshot_download(MODEL_ID, revision=MODEL_REVISION, allow_patterns=MODEL_FILES), **common)
    controlnet = ControlNetModel.from_pretrained(
        snapshot_download(CONTROLNET_ID, revision=CONTROLNET_REVISION, allow_patterns=CONTROLNET_FILES),
        torch_dtype=torch.float32, variant="fp16")
    with_depth = StableDiffusionControlNetImg2ImgPipeline(**img2img.components, controlnet=controlnet)

    jobs = plan()
    for n, (variant, person, shot) in enumerate(jobs, start=1):
        out = OUT_DIR / variant / person / f"{shot}.png"
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        cfg = VARIANTS[variant]
        seed = SEED_BASE + int(person.split("_")[1])
        args = {"prompt": PROMPT, "negative_prompt": NEGATIVE, "image": load(SHOTS_DIR / person / f"{shot}.png"),
                "strength": cfg["strength"], "num_inference_steps": STEPS, "guidance_scale": GUIDANCE,
                "generator": torch.Generator("cpu").manual_seed(seed)}
        t0 = time.perf_counter()
        if cfg["controlnet"]:
            image = with_depth(control_image=load(SHOTS_DIR / person / f"{shot}_depth.png"),
                               controlnet_conditioning_scale=CONTROLNET_SCALE, **args).images[0]
        else:
            image = img2img(**args).images[0]
        image.save(out)
        seconds = time.perf_counter() - t0
        record = {"variant": variant, "person": person, "shot": shot, "seed": seed, "seconds": round(seconds, 1),
                  **cfg, "controlnet_scale": CONTROLNET_SCALE if cfg["controlnet"] else None, "prompt": PROMPT,
                  "model": f"{MODEL_ID}@{MODEL_REVISION}",
                  "controlnet_model": f"{CONTROLNET_ID}@{CONTROLNET_REVISION}" if cfg["controlnet"] else None}
        with RUNS_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        log.info("[%d/%d] %s %s %s done in %.0fs", n, len(jobs), variant, person, shot, seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
