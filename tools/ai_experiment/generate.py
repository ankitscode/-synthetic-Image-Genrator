"""AI-generation experiment (approaches B and C). EXPERIMENT ONLY, see CLAUDE.md rule 1:
outputs go to output/ai_experiment/ and must not enter training data.

    venv_ai/bin/python tools/ai_experiment/generate.py

B  img2img: our 3D renders (same synthetic identity, 2 poses) -> photoreal.
   Question: does the same person stay the same person?
C  text2img: 2 images from one prompt describing one person, different seeds.
   Question: do we get the same person twice?

Model: Stable Diffusion 1.5 (CreativeML OpenRAIL-M), pinned revision, CPU, fp32.
Deterministic: fixed seeds, fixed revision.
"""

import json
import logging
import sys
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from huggingface_hub import snapshot_download
from PIL import Image

log = logging.getLogger("ai_experiment")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "output" / "ai_experiment"
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MODEL_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
MODEL_FILES = [
    "model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors", "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors", "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
]
SIZE = 512
STEPS = 25
GUIDANCE = 7.0
NEGATIVE = ("cartoon, 3d render, cgi, painting, drawing, deformed, disfigured, extra eyes, "
            "blurry face, low quality, watermark, text")

# B: two renders of the SAME synthetic identity (id_00011 of test_12x6), two poses.
B_INPUTS = [
    PROJECT_ROOT / "output" / "smoke" / "realism_preview" / "id11_C_1.png",
    PROJECT_ROOT / "output" / "smoke" / "realism_preview" / "id11_C_2.png",
]
B_PROMPT = ("close-up photo of a young man with short dark curly hair, natural skin texture, "
            "neutral expression, natural light, shot on a phone camera, photorealistic")
B_STRENGTH = 0.45  # 0 = keep the render, 1 = ignore it
B_SEED = 11

# C: one text description of one person, two seeds.
C_PROMPT = ("close-up portrait photo of a 35 year old Arab man with short black hair and a short "
            "beard, neutral expression, indoor natural light, shot on a phone camera, photorealistic")
C_SEEDS = [1, 2]


def load_model_dir() -> str:
    return snapshot_download(MODEL_ID, revision=MODEL_REVISION, allow_patterns=MODEL_FILES)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = load_model_dir()
    common = {"torch_dtype": torch.float32, "variant": "fp16", "safety_checker": None, "feature_extractor": None,
              "requires_safety_checker": False}
    record = {"model": MODEL_ID, "revision": MODEL_REVISION, "license": "creativeml-openrail-m",
              "steps": STEPS, "guidance": GUIDANCE, "size": SIZE, "negative_prompt": NEGATIVE, "images": []}

    img2img = StableDiffusionImg2ImgPipeline.from_pretrained(model_dir, **common)
    for i, src in enumerate(B_INPUTS, start=1):
        init = Image.open(src).convert("RGB").resize((SIZE, SIZE), Image.Resampling.LANCZOS)
        t0 = time.perf_counter()
        out = img2img(prompt=B_PROMPT, negative_prompt=NEGATIVE, image=init, strength=B_STRENGTH,
                      num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                      generator=torch.Generator("cpu").manual_seed(B_SEED)).images[0]
        name = f"B_{i}.png"
        out.save(OUT_DIR / name)
        init.save(OUT_DIR / f"B_{i}_input.png")
        seconds = time.perf_counter() - t0
        record["images"].append({"file": name, "approach": "B img2img", "input": str(src.relative_to(PROJECT_ROOT)),
                                 "prompt": B_PROMPT, "strength": B_STRENGTH, "seed": B_SEED, "seconds": seconds})
        log.info("%s done in %.0fs", name, seconds)
    del img2img

    txt2img = StableDiffusionPipeline.from_pretrained(model_dir, **common)
    for i, seed in enumerate(C_SEEDS, start=1):
        t0 = time.perf_counter()
        out = txt2img(prompt=C_PROMPT, negative_prompt=NEGATIVE, width=SIZE, height=SIZE,
                      num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                      generator=torch.Generator("cpu").manual_seed(seed)).images[0]
        name = f"C_{i}.png"
        out.save(OUT_DIR / name)
        seconds = time.perf_counter() - t0
        record["images"].append({"file": name, "approach": "C text2img", "prompt": C_PROMPT, "seed": seed,
                                 "seconds": seconds})
        log.info("%s done in %.0fs", name, seconds)

    (OUT_DIR / "generation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
