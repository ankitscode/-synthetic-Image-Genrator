"""Single-person reference test (approach B, experiment only).

    venv_ai/bin/python tools/ai_experiment/reference_test.py           # generate
    venv_ai/bin/python tools/ai_experiment/reference_test.py --report  # validate what exists
    venv_ai/bin/python tools/ai_experiment/reference_test.py --hair-fix [--report]

--hair-fix: the 3D hair texture (alpha cards, reads as a knitted cap after SD) is masked out
before SD: inside the hair mask (<shot>_hair.png from render_shots.py --hair-masks) the render
is replaced by its smooth local colour (shape and shading kept, strand texture removed), and
the prompt names the hairstyle, so SD paints the hair once in the reference; the variations
get it from the reference through IP-Adapter. Outputs go to .../reference_hairfix/.

1. Reference : the main person's frontal 3D render (shot s1) made photoreal with
               SD 1.5 img2img + ControlNet-depth (Blender depth map).
2. Variations: the other shots (new angles and lighting). Each one starts from that shot's
               3D render + its depth map (pose/geometry) and uses the reference through
               IP-Adapter Plus-Face (appearance/identity). Same seed for every shot.
3. Control   : a different person's frontal render made photoreal the same way as the
               reference; this is the "different person" level for every check.

Validation (no face detector, no face recognition model):
  - shape drift   : optical flow from each output back to its own 3D render, in the face mask
  - skin Delta-E  : CIELAB skin colour vs the reference (face mask)
  - CLIP face sim : cosine similarity of eye-aligned face crops to the reference, using the
                    CLIP image encoder (general image similarity, NOT a face recognizer)
Writes output/ai_experiment/consistency/reference[_hairfix]/{*.png, runs.jsonl, metrics.json, sheet.png}.
"""

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import GUIDANCE, MODEL_FILES, MODEL_ID, MODEL_REVISION, NEGATIVE, STEPS  # noqa: E402
from run_variants import CONTROLNET_FILES, CONTROLNET_ID, CONTROLNET_REVISION, SHOTS_DIR, SIZE  # noqa: E402

log = logging.getLogger("reference_test")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "output" / "ai_experiment" / "consistency" / "reference"
RUNS = OUT / "runs.jsonl"
IPA_ID = "h94/IP-Adapter"
IPA_REVISION = "018e402774aeeddd60609b4ecdb7e298259dc729"
IPA_WEIGHT = "ip-adapter-plus-face_sd15.safetensors"
IPA_FILES = ["models/image_encoder/config.json", "models/image_encoder/model.safetensors", f"models/{IPA_WEIGHT}"]

PERSON = "id_00011"
CONTROL = "id_00002"
REF_SHOT = "s1"
VARIATION_SHOTS = ["s2", "s3", "s6", "s4", "s5", "s7", "s8"]  # first three give an early answer
SEED = 1011
REF_STRENGTH = 0.55
VAR_STRENGTH = 0.55
IPA_SCALE = 0.7
CONTROLNET_SCALE = 1.0
PROMPT = ("close-up photo of a man{hair}, natural skin texture, neutral expression, natural light, "
          "shot on a phone camera, photorealistic")

HAIRFIX_OUT = OUT.parent / "reference_hairfix"
HAIRFIX_SHOTS = ["s2", "s3", "s6"]  # frontal-ish, side, warm light: 1 reference + 3 variations
HAIR_PROMPT = {"id_00011": " with short dark brown afro hair", "id_00002": " with short brown hair"}
HAIR_DILATE_PX = 5
HAIR_FEATHER_SIGMA = 3.0
HAIR_FILL_SIGMA = 10.0  # smoothing of the hair colour fill; removes strand/card texture
HAIR_NOISE = 6.0  # seeded grain (0-255) in the fill so SD has texture to work from
hair_fix = False  # set by --hair-fix

# eye-aligned face crop for CLIP / colour checks (eyes at these pixel positions in a 224 crop)
CROP = 224
CROP_EYES = np.float32([[80, 96], [144, 96]])  # viewer-left eye, viewer-right eye


def shot_png(person: str, shot: str, kind: str = "") -> Path:
    return SHOTS_DIR / person / f"{shot}{kind}.png"


def shot_meta(person: str, shot: str) -> dict:
    return json.loads((SHOTS_DIR / person / f"{shot}.json").read_text())


def out_png(name: str) -> Path:
    return OUT / f"{name}.png"


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.Resampling.LANCZOS)


def prompt(person: str) -> str:
    return PROMPT.format(hair=HAIR_PROMPT[person] if hair_fix else "")


def hair_masked_input(person: str, shot: str) -> Image.Image:
    """Render with the hair texture replaced by a smooth fill of its own local colour."""
    img = np.asarray(load_rgb(shot_png(person, shot)), np.float32)
    hair = cv2.imread(str(shot_png(person, shot, "_hair")), cv2.IMREAD_GRAYSCALE) > 127
    hair = cv2.dilate(hair.astype(np.uint8), np.ones((2 * HAIR_DILATE_PX + 1,) * 2, np.uint8)).astype(np.float32)
    # normalised convolution: average colour of nearby hair pixels only (no skin/background bleed)
    num = cv2.GaussianBlur(img * hair[..., None], (0, 0), HAIR_FILL_SIGMA)
    den = cv2.GaussianBlur(hair, (0, 0), HAIR_FILL_SIGMA)[..., None]
    fill = num / np.maximum(den, 1e-3)
    fill += np.random.default_rng(SEED).normal(0.0, HAIR_NOISE, fill.shape).astype(np.float32)
    alpha = cv2.GaussianBlur(hair, (0, 0), HAIR_FEATHER_SIGMA)[..., None]
    out = alpha * fill + (1.0 - alpha) * img
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def sd_input(person: str, shot: str) -> Image.Image:
    if not hair_fix:
        return load_rgb(shot_png(person, shot))
    image = hair_masked_input(person, shot)
    image.save(OUT / f"input_{person}_{shot}.png")
    return image


# --------------------------------------------------------------------------- generation
def generate() -> None:
    common = {"torch_dtype": torch.float32, "variant": "fp16", "safety_checker": None,
              "feature_extractor": None, "requires_safety_checker": False}
    controlnet = ControlNetModel.from_pretrained(
        snapshot_download(CONTROLNET_ID, revision=CONTROLNET_REVISION, allow_patterns=CONTROLNET_FILES),
        torch_dtype=torch.float32, variant="fp16")
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        snapshot_download(MODEL_ID, revision=MODEL_REVISION, allow_patterns=MODEL_FILES),
        controlnet=controlnet, **common)

    def run(name: str, person: str, shot: str, strength: float, **extra) -> None:
        if out_png(name).exists():
            return
        t0 = time.perf_counter()
        image = pipe(prompt=prompt(person), negative_prompt=NEGATIVE, image=sd_input(person, shot),
                     control_image=load_rgb(shot_png(person, shot, "_depth")), strength=strength,
                     controlnet_conditioning_scale=CONTROLNET_SCALE, num_inference_steps=STEPS,
                     guidance_scale=GUIDANCE, generator=torch.Generator("cpu").manual_seed(SEED), **extra).images[0]
        image.save(out_png(name))
        seconds = time.perf_counter() - t0
        rec = {"name": name, "person": person, "shot": shot, "strength": strength, "seed": SEED,
               "ip_adapter": "ip_adapter_image_embeds" in extra, "hair_fix": hair_fix,
               "prompt": prompt(person), "seconds": round(seconds, 1)}
        with RUNS.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        log.info("%s done in %.0fs", name, seconds)

    run("reference", PERSON, REF_SHOT, REF_STRENGTH)
    run("control_reference", CONTROL, REF_SHOT, REF_STRENGTH)

    ipa_dir = snapshot_download(IPA_ID, revision=IPA_REVISION, allow_patterns=IPA_FILES)
    pipe.load_ip_adapter(ipa_dir, subfolder="models", weight_name=IPA_WEIGHT,
                         image_encoder_folder="image_encoder", torch_dtype=torch.float32)
    pipe.set_ip_adapter_scale(IPA_SCALE)
    embeds = pipe.prepare_ip_adapter_image_embeds(
        ip_adapter_image=[Image.open(out_png("reference")).convert("RGB")], ip_adapter_image_embeds=None,
        device="cpu", num_images_per_prompt=1, do_classifier_free_guidance=True)
    pipe.image_encoder = None  # embeddings are computed once; free ~2.5 GB before generating
    gc.collect()
    for shot in VARIATION_SHOTS:
        run(f"var_{shot}", PERSON, shot, VAR_STRENGTH, ip_adapter_image_embeds=embeds)


# --------------------------------------------------------------------------- validation
def face_crop(img_bgr: np.ndarray, meta: dict) -> np.ndarray:
    """Eye-aligned crop. Subject's right eye is on the viewer's left."""
    eyes = np.float32([meta["eyes_px"]["subject_right"], meta["eyes_px"]["subject_left"]])
    scale = img_bgr.shape[1] / SIZE
    matrix, _ = cv2.estimateAffinePartial2D(eyes * scale, CROP_EYES)
    return cv2.warpAffine(img_bgr, matrix, (CROP, CROP), flags=cv2.INTER_AREA)


def skin_lab(path: Path, person: str, shot: str) -> np.ndarray:
    img = cv2.imread(str(path)).astype(np.float32) / 255.0
    mask = cv2.imread(str(shot_png(person, shot, "_mask")), cv2.IMREAD_GRAYSCALE) > 127
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[mask].mean(axis=0)


def flow_drift(render: Path, output: Path, person: str, shot: str) -> float:
    a = cv2.cvtColor(cv2.imread(str(render)), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(cv2.imread(str(output)), cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 21, 5, 7, 1.5, 0)
    mask = cv2.imread(str(shot_png(person, shot, "_mask")), cv2.IMREAD_GRAYSCALE) > 127
    return float(np.linalg.norm(flow, axis=2)[mask].mean())


def clip_embedder():
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    ipa_dir = snapshot_download(IPA_ID, revision=IPA_REVISION, allow_patterns=IPA_FILES)
    model = CLIPVisionModelWithProjection.from_pretrained(f"{ipa_dir}/models/image_encoder").eval()
    processor = CLIPImageProcessor()

    def embed(crop_bgr: np.ndarray) -> np.ndarray:
        rgb = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            vec = model(**processor(images=rgb, return_tensors="pt")).image_embeds[0].numpy()
        return vec / np.linalg.norm(vec)
    return embed


def report() -> dict:
    embed = clip_embedder()
    ref_meta = shot_meta(PERSON, REF_SHOT)
    ref_img = cv2.imread(str(out_png("reference")))
    ref_vec = embed(face_crop(ref_img, ref_meta))
    ref_lab = skin_lab(out_png("reference"), PERSON, REF_SHOT)

    def entry(path: Path, person: str, shot: str, render: Path | None) -> dict:
        img = cv2.imread(str(path))
        vec = embed(face_crop(img, shot_meta(person, shot)))
        e = {"clip_face_similarity_to_reference": float(vec @ ref_vec),
             "skin_delta_e_to_reference": float(np.linalg.norm(skin_lab(path, person, shot) - ref_lab))}
        if render is not None:
            e["shape_drift_px_vs_own_render"] = flow_drift(render, path, person, shot)
        return e

    result = {"different_person": entry(out_png("control_reference"), CONTROL, REF_SHOT, shot_png(CONTROL, REF_SHOT)),
              "reference": {"shape_drift_px_vs_own_render": flow_drift(shot_png(PERSON, REF_SHOT), out_png("reference"),
                                                                       PERSON, REF_SHOT)},
              "variations": {}, "plain_3d_renders": {}}
    for shot in VARIATION_SHOTS:
        if out_png(f"var_{shot}").exists():
            result["variations"][shot] = entry(out_png(f"var_{shot}"), PERSON, shot, shot_png(PERSON, shot))
        # same-person level for clean 3D: each plain render vs the plain frontal render
        render_vec = embed(face_crop(cv2.imread(str(shot_png(PERSON, shot))), shot_meta(PERSON, shot)))
        plain_ref = embed(face_crop(cv2.imread(str(shot_png(PERSON, REF_SHOT))), ref_meta))
        result["plain_3d_renders"][shot] = {"clip_face_similarity_to_plain_frontal": float(render_vec @ plain_ref)}
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    sheet(result)
    return result


def sheet(result: dict) -> None:
    tile, label_h = 200, 46
    items = [("reference (s1)", out_png("reference"), "photoreal frontal\n= identity reference"),
             ("control (other man)", out_png("control_reference"),
              f"CLIP {result['different_person']['clip_face_similarity_to_reference']:.3f}\n"
              f"skin dE {result['different_person']['skin_delta_e_to_reference']:.1f}")]
    for shot, m in result["variations"].items():
        meta = shot_meta(PERSON, shot)
        items.append((f"{shot} yaw {meta['yaw']:+.0f} pitch {meta['pitch']:+.0f}", out_png(f"var_{shot}"),
                      f"CLIP {m['clip_face_similarity_to_reference']:.3f}  dE {m['skin_delta_e_to_reference']:.1f}\n"
                      f"shape drift {m['shape_drift_px_vs_own_render']:.2f}px  ({meta['lighting']})"))
    cols = 3
    rows = -(-len(items) // cols)
    img = Image.new("RGB", (cols * (tile + 6) + 6, 24 + rows * (tile + label_h + 6)), (24, 24, 24))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((6, 6), "Reference test: does each variation look like the reference? (higher CLIP / lower dE = closer)",
              fill=(230, 230, 230), font=font)
    for i, (title, path, text) in enumerate(items):
        x, y = 6 + (i % cols) * (tile + 6), 24 + (i // cols) * (tile + label_h + 6)
        img.paste(Image.open(path).convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS), (x, y))
        draw.multiline_text((x, y + tile + 2), f"{title}\n{text}", fill=(230, 230, 230), font=font, spacing=1)
    img.save(OUT / "sheet.png")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="only validate existing outputs")
    parser.add_argument("--hair-fix", action="store_true", help="mask out the 3D hair texture (see docstring)")
    args = parser.parse_args()
    if args.hair_fix:
        global OUT, RUNS, VARIATION_SHOTS, hair_fix
        OUT, RUNS, VARIATION_SHOTS, hair_fix = HAIRFIX_OUT, HAIRFIX_OUT / "runs.jsonl", HAIRFIX_SHOTS, True
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.report:
        generate()
    result = report()
    d = result["different_person"]
    print(f"DIFFERENT PERSON  CLIP {d['clip_face_similarity_to_reference']:.3f}  skin dE {d['skin_delta_e_to_reference']:.1f}")
    for shot, m in result["variations"].items():
        print(f"VARIATION {shot}  CLIP {m['clip_face_similarity_to_reference']:.3f}  skin dE "
              f"{m['skin_delta_e_to_reference']:.1f}  shape drift {m['shape_drift_px_vs_own_render']:.2f}px")
    for shot, m in result["plain_3d_renders"].items():
        print(f"PLAIN 3D  {shot}  CLIP vs plain frontal {m['clip_face_similarity_to_plain_frontal']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
