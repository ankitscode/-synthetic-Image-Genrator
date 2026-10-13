"""Approach B blind test: N people x 4 shots with the hair fix (experiment only).

    venv_ai/bin/python tools/ai_experiment/blind_test.py           # 2x4 test, generate + report
    venv_ai/bin/python tools/ai_experiment/blind_test.py --report  # report only
    python tools/ai_experiment/blind_test.py --set colab_10 --steps 30   # a render_shots.py set (Colab GPU)

With --set, shots come from output/ai_experiment/sets/<set>/shots/ and outputs go to .../sets/<set>/b/;
the prompt is built from each person's own sex and hair (shot .json). GPU (CUDA, fp16) is used when
available, otherwise CPU (float32). Generation is resumable (existing outputs are skipped).

Per person: reference = frontal shot s1 made photoreal (hair-masked input, ControlNet-depth
img2img); variations s2/s3/s6 = own render + depth + IP-Adapter Plus-Face from that person's
reference (same recipe as reference_test.py --hair-fix). Needs <shot>_hair.png masks
(render_shots.py --hair-masks).

CPU speed-ups vs reference_test.py: one model load for all 8 images, DPM++ 2M Karras with
fewer steps (~9 denoising steps at strength 0.55 instead of ~13), both references first so the
CLIP image encoder is used once and then freed.

Report: CLIP face similarity / skin Delta-E of each variation to its own reference and to the
other people's references (different-person level), rank-1 (is the own reference the most similar?),
shape drift, plus two blind sheets (B = AI outputs, A = plain 3D renders) of numbered pairs,
half same / half different. Writes output/ai_experiment/consistency/blind_2x4/ (or the set's b/).
"""

import argparse
import gc
import itertools
import json
import logging
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import ControlNetModel, DPMSolverMultistepScheduler, StableDiffusionControlNetImg2ImgPipeline
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference_test as rt  # noqa: E402
from generate import GUIDANCE, MODEL_FILES, MODEL_ID, MODEL_REVISION, NEGATIVE  # noqa: E402
from run_variants import CONTROLNET_FILES, CONTROLNET_ID, CONTROLNET_REVISION  # noqa: E402

log = logging.getLogger("blind_test")

OUT = rt.OUT.parent / "blind_2x4"
RUNS = OUT / "runs.jsonl"
SETS_DIR = rt.PROJECT_ROOT / "output" / "ai_experiment" / "sets"
PEOPLE = ["id_00011", "id_00002"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
HAIR_STYLE_WORDS = {"afro01": "short afro", "bob01": "bob-cut", "bob02": "bob-cut", "braid01": "braided",
                    "long01": "long", "ponytail01": "ponytail", "short01": "short", "short02": "short",
                    "short03": "short", "short04": "short"}
REF_SHOT = "s1"
VAR_SHOTS = ["s2", "s3", "s6"]
STEPS = 18  # DPM++ 2M Karras; x strength 0.55 -> 9 denoising steps
NEGATIVE_EXTRA = ", earring, earrings, jewelry, piercing, tattoo, necklace"
BLIND_SEED = 2027
BLIND_PAIRS_EACH = 6
steps = STEPS  # --steps
pairs_each = BLIND_PAIRS_EACH  # --pairs
TILE = 200
TEXT = (230, 230, 230)
BG = (24, 24, 24)


def out_png(person: str, shot: str) -> Path:
    return OUT / f"{person}_{shot}.png"


def prompt(person: str) -> str:
    """Coarse attributes from the synthetic identity's own params (sex, hair); old shots: fixed text."""
    meta = rt.shot_meta(person, REF_SHOT)
    if "hair" not in meta:
        return rt.prompt(person)
    hair = meta["hair"]
    if hair["style"] == "none":
        desc = "with a shaved bald head"
    else:
        desc = f"with {HAIR_STYLE_WORDS.get(hair['style'], '')} {hair['color_name'].replace('_', ' ')} hair"
    noun = "woman" if meta["sex"] == "female" else "man"
    return rt.PROMPT.replace("a man", f"a {noun}").format(hair=" " + " ".join(desc.split()))


def generate() -> None:
    controlnet = ControlNetModel.from_pretrained(
        snapshot_download(CONTROLNET_ID, revision=CONTROLNET_REVISION, allow_patterns=CONTROLNET_FILES),
        torch_dtype=DTYPE, variant="fp16")
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        snapshot_download(MODEL_ID, revision=MODEL_REVISION, allow_patterns=MODEL_FILES), controlnet=controlnet,
        torch_dtype=DTYPE, variant="fp16", safety_checker=None, feature_extractor=None,
        requires_safety_checker=False).to(DEVICE)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    pipe.unet.to(memory_format=torch.channels_last)
    pipe.controlnet.to(memory_format=torch.channels_last)
    pipe.set_progress_bar_config(disable=True)

    def run(person: str, shot: str, strength: float, **extra) -> None:
        out = out_png(person, shot)
        if out.exists():
            return
        t0 = time.perf_counter()
        with torch.inference_mode():
            image = pipe(prompt=prompt(person), negative_prompt=NEGATIVE + NEGATIVE_EXTRA,
                         image=rt.sd_input(person, shot), control_image=rt.load_rgb(rt.shot_png(person, shot, "_depth")),
                         strength=strength, controlnet_conditioning_scale=rt.CONTROLNET_SCALE,
                         num_inference_steps=steps, guidance_scale=GUIDANCE,
                         generator=torch.Generator("cpu").manual_seed(rt.SEED), **extra).images[0]
        image.save(out)
        seconds = time.perf_counter() - t0
        rec = {"person": person, "shot": shot, "strength": strength, "seed": rt.SEED, "steps": steps,
               "scheduler": "DPMSolverMultistep karras", "ip_adapter": "ip_adapter_image_embeds" in extra,
               "device": DEVICE, "prompt": prompt(person), "negative_extra": NEGATIVE_EXTRA, "seconds": round(seconds, 1)}
        with RUNS.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        log.info("%s %s done in %.0fs", person, shot, seconds)

    for person in PEOPLE:
        run(person, REF_SHOT, rt.REF_STRENGTH)

    ipa_dir = snapshot_download(rt.IPA_ID, revision=rt.IPA_REVISION, allow_patterns=rt.IPA_FILES)
    if all(out_png(p, s).exists() for p in PEOPLE for s in VAR_SHOTS):
        return
    pipe.load_ip_adapter(ipa_dir, subfolder="models", weight_name=rt.IPA_WEIGHT,
                         image_encoder_folder="image_encoder", torch_dtype=DTYPE)
    pipe.image_encoder.to(DEVICE)
    pipe.set_ip_adapter_scale(rt.IPA_SCALE)
    embeds = {p: pipe.prepare_ip_adapter_image_embeds(
        ip_adapter_image=[Image.open(out_png(p, REF_SHOT)).convert("RGB")], ip_adapter_image_embeds=None,
        device=DEVICE, num_images_per_prompt=1, do_classifier_free_guidance=True) for p in PEOPLE}
    pipe.image_encoder = None  # free ~2.5 GB before the variations
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    for person in PEOPLE:
        for shot in VAR_SHOTS:
            run(person, shot, rt.VAR_STRENGTH, ip_adapter_image_embeds=embeds[person])


# --------------------------------------------------------------------------- report
def report() -> dict:
    embed = rt.clip_embedder()
    refs = {p: {"vec": embed(rt.face_crop(cv2.imread(str(out_png(p, REF_SHOT))), rt.shot_meta(p, REF_SHOT))),
                "lab": rt.skin_lab(out_png(p, REF_SHOT), p, REF_SHOT)} for p in PEOPLE}
    result: dict = {"settings": {"steps": steps, "scheduler": "DPMSolverMultistep karras",
                                 "strength": rt.VAR_STRENGTH, "ip_adapter_scale": rt.IPA_SCALE,
                                 "negative_extra": NEGATIVE_EXTRA, "people": PEOPLE}, "people": {}}
    same_clip, diff_clip, rank1 = [], [], []
    for p in PEOPLE:
        entries = {}
        for shot in VAR_SHOTS:
            path = out_png(p, shot)
            if not path.exists():
                continue
            vec = embed(rt.face_crop(cv2.imread(str(path)), rt.shot_meta(p, shot)))
            lab = rt.skin_lab(path, p, shot)
            to_ref = {q: float(vec @ refs[q]["vec"]) for q in PEOPLE}
            others = {q: v for q, v in to_ref.items() if q != p}
            closest_other = max(others, key=others.get)
            entries[shot] = {"prompt": prompt(p),
                             "clip_to_own_reference": to_ref[p],
                             "clip_to_closest_other_reference": others[closest_other],
                             "closest_other_person": closest_other,
                             "most_similar_reference_is_own": to_ref[p] > others[closest_other],
                             "skin_delta_e_to_own_reference": float(np.linalg.norm(lab - refs[p]["lab"])),
                             "shape_drift_px_vs_own_render": rt.flow_drift(rt.shot_png(p, shot), path, p, shot)}
            same_clip.append(to_ref[p])
            diff_clip += list(others.values())
            rank1.append(entries[shot]["most_similar_reference_is_own"])
        result["people"][p] = entries
    result["summary"] = {"variations": len(same_clip),
                         "same_person_clip_mean": float(np.mean(same_clip)),
                         "same_person_clip_min": float(np.min(same_clip)),
                         "different_person_clip_mean": float(np.mean(diff_clip)),
                         "different_person_clip_max": float(np.max(diff_clip)),
                         "rank1_own_reference_most_similar": float(np.mean(rank1)),
                         "fully_separated": bool(np.min(same_clip) > np.max(diff_clip))}
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    overview()
    blind_sheets()
    return result


def tile(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB").resize((TILE, TILE), Image.Resampling.LANCZOS)


def overview() -> None:
    """Rows: per person the plain 3D renders, then the AI outputs; columns: shots."""
    shots = [REF_SHOT] + VAR_SHOTS
    rows = []
    for p in PEOPLE:
        rows.append((f"{p}\n3D render", [rt.shot_png(p, s) for s in shots]))
        rows.append((f"{p}\nB output", [out_png(p, s) for s in shots]))
    label_w, head_h = 90, 20
    sheet = Image.new("RGB", (label_w + len(shots) * (TILE + 4), head_h + len(rows) * (TILE + 4)), BG)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for c, s in enumerate(shots):
        meta = rt.shot_meta(PEOPLE[0], s)
        title = f"{s} yaw {meta['yaw']:+.0f} pitch {meta['pitch']:+.0f}" + (" (reference)" if s == REF_SHOT else "")
        draw.text((label_w + c * (TILE + 4) + 4, 4), title, fill=TEXT, font=font)
    for r, (name, paths) in enumerate(rows):
        y = head_h + r * (TILE + 4)
        draw.multiline_text((4, y + 4), name, fill=TEXT, font=font)
        for c, path in enumerate(paths):
            if path.exists():
                sheet.paste(tile(path), (label_w + c * (TILE + 4), y))
    sheet.save(OUT / "overview.png")


def blind_sheets() -> None:
    shots = [REF_SHOT] + VAR_SHOTS
    rng = random.Random(BLIND_SEED)
    answers, csv_rows = {}, ["sheet,pair,your_answer (same/different)"]
    for variant, path in (("B", out_png), ("A", lambda p, s: rt.shot_png(p, s))):
        same_all = [((p, a), (p, b)) for p in PEOPLE for a, b in itertools.combinations(shots, 2)]
        diff_all = [((p, a), (q, b)) for p, q in itertools.combinations(PEOPLE, 2) for a in shots for b in shots]
        pairs = [(a, b, "same") for a, b in rng.sample(same_all, min(pairs_each, len(same_all)))]
        pairs += [(a, b, "different") for a, b in rng.sample(diff_all, min(pairs_each, len(diff_all)))]
        pairs = [p if rng.random() < 0.5 else (p[1], p[0], p[2]) for p in pairs]
        rng.shuffle(pairs)
        cols, gap, head_h = 3, 12, 24
        cell_w, cell_h = 2 * TILE + 4, TILE + 18
        rows = -(-len(pairs) // cols)
        sheet = Image.new("RGB", (cols * (cell_w + gap) + gap, head_h + rows * (cell_h + gap)), BG)
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default()
        draw.text((gap, 6), f"Blind test {variant}: is each pair the SAME person or DIFFERENT? "
                            "(answers in blind_answers.json - look only after answering)", fill=TEXT, font=font)
        answers[variant] = []
        for i, (a, b, truth) in enumerate(pairs, start=1):
            x = gap + ((i - 1) % cols) * (cell_w + gap)
            y = head_h + ((i - 1) // cols) * (cell_h + gap)
            sheet.paste(tile(path(*a)), (x, y))
            sheet.paste(tile(path(*b)), (x + TILE + 4, y))
            draw.text((x, y + TILE + 3), f"pair {i}", fill=TEXT, font=font)
            answers[variant].append({"pair": i, "left": list(a), "right": list(b), "truth": truth})
            csv_rows.append(f"{variant},{i},")
        sheet.save(OUT / f"blind_{variant}.png")
    (OUT / "blind_answers.json").write_text(json.dumps(answers, indent=2) + "\n", encoding="utf-8")
    (OUT / "blind_answer_sheet.csv").write_text("\n".join(csv_rows) + "\n", encoding="utf-8")


def main() -> int:
    global OUT, RUNS, PEOPLE, steps, pairs_each
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="only validate existing outputs")
    parser.add_argument("--set", help="render_shots.py set under output/ai_experiment/sets/")
    parser.add_argument("--people", help="comma list of person ids (default: every person in the set)")
    parser.add_argument("--steps", type=int, default=STEPS, help="DPM++ steps (x strength = denoising steps)")
    parser.add_argument("--pairs", type=int, default=BLIND_PAIRS_EACH, help="blind pairs per kind (same/different)")
    args = parser.parse_args()
    steps, pairs_each = args.steps, args.pairs
    if args.set:
        rt.SHOTS_DIR = SETS_DIR / args.set / "shots"
        OUT = SETS_DIR / args.set / "b"
        RUNS = OUT / "runs.jsonl"
        PEOPLE = sorted(d.name for d in rt.SHOTS_DIR.iterdir() if (d / f"{REF_SHOT}.json").exists())
    if args.people:
        PEOPLE = args.people.split(",")
    OUT.mkdir(parents=True, exist_ok=True)
    rt.OUT, rt.hair_fix = OUT, True  # hair-masked SD inputs are saved next to the outputs
    log.info("%d people on %s, %d steps -> %s", len(PEOPLE), DEVICE, steps, OUT)
    if not args.report:
        generate()
    result = report()
    for p, entries in result["people"].items():
        for shot, m in entries.items():
            print(f"{p} {shot}  CLIP own {m['clip_to_own_reference']:.3f} closest other "
                  f"{m['clip_to_closest_other_reference']:.3f} ({m['closest_other_person']})  "
                  f"dE {m['skin_delta_e_to_own_reference']:.1f}  drift {m['shape_drift_px_vs_own_render']:.2f}px")
    print("SUMMARY", json.dumps(result["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
