"""Consistency test, step 3: measure whether approach-B outputs stay the same person.

    venv_ai/bin/python tools/ai_experiment/validate.py

No face detector or face recognition model: geometry comes from the Blender shots.

1. Shape drift  : mean optical-flow magnitude (px at 512) from each B output back to its own
                  render, inside the face mask. Compared with the flow between the main
                  person's render and each control person's render of the same shot
                  ("different person" level).
2. Colour drift : mean skin colour (CIELAB) inside the face mask; mean pairwise Delta-E76
                  across the main person's shots, vs the plain renders (lighting only) and vs
                  the control people (different person).
3. Blind test   : per blind-test variant (and plain renders "A"), 12 shuffled numbered pairs
                  (6 same person, 6 different). Answer key in a separate file.

Writes output/ai_experiment/consistency/report/{metrics.json, overview.png,
blind_<variant>.png, blind_answers.json, blind_answer_sheet.csv}.
"""

import itertools
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_variants import (BASE, BLIND_TEST_VARIANTS, CONTROL_SHOTS, CONTROLS, MAIN, MAIN_SHOTS,  # noqa: E402
                          OUT_DIR, SHOTS_DIR, SIZE, VARIANTS)

REPORT = BASE / "report"
FLOW = {"pyr_scale": 0.5, "levels": 4, "winsize": 21, "iterations": 5, "poly_n": 7, "poly_sigma": 1.5, "flags": 0}
BLIND_SEED = 2026
BLIND_PAIRS_EACH = 6
TILE = 192
TEXT = (230, 230, 230)
BG = (24, 24, 24)


def gray(path: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)


def mask(person: str, shot: str) -> np.ndarray:
    return cv2.imread(str(SHOTS_DIR / person / f"{shot}_mask.png"), cv2.IMREAD_GRAYSCALE) > 127


def flow_drift(a: Path, b: Path, m: np.ndarray) -> float:
    flow = cv2.calcOpticalFlowFarneback(gray(a), gray(b), None, **FLOW)
    return float(np.linalg.norm(flow, axis=2)[m].mean())


def skin_lab(path: Path, m: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(cv2.imread(str(path)).astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    return lab[m].mean(axis=0)


def mean_pairwise_de(labs: list[np.ndarray]) -> float:
    return float(np.mean([np.linalg.norm(a - b) for a, b in itertools.combinations(labs, 2)]))


def render_path(person: str, shot: str) -> Path:
    return SHOTS_DIR / person / f"{shot}.png"


def output_path(variant: str, person: str, shot: str) -> Path:
    return OUT_DIR / variant / person / f"{shot}.png"


def metrics() -> dict:
    result: dict = {"different_person_baseline": {}, "plain_render": {}, "variants": {}}
    # different-person level: main render vs control render of the same shot
    shape, colour = [], []
    for person in CONTROLS:
        for shot in CONTROL_SHOTS:
            m = mask(MAIN, shot)
            shape.append(flow_drift(render_path(MAIN, shot), render_path(person, shot), m))
            colour.append(float(np.linalg.norm(skin_lab(render_path(MAIN, shot), m)
                                               - skin_lab(render_path(person, shot), mask(person, shot)))))
    result["different_person_baseline"] = {"shape_drift_px": float(np.mean(shape)),
                                           "skin_delta_e": float(np.mean(colour))}
    result["plain_render"] = {"skin_delta_e_across_shots": mean_pairwise_de(
        [skin_lab(render_path(MAIN, s), mask(MAIN, s)) for s in MAIN_SHOTS])}

    for variant, cfg in VARIANTS.items():
        shots = [s for s in MAIN_SHOTS if output_path(variant, MAIN, s).exists()]
        if not shots:
            continue
        drift = {s: flow_drift(render_path(MAIN, s), output_path(variant, MAIN, s), mask(MAIN, s)) for s in shots}
        labs = [skin_lab(output_path(variant, MAIN, s), mask(MAIN, s)) for s in shots]
        entry = {**cfg, "shots_done": len(shots), "shape_drift_px_mean": float(np.mean(list(drift.values()))),
                 "shape_drift_px_max": float(max(drift.values())), "shape_drift_px": drift,
                 "skin_delta_e_across_shots": mean_pairwise_de(labs) if len(labs) > 1 else None}
        ctrl = [float(np.linalg.norm(skin_lab(output_path(variant, MAIN, s), mask(MAIN, s))
                                     - skin_lab(output_path(variant, p, s), mask(p, s))))
                for p in CONTROLS for s in CONTROL_SHOTS
                if output_path(variant, p, s).exists() and output_path(variant, MAIN, s).exists()]
        if ctrl:
            entry["skin_delta_e_vs_controls"] = float(np.mean(ctrl))
        result["variants"][variant] = entry
    return result


def tile(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB").resize((TILE, TILE), Image.Resampling.LANCZOS)


def overview() -> None:
    """Rows: plain render, then each variant; columns: the main person's 8 shots."""
    rows = [("A render", [render_path(MAIN, s) for s in MAIN_SHOTS])]
    rows += [(v, [output_path(v, MAIN, s) for s in MAIN_SHOTS]) for v in VARIANTS]
    label_w, head_h = 90, 20
    sheet = Image.new("RGB", (label_w + len(MAIN_SHOTS) * (TILE + 2), head_h + len(rows) * (TILE + 2)), BG)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for c, s in enumerate(MAIN_SHOTS):
        draw.text((label_w + c * (TILE + 2) + 4, 4), s, fill=TEXT, font=font)
    for r, (name, paths) in enumerate(rows):
        y = head_h + r * (TILE + 2)
        strength = VARIANTS.get(name, {}).get("strength")
        draw.multiline_text((4, y + 4), name + (f"\nstrength {strength}" if strength else ""), fill=TEXT, font=font)
        for c, p in enumerate(paths):
            if p.exists():
                sheet.paste(tile(p), (label_w + c * (TILE + 2), y))
    sheet.save(REPORT / "overview.png")


def blind_tests() -> None:
    rng = random.Random(BLIND_SEED)
    answers, csv_rows = {}, ["sheet,pair,your_answer (same/different)"]
    for variant in ["A"] + BLIND_TEST_VARIANTS:
        path = (lambda p, s: render_path(p, s)) if variant == "A" else (lambda p, s, v=variant: output_path(v, p, s))
        same = rng.sample(list(itertools.combinations(MAIN_SHOTS, 2)), BLIND_PAIRS_EACH)
        pairs = [((MAIN, a), (MAIN, b), "same") for a, b in same]
        for _ in range(BLIND_PAIRS_EACH):
            a = rng.choice(MAIN_SHOTS)
            b = rng.choice([s for s in CONTROL_SHOTS if s != a])
            pairs.append(((MAIN, a), (rng.choice(CONTROLS), b), "different"))
        pairs = [p if rng.random() < 0.5 else (p[1], p[0], p[2]) for p in pairs]
        rng.shuffle(pairs)
        if not all(path(*x).exists() for pair in pairs for x in pair[:2]):
            continue
        cols, gap, head_h = 3, 10, 22
        cell_w, cell_h = 2 * TILE + 4, TILE + 18
        rows = -(-len(pairs) // cols)
        sheet = Image.new("RGB", (cols * (cell_w + gap) + gap, head_h + rows * (cell_h + gap)), BG)
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default()
        draw.text((gap, 4), f"Blind test '{variant}': is each pair the SAME person? Write answers in "
                            "blind_answer_sheet.csv", fill=TEXT, font=font)
        answers[variant] = []
        for i, (a, b, truth) in enumerate(pairs, start=1):
            x = gap + ((i - 1) % cols) * (cell_w + gap)
            y = head_h + ((i - 1) // cols) * (cell_h + gap)
            sheet.paste(tile(path(*a)), (x, y))
            sheet.paste(tile(path(*b)), (x + TILE + 4, y))
            draw.text((x, y + TILE + 3), f"pair {i}", fill=TEXT, font=font)
            answers[variant].append({"pair": i, "left": list(a), "right": list(b), "truth": truth})
            csv_rows.append(f"{variant},{i},")
        sheet.save(REPORT / f"blind_{variant}.png")
    (REPORT / "blind_answers.json").write_text(json.dumps(answers, indent=2) + "\n", encoding="utf-8")
    (REPORT / "blind_answer_sheet.csv").write_text("\n".join(csv_rows) + "\n", encoding="utf-8")


def main() -> int:
    REPORT.mkdir(parents=True, exist_ok=True)
    result = metrics()
    (REPORT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    overview()
    blind_tests()
    base = result["different_person_baseline"]
    print(f"different-person level: shape {base['shape_drift_px']:.2f}px, skin dE {base['skin_delta_e']:.1f}")
    print(f"plain renders across shots (lighting only): skin dE {result['plain_render']['skin_delta_e_across_shots']:.1f}")
    for v, e in result["variants"].items():
        de = e["skin_delta_e_across_shots"]
        print(f"{v:<6} strength {e['strength']:.2f} cn={e['controlnet']!s:<5} shots {e['shots_done']} "
              f"shape {e['shape_drift_px_mean']:.2f}px (max {e['shape_drift_px_max']:.2f}) "
              f"skin dE across shots {de if de is None else round(de, 1)}"
              + (f" vs controls {e['skin_delta_e_vs_controls']:.1f}" if "skin_delta_e_vs_controls" in e else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
