"""Landmark check sheet: each raw render with its 5 landmarks drawn, next to the aligned 112x112 crop.

    venv/bin/python tools/landmark_sheet.py output/<job_id> [--ids 2,3,19] [--images 4] [--out sheet.png]

Dots: left eye (blue), right eye (cyan), nose tip (yellow), left / right mouth corner (magenta /
red), viewer's left/right. A hollow ring = point marked not visible. Needs landmarks.json
(blender/compute_landmarks.py) and aligned/ (app/postprocess.py).
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

COLORS = [(60, 120, 255), (0, 220, 230), (255, 220, 0), (240, 60, 220), (255, 60, 60)]
TILE = 192
GAP = 4
LABEL_W = 120
HEADER_H = 20
BG = (24, 24, 24)
TEXT = (230, 230, 230)


def draw_points(image: Image.Image, points: list, visible: list, scale: float) -> Image.Image:
    image = image.convert("RGB").resize((TILE, TILE), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    r = 3
    for (x, y), vis, color in zip(points, visible, COLORS):
        x, y = x * scale, y * scale
        if vis:
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=(0, 0, 0))
        else:
            draw.ellipse((x - r - 1, y - r - 1, x + r + 1, y + r + 1), outline=color, width=2)
    return image


def build(job: Path, ids: list[str] | None, n_images: int) -> Image.Image:
    rows = []
    for identity_dir in sorted(job.glob("id_*")):
        if ids and identity_dir.name not in ids:
            continue
        lm = json.loads((identity_dir / "landmarks.json").read_text(encoding="utf-8"))
        quality = json.loads((identity_dir / "alignment.json").read_text(encoding="utf-8"))
        scale = TILE / lm["resolution"]
        tiles = []
        for stem in sorted(lm["images"])[:n_images]:
            entry = lm["images"][stem]
            raw = draw_points(Image.open(identity_dir / "raw" / f"{stem}.png"), entry["points"], entry["visible"], scale)
            aligned = Image.open(identity_dir / "aligned" / f"{stem}.jpg").convert("RGB").resize((TILE, TILE))
            tiles.append((stem, raw, aligned, quality[stem]["residual_px"]))
        rows.append((identity_dir.name, tiles))
    cols = max(len(t) for _, t in rows) * 2
    sheet = Image.new("RGB", (LABEL_W + cols * (TILE + GAP), HEADER_H + len(rows) * (TILE + 14 + GAP)), BG)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((GAP, 4), f"{job.name}: landmarks on the render | aligned 112x112 crop (residual px)", fill=TEXT, font=font)
    for r, (name, tiles) in enumerate(rows):
        y = HEADER_H + r * (TILE + 14 + GAP)
        draw.text((GAP, y + 4), name, fill=TEXT, font=font)
        for c, (stem, raw, aligned, residual) in enumerate(tiles):
            x = LABEL_W + c * 2 * (TILE + GAP)
            sheet.paste(raw, (x, y))
            sheet.paste(aligned, (x + TILE + GAP, y))
            draw.text((x, y + TILE + 1), f"{stem}  residual {residual:.1f}px", fill=TEXT, font=font)
    return sheet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("job", type=Path)
    parser.add_argument("--ids", help="identity numbers, e.g. 2,3,19 (default all)")
    parser.add_argument("--images", type=int, default=4, help="images per identity")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    ids = [f"id_{int(i):05d}" for i in args.ids.split(",")] if args.ids else None
    out = args.out or args.job / "landmark_sheet.png"
    build(args.job, ids, args.images).save(out)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
