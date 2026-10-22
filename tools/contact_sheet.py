"""Contact sheet for one job: one row per identity, one column per image.

    venv/bin/python tools/contact_sheet.py output/<job_id>
    venv/bin/python tools/contact_sheet.py output/<job_id> --images aligned --tile 112 --out sheet.png
    venv/bin/python tools/contact_sheet.py output/<job_id> --max-cols 4

Reads <job>/id_*/{raw|aligned}/img_*.{png,jpg}. Failed identities and missing images get a
grey placeholder tile. Default output: <job>/contact_sheet.png.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("contact_sheet")

LABEL_WIDTH = 175
HEADER_HEIGHT = 22
GAP = 2
BACKGROUND = (24, 24, 24)
PLACEHOLDER = (70, 70, 70)
TEXT = (230, 230, 230)
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("job_dir", type=Path)
    parser.add_argument("--images", choices=("raw", "aligned"), default="raw", help="image folder to use")
    parser.add_argument("--tile", type=int, default=128, help="tile size in pixels (square)")
    parser.add_argument("--out", type=Path, help="output path (default <job>/contact_sheet.png)")
    parser.add_argument("--max-cols", type=int, help="show at most this many images per identity")
    return parser.parse_args(argv)


def identity_label(identity_dir: Path) -> tuple[str, bool]:
    """Row label and whether the identity rendered OK (from params.json when present)."""
    params_file = identity_dir / "params.json"
    if not params_file.exists():
        return identity_dir.name, True
    record = json.loads(params_file.read_text(encoding="utf-8"))
    if record.get("status") != "done":
        return f"{identity_dir.name}\nFAILED", False
    ident = record["identity"]
    macros, skin = ident["macros"], ident["skin"]
    race = macros["race"]
    lines = [
        identity_dir.name,
        f"{ident['sex']} {ident['age_years']:.0f}y  gender {macros['gender']:.2f}",
        f"af {race['african']:.2f} as {race['asian']:.2f} ca {race['caucasian']:.2f}",
    ]
    if "tone" in skin:
        lines.append(f"tone {skin['tone']:.2f} {skin.get('tone_band', '')}")
    else:
        lines.append(skin.get("name", "").replace("_", " "))
    lines += [
        f"h{skin['hue_shift']:+.3f} s{skin['saturation']:.2f} v{skin['value']:.2f}",
        f"eyes {ident['eyes'].get('color', ident['eyes'].get('material'))}",
        f"hair {ident['hair']['style']} {ident['hair'].get('color_name', '')}  brow {ident.get('eyebrows', '')[-3:]}",
    ]
    beard = ident.get("beard", {})
    if beard.get("style", "none") != "none":
        lines.append(f"beard {beard['style']} {beard['length_mm']:.0f}mm {beard['color_name']}")
    return "\n".join(lines), True


def build_sheet(job_dir: Path, images: str, tile: int, max_cols: int | None = None) -> Image.Image:
    identity_dirs = sorted(p for p in job_dir.glob("id_*") if p.is_dir())
    if not identity_dirs:
        raise FileNotFoundError(f"No id_* folders in {job_dir}")
    rows = [sorted(f for f in (d / images).glob("img_*") if f.suffix.lower() in IMAGE_SUFFIXES)
            for d in identity_dirs]
    stems = sorted({f.stem for row in rows for f in row})[:max_cols]
    n_cols = max(len(stems), 1)

    width = LABEL_WIDTH + n_cols * (tile + GAP)
    height = HEADER_HEIGHT + len(identity_dirs) * (tile + GAP)
    sheet = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    draw.text((4, 4), job_dir.name, fill=TEXT, font=font)
    for col, stem in enumerate(stems):
        draw.text((LABEL_WIDTH + col * (tile + GAP) + 4, 4), stem, fill=TEXT, font=font)

    for row, (identity_dir, files) in enumerate(zip(identity_dirs, rows)):
        y = HEADER_HEIGHT + row * (tile + GAP)
        label, ok = identity_label(identity_dir)
        draw.multiline_text((4, y + 4), label, fill=TEXT, font=font, spacing=2)
        by_stem = {f.stem: f for f in files}
        for col, stem in enumerate(stems):
            x = LABEL_WIDTH + col * (tile + GAP)
            path = by_stem.get(stem)
            tile_img = None
            if ok and path is not None:
                try:
                    with Image.open(path) as img:
                        tile_img = img.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
                except OSError:  # damaged file (validate_job.py lists it as broken)
                    logging.warning("%s/%s: unreadable image", identity_dir.name, path.name)
            if tile_img is not None:
                sheet.paste(tile_img, (x, y))
            else:
                draw.rectangle((x, y, x + tile - 1, y + tile - 1), fill=PLACEHOLDER)
    return sheet


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    out = args.out or args.job_dir / "contact_sheet.png"
    sheet = build_sheet(args.job_dir, args.images, args.tile, args.max_cols)
    sheet.save(out)
    log.info("Wrote %s (%dx%d)", out, *sheet.size)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
