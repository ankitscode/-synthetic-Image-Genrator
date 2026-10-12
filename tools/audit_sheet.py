"""Labelled grid from an audit folder's manifest.json (written by blender/debug_*_audit.py).

    venv/bin/python tools/audit_sheet.py output/smoke/hair_audit
    venv/bin/python tools/audit_sheet.py output/smoke/skin_audit --tile 192

manifest.json: {"title": str, "columns": int, "tiles": [{"file": str, "label": str}, ...]}
Default output: <folder>.png next to the folder.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("audit_sheet")

HEADER_HEIGHT = 22
LINE_HEIGHT = 12
GAP = 4
BACKGROUND = (24, 24, 24)
TEXT = (230, 230, 230)


def build_sheet(folder: Path, tile: int) -> Image.Image:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    tiles = manifest["tiles"]
    columns = max(1, min(manifest["columns"], len(tiles)))
    rows = -(-len(tiles) // columns)
    label_lines = max(t["label"].count("\n") + 1 for t in tiles)
    cell_h = tile + label_lines * LINE_HEIGHT + GAP

    sheet = Image.new("RGB", (columns * (tile + GAP) + GAP, HEADER_HEIGHT + rows * cell_h), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((GAP, 4), manifest["title"], fill=TEXT, font=font)
    for i, entry in enumerate(tiles):
        x = GAP + (i % columns) * (tile + GAP)
        y = HEADER_HEIGHT + (i // columns) * cell_h
        with Image.open(folder / entry["file"]) as img:
            sheet.paste(img.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS), (x, y))
        draw.multiline_text((x, y + tile + 1), entry["label"], fill=TEXT, font=font, spacing=1)
    return sheet


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--tile", type=int, default=160)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    out = args.out or args.folder.with_suffix(".png")
    sheet = build_sheet(args.folder, args.tile)
    sheet.save(out)
    log.info("Wrote %s (%dx%d)", out, *sheet.size)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
