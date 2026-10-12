"""Print what this Blender + MPFB install provides, so other scripts use real names only.

Run headless:
    /snap/bin/blender --background --python blender/inspect_mpfb.py

Writes the same report to stdout and output/smoke/mpfb_inspect.txt.
"""

import json
import sys
from pathlib import Path

import bpy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "blender"))

sys.path.insert(0, str(PROJECT_ROOT))

from human import load_mpfb, service  # noqa: E402
from shared.constants import FACE_REGION_RULES, FACE_TARGET_EXCLUDE, FACE_TARGET_GROUPS  # noqa: E402

mpfb = load_mpfb()
AssetService = service("assetservice", "AssetService")
TargetService = service("targetservice", "TargetService")
LocationService = service("locationservice", "LocationService")

FACE_TARGETS_FILE = PROJECT_ROOT / "config" / "face_targets.json"
FACE_TARGETS_TXT = PROJECT_ROOT / "output" / "smoke" / "face_targets.txt"

OUT_FILE = PROJECT_ROOT / "output" / "smoke" / "mpfb_inspect.txt"

# (label, asset subdir, list function)
ASSET_GROUPS = [
    ("skins", "skins", AssetService.list_mhmat_assets),
    ("eyes", "eyes", AssetService.list_mhclo_assets),
    ("eye materials", "eyes", AssetService.list_mhmat_assets),
    ("eyebrows", "eyebrows", AssetService.list_mhclo_assets),
    ("eyelashes", "eyelashes", AssetService.list_mhclo_assets),
    ("hair", "hair", AssetService.list_mhclo_assets),
]


def build_report() -> list[str]:
    lines: list[str] = []
    add = lines.append

    add(f"Blender: {bpy.app.version_string}")
    add(f"MPFB version: {'.'.join(str(v) for v in mpfb.VERSION)}")
    add("")

    add("== Default macro info dict (TargetService.get_default_macro_info_dict) ==")
    add(json.dumps(TargetService.get_default_macro_info_dict(), indent=2))
    add("")

    add("== Data roots (AssetService.get_available_data_roots) ==")
    for root in AssetService.get_available_data_roots():
        add(f"  {root}")
    add("")

    pack_names = AssetService.get_pack_names()
    add("== Installed asset packs ==")
    add(f"  makehuman_system_assets installed: {AssetService.system_assets_pack_is_installed()}")
    for name in pack_names:
        add(f"  {name} ({len(AssetService.get_asset_names_in_pack(name))} assets)")
    if not pack_names:
        add("  (none)")
    add("")

    for label, subdir, list_fn in ASSET_GROUPS:
        roots = AssetService.get_asset_roots(subdir)
        files = sorted(list_fn(subdir), key=str)
        add(f"== {label} (subdir '{subdir}', {len(files)} files) ==")
        for root in roots:
            add(f"  root: {root}")
        for path in files:
            add(f"  {path.stem:<40} {path}")
        if not files:
            add("  (none found)")
        add("")

    return lines


def face_region(group: str, name: str) -> str:
    for region, rule_group, keywords in FACE_REGION_RULES:
        if group == rule_group and (not keywords or any(k in name for k in keywords)):
            return region
    raise ValueError(f"No face region for {group}/{name}")


def build_face_targets() -> dict:
    """Face target categories from MPFB's target.json.

    value < 0 loads the "negative" targets, value > 0 the "positive" ones (left and right
    together for sided categories). Single-target categories have only "positive".
    """
    with open(LocationService.get_mpfb_data("targets/target.json"), encoding="utf-8") as fh:
        table = json.load(fh)
    categories = []
    for group in FACE_TARGET_GROUPS:
        for cat in table[group]["categories"]:
            if cat["name"] in FACE_TARGET_EXCLUDE:
                continue
            opp = cat.get("opposites")
            if opp:
                sides = ("left", "right") if cat["has_left_and_right"] else ("unsided",)
                negative = [opp[f"negative-{s}"] for s in sides if opp[f"negative-{s}"]]
                positive = [opp[f"positive-{s}"] for s in sides if opp[f"positive-{s}"]]
            else:
                negative, positive = [], list(cat["targets"])
            for name in negative + positive:
                path = TargetService.target_full_path(name)
                if path is None or not Path(path).name.startswith(name + ".target"):
                    raise FileNotFoundError(f"No exact target file for {name} (got {path})")
            categories.append({"name": cat["name"], "group": group, "region": face_region(group, cat["name"]),
                               "negative": negative, "positive": positive})
    return {"mpfb_version": ".".join(str(v) for v in mpfb.VERSION), "categories": categories}


def face_targets_text(face_targets: dict) -> str:
    """Human-readable list: region -> category -> target files resolved by TargetService."""
    lines = [f"Face target categories (MPFB {face_targets['mpfb_version']}), resolved via "
             "TargetService.target_full_path", ""]
    for region, _, _ in FACE_REGION_RULES:
        cats = [c for c in face_targets["categories"] if c["region"] == region]
        lines.append(f"== {region} ({len(cats)} categories) ==")
        for cat in cats:
            lines.append(f"  {cat['name']}")
            for side, names in (("-", cat["negative"]), ("+", cat["positive"])):
                for name in names:
                    lines.append(f"      {side} {name:<36} {TargetService.target_full_path(name)}")
        lines.append("")
    lines.append(f"Excluded categories: {', '.join(FACE_TARGET_EXCLUDE)}")
    return "\n".join(lines) + "\n"


def main() -> None:
    face_targets = build_face_targets()
    FACE_TARGETS_FILE.write_text(json.dumps(face_targets, indent=2) + "\n", encoding="utf-8")
    FACE_TARGETS_TXT.parent.mkdir(parents=True, exist_ok=True)
    FACE_TARGETS_TXT.write_text(face_targets_text(face_targets), encoding="utf-8")

    report = "\n".join(build_report()) + "\n"
    report += f"== Face target categories: {len(face_targets['categories'])} written to {FACE_TARGETS_FILE} ==\n"
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(report, encoding="utf-8")
    sys.stdout.write(report)
    sys.stdout.write(f"Report written to {OUT_FILE}\n")


main()
