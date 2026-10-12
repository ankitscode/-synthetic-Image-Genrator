"""Shared setup for audit renders: frontal camera, neutral light, grey background, manifest."""

import json
from pathlib import Path

import bpy

import scene

NEUTRAL_CAMERA = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "distance_m": 0.55, "focal_mm": 50.0, "face_fill": 0.58}
NEUTRAL_LIGHT = {"azimuth": 25.0, "elevation": 20.0, "kelvin": 5500.0, "intensity": 1.0, "size_m": 0.8}
NEUTRAL_BACKGROUND = {"kind": "solid", "top_rgb": [0.35, 0.35, 0.35], "bottom_rgb": [0.35, 0.35, 0.35],
                      "ambient_strength": 0.25, "ambient_tint": 0.0}
AUDIT_SAMPLES = 48


def frontal_neutral(eye_mid, iod) -> bpy.types.Object:
    """Create camera + light for the current human and set the neutral audit look."""
    camera, light = scene.create_rig()
    scene.configure_render(AUDIT_SAMPLES, "CPU")
    scene.place_camera(camera, eye_mid, iod, NEUTRAL_CAMERA)
    scene.place_key_light(light, eye_mid, NEUTRAL_LIGHT)
    scene.set_background(NEUTRAL_BACKGROUND)
    return camera


def render_tile(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def write_manifest(out_dir: Path, title: str, columns: int, tiles: list[dict]) -> Path:
    """tiles: [{"file": name relative to out_dir, "label": "line1\\nline2"}] for tools/audit_sheet.py"""
    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps({"title": title, "columns": columns, "tiles": tiles}, indent=2) + "\n",
                        encoding="utf-8")
    return manifest
