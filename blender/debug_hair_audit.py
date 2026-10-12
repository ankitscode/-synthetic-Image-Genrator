"""Hairstyle audit: every hairstyle on the same frontal, neutral identity.

Run headless:
    /snap/bin/blender --background --python blender/debug_hair_audit.py
    venv/bin/python tools/audit_sheet.py output/smoke/hair_audit

For each style, rays from the camera go to 5 points per eye (centre + 4 around the pupil).
A style "covers" an eye when any of those points is blocked by the hair or its underlayer.
Writes output/smoke/hair_audit/{<style>.png, manifest.json, hair_audit.json}.
"""

import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "blender"))
sys.path.insert(0, str(PROJECT_ROOT))

import audit_common  # noqa: E402
import human  # noqa: E402
import scene  # noqa: E402
from shared import constants as C  # noqa: E402

OUT_DIR = PROJECT_ROOT / "output" / "smoke" / "hair_audit"
HAIR_RGB = list(C.HAIR_COLORS["dark_brown"])
EYE_PROBE_OFFSETS_M = [(0.0, 0.0), (0.005, 0.0), (-0.005, 0.0), (0.0, 0.004), (0.0, -0.004)]  # (x, z)

NEUTRAL_IDENTITY = {
    "macros": dict(C.MPFB_MACRO_DEFAULTS),  # gender 0.5, all macros neutral
    "face_targets": [],
    "skin": {"light": "young_caucasian_male", "dark": "young_african_male", "tone": 0.5,
             "hue_shift": 0.0, "saturation": 1.0, "value": 1.0,
             "roughness": 0.6, "specular": 0.35, "clearcoat": 0.02},
    "eyes": {"mesh": C.EYES_MESH, "color": "brown", "material": "brown", "iris_darken": None},
    "eyebrows": "eyebrow001",
    "eyebrow_rgb": HAIR_RGB,
    "eyelashes": "eyelashes01",
    "hair": {"style": C.HAIR_NONE, "rgb": HAIR_RGB},
}


def eye_probes(eyes) -> dict[str, Vector]:
    left, right = human.eye_centres(eyes)
    probes = {}
    for side, centre in (("subject_left", left), ("subject_right", right)):
        for i, (dx, dz) in enumerate(EYE_PROBE_OFFSETS_M):
            # probes sit on the front of the eyeball (toward the camera at -Y)
            probes[f"{side}_{i}"] = centre + Vector((dx, -0.005, dz))
    return probes


def main() -> None:
    human.load_mpfb()
    human.clear_scene()
    basemesh, eyes = human.build_human(NEUTRAL_IDENTITY, {})
    eye_mid, iod = human.eye_geometry(eyes)
    camera = audit_common.frontal_neutral(eye_mid, iod)
    see_through = {o.name for o in bpy.data.objects if o.name.endswith((".eyebrow001", ".eyelashes01"))}
    probes = eye_probes(eyes)

    styles = [C.HAIR_NONE] + sorted({s for w in C.HAIR_STYLE_WEIGHTS.values() for s in w} - {C.HAIR_NONE})
    tiles, results = [], {}
    for style in styles:
        hair_objs = [] if style == C.HAIR_NONE else human.add_hair(basemesh, style, HAIR_RGB)
        hair_names = {o.name for o in hair_objs}
        vis = scene.eye_visibility(camera, eyes, probes, see_through)
        blocked = {k: v["blocked_by"] for k, v in vis.items() if not v["visible"]}
        by_hair = {k: b for k, b in blocked.items() if b in hair_names}
        covered = sorted({k.rsplit("_", 1)[0] for k in by_hair})
        results[style] = {"covers_eye": bool(covered), "covered_eyes": covered,
                          "blocked_probes": blocked, "probes": len(probes)}
        audit_common.render_tile(OUT_DIR / f"{style}.png")
        verdict = "COVERS " + "+".join(c.replace("subject_", "") for c in covered) if covered else "eyes clear"
        excluded = " (excluded)" if style in C.HAIR_STYLE_EXCLUDE else ""
        tiles.append({"file": f"{style}.png", "label": f"{style}{excluded}\n{verdict}"})
        for obj in hair_objs:
            bpy.data.objects.remove(obj, do_unlink=True)

    (OUT_DIR / "hair_audit.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    audit_common.write_manifest(OUT_DIR, "Hairstyle audit (frontal, neutral identity)", 6, tiles)
    for style, res in results.items():
        print(f"HAIR_AUDIT {style:<12} covers_eye={res['covers_eye']} {res['covered_eyes']}")


main()
