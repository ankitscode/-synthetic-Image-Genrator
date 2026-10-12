"""Skin tone audit: 12 identities from the default parameters, frontal, neutral light.

Run headless:
    /snap/bin/blender --background --python blender/debug_skin_audit.py
    venv/bin/python tools/audit_sheet.py output/smoke/skin_audit

Identities come from shared.sampling with AUDIT_SEED (same code path as a real job) and are
shown sorted by tone, light to dark. Writes output/smoke/skin_audit/{id_XX.png, manifest.json}.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "blender"))
sys.path.insert(0, str(PROJECT_ROOT))

import audit_common  # noqa: E402
import human  # noqa: E402
from shared import sampling  # noqa: E402

OUT_DIR = PROJECT_ROOT / "output" / "smoke" / "skin_audit"
FACE_TARGETS_FILE = PROJECT_ROOT / "config" / "face_targets.json"
AUDIT_SEED = 777
N_IDENTITIES = 12


def main() -> None:
    categories = {c["name"]: c for c in json.loads(FACE_TARGETS_FILE.read_text())["categories"]}
    regions = {n: c["region"] for n, c in categories.items()}
    params = sampling.resolve_params()
    identities = [sampling.sample_identity(AUDIT_SEED, i, params, regions) for i in range(1, N_IDENTITIES + 1)]
    identities.sort(key=lambda ident: ident["skin"]["tone"])

    human.load_mpfb()
    tiles = []
    for ident in identities:
        human.clear_scene()
        _, eyes = human.build_human(ident, categories)
        eye_mid, iod = human.eye_geometry(eyes)
        audit_common.frontal_neutral(eye_mid, iod)
        name = f"id_{ident['identity_index']:02d}.png"
        audit_common.render_tile(OUT_DIR / name)
        skin, race = ident["skin"], ident["macros"]["race"]
        tiles.append({"file": name, "label": "\n".join([
            f"tone {skin['tone']:.2f} {skin['tone_band']}",
            f"{skin['light'].replace('_', ' ')}",
            f"+ {skin['dark'].replace('_', ' ')}",
            f"af {race['african']:.2f} as {race['asian']:.2f} ca {race['caucasian']:.2f}",
            f"h{skin['hue_shift']:+.3f} s{skin['saturation']:.2f} v{skin['value']:.2f}",
        ])})
        print(f"SKIN_AUDIT {name} tone={skin['tone']:.2f} {skin['tone_band']}")
    audit_common.write_manifest(OUT_DIR, f"Skin tone audit: {N_IDENTITIES} identities, seed {AUDIT_SEED}, "
                                         "sorted light -> dark, frontal, 5500 K", 6, tiles)


main()
