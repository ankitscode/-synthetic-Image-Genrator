"""Render-only fixes from the p1_chunk02 review (functions exec'd from blender/scene.py, which imports bpy)."""

import ast
from pathlib import Path

from shared import constants as C

ROOT = Path(__file__).resolve().parent.parent


def load_white_balanced_kelvin():
    tree = ast.parse((ROOT / "blender" / "scene.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "white_balanced_kelvin")
    namespace = {"C": C}
    exec(compile(ast.Module([fn], []), "scene", "exec"), namespace)
    return namespace["white_balanced_kelvin"]


def test_white_balance_pulls_warm_and_cool_light_toward_neutral() -> None:
    wb = load_white_balanced_kelvin()
    low, high = C.DEFAULT_PARAMS["light_kelvin_range"]
    assert wb(C.WHITE_BALANCE_NEUTRAL_K) == C.WHITE_BALANCE_NEUTRAL_K
    assert low < wb(low) < C.WHITE_BALANCE_NEUTRAL_K < wb(high) < high
    assert abs(wb(3000.0) - 4250.0) < 1e-6  # with strength 0.5


def test_skin_saturation_fix_only_desaturates_known_textures() -> None:
    known = {s for group in C.SKINS_BY_GROUP.values() for s in group}
    assert set(C.SKIN_TEXTURE_SATURATION_FIX) <= known
    assert all(0.5 < f < 1.0 for f in C.SKIN_TEXTURE_SATURATION_FIX.values())


def test_afro_texture_softened_only() -> None:
    assert 0.0 < C.HAIR_TEXTURE_DETAIL["afro01"] < 1.0
    assert set(C.HAIR_TEXTURE_DETAIL) == {"afro01"}
