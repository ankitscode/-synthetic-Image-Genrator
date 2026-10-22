"""Identity distinctness rules. Standard library only. Face shape is measured in Blender
(blender/select_identities.py, blender/face_shape_distance.py); these are the rules.

Near-duplicate (rejected): same sex and face shape < NEAR_DUPLICATE_SHAPE_IOD with skin tone within
NEAR_DUPLICATE_TONE_DIFF, or face shape < NEAR_DUPLICATE_ANY_TONE_IOD whatever the tone: practically the
same face, which would be label noise (or teach the model that skin tone alone separates people).
Look-alike (reported only): face shape < LOOKALIKE_SHAPE_IOD and >= LOOKALIKE_MIN_SHARED shared traits
(sex, skin tone, hairstyle, hair colour family, and beard style between two men). Face shape is measured in Blender
(blender/select_identities.py, blender/face_shape_distance.py); this module is the trait part.
"""

from shared import constants as C


def shared_traits(a: dict, b: dict) -> list[str]:
    """a, b: identities from shared.sampling.sample_identity (or params.json "identity")."""
    shared = []
    if a["sex"] == b["sex"]:
        shared.append("sex")
    if abs(a["skin"]["tone"] - b["skin"]["tone"]) < C.LOOKALIKE_TONE_DIFF:
        shared.append("skin tone")
    if a["hair"]["style"] == b["hair"]["style"]:
        shared.append("hairstyle")
    if C.HAIR_COLOR_FAMILY[a["hair"]["color_name"]] == C.HAIR_COLOR_FAMILY[b["hair"]["color_name"]]:
        shared.append("hair colour")
    if a["sex"] == b["sex"] == "male" and a["beard"]["style"] == b["beard"]["style"]:
        shared.append("beard")  # only between men: two women always share "no beard"
    return shared


def too_similar(shape_iod: float, shared: list[str]) -> bool:
    return shape_iod < C.LOOKALIKE_SHAPE_IOD and len(shared) >= C.LOOKALIKE_MIN_SHARED


def near_duplicate(shape_iod: float, a: dict, b: dict) -> bool:
    if a["sex"] != b["sex"]:
        return False
    if shape_iod < C.NEAR_DUPLICATE_ANY_TONE_IOD:
        return True
    return (shape_iod < C.NEAR_DUPLICATE_SHAPE_IOD
            and abs(a["skin"]["tone"] - b["skin"]["tone"]) < C.NEAR_DUPLICATE_TONE_DIFF)


def excluded(identity: dict) -> str | None:
    """Reason an identity is never selected (None = allowed)."""
    if C.SELECTION_EXCLUDE_BALD_WOMEN and identity["sex"] == "female" and identity["hair"]["style"] == C.HAIR_NONE:
        return "bald woman"
    return None
