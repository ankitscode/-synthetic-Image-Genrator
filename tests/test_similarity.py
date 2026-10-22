"""Identity distinctness rules (shared/similarity.py)."""

import json
from pathlib import Path

from shared import constants as C
from shared import sampling, similarity

CATEGORIES = {c["name"]: c["region"] for c in json.loads(
    (Path(__file__).parent.parent / "config" / "face_targets.json").read_text())["categories"]}
PARAMS = sampling.resolve_params()


def person(sex: str, tone: float, hair: str = "short01", color: str = "black", beard: str = "none") -> dict:
    return {"sex": sex, "skin": {"tone": tone}, "hair": {"style": hair, "color_name": color}, "beard": {"style": beard}}


def test_two_women_do_not_share_beard() -> None:
    assert "beard" not in similarity.shared_traits(person("female", 0.5), person("female", 0.5))
    assert "beard" in similarity.shared_traits(person("male", 0.5), person("male", 0.5))


def test_hair_colour_families() -> None:
    assert "hair colour" in similarity.shared_traits(person("male", 0.1, color="black"), person("male", 0.9, color="dark_brown"))
    assert "hair colour" not in similarity.shared_traits(person("male", 0.1, color="black"), person("male", 0.9, color="blonde"))


def test_near_duplicate_needs_close_shape_same_sex_and_tone() -> None:
    a, b = person("male", 0.50), person("male", 0.55)
    close = (C.NEAR_DUPLICATE_ANY_TONE_IOD + C.NEAR_DUPLICATE_SHAPE_IOD) / 2  # tone decides in this range
    assert similarity.near_duplicate(close, a, b)
    assert not similarity.near_duplicate(C.NEAR_DUPLICATE_SHAPE_IOD * 1.5, a, b)
    assert not similarity.near_duplicate(close, a, person("female", 0.55))
    assert not similarity.near_duplicate(close, a, person("male", 0.50 + 2 * C.NEAR_DUPLICATE_TONE_DIFF))


def test_lookalike_counts_traits() -> None:
    shared = similarity.shared_traits(person("male", 0.5, "short04", "blonde"), person("male", 0.55, "short04", "blonde"))
    assert similarity.too_similar(C.LOOKALIKE_SHAPE_IOD / 2, shared)
    assert not similarity.too_similar(C.LOOKALIKE_SHAPE_IOD * 2, shared)


def test_hair_colour_family_covers_every_colour() -> None:
    assert set(C.HAIR_COLOR_FAMILY) == set(C.HAIR_COLORS)


def test_works_on_sampled_identities() -> None:
    a, b = (sampling.sample_identity(7, i, PARAMS, CATEGORIES) for i in (1, 2))
    assert isinstance(similarity.shared_traits(a, b), list)


def test_near_duplicate_very_close_shape_ignores_skin_tone() -> None:
    a, b = person("male", 0.20), person("male", 0.80)
    assert similarity.near_duplicate(C.NEAR_DUPLICATE_ANY_TONE_IOD * 0.8, a, b)
    assert not similarity.near_duplicate(C.NEAR_DUPLICATE_ANY_TONE_IOD * 1.1, a, b)
    assert not similarity.near_duplicate(C.NEAR_DUPLICATE_ANY_TONE_IOD * 0.8, a, person("female", 0.20))
