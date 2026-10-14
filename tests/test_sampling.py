import json
import random
from collections import Counter
from pathlib import Path

import pytest

from shared import constants as C
from shared import sampling

FACE_TARGETS = json.loads((Path(__file__).parent.parent / "config" / "face_targets.json").read_text())
CATEGORIES = {c["name"]: c["region"] for c in FACE_TARGETS["categories"]}
REGIONS = {region for region, _, _ in C.FACE_REGION_RULES}
PARAMS = sampling.resolve_params()
N = 500


def identities(seed: int = 7, n: int = N) -> list[dict]:
    return [sampling.sample_identity(seed, i, PARAMS, CATEGORIES) for i in range(1, n + 1)]


def images(seed: int = 7, n: int = N) -> list[dict]:
    return [sampling.sample_image(seed, 1, i, PARAMS) for i in range(1, n + 1)]


def in_range(value: float, bounds: list[float]) -> bool:
    return bounds[0] <= value <= bounds[1]


# --- determinism -----------------------------------------------------------
def test_same_seed_gives_identical_params() -> None:
    assert identities(seed=42, n=20) == identities(seed=42, n=20)
    assert images(seed=42, n=20) == images(seed=42, n=20)


def test_different_indices_differ() -> None:
    a = sampling.sample_identity(42, 1, PARAMS, CATEGORIES)
    b = sampling.sample_identity(42, 2, PARAMS, CATEGORIES)
    c = sampling.sample_identity(43, 1, PARAMS, CATEGORIES)
    assert a != b and a != c


def test_derive_seed_is_stable() -> None:
    # Fixed value: guards against accidental changes to the seed derivation.
    assert sampling.derive_seed(1, 2, 3) == 12695445271409270800
    assert sampling.derive_seed(1, 2) != sampling.derive_seed(2, 1)


def test_image_params_do_not_depend_on_identity_sampling_order() -> None:
    first = sampling.sample_image(5, 3, 2, PARAMS)
    identities(seed=5, n=3)
    assert sampling.sample_image(5, 3, 2, PARAMS) == first


def test_image_resample_attempts_are_deterministic_and_distinct() -> None:
    base = sampling.sample_image(5, 3, 2, PARAMS)
    retry = sampling.sample_image(5, 3, 2, PARAMS, attempt=1)
    assert base["seed"] == sampling.derive_seed(5, 3, 2)  # attempt 0 keeps the original seed
    assert retry == sampling.sample_image(5, 3, 2, PARAMS, attempt=1)
    assert retry["camera"] != base["camera"]


# --- macros -------------------------------------------------------------------
def test_macro_keys_match_mpfb_exactly() -> None:
    for ident in identities(n=20):
        assert set(ident["macros"]) == set(C.MPFB_MACRO_DEFAULTS)
        assert set(ident["macros"]["race"]) == set(C.MPFB_MACRO_DEFAULTS["race"])


def test_race_weights_sum_to_one() -> None:
    for ident in identities():
        race = ident["macros"]["race"]
        assert sum(race.values()) == pytest.approx(1.0)
        assert all(0.0 <= v <= 1.0 for v in race.values())


def test_macro_values_in_range() -> None:
    for ident in identities():
        m = ident["macros"]
        assert 0.0 <= m["gender"] <= 1.0
        assert in_range(ident["age_years"], PARAMS["age_years_range"])
        assert 0.0 <= m["age"] <= 1.0
        assert in_range(m["weight"], [0.1, 0.9])
        assert in_range(m["muscle"], [0.2, 0.8])
        assert in_range(m["height"], PARAMS["height_range"])
        assert in_range(m["proportions"], [0.2, 0.8])


def test_gender_is_bimodal() -> None:
    genders = [i["macros"]["gender"] for i in identities(n=4000)]
    female = sum(g < 0.25 for g in genders) / len(genders)
    male = sum(g > 0.75 for g in genders) / len(genders)
    between = 1.0 - female - male
    assert 0.07 <= between <= 0.13
    assert 0.40 <= female <= 0.50 and 0.40 <= male <= 0.50


@pytest.mark.parametrize("years, macro", [(1, 0.0), (11, 0.1875), (25, 0.5), (90, 1.0), (57.5, 0.75)])
def test_age_mapping(years: float, macro: float) -> None:
    assert sampling.age_years_to_macro(years) == pytest.approx(macro)


# --- face targets -------------------------------------------------------------
def test_face_targets_count_strength_and_regions() -> None:
    lo, hi = PARAMS["face_targets_count_range"]
    for ident in identities():
        targets = ident["face_targets"]
        assert lo <= len(targets) <= hi
        assert len({t["category"] for t in targets}) == len(targets)
        cap = PARAMS["face_target_strength"]
        assert all(t["category"] in CATEGORIES and -cap <= t["value"] <= cap for t in targets)
        assert {t["region"] for t in targets} == REGIONS  # at least one per region


def test_every_region_has_categories() -> None:
    assert set(CATEGORIES.values()) == REGIONS


def test_face_targets_exclude_list_respected() -> None:
    assert not set(CATEGORIES) & set(C.FACE_TARGET_EXCLUDE)


# --- skin -----------------------------------------------------------------------
def test_skin_bases_match_gender_age_and_race() -> None:
    for ident in identities():
        skin, race = ident["skin"], ident["macros"]["race"]
        bucket = sampling.age_bucket(ident["age_years"])
        light_race = max(C.SKIN_LIGHT_RACES, key=lambda r: race[r])
        assert skin["light"] in C.SKINS_BY_GROUP[f"{bucket}_{light_race}_{ident['sex']}"]
        assert skin["dark"] in C.SKINS_BY_GROUP[f"{bucket}_african_{ident['sex']}"]
        for name in (skin["light"], skin["dark"]):
            assert "toon" not in name and "special_suit" not in name
        assert 0.0 <= skin["tone"] <= 1.0
        assert skin["tone_band"] == sampling.skin_tone_band(skin["tone"])
        assert in_range(skin["hue_shift"], PARAMS["skin_hue_shift_range"])
        assert in_range(skin["saturation"], PARAMS["skin_saturation_range"])
        assert in_range(skin["value"], PARAMS["skin_value_range"])


def test_skin_tones_mostly_medium_then_dark_then_light() -> None:
    bands = Counter(i["skin"]["tone_band"] for i in identities(n=4000))
    assert bands["medium"] > bands["dark"] > bands["light"] > 0
    assert bands["medium"] / 4000 > 0.45


def test_darker_race_weight_gives_darker_tone() -> None:
    ids = identities(n=2000)
    african = [i["skin"]["tone"] for i in ids if sampling.dominant_race(i["macros"]) == "african"]
    caucasian = [i["skin"]["tone"] for i in ids if sampling.dominant_race(i["macros"]) == "caucasian"]
    assert sum(african) / len(african) > sum(caucasian) / len(caucasian) + 0.2


@pytest.mark.parametrize("years, bucket", [(18, "young"), (34.9, "young"), (35, "middleage"), (55, "middleage"), (55.1, "old")])
def test_age_buckets(years: float, bucket: str) -> None:
    assert sampling.age_bucket(years) == bucket


def test_skin_shading_in_range() -> None:
    for ident in identities():
        skin = ident["skin"]
        assert in_range(skin["roughness"], PARAMS["skin_roughness_range"])
        assert in_range(skin["specular"], PARAMS["skin_specular_range"])
        assert in_range(skin["clearcoat"], PARAMS["skin_clearcoat_range"])


def test_skin_is_identity_level_only() -> None:
    """Skin is sampled once per identity; per-image params must not carry any skin values."""
    for img in images(n=50):
        assert "skin" not in img
        assert not any("skin" in key for group in img.values() if isinstance(group, dict) for key in group)


# --- eyes, brows, lashes, hair ---------------------------------------------------
def test_eye_colour_mostly_brown() -> None:
    ids = identities(n=2000)
    browns = {"brown", "darkbrown"}
    assert all(i["eyes"]["mesh"] == "high-poly" for i in ids)
    assert all(i["eyes"]["color"] != "brownlight" for i in ids)  # renders amber, dropped
    for race, minimum in (("african", 0.93), ("asian", 0.90)):
        group = [i for i in ids if sampling.dominant_race(i["macros"]) == race]
        share = sum(i["eyes"]["color"] in browns for i in group) / len(group)
        assert share >= minimum, (race, share)


def test_eye_material_matches_colour() -> None:
    for ident in identities(n=200):
        eyes = ident["eyes"]
        assert (eyes["material"], eyes["iris_darken"]) == C.EYE_MATERIALS[eyes["color"]]


def test_eyebrows_dealt_evenly() -> None:
    ids = identities(n=12 * 5)
    for block in range(5):
        styles = [i["eyebrows"] for i in ids[block * 12:(block + 1) * 12]]
        assert sorted(styles) == sorted(C.EYEBROWS)  # every style exactly once per 12


def test_eyebrow_colour_varies_with_darkness() -> None:
    for ident in identities():
        d = ident["eyebrow_darkness"]
        assert in_range(d, PARAMS["eyebrow_darkness_range"])
        assert ident["eyebrow_rgb"] == pytest.approx([c * d for c in ident["hair"]["rgb"]])
    assert len({round(i["eyebrow_darkness"], 3) for i in identities(n=50)}) > 40


def test_hair_excludes_eye_covering_styles() -> None:
    styles = Counter(i["hair"]["style"] for i in identities(n=2000))
    assert not set(styles) & set(C.HAIR_STYLE_EXCLUDE)
    assert C.HAIR_NONE in styles  # some bald identities
    for sex in ("male", "female"):
        assert sum(sampling.hair_style_weights(sex).values()) == pytest.approx(1.0)


def test_brows_lashes_hair_from_known_names() -> None:
    for ident in identities():
        assert ident["eyebrows"] in C.EYEBROWS
        assert ident["eyelashes"] in C.EYELASHES
        hair = ident["hair"]
        assert hair["style"] in sampling.hair_style_weights(ident["sex"])
        assert hair["color_name"] in C.HAIR_COLORS


def test_hair_colour_mostly_dark() -> None:
    ids = [i for i in identities(n=2000) if sampling.age_bucket(i["age_years"]) != "old"]
    counts = Counter(i["hair"]["color_name"] for i in ids)
    assert (counts["black"] + counts["dark_brown"]) / len(ids) > 0.6


def test_weight_tables_sum_to_one() -> None:
    tables = [*C.EYE_COLOR_WEIGHTS_BY_RACE.values(), C.HAIR_COLOR_WEIGHTS, C.HAIR_COLOR_WEIGHTS_GREYING,
              C.HAIR_COLOR_WEIGHTS_OLD, *C.HAIR_STYLE_WEIGHTS.values(), *C.BEARD_STYLE_WEIGHTS.values()]
    for table in tables:
        assert sum(table.values()) == pytest.approx(1.0)


def test_face_targets_mostly_moderate() -> None:
    values = [abs(t["value"]) for i in identities() for t in i["face_targets"]]
    assert sum(v > 0.6 for v in values) / len(values) < 0.15  # was ~40% with uniform +-1.0
    assert sum(v < 0.3 for v in values) / len(values) > 0.5


def test_no_grey_or_white_hair_before_greying_age() -> None:
    for ident in identities(n=2000):
        if ident["age_years"] < C.HAIR_GREYING_MIN_AGE:
            assert ident["hair"]["color_name"] not in ("grey", "white")
            if ident["beard"]["style"] != C.BEARD_NONE:
                assert ident["beard"]["color_name"] not in ("grey", "white")


def test_beards_men_only_and_common() -> None:
    ids = identities(n=2000)
    assert all(i["beard"]["style"] == C.BEARD_NONE for i in ids if i["sex"] == "female")
    men = [i for i in ids if i["sex"] == "male"]
    bearded = [i for i in men if i["beard"]["style"] != C.BEARD_NONE]
    assert 0.55 < len(bearded) / len(men) < 0.8
    assert {i["beard"]["style"] for i in bearded} == set(C.BEARD_STYLES)
    for i in bearded:
        b = i["beard"]
        assert in_range(b["length_mm"], C.BEARD_STYLES[b["style"]]["length_mm"])
        assert in_range(b["density"], C.BEARD_DENSITY_RANGE)


def test_beard_stream_does_not_change_other_identity_values() -> None:
    off = sampling.resolve_params({"beard_enabled": False})
    for i in range(1, 50):
        a = sampling.sample_identity(7, i, PARAMS, CATEGORIES)
        b = sampling.sample_identity(7, i, off, CATEGORIES)
        assert {k: v for k, v in a.items() if k != "beard"} == {k: v for k, v in b.items() if k != "beard"}


def test_environment_mix_and_exposure_jitter() -> None:
    imgs = images(n=2000)
    hdri = [i["environment"] for i in imgs if i["environment"]["kind"] == "hdri"]
    assert abs(len(hdri) / len(imgs) - PARAMS["hdri_prob"]) < 0.05
    assert {e["hdri"] for e in hdri} == set(PARAMS["hdri_ids"])
    assert all(in_range(e["fstop"], PARAMS["hdri_fstop_range"]) for e in hdri)
    limit = PARAMS["exposure_jitter_ev_limit"]
    assert all(abs(i["exposure_jitter_ev"]) <= limit for i in imgs)


def test_hdri_ids_exist_in_manifest() -> None:
    manifest = json.loads((Path(__file__).parent.parent / "assets" / "hdri" / "hdris.json").read_text())
    assert set(PARAMS["hdri_ids"]) <= {h["id"] for h in manifest}


# --- per image -------------------------------------------------------------------
def test_pose_distribution() -> None:
    cams = [img["camera"] for img in images(n=4000)]
    for cam in cams:
        assert abs(cam["yaw"]) <= 40.0
        assert abs(cam["pitch"]) <= 20.0
        assert abs(cam["roll"]) <= 10.0
        assert in_range(cam["distance_m"], PARAMS["camera_distance_range"])
        assert in_range(cam["focal_mm"], PARAMS["focal_mm_range"])
        assert in_range(cam["face_fill"], PARAMS["face_fill_range"])
    yaw = [abs(c["yaw"]) for c in cams]
    share = lambda lo, hi: sum(lo <= y < hi for y in yaw) / len(yaw)  # noqa: E731
    assert share(0, 15) == pytest.approx(0.6, abs=0.03)
    assert share(15, 30) == pytest.approx(0.3, abs=0.03)
    assert share(30, 40.001) == pytest.approx(0.1, abs=0.02)
    assert 0.4 < sum(c["yaw"] > 0 for c in cams) / len(cams) < 0.6  # both directions


def test_light_and_background_in_range() -> None:
    for img in images():
        light = img["light"]
        assert abs(light["azimuth"]) <= PARAMS["light_azimuth_limit"]
        assert in_range(light["elevation"], PARAMS["light_elevation_range"])
        assert in_range(light["kelvin"], PARAMS["light_kelvin_range"])
        assert in_range(light["intensity"], PARAMS["light_intensity_range"])
        bg = img["background"]
        assert bg["kind"] in ("solid", "gradient")
        assert all(0.0 <= c <= 1.0 for c in bg["top_rgb"] + bg["bottom_rgb"])
        assert in_range(bg["ambient_strength"], PARAMS["ambient_strength_range"])
        if bg["kind"] == "solid":
            assert bg["top_rgb"] == bg["bottom_rgb"]


# --- params --------------------------------------------------------------------
def test_unknown_override_rejected() -> None:
    with pytest.raises(KeyError):
        sampling.resolve_params({"not_a_param": 1})


def test_override_applies() -> None:
    params = sampling.resolve_params({"race_mix": {"african": 1.0, "asian": 0.0, "caucasian": 0.0},
                                      "race_concentration": 50.0})
    rng = random.Random(0)
    races = [sampling.sample_race(rng, params) for _ in range(50)]
    assert all(max(r, key=r.get) == "african" for r in races)


def test_shared_is_stdlib_only() -> None:
    import ast

    stdlib = {"colorsys", "copy", "hashlib", "math", "random", "statistics", "shared", "__future__"}
    for path in (Path(__file__).parent.parent / "shared").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            assert set(names) <= stdlib, f"{path.name} imports {names}"
