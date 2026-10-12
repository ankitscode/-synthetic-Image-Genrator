"""Seeded parameter sampling. Standard library only (imported by app and Blender).

Every function takes an explicit random.Random. Seeds come from derive_seed():
identity = (job_seed, identity_index), image = (job_seed, identity_index, image_index).
"""

import colorsys
import copy
import hashlib
import math
import random

from shared import constants as C


def derive_seed(*parts: int) -> int:
    """Stable 64-bit seed from integer parts (independent of PYTHONHASHSEED)."""
    key = "synthface:" + ":".join(str(int(p)) for p in parts)
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def resolve_params(overrides: dict | None = None) -> dict:
    """Defaults from constants.DEFAULT_PARAMS with job-level overrides applied."""
    params = copy.deepcopy(C.DEFAULT_PARAMS)
    for key, value in (overrides or {}).items():
        if key not in params:
            raise KeyError(f"Unknown sampling parameter: {key}")
        params[key] = value
    return params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _uniform(rng: random.Random, bounds: list[float]) -> float:
    return rng.uniform(bounds[0], bounds[1])


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    names = list(weights)
    return rng.choices(names, weights=[weights[n] for n in names], k=1)[0]


def _near_frontal(rng: random.Random, sigma: float, limit: float) -> float:
    """Gaussian around 0, resampled until inside [-limit, limit] (no pile-up at the edges)."""
    while True:
        value = rng.gauss(0.0, sigma)
        if abs(value) <= limit:
            return value


def age_years_to_macro(age_years: float) -> float:
    anchors = C.AGE_YEARS_TO_MACRO
    if age_years <= anchors[0][0]:
        return anchors[0][1]
    for (y0, m0), (y1, m1) in zip(anchors, anchors[1:]):
        if age_years <= y1:
            return m0 + (m1 - m0) * (age_years - y0) / (y1 - y0)
    return anchors[-1][1]


def age_bucket(age_years: float) -> str:
    if age_years < C.AGE_BUCKET_YOUNG_MAX:
        return "young"
    if age_years <= C.AGE_BUCKET_MIDDLEAGE_MAX:
        return "middleage"
    return "old"


def sex_of(gender: float) -> str:
    return "male" if gender >= C.MALE_GENDER_THRESHOLD else "female"


# ---------------------------------------------------------------------------
# Identity parameter groups
# ---------------------------------------------------------------------------
def sample_race(rng: random.Random, params: dict) -> dict[str, float]:
    """Dirichlet sample around the job-level race mix, normalized to sum to 1."""
    mix = params["race_mix"]
    concentration = params["race_concentration"]
    raw = {race: rng.gammavariate(max(mix[race] * concentration, 1e-3), 1.0) for race in C.RACES}
    total = sum(raw.values())
    return {race: raw[race] / total for race in C.RACES}


def sample_macros(rng: random.Random, params: dict) -> tuple[dict, float]:
    """Return (MPFB macro dict with the exact default keys, age in years).

    Gender is bimodal: gender_between_prob in the middle range, the rest split female/male.
    """
    if rng.random() < params["gender_between_prob"]:
        gender = _uniform(rng, params["gender_between_range"])
    elif rng.random() < params["male_prob"]:
        gender = _uniform(rng, params["gender_male_range"])
    else:
        gender = _uniform(rng, params["gender_female_range"])

    if rng.random() < params["age_years_core_prob"]:
        age_years = _uniform(rng, params["age_years_core_range"])
    else:
        age_years = _uniform(rng, params["age_years_range"])

    macros = copy.deepcopy(C.MPFB_MACRO_DEFAULTS)
    macros["gender"] = gender
    macros["age"] = age_years_to_macro(age_years)
    macros["muscle"] = _uniform(rng, params["muscle_range"])
    macros["weight"] = _uniform(rng, params["weight_range"])
    macros["proportions"] = _uniform(rng, params["proportions_range"])
    macros["height"] = _uniform(rng, params["height_range"])
    macros["race"] = sample_race(rng, params)
    return macros, age_years


def sample_face_targets(rng: random.Random, params: dict, categories: dict[str, str]) -> list[dict]:
    """Pick face target categories ({name: region}) with strength in [-s, s] each.

    One category from every region first, then random others up to the sampled count.
    """
    lo, hi = params["face_targets_count_range"]
    regions = sorted(set(categories.values()))
    count = min(max(rng.randint(lo, hi), len(regions)), len(categories))
    chosen = [rng.choice(sorted(n for n, r in categories.items() if r == region)) for region in regions]
    rest = sorted(set(categories) - set(chosen))
    chosen += rng.sample(rest, count - len(chosen))
    strength = params["face_target_strength"]
    return [{"category": name, "region": categories[name], "value": rng.uniform(-strength, strength)}
            for name in chosen]


def dominant_race(macros: dict) -> str:
    return max(C.RACES, key=lambda r: macros["race"][r])


def skin_tone_band(tone: float) -> str:
    if tone < C.SKIN_TONE_BANDS["light_max"]:
        return "light"
    if tone < C.SKIN_TONE_BANDS["medium_max"]:
        return "medium"
    return "dark"


def sample_skin(rng: random.Random, params: dict, macros: dict, age_years: float) -> dict:
    """Skin textures + tone + HSV shift + shading. Sampled once per identity; every image of
    the identity uses exactly these values.

    Two base skins of the same age bucket and sex are blended by `tone`: the light one
    (asian or caucasian, whichever race weight is higher) and the african one.
    """
    race = macros["race"]
    sex = sex_of(macros["gender"])
    bucket = age_bucket(age_years)
    light_race = max(C.SKIN_LIGHT_RACES, key=lambda r: race[r])
    light_group = f"{bucket}_{light_race}_{sex}"
    dark_group = f"{bucket}_{C.SKIN_DARK_RACE}_{sex}"
    anchors = params["skin_tone_anchor"]
    tone = sum(race[r] * anchors[r] for r in C.RACES) + rng.gauss(0.0, params["skin_tone_sigma"])
    tone = min(1.0, max(0.0, tone))
    return {
        "light": rng.choice(C.SKINS_BY_GROUP[light_group]),
        "dark": rng.choice(C.SKINS_BY_GROUP[dark_group]),
        "tone": tone,
        "tone_band": skin_tone_band(tone),
        "hue_shift": _uniform(rng, params["skin_hue_shift_range"]),
        "saturation": _uniform(rng, params["skin_saturation_range"]),
        "value": _uniform(rng, params["skin_value_range"]),
        "roughness": _uniform(rng, params["skin_roughness_range"]),
        "specular": _uniform(rng, params["skin_specular_range"]),
        "clearcoat": _uniform(rng, params["skin_clearcoat_range"]),
    }


def sample_eyes(rng: random.Random, macros: dict) -> dict:
    """Eye colour weighted by the dominant race weight."""
    color = _weighted_choice(rng, C.EYE_COLOR_WEIGHTS_BY_RACE[dominant_race(macros)])
    material, iris_darken = C.EYE_MATERIALS[color]
    return {"mesh": C.EYES_MESH, "color": color, "material": material, "iris_darken": iris_darken}


def eyebrow_for_identity(job_seed: int, identity_index: int) -> str:
    """Eyebrow styles dealt in shuffled blocks of 12: every block of 12 identities uses each
    style exactly once. Deterministic from (job_seed, identity_index)."""
    block, slot = divmod(identity_index - 1, len(C.EYEBROWS))
    order = list(C.EYEBROWS)
    random.Random(derive_seed(job_seed, C.EYEBROW_STREAM_ID, block)).shuffle(order)
    return order[slot]


def sample_brows_lashes(rng: random.Random, params: dict, hair: dict, eyebrows: str) -> dict:
    darkness = _uniform(rng, params["eyebrow_darkness_range"])
    return {
        "eyebrows": eyebrows,
        "eyebrow_darkness": darkness,
        "eyebrow_rgb": [c * darkness for c in hair["rgb"]],
        "eyelashes": rng.choice(C.EYELASHES),
    }


def hair_style_weights(sex: str) -> dict[str, float]:
    """Style weights without the excluded (eye-covering) styles, renormalized."""
    weights = {k: v for k, v in C.HAIR_STYLE_WEIGHTS[sex].items() if k not in C.HAIR_STYLE_EXCLUDE}
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def sample_hair(rng: random.Random, macros: dict, age_years: float) -> dict:
    style = _weighted_choice(rng, hair_style_weights(sex_of(macros["gender"])))
    color_weights = C.HAIR_COLOR_WEIGHTS_OLD if age_bucket(age_years) == "old" else C.HAIR_COLOR_WEIGHTS
    color_name = _weighted_choice(rng, color_weights)
    jitter = _uniform(rng, list(C.HAIR_COLOR_VALUE_JITTER))
    rgb = [c * jitter for c in C.HAIR_COLORS[color_name]]
    return {"style": style, "color_name": color_name, "rgb": rgb}


def sample_identity(job_seed: int, identity_index: int, params: dict, face_categories: dict[str, str]) -> dict:
    """face_categories: {category name: region} from config/face_targets.json."""
    seed = derive_seed(job_seed, identity_index)
    rng = random.Random(seed)
    macros, age_years = sample_macros(rng, params)
    face_targets = sample_face_targets(rng, params, face_categories)
    skin = sample_skin(rng, params, macros, age_years)
    eyes = sample_eyes(rng, macros)
    hair = sample_hair(rng, macros, age_years)
    return {
        "identity_index": identity_index,
        "seed": seed,
        "age_years": age_years,
        "sex": sex_of(macros["gender"]),
        "macros": macros,
        "face_targets": face_targets,
        "skin": skin,
        "eyes": eyes,
        "hair": hair,
        **sample_brows_lashes(rng, params, hair, eyebrow_for_identity(job_seed, identity_index)),
    }


# ---------------------------------------------------------------------------
# Per-image parameter groups
# ---------------------------------------------------------------------------
def _banded_yaw(rng: random.Random, bands: list[list[float]]) -> float:
    lo, hi, _ = rng.choices(bands, weights=[b[2] for b in bands], k=1)[0]
    return rng.choice((-1.0, 1.0)) * rng.uniform(lo, hi)


def sample_camera(rng: random.Random, params: dict) -> dict:
    """Degrees. +yaw: camera toward subject's left (+X); +pitch: camera above; +roll: CCW in image.

    |yaw| comes from weighted bands (default 60% 0-15, 30% 15-30, 10% 30-40)."""
    return {
        "yaw": _banded_yaw(rng, params["yaw_bands"]),
        "pitch": _near_frontal(rng, params["pitch_sigma"], params["pitch_limit"]),
        "roll": _near_frontal(rng, params["roll_sigma"], params["roll_limit"]),
        "distance_m": _uniform(rng, params["camera_distance_range"]),
        "focal_mm": _uniform(rng, params["focal_mm_range"]),
        "face_fill": _uniform(rng, params["face_fill_range"]),
    }


def sample_light(rng: random.Random, params: dict) -> dict:
    lo, hi = params["light_intensity_range"]
    limit = params["light_azimuth_limit"]
    return {
        "azimuth": rng.uniform(-limit, limit),
        "elevation": _uniform(rng, params["light_elevation_range"]),
        "kelvin": _uniform(rng, params["light_kelvin_range"]),
        "intensity": math.exp(rng.uniform(math.log(lo), math.log(hi))),
        "size_m": _uniform(rng, params["light_size_range"]),
    }


def _random_color(rng: random.Random, params: dict) -> list[float]:
    hue = rng.random()
    sat = _uniform(rng, params["background_saturation_range"])
    val = _uniform(rng, params["background_value_range"])
    return list(colorsys.hsv_to_rgb(hue, sat, val))


def sample_background(rng: random.Random, params: dict) -> dict:
    gradient = rng.random() < params["background_gradient_prob"]
    bottom = _random_color(rng, params)
    top = _random_color(rng, params) if gradient else bottom
    return {
        "kind": "gradient" if gradient else "solid",
        "top_rgb": top,
        "bottom_rgb": bottom,
        "ambient_strength": _uniform(rng, params["ambient_strength_range"]),
        "ambient_tint": params["ambient_tint"],
    }


def sample_image(job_seed: int, identity_index: int, image_index: int, params: dict, attempt: int = 0) -> dict:
    """Per-image params. attempt > 0 = deterministic re-sample (e.g. an eye was hidden)."""
    parts = (job_seed, identity_index, image_index) + ((attempt,) if attempt else ())
    seed = derive_seed(*parts)
    rng = random.Random(seed)
    return {
        "image_index": image_index,
        "attempt": attempt,
        "seed": seed,
        "camera": sample_camera(rng, params),
        "light": sample_light(rng, params),
        "background": sample_background(rng, params),
    }
