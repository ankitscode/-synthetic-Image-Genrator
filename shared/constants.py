"""Shared defaults and fixed tables. Standard library only (imported by app and Blender).

Asset names below are exactly as printed by blender/inspect_mpfb.py (makehuman_system_assets, CC0).
"""

# ---------------------------------------------------------------------------
# Alignment template (also used by the mobile app)
# ---------------------------------------------------------------------------
ARCFACE_112: list[list[float]] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
]

# Bump whenever sampling changes what a (seed, index) produces; stored in params.json.
SAMPLER_VERSION = 3

# ---------------------------------------------------------------------------
# Output naming
# ---------------------------------------------------------------------------
IDENTITY_DIR_FMT = "id_{:05d}"
IMAGE_STEM_FMT = "img_{:03d}"

# ---------------------------------------------------------------------------
# Render defaults (env vars / job.json override these)
# ---------------------------------------------------------------------------
DEFAULT_RENDER_SAMPLES = 48
DEFAULT_RENDER_DEVICE = "CPU"
RENDER_RESOLUTION = 256

# ---------------------------------------------------------------------------
# MPFB macros: keys exactly as TargetService.get_default_macro_info_dict() (MPFB 2.0.17)
# ---------------------------------------------------------------------------
MPFB_MACRO_DEFAULTS: dict = {
    "gender": 0.5,
    "age": 0.5,
    "muscle": 0.5,
    "weight": 0.5,
    "proportions": 0.5,
    "height": 0.5,
    "cupsize": 0.5,
    "firmness": 0.5,
    "race": {"asian": 0.33, "caucasian": 0.33, "african": 0.33},
}
RACES: tuple[str, ...] = ("african", "asian", "caucasian")

# Age in years -> MPFB "age" macro (MakeHuman convention, piecewise linear).
AGE_YEARS_TO_MACRO: list[tuple[float, float]] = [(1.0, 0.0), (11.0, 0.1875), (25.0, 0.5), (90.0, 1.0)]

# ---------------------------------------------------------------------------
# Skins
# ---------------------------------------------------------------------------
AGE_BUCKET_YOUNG_MAX = 35.0  # young < 35
AGE_BUCKET_MIDDLEAGE_MAX = 55.0  # 35 <= middleage <= 55, old > 55
MALE_GENDER_THRESHOLD = 0.5  # gender macro >= this -> male skin / hair weights


def _skin_table() -> dict[str, list[str]]:
    table: dict[str, list[str]] = {}
    for age in ("young", "middleage", "old"):
        for race in RACES:
            for sex in ("female", "male"):
                table[f"{age}_{race}_{sex}"] = [f"{age}_{race}_{sex}"]
    # Extra variants present in the pack (toon01 and *_special_suit are excluded on purpose)
    table["young_caucasian_female"].append("young_caucasian_female2")
    table["young_caucasian_male"].append("young_caucasian_male2")
    return table


# key "<age bucket>_<race>_<sex>" -> candidate skin names
SKINS_BY_GROUP: dict[str, list[str]] = _skin_table()

# ---------------------------------------------------------------------------
# Eyes / eyebrows / eyelashes / hair
# ---------------------------------------------------------------------------
EYES_MESH = "high-poly"
# Eye colours: name -> (MPFB eye material, iris darkening factor or None).
# "brownlight" is not used: it renders amber/orange. "darkbrown" = brown with a darkened iris.
EYE_MATERIALS: dict[str, tuple[str, float | None]] = {
    "darkbrown": ("brown", 0.45),
    "brown": ("brown", None),
    "green": ("green", None),
    "grey": ("grey", None),
    "bluegreen": ("bluegreen", None),
    "blue": ("blue", None),
    "lightblue": ("lightblue", None),
    "deepblue": ("deepblue", None),
    "ice": ("ice", None),
}
# Eye colour weights by dominant race weight (darker-skinned groups ~95% brown/dark brown).
EYE_COLOR_WEIGHTS_BY_RACE: dict[str, dict[str, float]] = {
    "african": {"darkbrown": 0.70, "brown": 0.26, "green": 0.01, "grey": 0.01, "bluegreen": 0.01, "blue": 0.01},
    "asian": {"darkbrown": 0.60, "brown": 0.35, "green": 0.02, "grey": 0.01, "bluegreen": 0.01, "blue": 0.01},
    "caucasian": {
        "darkbrown": 0.15, "brown": 0.35, "green": 0.12, "grey": 0.08, "bluegreen": 0.08,
        "blue": 0.08, "lightblue": 0.07, "deepblue": 0.05, "ice": 0.02,
    },
}
EYEBROWS: list[str] = [f"eyebrow{i:03d}" for i in range(1, 13)]
EYELASHES: list[str] = [f"eyelashes{i:02d}" for i in range(1, 5)]

HAIR_NONE = "none"  # bald
HAIR_STYLE_WEIGHTS: dict[str, dict[str, float]] = {
    "male": {
        HAIR_NONE: 0.20, "short01": 0.15, "short02": 0.20, "short03": 0.15, "short04": 0.15,
        "afro01": 0.08, "long01": 0.03, "ponytail01": 0.02, "bob01": 0.01, "bob02": 0.01,
    },
    "female": {
        HAIR_NONE: 0.02, "short01": 0.04, "short02": 0.04, "short03": 0.04, "short04": 0.04,
        "afro01": 0.10, "bob01": 0.12, "bob02": 0.12, "braid01": 0.10, "long01": 0.20,
        "ponytail01": 0.18,
    },
}

# Hair colours as linear RGB targets for the mean strand colour.
HAIR_COLORS: dict[str, tuple[float, float, float]] = {
    "black": (0.020, 0.018, 0.016),
    "dark_brown": (0.060, 0.035, 0.020),
    "brown": (0.130, 0.070, 0.035),
    "light_brown": (0.250, 0.150, 0.080),
    "blonde": (0.550, 0.400, 0.220),
    "auburn": (0.220, 0.070, 0.030),
    "grey": (0.350, 0.350, 0.350),
    "white": (0.700, 0.700, 0.680),
}
HAIR_COLOR_WEIGHTS: dict[str, float] = {
    "black": 0.42, "dark_brown": 0.30, "brown": 0.12, "light_brown": 0.06,
    "blonde": 0.04, "auburn": 0.03, "grey": 0.02, "white": 0.01,
}
HAIR_COLOR_WEIGHTS_OLD: dict[str, float] = {  # age > AGE_BUCKET_MIDDLEAGE_MAX
    "black": 0.15, "dark_brown": 0.12, "brown": 0.05, "light_brown": 0.03,
    "blonde": 0.01, "auburn": 0.01, "grey": 0.43, "white": 0.20,
}
HAIR_COLOR_VALUE_JITTER = (0.85, 1.15)  # multiplier on the colour
# Hair styles that cover an eye on a frontal neutral face (blender/debug_hair_audit.py).
HAIR_STYLE_EXCLUDE: tuple[str, ...] = ("bob01", "bob02", "braid01", "short03")

# Skin tone bands (for audits / tests): tone < light_max = light, < medium_max = medium, else dark
SKIN_TONE_BANDS = {"light_max": 0.30, "medium_max": 0.65}
SKIN_LIGHT_RACES: tuple[str, ...] = ("asian", "caucasian")
SKIN_DARK_RACE = "african"

# Eyebrow styles are dealt in shuffled blocks of len(EYEBROWS) per job so they come out even.
EYEBROW_STREAM_ID = 1  # extra seed part for the per-block eyebrow shuffle

MAX_IMAGE_ATTEMPTS = 10  # re-sample an image's pose when an eye centre is hidden

# ---------------------------------------------------------------------------
# Face shape targets (config/face_targets.json is generated by blender/inspect_mpfb.py)
# ---------------------------------------------------------------------------
FACE_TARGET_GROUPS: tuple[str, ...] = (
    "cheek", "chin", "ears", "eyebrows", "eyes", "forehead", "head", "mouth", "nose",
)
# Region of each category: first matching (MPFB group, name keywords) rule wins; empty
# keywords = rest of the group. Every identity gets at least one target from every region.
FACE_REGION_RULES: list[tuple[str, str, tuple[str, ...]]] = [
    ("nose", "nose", ()),
    ("eyes", "eyes", ()),
    ("eyebrows", "eyebrows", ()),
    ("cheeks", "cheek", ()),
    ("jaw", "chin", ("jaw", "bones", "prognathism")),
    ("chin", "chin", ()),
    ("lips", "mouth", ("lip", "cupidsbow", "philtrum")),
    ("mouth", "mouth", ()),
    ("forehead", "forehead", ()),
    ("ears", "ears", ()),
    ("head_shape", "head", ()),
]
# Excluded: age is handled by the macro; whole-head and sideways feature shifts look broken.
FACE_TARGET_EXCLUDE: tuple[str, ...] = (
    "head-age-decr-incr",
    "head-trans-backward-forward", "head-trans-down-up", "head-trans-in-out",
    "nose-trans-in-out", "mouth-trans-in-out",
)

# ---------------------------------------------------------------------------
# Default sampling parameters (all overridable per job via job.json "params")
# ---------------------------------------------------------------------------
DEFAULT_PARAMS: dict = {
    # identity
    "gender_female_range": [0.0, 0.25],
    "gender_male_range": [0.75, 1.0],
    "gender_between_range": [0.25, 0.75],
    "gender_between_prob": 0.10,  # rest split female / male by male_prob
    "male_prob": 0.5,
    "age_years_range": [18.0, 70.0],
    "age_years_core_range": [20.0, 50.0],
    "age_years_core_prob": 0.75,
    "weight_range": [0.1, 0.9],
    "muscle_range": [0.2, 0.8],
    "height_range": [0.3, 0.7],
    "proportions_range": [0.2, 0.8],
    "race_mix": {"african": 0.30, "asian": 0.35, "caucasian": 0.35},
    "race_concentration": 3.0,  # Dirichlet concentration around race_mix (higher = closer)
    "face_targets_count_range": [20, 40],  # at least one per FACE_REGION_RULES region
    "face_target_strength": 1.0,
    # skin tone: 0 = light base texture, 1 = dark (african) base texture; the two are blended.
    # tone = sum(race weight x anchor) + N(0, sigma). With the default race mix this gives
    # ~58% medium (Arab / South Asian range), ~26% dark, ~16% light (see SKIN_TONE_BANDS).
    "skin_tone_anchor": {"african": 0.95, "asian": 0.45, "caucasian": 0.20},
    "skin_tone_sigma": 0.15,
    "skin_hue_shift_range": [-0.02, 0.02],  # HSV node hue offset around 0.5
    "skin_saturation_range": [0.9, 1.1],
    "skin_value_range": [0.92, 1.08],
    # skin shading (MPFB defaults: roughness 0.45, specular 0.5, clearcoat 0.1 -> too shiny)
    "skin_roughness_range": [0.55, 0.65],
    "skin_specular_range": [0.30, 0.40],
    "skin_clearcoat_range": [0.0, 0.04],
    "eyebrow_darkness_range": [0.45, 1.1],  # multiplier on the hair colour
    # per image: camera (degrees / metres)
    "yaw_bands": [[0.0, 15.0, 0.6], [15.0, 30.0, 0.3], [30.0, 40.0, 0.1]],  # |yaw| lo, hi, prob
    "pitch_limit": 20.0,
    "pitch_sigma": 10.0,  # truncated Gaussian around 0
    "roll_limit": 10.0,
    "roll_sigma": 5.0,
    "camera_distance_range": [0.45, 0.65],
    "focal_mm_range": [40.0, 55.0],
    "face_fill_range": [0.55, 0.60],  # face height (chin to hairline) as a fraction of the frame
    # per image: key light
    "light_azimuth_limit": 75.0,
    "light_elevation_range": [-10.0, 60.0],
    "light_kelvin_range": [3000.0, 6500.0],
    "light_intensity_range": [0.3, 3.0],  # sampled log-uniform
    "light_size_range": [0.2, 1.0],
    # per image: background
    "background_gradient_prob": 0.5,
    "background_saturation_range": [0.0, 0.5],
    "background_value_range": [0.05, 0.8],
    # world light on the face, separate from the visible background colour
    "ambient_strength_range": [0.1, 0.4],
    "ambient_tint": 0.3,  # how much of the background colour tints the ambient light (0 = neutral)
}

# ---------------------------------------------------------------------------
# Scene geometry (Blender side)
# ---------------------------------------------------------------------------
# Face box from the inter-ocular distance (IOD, between eye centres):
FACE_HEIGHT_PER_IOD = 3.0  # chin to hairline ~= 3 x IOD
FACE_CENTER_BELOW_EYES_PER_IOD = 0.3  # eyes sit above the middle of the face box
KEY_LIGHT_DISTANCE_M = 1.0
KEY_LIGHT_BASE_POWER_W = 12.0  # 40 W washed out pale skin (mean face luma 0.82 at 1x); see params.json qa

# Colour management (Blender scene view settings)
VIEW_TRANSFORM = "AgX"
VIEW_LOOK = "None"
VIEW_EXPOSURE = 0.0

# Exposure QA: face pixels = basemesh hits within this radius of the face centre
QA_FACE_RADIUS_PER_IOD = 1.4
QA_SAMPLE_STEP_PX = 2  # ray-cast every Nth pixel in x and y
QA_WHITE_LEVEL = 254.5 / 255.0  # all three channels at or above this = pure white
QA_MAX_WHITE_FRACTION = 0.01
QA_NEAR_WHITE_LUMA = 0.95  # stricter "washed out" check: display luma at or above this
