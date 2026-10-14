"""Pick identity indices of a job seed that cover a requested mix (women, clean-shaven men,
each beard style...). Only runs the sampler (no rendering), so it takes a second.

    venv/bin/python tools/pick_identities.py --seed 20261013 --mix female=8,male_clean=4,male_stubble=2,male_short=2,male_full=2,male_goatee=2
    -> 1,2,3,4,...   (put into job.json "identity_indices", or the Colab notebooks' MIX)

Groups: female, male_clean, male_<beard style> (stubble / short / full / goatee), male_beard (any beard).
Deterministic: walks indices 1, 2, 3, ... and keeps an identity while its group still needs one.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared import constants as C  # noqa: E402
from shared import sampling  # noqa: E402

log = logging.getLogger("pick_identities")
FACE_TARGETS_FILE = PROJECT_ROOT / "config" / "face_targets.json"
MAX_SEARCH = 100_000


def groups_of(identity: dict) -> list[str]:
    """Every group name this identity counts for (most specific first)."""
    if identity["sex"] == "female":
        return ["female"]
    style = identity["beard"]["style"]
    if style == C.BEARD_NONE:
        return ["male_clean"]
    return [f"male_{style}", "male_beard"]


def parse_mix(text: str) -> dict[str, int]:
    mix = {}
    for part in text.split(","):
        name, _, count = part.partition("=")
        mix[name.strip()] = int(count)
    known = {"female", "male_clean", "male_beard", *(f"male_{s}" for s in C.BEARD_STYLES)}
    unknown = set(mix) - known
    if unknown:
        raise ValueError(f"unknown groups {sorted(unknown)}; known: {sorted(known)}")
    return mix


def pick(seed: int, mix: dict[str, int], params: dict | None = None) -> list[dict]:
    """[{"index": i, "group": name, "sex", "age_years", "beard", "hair"}] covering `mix`."""
    params = sampling.resolve_params(params)
    categories = {c["name"]: c["region"] for c in json.loads(FACE_TARGETS_FILE.read_text())["categories"]}
    need = dict(mix)
    chosen = []
    for index in range(1, MAX_SEARCH + 1):
        if not any(need.values()):
            return chosen
        identity = sampling.sample_identity(seed, index, params, categories)
        group = next((g for g in groups_of(identity) if need.get(g, 0) > 0), None)
        if group is None:
            continue
        need[group] -= 1
        chosen.append({"index": index, "group": group, "sex": identity["sex"],
                       "age_years": round(identity["age_years"], 1), "beard": identity["beard"]["style"],
                       "hair": f"{identity['hair']['style']} {identity['hair']['color_name']}"})
    raise RuntimeError(f"could not fill {need} within {MAX_SEARCH} identities")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mix", required=True, help="e.g. female=8,male_clean=4,male_beard=8")
    args = parser.parse_args()
    chosen = pick(args.seed, parse_mix(args.mix))
    for c in chosen:
        log.info("id_%05d  %-12s %-6s %4.0fy  beard %-7s hair %s", c["index"], c["group"], c["sex"],
                 c["age_years"], c["beard"], c["hair"])
    print(",".join(str(c["index"]) for c in chosen))
    return 0


if __name__ == "__main__":
    sys.exit(main())
