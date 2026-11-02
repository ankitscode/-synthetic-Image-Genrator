"""Pack what training/report.py needs into one zip, to run the evaluation on another machine (e.g. a laptop).

    python -m training.bundle --root <synthface_output> --run <synthface_output>/models/<run> --out <zip> \
        [--real-dir <synthface_output>/raw_photos]

The zip keeps the synthface_output layout, so after unzipping:

    python -m training.report --root <unzipped> --run <unzipped>/models/<run> [--real-dir <photos>]

Contents: models/<run>/{best.pt, index.json, metrics.json, config.json}; for each held-out person
<chunk>/<id>/params.json + aligned/*.jpg; selections/<chunk>.shapes.npz with only the held-out people's face shapes;
with --real-dir only its landmarks.json (as real_landmarks.json; the photos themselves are not packed).
About 10 MB for 90 held-out people.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from training import data

RUN_FILES = ("best.pt", "index.json", "metrics.json", "config.json")


def export(root: Path, run: Path, out: Path, real_dir: Path | None = None) -> dict:
    people = [data.Person(**p) for p in json.loads((run / "index.json").read_text())["people"]]
    holdout = [p for p in people if p.split == "holdout"]
    counts = {"people": len(holdout), "images": 0, "shapes": 0}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name in RUN_FILES:
            if (run / name).exists():
                z.write(run / name, f"models/{run.name}/{name}")
        for p in holdout:
            z.write(root / p.chunk / p.identity_id / "params.json", f"{p.chunk}/{p.identity_id}/params.json")
            for rel in p.images:
                z.write(root / rel, rel)
                counts["images"] += 1
        for chunk in sorted({p.chunk for p in holdout}):
            path = root / "selections" / f"{chunk}.shapes.npz"
            if not path.exists():
                continue
            cache = np.load(path)
            wanted = {int(p.key.split(":")[1]) for p in holdout if p.chunk == chunk}
            keep = np.array([k for k, i in enumerate(cache["indices"]) if int(i) in wanted], dtype=np.int64)
            buf = io.BytesIO()
            np.savez_compressed(buf, shapes=cache["shapes"][keep], indices=cache["indices"][keep])
            z.writestr(f"selections/{chunk}.shapes.npz", buf.getvalue())
            counts["shapes"] += len(keep)
        if real_dir is not None and (real_dir / "landmarks.json").exists():
            z.write(real_dir / "landmarks.json", "real_landmarks.json")
            counts["real_landmarks"] = True
    tmp.replace(out)
    counts["mb"] = round(out.stat().st_size / 1e6, 1)
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True, help="folder with the chunk folders (synthface_output)")
    p.add_argument("--run", type=Path, required=True, help="trained run folder (best.pt, index.json)")
    p.add_argument("--out", type=Path, required=True, help="zip to write")
    p.add_argument("--real-dir", type=Path, help="real test photos folder: its landmarks.json is packed")
    args = p.parse_args()
    counts = export(args.root, args.run, args.out, args.real_dir)
    print(f"BUNDLE {args.out}: {counts}")


if __name__ == "__main__":
    main()
