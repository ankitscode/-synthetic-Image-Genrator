"""Alignment to the ArcFace 112x112 template from the 3D landmarks (no face detector).

    venv/bin/python -m app.postprocess output/<job_id>

Per identity: reads landmarks.json + raw/*.png, writes aligned/img_XXX.jpg (JPEG 95) and
alignment.json (per image: template residual in pixels, scale, rotation).
"""

import json
import logging
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from shared import constants as C

log = logging.getLogger("postprocess")

ALIGNED_SIZE = 112
ALIGNED_JPEG_QUALITY = 95
TEMPLATE = np.array(C.ARCFACE_112, dtype=np.float32)


def similarity_transform(points: np.ndarray) -> np.ndarray:
    """2x3 similarity transform (rotation, uniform scale, translation) from 5 points to the template."""
    matrix, _ = cv2.estimateAffinePartial2D(points.astype(np.float32), TEMPLATE, method=cv2.LMEDS)
    if matrix is None:
        raise ValueError("alignment failed (degenerate landmarks)")
    return matrix


def align(image: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, dict]:
    """Aligned 112x112 crop + quality numbers: residual = mean distance (template pixels) between
    the transformed landmarks and the template (0 = perfect fit; large = unusual pose/shape)."""
    matrix = similarity_transform(points)
    aligned = cv2.warpAffine(image, matrix, (ALIGNED_SIZE, ALIGNED_SIZE), flags=cv2.INTER_AREA,
                             borderMode=cv2.BORDER_REPLICATE)
    moved = points @ matrix[:, :2].T + matrix[:, 2]
    errors = np.linalg.norm(moved - TEMPLATE, axis=1)
    scale = float(math.hypot(matrix[0, 0], matrix[1, 0]))
    return aligned, {
        "residual_px": float(errors.mean()),
        "max_point_error_px": float(errors.max()),
        "scale": scale,
        "rotation_deg": float(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))),
    }


def align_identity(identity_dir: Path) -> dict:
    landmarks = json.loads((identity_dir / "landmarks.json").read_text(encoding="utf-8"))
    out_dir = identity_dir / "aligned"
    out_dir.mkdir(exist_ok=True)
    report = {}
    for stem, lm in landmarks["images"].items():
        image = cv2.imread(str(identity_dir / "raw" / f"{stem}.png"), cv2.IMREAD_COLOR)
        aligned, quality = align(image, np.array(lm["points"], dtype=np.float32))
        cv2.imwrite(str(out_dir / f"{stem}.jpg"), aligned, [cv2.IMWRITE_JPEG_QUALITY, ALIGNED_JPEG_QUALITY])
        report[stem] = {**quality, "all_visible": all(lm["visible"])}
    (identity_dir / "alignment.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return report


def align_job(job_dir: Path) -> None:
    for identity_dir in sorted(job_dir.glob("id_*")):
        if (identity_dir / "landmarks.json").exists():
            report = align_identity(identity_dir)
            worst = max(r["residual_px"] for r in report.values())
            log.info("%s aligned %d images, worst residual %.2f px", identity_dir.name, len(report), worst)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    align_job(Path(sys.argv[1]))
