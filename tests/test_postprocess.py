"""Alignment: landmarks that are a rotated / scaled / shifted template land back on the template."""

import math

import numpy as np
import pytest

from app import postprocess
from shared import constants as C

TEMPLATE = np.array(C.ARCFACE_112, dtype=np.float32)


@pytest.mark.parametrize("angle, scale, shift", [(0, 1.0, (0, 0)), (12, 2.1, (40, 30)), (-25, 1.6, (75, 60))])
def test_similarity_alignment_lands_on_template(angle: float, scale: float, shift: tuple) -> None:
    a = math.radians(angle)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], dtype=np.float32)
    points = (TEMPLATE @ rot.T) * scale + np.array(shift, dtype=np.float32)
    image = np.zeros((256, 256, 3), np.uint8)
    aligned, quality = postprocess.align(image, points)
    assert aligned.shape == (112, 112, 3)
    assert quality["residual_px"] < 1.0
    assert quality["rotation_deg"] == pytest.approx(-angle, abs=0.5)
