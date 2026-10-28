"""Training-time augmentation (the saved renders stay clean, CLAUDE.md). numpy + Pillow, one image at a time.

Aims at what a phone camera adds to a clean, well-aligned face: alignment jitter, colour / exposure /
white-balance changes, grey images, blur and low resolution, sensor noise and JPEG compression.
Every call takes an explicit numpy Generator (deterministic per worker seed).
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

SIZE = 112
AUG = {
    "flip_p": 0.5,
    "shift_px": 4, "scale": (0.93, 1.07), "rotate_deg": 6.0, "geometry_p": 0.7,
    "brightness": 0.35, "contrast": 0.3, "saturation": 0.35, "color_p": 0.8,
    "wb_shift": 0.08,                       # per-channel gain +- (white balance)
    "gray_p": 0.08,
    "blur_p": 0.2, "blur_radius": (0.4, 1.4),
    "lowres_p": 0.2, "lowres_size": (36, 80),
    "noise_p": 0.2, "noise_sigma": (2.0, 8.0),
    "jpeg_p": 0.4, "jpeg_quality": (30, 90),
}


def augment(image: np.ndarray, rng: np.random.Generator, cfg: dict = AUG) -> np.ndarray:
    """uint8 (112, 112, 3) RGB -> augmented uint8 (112, 112, 3)."""
    im = Image.fromarray(image)
    if rng.random() < cfg["flip_p"]:
        im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rng.random() < cfg["geometry_p"]:  # small re-alignment error: shift, scale, rotation about the centre
        s = rng.uniform(*cfg["scale"])
        a = np.deg2rad(rng.uniform(-cfg["rotate_deg"], cfg["rotate_deg"]))
        dx, dy = rng.uniform(-cfg["shift_px"], cfg["shift_px"], 2)
        c = SIZE / 2
        cos, sin = np.cos(a) / s, np.sin(a) / s     # inverse map (output -> input) for Image.transform
        coeffs = (cos, sin, c - cos * c - sin * c + dx, -sin, cos, c + sin * c - cos * c + dy)
        im = im.transform((SIZE, SIZE), Image.Transform.AFFINE, coeffs, Image.Resampling.BILINEAR,
                          fillcolor=tuple(int(v) for v in np.asarray(im).reshape(-1, 3).mean(0)))
    if rng.random() < cfg["color_p"]:
        im = ImageEnhance.Brightness(im).enhance(1 + rng.uniform(-cfg["brightness"], cfg["brightness"]))
        im = ImageEnhance.Contrast(im).enhance(1 + rng.uniform(-cfg["contrast"], cfg["contrast"]))
        im = ImageEnhance.Color(im).enhance(1 + rng.uniform(-cfg["saturation"], cfg["saturation"]))
        gains = 1 + rng.uniform(-cfg["wb_shift"], cfg["wb_shift"], 3)
        im = Image.fromarray(np.clip(np.asarray(im, np.float32) * gains, 0, 255).astype(np.uint8))
    if rng.random() < cfg["gray_p"]:
        im = im.convert("L").convert("RGB")
    if rng.random() < cfg["blur_p"]:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(*cfg["blur_radius"])))
    if rng.random() < cfg["lowres_p"]:
        small = int(rng.integers(cfg["lowres_size"][0], cfg["lowres_size"][1] + 1))
        im = im.resize((small, small), Image.Resampling.BILINEAR).resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    out = np.asarray(im, np.float32)
    if rng.random() < cfg["noise_p"]:
        out = out + rng.normal(0, rng.uniform(*cfg["noise_sigma"]), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < cfg["jpeg_p"]:
        buf = io.BytesIO()
        Image.fromarray(out).save(buf, "JPEG", quality=int(rng.integers(cfg["jpeg_quality"][0], cfg["jpeg_quality"][1] + 1)))
        out = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    return out
