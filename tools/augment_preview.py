"""Preview of training-time camera effects (NOT applied to stored data; CLAUDE.md keeps
saved renders clean). Shows what the model would see after augmentation.

    venv/bin/python tools/augment_preview.py in.png out.png --seed 1
"""

import argparse
import io
import random
import sys

from PIL import Image, ImageChops, ImageEnhance, ImageFilter

BLUR_RADIUS = (0.3, 0.9)
NOISE_SIGMA = (3.0, 8.0)
JPEG_QUALITY = (45, 75)
COLOR_JITTER = (0.9, 1.1)  # brightness / contrast / saturation multipliers


def augment(img: Image.Image, rng: random.Random) -> Image.Image:
    img = img.convert("RGB").filter(ImageFilter.GaussianBlur(rng.uniform(*BLUR_RADIUS)))
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        img = enhancer(img).enhance(rng.uniform(*COLOR_JITTER))
    noise = Image.effect_noise(img.size, rng.uniform(*NOISE_SIGMA)).convert("RGB")
    img = ImageChops.add(img, noise, scale=1.0, offset=-128)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=rng.randint(*JPEG_QUALITY))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    augment(Image.open(args.src), random.Random(args.seed)).save(args.dst)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
