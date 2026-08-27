"""Light train-time image augmentations for OCR robustness."""

from __future__ import annotations

import io
import random

from PIL import Image, ImageEnhance, ImageFilter


def _gaussian_blur(image: Image.Image, rng: random.Random) -> Image.Image:
    radius = rng.uniform(0.3, 1.2)
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def _contrast_jitter(image: Image.Image, rng: random.Random) -> Image.Image:
    factor = rng.uniform(0.75, 1.25)
    return ImageEnhance.Contrast(image).enhance(factor)


def _jpeg_recompress(image: Image.Image, rng: random.Random) -> Image.Image:
    quality = rng.randint(40, 85)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _small_rotate(image: Image.Image, rng: random.Random) -> Image.Image:
    angle = rng.uniform(-3.0, 3.0)
    return image.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=(255, 255, 255))


_AUG_OPS = (_gaussian_blur, _contrast_jitter, _jpeg_recompress, _small_rotate)


def maybe_augment(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    """Return the image unchanged 50% of the time; otherwise apply a random subset of light augs."""
    if rng is None:
        rng = random.Random()

    if rng.random() < 0.5:
        return image

    out = image
    for op in _AUG_OPS:
        if rng.random() < 0.5:
            out = op(out, rng)
    return out
