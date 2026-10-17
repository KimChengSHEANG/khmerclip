"""TrOCR preprocessing, ported from apps/mac/Sources/OCR/TrOCREngine.swift.

Keep the constants and integer arithmetic identical to the Swift version so both apps
split lines and crop the same way.
"""

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

SCAN_MAX = 512
THRESHOLD = 40          # per-channel difference from background that counts as ink
MERGE_PERCENT = 35      # runs shorter than this % of the tallest merge into a neighbour
SIZE = 384              # TrOCR ViT input


@dataclass
class InkMask:
    ink: np.ndarray  # bool, shape (h, w)
    scale: int       # mask coords * scale = full-res coords


def ink_mask(image: Image.Image, scan_max: int = SCAN_MAX) -> InkMask | None:
    w, h = image.size
    if w <= 1 or h <= 1:
        return None
    scale = max(1, -(-max(w, h) // scan_max))
    sw, sh = w // scale, h // scale
    if sw <= 1 or sh <= 1:
        return None
    px = np.asarray(image.convert("RGB").resize((sw, sh), Image.BILINEAR), dtype=np.int32)
    corners = np.stack([px[0, 0], px[0, -1], px[-1, 0], px[-1, -1]])
    bg = corners.sum(axis=0) // 4
    return InkMask((np.abs(px - bg) > THRESHOLD).any(axis=2), scale)


def line_strips(image: Image.Image) -> list[Image.Image]:
    """Split a clip into horizontal text-line strips by row ink projection."""
    mask = ink_mask(image)
    if mask is None:
        return [image]
    runs: list[list[int]] = []
    for y, inked in enumerate(mask.ink.any(axis=1)):
        if not inked:
            continue
        if runs and runs[-1][1] == y - 1:
            runs[-1][1] = y
        else:
            runs.append([y, y])
    # ponytail: fixed 35%-of-tallest merge cutoff (same as mac); tune if headings next to
    # small body text get merged.
    while len(runs) > 1:
        heights = [hi - lo + 1 for lo, hi in runs]
        tallest = max(heights)
        i = next((k for k, hgt in enumerate(heights) if hgt * 100 < tallest * MERGE_PERCENT), None)
        if i is None:
            break
        gap_above = runs[i][0] - runs[i - 1][1] if i > 0 else math.inf
        gap_below = runs[i + 1][0] - runs[i][1] if i < len(runs) - 1 else math.inf
        j = i - 1 if gap_above <= gap_below else i + 1
        runs[min(i, j)] = [min(runs[i][0], runs[j][0]), max(runs[i][1], runs[j][1])]
        del runs[max(i, j)]
    if len(runs) <= 1:
        return [image]

    s, mh = mask.scale, mask.ink.shape[0]
    strips = []
    for k in range(len(runs)):
        top = 0 if k == 0 else (runs[k - 1][1] + runs[k][0] + 1) // 2
        bottom = mh if k == len(runs) - 1 else (runs[k][1] + runs[k + 1][0] + 1) // 2
        strips.append(image.crop((0, top * s, image.width, min(image.height, bottom * s))))
    return strips


def content_crop(image: Image.Image) -> Image.Image:
    """Trim uniform background margin so the text fills the frame (TrOCR decodes garbage
    past the real text when given a loose crop)."""
    mask = ink_mask(image)
    if mask is None:
        return image
    ys, xs = np.nonzero(mask.ink)
    if xs.size == 0:
        return image
    sh, sw = mask.ink.shape
    min_x, max_x, min_y, max_y = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    # A mis-detected background (corner on ink) collapses the box to a sliver: don't trust it.
    if (max_x - min_x + 1) * 20 < sw or (max_y - min_y + 1) * 20 < sh:
        return image
    pad_x = max(1, (max_x - min_x + 1) * 4 // 100)
    pad_y = max(1, (max_y - min_y + 1) * 4 // 100)
    s = mask.scale
    x0, y0 = max(0, (min_x - pad_x) * s), max(0, (min_y - pad_y) * s)
    x1 = min(image.width, (max_x + 1 + pad_x) * s)
    y1 = min(image.height, (max_y + 1 + pad_y) * s)
    return image.crop((x0, y0, x1, y1))


def pixel_values(image: Image.Image) -> np.ndarray:
    """HF ViTImageProcessor equivalent: bilinear resize to 384x384, mean = std = 0.5."""
    img = content_crop(image).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - 0.5) / 0.5
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None])
