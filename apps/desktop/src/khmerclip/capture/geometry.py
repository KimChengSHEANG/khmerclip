import math
import os
import sys
from collections.abc import Mapping

MIN_SIDE = 4  # logical px; anything smaller is a click, not a selection


def normalize_drag(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float]:
    return min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)


def to_physical(rect: tuple[float, float, float, float], ratio: float, image_w: int, image_h: int
                ) -> tuple[int, int, int, int] | None:
    """Map a logical selection on an overlay to device pixels of the grabbed screenshot.
    `ratio` = screenshot pixels per overlay pixel (1.25/1.5/2.0 under display scaling).
    Rounds outward so no selected pixel is lost; clamps to the image."""
    x, y, w, h = rect
    if w < MIN_SIDE or h < MIN_SIDE:
        return None
    px0, py0 = max(0, math.floor(x * ratio)), max(0, math.floor(y * ratio))
    px1, py1 = min(image_w, math.ceil((x + w) * ratio)), min(image_h, math.ceil((y + h) * ratio))
    if px1 - px0 < 1 or py1 - py0 < 1:
        return None
    return px0, py0, px1 - px0, py1 - py0


def capture_backend(platform: str | None = None, env: Mapping[str, str] | None = None) -> str:
    """'portal' on Wayland (apps can't read the screen or grab global keys there), else 'qt'."""
    platform = sys.platform if platform is None else platform
    env = os.environ if env is None else env
    if not platform.startswith("linux"):
        return "qt"
    session = env.get("XDG_SESSION_TYPE")
    if session:
        return "portal" if session == "wayland" else "qt"
    return "portal" if env.get("WAYLAND_DISPLAY") else "qt"
