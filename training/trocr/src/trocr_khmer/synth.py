"""Render synthetic Khmer text-line images across many fonts and styles.

Targets gaps in the existing training data: font variety (Google Fonts Khmer families), italic
(Kantumruy Pro Italic plus a synthetic shear, since almost no Khmer font ships an italic),
handwriting-like fonts (Fasthand, Freehand), scanned / low-res documents, and light/dark UI
panels.

Khmer needs complex shaping (COENG subscripts, reordered vowel signs). Pillow only shapes it with
the libraqm layout engine, so fonts are always opened with ``Layout.RAQM`` and ``require_raqm``
fails closed when raqm is missing — unshaped renders would silently produce wrong labels.
"""
from __future__ import annotations

import io
import math
import random
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

# google/fonts repo path -> kind. Kinds steer font sampling in render_sample.
GOOGLE_KHMER_FONTS = {
    "ofl/angkor/Angkor-Regular.ttf": "display",
    "ofl/battambang/Battambang-Regular.ttf": "text",
    "ofl/battambang/Battambang-Bold.ttf": "text",
    "ofl/battambang/Battambang-Light.ttf": "text",
    "ofl/bayon/Bayon-Regular.ttf": "display",
    "ofl/bokor/Bokor-Regular.ttf": "display",
    "ofl/chenla/Chenla.ttf": "display",
    "ofl/content/Content-Regular.ttf": "text",
    "ofl/content/Content-Bold.ttf": "text",
    "ofl/dangrek/Dangrek-Regular.ttf": "display",
    "ofl/fasthand/Fasthand-Regular.ttf": "handwriting",
    "ofl/freehand/Freehand-Regular.ttf": "handwriting",
    "ofl/hanuman/Hanuman[wght].ttf": "text",
    "ofl/kantumruypro/KantumruyPro[wght].ttf": "text",
    "ofl/kantumruypro/KantumruyPro-Italic[wght].ttf": "italic",
    "ofl/kdamthmorpro/KdamThmorPro-Regular.ttf": "text",
    "ofl/khmer/Khmer.ttf": "text",
    "ofl/kohsantepheap/KohSantepheap-Regular.ttf": "text",
    "ofl/kohsantepheap/KohSantepheap-Bold.ttf": "text",
    "ofl/konkhmersleokchher/KonkhmerSleokchher-Regular.ttf": "display",
    "ofl/koulen/Koulen-Regular.ttf": "display",
    "ofl/metal/Metal-Regular.ttf": "display",
    "ofl/moul/Moul-Regular.ttf": "display",
    "ofl/moulpali/Moulpali-Regular.ttf": "display",
    "ofl/nokora/Nokora[wght].ttf": "text",
    "ofl/notosanskhmer/NotoSansKhmer[wdth,wght].ttf": "text",
    "ofl/notoserifkhmer/NotoSerifKhmer[wdth,wght].ttf": "text",
    "ofl/odormeanchey/OdorMeanChey-Regular.ttf": "text",
    "ofl/preahvihear/Preahvihear-Regular.ttf": "text",
    "ofl/siemreap/Siemreap.ttf": "text",
    "ofl/suwannaphum/Suwannaphum-Regular.ttf": "text",
    "ofl/suwannaphum/Suwannaphum-Bold.ttf": "text",
    "ofl/taprom/Taprom-Regular.ttf": "display",
}

KIND_WEIGHTS = {"text": 0.55, "handwriting": 0.2, "display": 0.15, "italic": 0.1}
SYNTHETIC_ITALIC_P = 0.25  # extra shear on non-italic fonts
COENG = "្"
_IGNORABLE = {0x20, 0x200B, 0x200C, 0x200D}  # spaces / zero-width: need no glyph


@dataclass(frozen=True)
class FontInfo:
    path: Path
    kind: str
    codepoints: frozenset[int]


def font_info(path: Path, kind: str) -> FontInfo:
    from fontTools.ttLib import TTFont

    return FontInfo(Path(path), kind, frozenset(TTFont(str(path), lazy=True).getBestCmap()))


def load_fonts(font_dir: Path) -> list[FontInfo]:
    fonts = []
    for repo_path, kind in GOOGLE_KHMER_FONTS.items():
        path = Path(font_dir) / Path(repo_path).name
        if not path.is_file():
            raise FileNotFoundError(f"{path} missing; run scripts/fetch_fonts.py")
        fonts.append(font_info(path, kind))
    return fonts


def covers(font: FontInfo, text: str) -> bool:
    """True when the font has a glyph for every visible character (no tofu boxes)."""
    return all(ord(c) in font.codepoints or ord(c) in _IGNORABLE for c in text)


def require_raqm() -> None:
    if not features.check("raqm"):
        raise RuntimeError("Pillow lacks libraqm; Khmer would render unshaped. Install libraqm and reinstall Pillow.")


def render_text(text, font_path, size, fg, bg, pad=(8, 6)) -> Image.Image:
    """Shaped single-line render on a tight canvas plus padding."""
    font = ImageFont.truetype(str(font_path), size, layout_engine=ImageFont.Layout.RAQM)
    left, top, right, bottom = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), text, font=font)
    px, py = pad
    image = Image.new("RGB", (right - left + 2 * px, bottom - top + 2 * py), bg)
    ImageDraw.Draw(image).text((px - left, py - top), text, font=font, fill=fg)
    return image


def shear(image: Image.Image, k: float, bg) -> Image.Image:
    """Synthetic italic: lean right by ``k`` (dx per dy). The canvas widens by ``ceil(k*h)`` so
    nothing clips; the bottom row stays put and the top row shifts right."""
    w, h = image.size
    return image.transform(
        (w + math.ceil(k * h), h),
        Image.Transform.AFFINE,
        (1, k, -k * h, 0, 1, 0),
        resample=Image.Resampling.BICUBIC,
        fillcolor=bg,
    )


def degrade_document(image: Image.Image, rng: random.Random) -> Image.Image:
    """Scanned / photocopied look: low resolution, optional blur, sensor noise, JPEG artifacts.
    Noise is drawn from ``rng`` so renders are reproducible."""
    w, h = image.size
    scale = rng.uniform(0.35, 0.8)
    out = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)
    out = out.resize((w, h), Image.Resampling.BILINEAR)
    if rng.random() < 0.5:
        out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    noise = np.random.default_rng(rng.getrandbits(32)).normal(0, rng.uniform(4, 14), (h, w, 1))
    out = Image.fromarray(np.clip(np.asarray(out, dtype=np.float32) + noise, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=rng.randint(30, 75))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _colors(style: str, rng: random.Random):
    if style == "ui_dark":
        bg = rng.randint(15, 45)
        fg = rng.randint(205, 250)
        return (fg, fg, fg), (bg, bg, bg + rng.randint(0, 8))
    if style == "ui_light":
        bg = rng.randint(235, 255)
        fg = rng.randint(10, 60)
        return (fg, fg, fg), (bg, bg, bg)
    paper = rng.randint(225, 255)
    ink = rng.randint(0, 70)
    return (ink, ink, ink + rng.randint(0, 40)), (paper, paper, rng.randint(paper - 15, paper))


def render_sample(text: str, fonts: list[FontInfo], rng: random.Random):
    """Render one line in a random font/style. Returns ``(image, meta)``, or ``None`` when no
    font has glyphs for every character of ``text``."""
    usable = [f for f in fonts if covers(f, text)]
    if not usable:
        return None
    kinds = sorted({f.kind for f in usable})
    kind = rng.choices(kinds, weights=[KIND_WEIGHTS[k] for k in kinds])[0]
    font = rng.choice([f for f in usable if f.kind == kind])
    style = rng.choices(["plain", "document", "ui_dark", "ui_light"], weights=[0.35, 0.35, 0.15, 0.15])[0]
    fg, bg = _colors(style, rng)
    image = render_text(text, font.path, rng.randint(22, 64), fg, bg, pad=(rng.randint(4, 24), rng.randint(3, 16)))
    italic = kind == "italic"
    if not italic and rng.random() < SYNTHETIC_ITALIC_P:
        image = shear(image, rng.uniform(0.12, 0.3), bg)
        italic = True
    if style == "document":
        image = degrade_document(image, rng)
    return image, {"font": font.path.name, "kind": kind, "style": style, "italic": italic}


def _is_khmer(ch: str) -> bool:
    return "ក" <= ch <= "៿"


def _starts_cluster(ch: str, prev: str) -> bool:
    # A consonant / independent vowel starts a cluster unless a COENG just subscripted it;
    # any non-Khmer character starts its own unit.
    return not _is_khmer(ch) or ("ក" <= ch <= "ឳ" and prev != COENG)


def split_clusters(run: str) -> list[str]:
    """Split an unbroken run into Khmer orthographic clusters (safe cut points)."""
    clusters: list[str] = []
    for i, ch in enumerate(run):
        if clusters and not _starts_cluster(ch, run[i - 1]):
            clusters[-1] += ch
        else:
            clusters.append(ch)
    return clusters


def chunk_line(text: str, max_chars: int) -> list[str]:
    """Pack a label into renderable lines of at most ``max_chars``. Words split on spaces and
    ZWSP; words longer than the limit (common — Khmer rarely uses spaces) are cut at cluster
    boundaries. ZWSP is dropped: it is invisible, so it can't be learned from pixels."""
    chunks: list[str] = []
    cur = ""

    def push(piece: str, sep: str) -> None:
        nonlocal cur
        cand = f"{cur}{sep}{piece}" if cur else piece
        if len(cand) <= max_chars:
            cur = cand
        else:
            if cur:
                chunks.append(cur)
            cur = piece

    for word in text.replace("​", " ").split():
        if len(word) <= max_chars:
            push(word, " ")
        else:
            for i, cluster in enumerate(split_clusters(word)):
                push(cluster, " " if i == 0 else "")
    if cur:
        chunks.append(cur)
    return [c for c in chunks if len(c) <= max_chars]


def build_corpus(texts, exclude, max_chars: int = 32, min_chars: int = 2) -> list[str]:
    """Chunk labels into unique renderable lines, dropping any that equal, or contain, eval text
    (fixtures + UI strings) or one of its chunks."""
    banned = {e.strip() for e in exclude if e.strip()}
    for text in list(banned):
        banned.update(chunk_line(text, max_chars))
    contained = [b for b in banned if len(b) >= 4]
    seen: set[str] = set()
    out: list[str] = []
    for text in texts:
        for chunk in chunk_line(text, max_chars):
            if len(chunk) < min_chars or chunk in seen or chunk in banned:
                continue
            if any(b in chunk for b in contained):
                continue
            seen.add(chunk)
            out.append(chunk)
    return out


def corpus_part(text: str) -> str:
    """Stable ~2% val / 98% train assignment by content hash, so train and val renders never
    share a line across runs or seeds."""
    return "val" if zlib.crc32(text.encode("utf-8")) % 100 < 2 else "train"
