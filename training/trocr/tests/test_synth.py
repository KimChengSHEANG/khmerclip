import random
from pathlib import Path

import pytest
from PIL import Image, ImageFont, features

from trocr_khmer.synth import (
    COENG,
    build_corpus,
    chunk_line,
    corpus_part,
    covers,
    degrade_document,
    font_info,
    render_sample,
    render_text,
    shear,
)

NOTO = Path(__file__).resolve().parents[2] / "eval/fonts/NotoSansKhmer-Regular.ttf"
CLUSTER = "ស្ត្រ"  # s + coeng t + coeng r: subscripts stack only when shaped
# Rendering needs libraqm (render_text fails closed without it). Skip, don't fail,
# on dev machines whose Pillow can't load it; the render host must have it.
needs_raqm = pytest.mark.skipif(not features.check("raqm"), reason="Pillow lacks raqm: wheels need libfribidi (brew install fribidi)")


@needs_raqm
def test_raqm_shapes_coeng_cluster():
    raqm = ImageFont.truetype(str(NOTO), 48, layout_engine=ImageFont.Layout.RAQM).getbbox(CLUSTER)
    basic = ImageFont.truetype(str(NOTO), 48, layout_engine=ImageFont.Layout.BASIC).getbbox(CLUSTER)
    assert raqm != basic  # shaping repositioned the subscripts


@needs_raqm
def test_render_text_is_rgb_with_padding():
    image = render_text(CLUSTER, NOTO, 40, fg=(0, 0, 0), bg=(255, 255, 255), pad=(10, 5))
    assert image.mode == "RGB" and image.width > 20 and image.height > 10
    assert image.getpixel((0, 0)) == (255, 255, 255)


def test_shear_widens_canvas_and_keeps_corners_background():
    image = Image.new("RGB", (100, 40), (255, 255, 255))
    out = shear(image, 0.25, (255, 255, 255))
    assert out.size == (110, 40)
    assert out.getpixel((0, 0)) == (255, 255, 255)
    assert out.getpixel((109, 39)) == (255, 255, 255)


@needs_raqm
def test_degrade_document_keeps_size_and_mode():
    image = render_text(CLUSTER, NOTO, 40, fg=(0, 0, 0), bg=(250, 250, 245))
    out = degrade_document(image, random.Random(0))
    assert out.size == image.size and out.mode == "RGB"


@needs_raqm
def test_render_sample_is_deterministic():
    fonts = [font_info(NOTO, "text")]
    a, meta_a = render_sample("ភាសាខ្មែរ ២០២៤", fonts, random.Random(7))
    b, meta_b = render_sample("ភាសាខ្មែរ ២០២៤", fonts, random.Random(7))
    assert meta_a == meta_b and a.tobytes() == b.tobytes()
    assert set(meta_a) == {"font", "kind", "style", "italic"}


def test_render_sample_skips_fonts_missing_glyphs():
    fonts = [font_info(NOTO, "text")]
    assert not covers(fonts[0], "ក\U0001F600")
    assert render_sample("ក\U0001F600", fonts, random.Random(0)) is None


def test_chunk_line_respects_limit_and_keeps_coeng_attached():
    text = "ព្រះរាជាណាចក្រកម្ពុជាជាតិសាសនាព្រះមហាក្សត្រ​សួស្តី world"
    chunks = chunk_line(text, max_chars=12)
    assert chunks and all(len(c) <= 12 for c in chunks)
    assert all(not c.endswith(COENG) and not c.startswith(COENG) for c in chunks)
    assert "".join(chunks).replace(" ", "") == text.replace("​", "").replace(" ", "")


def test_build_corpus_excludes_eval_text():
    fixture = "រក្សាទុកជា"
    corpus = build_corpus(["សួស្តី", fixture, f"ចុច {fixture} ឥឡូវ", "សួស្តី"], exclude=[fixture], max_chars=32)
    assert corpus == ["សួស្តី"]


def test_corpus_part_is_stable_and_mostly_train():
    parts = [corpus_part(f"line {i}") for i in range(2000)]
    assert parts == [corpus_part(f"line {i}") for i in range(2000)]
    assert 0 < parts.count("val") < 100
