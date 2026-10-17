import numpy as np
from PIL import Image, ImageDraw, ImageFont

from khmerclip.ocr.preprocess import content_crop, ink_mask, line_strips, pixel_values
from khmerclip.paths import resource_dir


def bars(size, rows, bg="white", fg="black"):
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    for top, bottom in rows:
        draw.rectangle((20, top, size[0] - 20, bottom), fill=fg)  # inclusive rows top..bottom
    return img


def test_ink_mask_none_for_degenerate_image():
    assert ink_mask(Image.new("RGB", (1, 50))) is None


def test_ink_mask_downscales_long_edge_to_scan_max():
    mask = ink_mask(Image.new("RGB", (2000, 100), "white"))
    assert mask.scale == 4
    assert mask.ink.shape == (25, 500)


def test_line_strips_splits_two_bars():
    strips = line_strips(bars((400, 200), [(30, 60), (120, 150)]))
    assert [s.size for s in strips] == [(400, 90), (400, 110)]  # boundary at (60 + 120 + 1) // 2


def test_line_strips_merges_small_mark_into_nearest_line():
    # 4-row dot (< 35% of the 31-row line) sits 5 rows below line 1, 42 above line 2.
    strips = line_strips(bars((400, 200), [(20, 50), (55, 58), (100, 130)]))
    assert [s.size for s in strips] == [(400, 79), (400, 121)]


def test_line_strips_single_line_returns_original():
    img = bars((400, 100), [(30, 60)])
    strips = line_strips(img)
    assert len(strips) == 1 and strips[0] is img


def test_line_strips_three_khmer_lines():
    font = ImageFont.truetype(str(resource_dir() / "fonts" / "NotoSansKhmer-Regular.ttf"), 32)
    img = Image.new("RGB", (600, 290), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(["ច្បាប់ភូមិបាល", "ផ្តល់សិទ្ធិអំណាចឲ្យរដ្ឋ", "Hello world"]):
        draw.text((20, 10 + i * 90), line, font=font, fill="black")
    assert len(line_strips(img)) == 3


def test_content_crop_pads_four_percent_around_ink():
    img = bars((400, 200), [])
    ImageDraw.Draw(img).rectangle((100, 80, 299, 119), fill="black")
    assert content_crop(img).size == (216, 42)  # x 92..308, y 79..121


def test_content_crop_works_in_dark_mode():
    img = Image.new("RGB", (400, 200), "black")
    ImageDraw.Draw(img).rectangle((100, 80, 299, 119), fill="white")
    assert content_crop(img).size == (216, 42)


def test_content_crop_keeps_blank_and_sliver_images():
    blank = Image.new("RGB", (400, 200), "white")
    assert content_crop(blank) is blank
    sliver = Image.new("RGB", (400, 200), "white")
    ImageDraw.Draw(sliver).rectangle((10, 10, 11, 11), fill="black")
    assert content_crop(sliver) is sliver


def test_pixel_values_shape_and_normalization():
    arr = pixel_values(Image.new("RGB", (50, 20), "white"))
    assert arr.shape == (1, 3, 384, 384) and arr.dtype == np.float32
    assert arr.flags["C_CONTIGUOUS"]
    assert np.allclose(arr, 1.0)
    rgba = Image.new("RGBA", (50, 20), (0, 0, 0, 255))
    assert np.allclose(pixel_values(rgba), -1.0)
