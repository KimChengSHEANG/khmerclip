from PIL import Image, ImageDraw

from khmerclip_eval.pipeline import blocked_token, line_strips, split_wide


def blocks(size, boxes, fg="black", bg="white"):
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    for box in boxes:
        d.rectangle(box, fill=fg)
    return img


def test_blocked_token():
    assert blocked_token([1, 5, 5, 5, 5]) == 5
    assert blocked_token([1, 5, 5, 5]) is None
    assert blocked_token([5, 5, 5, 6, 5]) is None


def test_line_strips_merges_detached_mark():
    # 21px base line with a 4px mark 2px above it: one line, not two.
    img = blocks((200, 60), [(10, 20, 190, 40), (50, 14, 60, 17)])
    assert len(line_strips(img)) == 1


def test_line_strips_three_lines_cover_image():
    img = blocks((200, 120), [(10, 10, 190, 30), (10, 50, 190, 70), (10, 90, 190, 110)])
    strips = line_strips(img)
    assert len(strips) == 3
    assert sum(s.height for s in strips) == 120


def test_split_wide_gap():
    # 400x20 line, blank 20px gap in the middle -> two pieces joined by a space.
    img = blocks((400, 24), [(0, 2, 189, 21), (210, 2, 399, 21)])
    pieces = split_wide(img, max_aspect=10)
    assert [sep for sep, _ in pieces] == ["", " "]
    assert sum(p.width for _, p in pieces) <= 400


def test_split_wide_join_rule_narrow_gap():
    img = blocks((400, 24), [(0, 2, 198, 21), (201, 2, 399, 21)])
    assert [sep for sep, _ in split_wide(img, max_aspect=10)] == ["", ""]


def test_split_wide_no_gap_keeps_whole():
    img = blocks((400, 24), [(0, 2, 399, 21)])
    pieces = split_wide(img, max_aspect=10)
    assert len(pieces) == 1 and pieces[0][1].size == (400, 24)


def test_split_wide_dark():
    img = blocks((400, 24), [(0, 2, 189, 21), (210, 2, 399, 21)], fg=(230, 230, 230), bg=(25, 25, 25))
    assert len(split_wide(img, max_aspect=10)) == 2


def test_split_wide_narrow_strip_untouched():
    img = blocks((100, 24), [(0, 2, 40, 21), (60, 2, 99, 21)])
    assert len(split_wide(img, max_aspect=10)) == 1
