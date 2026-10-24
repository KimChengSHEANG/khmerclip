import importlib.util
import json
import random
from pathlib import Path

from PIL import Image

from khmerclipper_eval.fixtures import load_fixtures

_spec = importlib.util.spec_from_file_location(
    "build_screen_fixtures", Path(__file__).parents[1] / "scripts/build_screen_fixtures.py"
)
screen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(screen)

PARA = ("ប្រទេសកម្ពុជា​ជា​ប្រទេស​មួយ នៅ​អាស៊ីអាគ្នេយ៍ ដែល​មាន​រាជធានី "
        "គឺ​ភ្នំពេញ ហើយ​មាន​ប្រជាជន​ប្រហែល ១៧ លាន​នាក់ ។ ") * 6


def test_wrap_lines_bounds_and_no_zwsp():
    lines = screen.wrap_lines(PARA, random.Random(0), 20, 60)
    assert lines
    for line in lines:
        assert 20 <= len(line) <= 60
        assert "​" not in line
        assert line == line.strip() and "  " not in line


def test_wrap_lines_only_cuts_between_words():
    # Cut points are spaces and ZWSPs; ZWSP-joined words stay glued (ZWSP is invisible).
    pieces = set(PARA.replace(" ", "​").split("​")) - {""}
    clean = PARA.replace("​", "")
    for line in screen.wrap_lines(PARA, random.Random(1), 20, 60):
        assert line in clean
        assert any(line.startswith(p) for p in pieces) and any(line.endswith(p) for p in pieces)


def test_label_joins_lines_with_single_space():
    assert screen.label(["ក ខ", "គ"]) == "ក ខ គ"


def test_loader_accepts_screen_subsets(tmp_path: Path):
    Image.new("RGB", (8, 8), "white").save(tmp_path / "a.png")
    rows = [{"id": i, "image": "a.png", "text": "ក", "subset": s} for i, s in enumerate(["screen", "screen_long"])]
    (tmp_path / "labels.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert [i.subset for i in load_fixtures(tmp_path)] == ["screen", "screen_long"]


def test_slice_jobs_cover_every_slice_with_its_style():
    pool = screen.wrap_lines(PARA, random.Random(0), 20, 60)
    jobs = screen.slice_jobs(pool, random.Random(0), per_slice=10)
    by = {}
    for j in jobs:
        by.setdefault(j["subset"], []).append(j)
    assert set(by) == {f"slice_{s}" for s in screen.SLICES}
    assert all(len(v) == 10 and all(len(j["lines"]) == 1 for j in v) for v in by.values())
    assert all(j["size"] <= 13 and j["scale"] == 1 for j in by["slice_small"])
    assert all(j["bold"] for j in by["slice_bold"])
    assert all(j["font"] in screen.DISPLAY_FONTS for j in by["slice_display"])
    assert all(sum(c.isdigit() for c in j["lines"][0]) >= 4 for j in by["slice_numbers"])
    assert all(sum(t in j["lines"][0] for t in screen.LATIN) >= 2 for j in by["slice_mixed"])
    assert all(j["bg"] != [255, 255, 255] and j["fg"].startswith("rgb(") for j in by["slice_lowcontrast"])


def test_loader_accepts_slice_subsets(tmp_path: Path):
    Image.new("RGB", (8, 8), "white").save(tmp_path / "a.png")
    (tmp_path / "labels.jsonl").write_text(
        json.dumps({"id": "a", "image": "a.png", "text": "ក", "subset": "slice_small"}) + "\n", encoding="utf-8")
    assert load_fixtures(tmp_path)[0].subset == "slice_small"


def test_noisy_jobs_cover_every_kind_deterministically():
    from collections import Counter

    pool = ["សួស្តី ពិភពលោក", "ការអប់រំ និងបច្ចេកវិទ្យា", "Settings ការកំណត់"] * 20
    a = screen.noisy_jobs(pool, random.Random(1), per_kind=5)
    b = screen.noisy_jobs(pool, random.Random(1), per_kind=5)
    assert a == b
    assert Counter(j["kind"] for j in a) == {k: 5 for k in screen.NOISE_KINDS}
    assert all(j["subset"] == f"noisy_{j['kind']}" for j in a)


def test_noisy_labels_hold_only_the_text():
    for job in screen.noisy_jobs(["សួស្តី ពិភពលោក"] * 40, random.Random(2), per_kind=3):
        assert job["lines"] == ["សួស្តី ពិភពលោក"] or all(l in ("សួស្តី ពិភពលោក",) + tuple(screen.LATIN) for l in job["lines"])


def test_noisy_page_draws_the_noise_around_the_text():
    jobs = screen.noisy_jobs(["សួស្តី"] * 40, random.Random(3), per_kind=2)
    for job in jobs:
        html = screen.noisy_page(job)
        assert all(l in html for l in job["lines"])
    under = next(j for j in jobs if j["kind"] == "underline")
    assert "underline" in screen.noisy_page(under) or "border" in screen.noisy_page(under)


def test_some_background_and_chrome_clips_are_latin_only():
    jobs = screen.noisy_jobs(["សួស្តី ពិភពលោក"] * 40, random.Random(4), per_kind=20)
    assert any(j["kind"] in ("background", "chrome") and all(l in screen.LATIN for l in j["lines"]) for j in jobs)


def test_noisy_subsets_load(tmp_path):
    (tmp_path / "noisy_icons").mkdir()
    Image.new("RGB", (20, 10), "white").save(tmp_path / "noisy_icons/noisy-icons-001.png")
    row = {"id": "noisy-icons-001", "image": "noisy_icons/noisy-icons-001.png", "text": "ក", "subset": "noisy_icons"}
    (tmp_path / "labels.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    assert [f.subset for f in load_fixtures(tmp_path)] == ["noisy_icons"]
