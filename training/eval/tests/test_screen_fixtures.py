import importlib.util
import json
import random
from pathlib import Path

from PIL import Image

from khmerclip_eval.fixtures import load_fixtures

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
