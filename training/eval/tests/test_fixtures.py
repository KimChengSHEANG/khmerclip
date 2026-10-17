# training/eval/tests/test_fixtures.py
import json
from pathlib import Path

import pytest
from PIL import Image

from khmerclip_eval.fixtures import load_fixtures


@pytest.fixture()
def fixture_root(tmp_path: Path) -> Path:
    (tmp_path / "printed").mkdir()
    img = tmp_path / "printed" / "a.png"
    Image.new("RGB", (8, 8), "white").save(img)
    (tmp_path / "labels.jsonl").write_text(
        json.dumps(
            {"id": "a", "image": "printed/a.png", "text": "ឆ", "subset": "printed"},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return tmp_path


def test_load_fixtures_ok(fixture_root: Path):
    items = load_fixtures(fixture_root)
    assert len(items) == 1
    assert items[0].text == "ឆ"
    assert items[0].image_path.exists()


def test_missing_image_raises(fixture_root: Path):
    (fixture_root / "labels.jsonl").write_text(
        '{"id":"x","image":"printed/missing.png","text":"a","subset":"printed"}\n',
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError):
        load_fixtures(fixture_root)


def test_unverified_rows_are_skipped(fixture_root: Path):
    with (fixture_root / "labels.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"id":"b","image":"printed/a.png","text":"x","subset":"real","verified":false}\n')
    with pytest.warns(UserWarning, match="skipped 1 unverified"):
        items = load_fixtures(fixture_root)
    assert [i.id for i in items] == ["a"]
