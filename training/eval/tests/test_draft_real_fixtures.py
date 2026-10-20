import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from khmerclipper_eval.fixtures import load_fixtures

_spec = importlib.util.spec_from_file_location(
    "draft_real_fixtures", Path(__file__).parents[1] / "scripts/draft_real_fixtures.py"
)
draft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(draft)


def test_add_images_prefills_dedupes_and_stays_unverified(tmp_path: Path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    Image.new("RGB", (20, 10), "white").save(src / "a.png")
    Image.new("RGB", (20, 10), "black").save(src / "b.jpg")
    calls = []

    def fake(paths):
        calls.append(list(paths))
        return [f"pred-{p.stem}" for p in paths]

    rows = draft.add_images(src, out, fake)
    assert [r["text"] for r in rows] == ["pred-real-001", "pred-real-002"]
    assert all(r["verified"] is False and r["subset"] == "real" for r in rows)
    assert (out / "real/real-002.png").is_file()
    assert "real-001" in (out / "review.md").read_text(encoding="utf-8")

    # Same images again (one renamed): nothing new, recognizer not called.
    (src / "a.png").rename(src / "c.png")
    assert draft.add_images(src, out, fake) == []
    assert len(calls) == 1

    # Unverified drafts never score; a verified one does.
    with pytest.raises(ValueError, match="no verified fixtures"), pytest.warns(UserWarning):
        load_fixtures(out)
    lines = (out / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0]) | {"verified": True}
    (out / "labels.jsonl").write_text("\n".join([json.dumps(first)] + lines[1:]) + "\n", encoding="utf-8")
    with pytest.warns(UserWarning, match="skipped 1"):
        assert [i.id for i in load_fixtures(out)] == ["real-001"]


def test_flagged_app_capture_uses_its_own_ocr_text(tmp_path: Path):
    src, out = tmp_path / "feedback", tmp_path / "out"
    src.mkdir()
    Image.new("RGB", (20, 10), "white").save(src / "abc.png")
    (src / "abc.json").write_text(json.dumps({"ocr_text": "ក\nខ ", "engine": "trocr", "app_version": "1.2"}),
                                  encoding="utf-8")
    Image.new("RGB", (20, 10), "black").save(src / "plain.png")

    rows = draft.add_images(src, out, lambda paths: [f"pred-{p.stem}" for p in paths])
    by_source = {r["source"]: r for r in rows}
    assert by_source["abc.png"]["text"] == "ក ខ"  # space-joined lines, like the labels
    assert by_source["abc.png"]["app"] == {"engine": "trocr", "app_version": "1.2"}
    assert by_source["plain.png"]["text"] == "pred-real-002"  # no sidecar -> recognizer
    assert "app: trocr/1.2" in (out / "review.md").read_text(encoding="utf-8")
