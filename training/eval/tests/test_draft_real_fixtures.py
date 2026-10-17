import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from khmerclip_eval.fixtures import load_fixtures

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
