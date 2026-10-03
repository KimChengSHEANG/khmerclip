from types import SimpleNamespace

import pytest
from datasets import Dataset, Features, Value
from datasets import Image as DSImage
from PIL import Image

from trocr_khmer import data_sources
from trocr_khmer.data_sources import filter_ocr_rows, load_ocr_split, parse_source, resample

FEATS = Features({"image": DSImage(), "text": Value("string")})


def _source(prefix, n):
    return Dataset.from_dict(
        {"image": [Image.new("RGB", (64, 32), "white") for _ in range(n)], "text": [f"{prefix}{i}" for i in range(n)]},
        features=FEATS,
    )


class _FakeTokenizer:
    """One token per character plus <s>/</s>, like the Khmer-extended tokenizer on Khmer text."""

    cls_token_id = 0

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=[0, *range(3, 3 + len(text)), 2])


@pytest.fixture
def fake_hub(monkeypatch):
    sources = {"fake/a": _source("a", 100), "fake/b": _source("b", 50)}
    monkeypatch.setattr(data_sources, "load_dataset", lambda path, split, **kw: sources[path])
    return sources


def test_parse_source():
    assert parse_source("handwriting:60000") == ("handwriting", 60000)
    assert parse_source("org/name") == ("org/name", None)
    assert parse_source("odd:name") == ("odd:name", None)


def test_resample_upsamples_with_every_row_and_subsamples_uniquely():
    ds = _source("x", 10)
    up = resample(ds, 25, seed=0)
    assert len(up) == 25 and set(up["text"]) == set(ds["text"])
    down = resample(ds, 4, seed=0)
    assert len(down) == 4 and len(set(down["text"])) == 4
    assert resample(ds, 4, seed=0)["text"] == down["text"]  # deterministic


def test_resample_empty_raises():
    with pytest.raises(ValueError):
        resample(_source("x", 0), 5, seed=0)


def test_split_keeps_upsampled_rows_out_of_eval(fake_hub):
    split = load_ocr_split(["fake/a:1000", "fake/b"], test_size=0.1, seed=42)
    train, test = set(split["train"]["text"]), set(split["test"]["text"])
    assert not train & test
    assert sum(t.startswith("a") for t in split["train"]["text"]) == 1000


def test_eval_is_capped_per_source(fake_hub):
    split = load_ocr_split(["fake/a", "fake/b"], test_size=0.2, seed=42, max_eval_per_source=3)
    texts = split["test"]["text"]
    assert len(texts) == 6
    assert sum(t.startswith("a") for t in texts) == 3


def test_filter_drops_long_labels_and_wide_images():
    ds = Dataset.from_dict(
        {
            "image": [Image.new("RGB", s, "white") for s in [(64, 32), (64, 32), (2000, 32)]],
            "text": ["ok", "x" * 200, "wide"],
        },
        features=FEATS,
    )
    kept = filter_ocr_rows(ds, tokenizer=_FakeTokenizer(), max_label_len=128, max_aspect=20)
    assert kept["text"] == ["ok"]


def test_filter_is_noop_without_criteria():
    ds = _source("x", 3)
    assert filter_ocr_rows(ds) is ds
