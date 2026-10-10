from datasets import Dataset, Features, Value
from datasets import Image as DSImage
from PIL import Image

from trocr_khmer import data_sources
from trocr_khmer.data_sources import DatasetSpec, load_ocr_datasets, resolve_dataset

CANONICAL = Features({"image": DSImage(), "text": Value("string")})


def _rows(texts, text_col="text", size=(64, 32), extra=None, text_type="string"):
    data = {"image": [Image.new("RGB", size, "white") for _ in texts], text_col: list(texts)}
    feats = {"image": DSImage(), text_col: Value(text_type)}
    if extra:
        data[extra] = ["x"] * len(texts)
        feats[extra] = Value("string")
    return Dataset.from_dict(data, features=Features(feats))


def test_soyvitou_excludes_testset():
    spec = resolve_dataset("soyvitou")
    assert spec.path == "parquet"
    assert spec.data_files and all("testset" not in f for f in spec.data_files)
    assert any("trainset.parquet" in f for f in spec.data_files)


def test_handwriting_train_and_test_are_disjoint_files():
    train = set(resolve_dataset("handwriting").data_files)
    test = set(resolve_dataset("handwriting_test").data_files)
    assert not train & test
    assert all(f.endswith("testset.parquet") for f in test)


def test_darayut_reads_label_column():
    assert resolve_dataset("darayut").text_col == "label"


def test_unknown_name_is_raw_hub_id():
    assert resolve_dataset("org/some-dataset") == DatasetSpec("org/some-dataset")


def test_existing_directory_is_imagefolder(tmp_path):
    spec = resolve_dataset(str(tmp_path))
    assert spec == DatasetSpec("imagefolder", data_dir=str(tmp_path))
    assert spec.load_kwargs() == {"data_dir": str(tmp_path)}


def test_loader_passes_pinned_files(monkeypatch):
    calls = []

    def fake_load(path, split, **kwargs):
        calls.append((path, split, kwargs))
        return _rows(["ក", "ខ"])

    monkeypatch.setattr(data_sources, "load_dataset", fake_load)
    load_ocr_datasets(["handwriting"])
    assert calls == [("parquet", "train", {"data_files": resolve_dataset("handwriting").data_files})]


def test_loader_normalizes_columns_and_features(monkeypatch):
    monkeypatch.setattr(
        data_sources,
        "load_dataset",
        lambda path, split, **kw: _rows(["ក", "ខ"], text_col="label", extra="source", text_type="large_string"),
    )
    ds = load_ocr_datasets(["darayut"])
    assert ds.column_names == ["image", "text"]
    assert ds.features == CANONICAL
