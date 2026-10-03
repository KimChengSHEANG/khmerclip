"""Load and combine the Khmer OCR datasets we fine-tune / evaluate on.

Several interchangeable Hugging Face datasets are supported out of the box, each an
``image``/``text`` pair of printed or synthetic Khmer (some also English) lines. Train on one,
or combine several into a single set — they share the same column schema, so combining is just a
concatenation.

    python scripts/train.py --datasets soyvitou           # one dataset (the default)
    python scripts/train.py --datasets sokheng hanuman    # combined
    python scripts/train.py --datasets soyvitou sokheng hanuman mrrtmob kiteaether   # all

Registry aliases include ``handwriting``, ``handwriting_test``, ``darayut``, ``siemreap``,
``synth``, and ``synth_val``. An existing directory path loads as a local ``imagefolder``
(images plus ``metadata.jsonl`` with ``file_name`` and ``text``).

Any string that isn't a known alias is treated as a raw Hugging Face hub id (assumed to expose
``image``/``text`` columns), so new datasets work without a code change.

Note: ``mrrtmob`` (12M lines, Khmer+English) is a gated dataset — you must accept its terms on
the Hub and authenticate (``huggingface-cli login``) before it can be downloaded.
"""
from pathlib import Path
from typing import NamedTuple

from datasets import Dataset, Features, Value, concatenate_datasets, load_dataset
from datasets import Image as DSImage

_SOYVITOU = "hf://datasets/SoyVitou/62k-images-khmer-printed-dataset"
_HANDWRITING = "hf://datasets/SoyVitou/khmer-handwritten-dataset-4.2k"

# Every source is normalized to exactly these columns/types so sources concatenate cleanly.
_FEATURES = Features({"image": DSImage(), "text": Value("string")})


class DatasetSpec(NamedTuple):
    """How to load one source. ``path`` is a hub id, or a builder name ("parquet",
    "imagefolder") when ``data_files`` / ``data_dir`` pin exactly which files to read."""

    path: str
    image_col: str = "image"
    text_col: str = "text"
    data_files: list[str] | None = None
    data_dir: str | None = None

    def load_kwargs(self) -> dict:
        return {k: v for k, v in (("data_files", self.data_files), ("data_dir", self.data_dir)) if v is not None}


DATASET_REGISTRY = {
    # The hub card lumps train/valid/test parquets into one "train" split. testset.parquet is
    # where training/eval/fixtures/printed comes from, so it must never be trained on.
    "soyvitou": DatasetSpec("parquet", data_files=[f"{_SOYVITOU}/trainset.parquet", f"{_SOYVITOU}/validset.parquet"]),
    "sokheng": DatasetSpec("Sokheng/khmer-synthetic-ocr-v1-100k"),
    "hanuman": DatasetSpec("seanghay/khmer-hanuman-100k"),
    "mrrtmob": DatasetSpec("mrrtmob/khmer_english_ocr_image_line"),  # gated: 12M Khmer+English lines
    "kiteaether": DatasetSpec("KiteAether/c_khmer_gemma_ocr_train"),
    # Real handwriting; testset.parquet is held out for evaluation (handwriting_test).
    "handwriting": DatasetSpec("parquet", data_files=[f"{_HANDWRITING}/trainset.parquet", f"{_HANDWRITING}/valset.parquet"]),
    "handwriting_test": DatasetSpec("parquet", data_files=[f"{_HANDWRITING}/testset.parquet"]),
    "darayut": DatasetSpec("Darayut/khmer-document-synthetic-low-res", text_col="label"),  # low-res document lines
    "siemreap": DatasetSpec("KimkosalYon/khmer-ocr-200k-siemreap-arial"),  # Siemreap/Arial font lines
    # Rendered locally by scripts/render_synth.py (paths relative to training/trocr).
    "synth": DatasetSpec("parquet", data_files=["cache/synth/train/*.parquet"]),
    "synth_val": DatasetSpec("parquet", data_files=["cache/synth/val/*.parquet"]),
}


# What the scripts train/eval on when --datasets isn't passed.
DEFAULT_DATASETS = ["soyvitou"]


def resolve_dataset(name: str) -> DatasetSpec:
    """Registry alias -> its spec; an existing directory -> a local imagefolder (images plus a
    ``metadata.jsonl`` with ``file_name`` and ``text``); anything else -> a raw hub id."""
    if name in DATASET_REGISTRY:
        return DATASET_REGISTRY[name]
    if Path(name).is_dir():
        return DatasetSpec("imagefolder", data_dir=name)
    return DatasetSpec(name)


def _normalize_columns(ds: Dataset, image_col: str, text_col: str) -> Dataset:
    """Rename the image/text columns to the canonical names and drop everything else, so
    datasets with differing schemas concatenate cleanly."""
    rename = {}
    if image_col != "image":
        rename[image_col] = "image"
    if text_col != "text":
        rename[text_col] = "text"
    if rename:
        ds = ds.rename_columns(rename)
    extra = [c for c in ds.column_names if c not in ("image", "text")]
    if extra:
        ds = ds.remove_columns(extra)
    return ds


def _canonical(ds: Dataset) -> Dataset:
    """Cast to the shared features only when they differ. Call this on already-subsampled data:
    a cast rewrites every row, which is ruinous on the 12M-row mrrtmob."""
    return ds if ds.features == _FEATURES else ds.cast(_FEATURES)


def _load_one(spec: DatasetSpec, smoke_test: bool = False, smoke_samples: int = 200) -> Dataset:
    if smoke_test:
        stream = load_dataset(spec.path, split="train", streaming=True, **spec.load_kwargs())
        ds = Dataset.from_list(list(stream.take(smoke_samples)))
    else:
        ds = load_dataset(spec.path, split="train", **spec.load_kwargs())
    return _normalize_columns(ds, spec.image_col, spec.text_col)


def load_ocr_datasets(names, smoke_test: bool = False, smoke_samples: int = 200) -> Dataset:
    """Load every row of each named source and concatenate (no holdout carved).

    ``smoke_test`` streams only the first ``smoke_samples`` rows of each (no full download).
    """
    parts = []
    for name in names:
        spec = resolve_dataset(name)
        print(f"Loading dataset {name} ({spec.path})" + (f" [smoke: {smoke_samples}]" if smoke_test else ""))
        parts.append(_canonical(_load_one(spec, smoke_test, smoke_samples)))
    if len(parts) == 1:
        return parts[0]
    combined = concatenate_datasets(parts)
    print(f"Combined {len(names)} datasets -> {len(combined)} samples")
    return combined


def load_ocr_split(names, test_size: float = 0.03, seed: int = 42,
                   smoke_test: bool = False, smoke_samples: int = 200):
    """Load + combine the named datasets and carve a reproducible train/validation split.

    The same ``names``, ``seed`` and ``test_size`` always yield the same held-out validation
    split, so ``scripts/evaluate_model.py`` can reproduce training's eval set exactly.
    """
    dataset = load_ocr_datasets(names, smoke_test=smoke_test, smoke_samples=smoke_samples)
    return dataset.train_test_split(test_size=test_size, seed=seed)
