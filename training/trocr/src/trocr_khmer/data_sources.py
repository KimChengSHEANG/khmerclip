"""Load and combine the Khmer OCR datasets we fine-tune / evaluate on.

Several interchangeable Hugging Face datasets are supported out of the box, each an
``image``/``text`` pair of printed or synthetic Khmer (some also English) lines. Train on one,
or combine several into a single set — they share the same column schema, so combining is just a
concatenation.

    python scripts/train.py --datasets soyvitou           # one dataset (the default)
    python scripts/train.py --datasets sokheng hanuman    # combined
    python scripts/train.py --datasets soyvitou sokheng hanuman mrrtmob kiteaether   # all

Any string that isn't a known alias is treated as a raw Hugging Face hub id (assumed to expose
``image``/``text`` columns), so new datasets work without a code change.

Note: ``mrrtmob`` (12M lines, Khmer+English) is a gated dataset — you must accept its terms on
the Hub and authenticate (``huggingface-cli login``) before it can be downloaded.
"""
from datasets import Dataset, concatenate_datasets, load_dataset

# Short aliases -> (hub id, image column, text column). All three currently use the standard
# image/text columns, but the mapping keeps us honest if a future dataset names them otherwise.
DATASET_REGISTRY = {
    "soyvitou": ("SoyVitou/62k-images-khmer-printed-dataset", "image", "text"),
    "sokheng": ("Sokheng/khmer-synthetic-ocr-v1-100k", "image", "text"),
    "hanuman": ("seanghay/khmer-hanuman-100k", "image", "text"),
    "mrrtmob": ("mrrtmob/khmer_english_ocr_image_line", "image", "text"),  # gated: 12M Khmer+English lines
    "kiteaether": ("KiteAether/c_khmer_gemma_ocr_train", "image", "text"),
}


# What the scripts train/eval on when --datasets isn't passed.
DEFAULT_DATASETS = ["soyvitou"]


def resolve_dataset(name: str) -> tuple[str, str, str]:
    """Return ``(hub_id, image_col, text_col)`` for a registry alias or a raw hub id."""
    if name in DATASET_REGISTRY:
        return DATASET_REGISTRY[name]
    return (name, "image", "text")  # unknown -> raw hub id with standard columns


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


def load_ocr_datasets(names, smoke_test: bool = False, smoke_samples: int = 200) -> Dataset:
    """Load each named dataset's ``train`` split, normalize columns, and concatenate.

    ``smoke_test`` streams only the first ``smoke_samples`` rows of each (no full download) for a
    fast pipeline sanity check.
    """
    parts = []
    for name in names:
        hub_id, image_col, text_col = resolve_dataset(name)
        print(f"Loading dataset {name} ({hub_id})" + (f" [smoke: {smoke_samples}]" if smoke_test else ""))
        if smoke_test:
            stream = load_dataset(hub_id, split="train", streaming=True)
            ds = Dataset.from_list(list(stream.take(smoke_samples)))
        else:
            ds = load_dataset(hub_id, split="train")
        parts.append(_normalize_columns(ds, image_col, text_col))
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
