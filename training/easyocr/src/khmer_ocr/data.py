"""Convert Khmer OCR Hugging Face datasets into LMDB recognition datasets.

Each dataset is written under ``{out_dir}/lmdb/{split}/{name}/`` so the EasyOCR
trainer can mix sources via ``select_data`` + ``batch_ratio``.
"""
import io
import os

import lmdb

# Canonical order used by --dataset-count 1|2|3|4 and --datasets all.
DATASET_ORDER = ["soyvitou", "sokheng", "hanuman", "km_en_line", "gemma"]

# Registry of Hugging Face OCR line-image datasets (image + text columns).
DATASETS = {
    "soyvitou": {
        "repo_id": "SoyVitou/62k-images-khmer-printed-dataset",
        "url": "https://huggingface.co/datasets/SoyVitou/62k-images-khmer-printed-dataset",
        "load_kwargs": {
            "data_files": {
                "train": "trainset.parquet",
                "val": "validset.parquet",
                "test": "testset.parquet",
            },
        },
        "splits": {"train": "train", "val": "val", "test": "test"},
    },
    "sokheng": {
        "repo_id": "Sokheng/khmer-synthetic-ocr-v1-100k",
        "url": "https://huggingface.co/datasets/Sokheng/khmer-synthetic-ocr-v1-100k",
        "splits": {"train": "train", "test": "test"},
    },
    "hanuman": {
        "repo_id": "seanghay/khmer-hanuman-100k",
        "url": "https://huggingface.co/datasets/seanghay/khmer-hanuman-100k",
        "splits": {"train": "train"},
    },
    "gemma": {
        "repo_id": "KiteAether/c_khmer_gemma_ocr_train",
        "url": "https://huggingface.co/datasets/KiteAether/c_khmer_gemma_ocr_train",
        "splits": {"train": "train"},
    },
    "km_en_line": {
        "repo_id": "mrrtmob/khmer_english_ocr_image_line",
        "url": "https://huggingface.co/datasets/mrrtmob/khmer_english_ocr_image_line",
        "splits": {"train": "train", "validation": "val"},
        "map_size_gb": 45.0,
        "gated": True,
        "description": "12M lines (~40 GB); accept the Hub terms and run `huggingface-cli login`",
    },
}


def resolve_dataset_names(datasets=None, dataset_count=None):
    """Return dataset keys from explicit names, a 1–4 count, or the default."""
    if datasets is not None and dataset_count is not None:
        raise ValueError("Use only one of --datasets or --dataset-count")

    if datasets is not None:
        if datasets.strip().lower() == "all":
            return list(DATASET_ORDER)
        names = [n.strip() for n in datasets.split(",") if n.strip()]
    elif dataset_count is not None:
        if dataset_count not in (1, 2, 3, 4):
            raise ValueError("--dataset-count must be 1, 2, 3, or 4")
        names = DATASET_ORDER[:dataset_count]
    else:
        return ["soyvitou"]

    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        known = ", ".join(DATASET_ORDER)
        raise ValueError(f"Unknown dataset(s): {', '.join(unknown)}. Choose from: {known}, or all")
    return names


def equal_batch_ratios(n):
    """Return ``n`` batch_ratio strings that sum to 1.0 (for Batch_Balanced_Dataset)."""
    if n <= 0:
        raise ValueError("Need at least one dataset for batch_ratio")
    if n == 1:
        return ["1"]
    base = round(1.0 / n, 4)
    ratios = [base] * n
    ratios[-1] = round(1.0 - base * (n - 1), 4)
    return [str(r) for r in ratios]


def encode_image_to_jpeg(pil_image):
    """Return JPEG-encoded bytes for a PIL image (RGB/L, high quality)."""
    if pil_image.mode not in ("RGB", "L"):
        pil_image = pil_image.convert("RGB")
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def write_lmdb(dataset, out_dir, map_size, max_samples=None):
    """Write a Hugging Face dataset split into an LMDB with EasyOCR trainer keys.

    Keys: ``num-samples``, ``image-%09d`` (JPEG bytes), ``label-%09d`` (utf-8).
    Returns the set of characters seen in the labels.
    """
    os.makedirs(out_dir, exist_ok=True)
    env = lmdb.open(out_dir, map_size=map_size)

    cache = {}
    cnt = 1
    charset = set()
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    for i in range(n):
        sample = dataset[i]
        label = sample["text"]
        if label is None:
            continue
        label = str(label)
        image = sample["image"]
        try:
            image_bin = encode_image_to_jpeg(image)
        except Exception as e:  # noqa: BLE001
            print(f"  skip sample {i}: cannot encode image ({e})")
            continue

        cache[f"image-{cnt:09d}".encode()] = image_bin
        cache[f"label-{cnt:09d}".encode()] = label.encode("utf-8")
        charset.update(label)
        cnt += 1

        if cnt % 1000 == 0:
            with env.begin(write=True) as txn:
                for k, v in cache.items():
                    txn.put(k, v)
            cache = {}
            print(f"  {out_dir}: written {cnt - 1}/{n}")

    cache["num-samples".encode()] = str(cnt - 1).encode()
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)
    env.close()
    print(f"  {out_dir}: done, {cnt - 1} samples")
    return charset


def _load_hf_dataset(spec):
    from datasets import load_dataset

    kwargs = dict(spec.get("load_kwargs", {}))
    return load_dataset(spec["repo_id"], **kwargs)


def _prepare_one(name, spec, lmdb_root, max_samples, default_map_size_gb):
    map_size_gb = spec.get("map_size_gb", default_map_size_gb)
    map_size = int(map_size_gb * (1024 ** 3))

    if spec.get("gated"):
        print(f"[info] {name} is gated on the Hub — accept terms and run `huggingface-cli login`")

    print(f"Loading {name} ({spec['repo_id']}) ...")
    ds = _load_hf_dataset(spec)
    print({k: len(v) for k, v in ds.items()})

    charset = set()
    for hf_split, lmdb_split in spec["splits"].items():
        if hf_split not in ds:
            print(f"[warn] {name}: HF split '{hf_split}' not found, skipping")
            continue
        out_dir = os.path.join(lmdb_root, lmdb_split, name)
        print(f"Converting {name}/{hf_split} -> {out_dir} ...")
        charset |= write_lmdb(
            ds[hf_split], out_dir, map_size=map_size, max_samples=max_samples,
        )
    return charset


def build_dataset(
    out_dir="data",
    dataset_names=None,
    max_samples=None,
    map_size_gb=6.0,
):
    """Download selected datasets and convert them to LMDB, writing the charset.

    Returns the path to the written charset file.
    """
    from .config import load_charset, save_charset

    if dataset_names is None:
        dataset_names = ["soyvitou"]

    lmdb_root = os.path.join(out_dir, "lmdb")
    os.makedirs(lmdb_root, exist_ok=True)

    all_chars = set()
    for name in dataset_names:
        all_chars |= _prepare_one(name, DATASETS[name], lmdb_root, max_samples, map_size_gb)

    charset_path = os.path.join(out_dir, "charset.txt")
    if os.path.exists(charset_path):
        all_chars |= set(load_charset(charset_path))

    save_charset(all_chars, charset_path)
    n_chars = len([c for c in all_chars if c != "\n"])
    print(f"Charset ({n_chars} chars) -> {charset_path}")
    return charset_path
