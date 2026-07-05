"""Convert the Khmer printed-text dataset into LMDB recognition datasets.

Dataset: https://huggingface.co/datasets/SoyVitou/62k-images-khmer-printed-dataset
Columns: `image` (PIL image) and `text` (ground-truth string).
"""
import io
import os

import lmdb

REPO_ID = "SoyVitou/62k-images-khmer-printed-dataset"
SPLIT_FILES = {
    "train": "trainset.parquet",
    "val": "validset.parquet",
    "test": "testset.parquet",
}


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


def build_dataset(out_dir="data", max_samples=None, map_size_gb=6.0):
    """Download every split and convert to LMDB, writing the charset too.

    Returns the path to the written charset file.
    """
    from datasets import load_dataset

    from .config import save_charset

    lmdb_root = os.path.join(out_dir, "lmdb")
    os.makedirs(lmdb_root, exist_ok=True)
    map_size = int(map_size_gb * (1024 ** 3))

    print(f"Loading dataset splits from {REPO_ID} ...")
    ds = load_dataset(REPO_ID, data_files=dict(SPLIT_FILES))
    print({k: len(v) for k, v in ds.items()})

    all_chars = set()
    for split in ("train", "val", "test"):
        if split not in ds:
            print(f"[warn] split '{split}' not found, skipping")
            continue
        print(f"Converting split '{split}' -> LMDB ...")
        all_chars |= write_lmdb(
            ds[split], os.path.join(lmdb_root, split),
            map_size=map_size, max_samples=max_samples,
        )

    charset_path = os.path.join(out_dir, "charset.txt")
    save_charset(all_chars, charset_path)
    n_chars = len([c for c in all_chars if c != "\n"])
    print(f"Charset ({n_chars} chars) -> {charset_path}")
    return charset_path
