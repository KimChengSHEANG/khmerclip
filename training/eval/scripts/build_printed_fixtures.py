#!/usr/bin/env python3
"""Freeze evenly spaced OCR fixtures from SoyVitou's held-out test parquet."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"
TESTSET_URI = (
    "hf://datasets/SoyVitou/62k-images-khmer-printed-dataset/testset.parquet"
)
MIXED_RE = re.compile(r"[A-Za-z0-9]")


def evenly_spaced_indices(size: int, count: int) -> list[int]:
    if not 1 <= count <= size:
        raise ValueError(f"count must be between 1 and dataset size ({size})")
    if count == 1:
        return [0]
    return [round(i * (size - 1) / (count - 1)) for i in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    args = parser.parse_args()

    # The dataset card currently declares all parquet files as one train split.
    # Naming the repository's testset parquet explicitly prevents train leakage.
    dataset = load_dataset(
        "parquet", data_files={"test": TESTSET_URI}, split="test"
    )
    indices = evenly_spaced_indices(len(dataset), args.count)
    printed_dir = args.fixtures / "printed"
    if printed_dir.exists():
        shutil.rmtree(printed_dir)
    printed_dir.mkdir(parents=True)

    rows = []
    for number, dataset_index in enumerate(indices, 1):
        sample = dataset[dataset_index]
        fixture_id = f"printed-{number:03d}"
        relative_image = f"printed/{fixture_id}.png"
        sample["image"].convert("RGB").save(args.fixtures / relative_image)
        text = str(sample["text"])
        rows.append(
            {
                "id": fixture_id,
                "image": relative_image,
                "text": text,
                "subset": "mixed" if MIXED_RE.search(text) else "printed",
                "source": "SoyVitou/62k-images-khmer-printed-dataset",
                "source_split": "test",
                "source_index": dataset_index,
            }
        )

    args.fixtures.mkdir(parents=True, exist_ok=True)
    labels_path = args.fixtures / "labels.jsonl"
    with labels_path.open("w", encoding="utf-8") as labels:
        for row in rows:
            labels.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} printed fixtures to {printed_dir}")


if __name__ == "__main__":
    main()
