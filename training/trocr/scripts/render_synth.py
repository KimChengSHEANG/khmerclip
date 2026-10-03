#!/usr/bin/env python3
"""Pre-render synthetic Khmer line images to parquet shards (deterministic per --seed).

    python scripts/fetch_fonts.py
    python scripts/render_synth.py --part val --count 3000 --seed 2 --out cache/synth/val
    python scripts/render_synth.py --part train --count 400000 --seed 1 --out cache/synth/train

Text comes from existing datasets' labels, chunked to <= --max_chars, minus every eval fixture
and UI string. --part picks the train or val side of a stable content-hash split.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, Features, Value, load_dataset
from datasets import Image as DSImage

from trocr_khmer.data_sources import resolve_dataset
from trocr_khmer.synth import build_corpus, corpus_part, load_fonts, render_sample, require_raqm

EVAL = Path(__file__).resolve().parents[2] / "eval"
FEATURES = Features(
    {
        "image": DSImage(),
        "text": Value("string"),
        "font": Value("string"),
        "kind": Value("string"),
        "style": Value("string"),
        "italic": Value("bool"),
    }
)


def load_texts(name: str, k: int, seed: int) -> list[str]:
    """k random labels from a source's *training* side — text column only, images never decoded.

    Drops the same seeded 3% holdout that load_ocr_split carves (test_size=0.03, seed=42), so
    replay-holdout labels never reach training re-rendered in other fonts."""
    spec = resolve_dataset(name)
    ds = load_dataset(spec.path, split="train", **spec.load_kwargs()).select_columns([spec.text_col])
    ds = ds.train_test_split(test_size=0.03, seed=42)["train"]
    idx = random.Random(seed).sample(range(len(ds)), min(k, len(ds)))
    return ds.select(idx)[spec.text_col]


def eval_texts() -> list[str]:
    labels = (EVAL / "fixtures/labels.jsonl").read_text(encoding="utf-8").splitlines()
    texts = [json.loads(line)["text"] for line in labels if line.strip()]
    return texts + (EVAL / "ui_strings.txt").read_text(encoding="utf-8").splitlines()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--part", choices=["train", "val"], default="train")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--font_dir", type=Path, default=Path("cache/fonts"))
    p.add_argument("--text_sources", nargs="+", default=["hanuman", "sokheng", "soyvitou", "mrrtmob"])
    p.add_argument("--texts_per_source", type=int, default=200_000)
    p.add_argument("--max_chars", type=int, default=32)
    p.add_argument("--max_aspect", type=float, default=20.0)
    p.add_argument("--shard_size", type=int, default=10_000)
    args = p.parse_args()

    require_raqm()
    fonts = load_fonts(args.font_dir)
    texts = [t for name in args.text_sources for t in load_texts(name, args.texts_per_source, args.seed)]
    corpus = [c for c in build_corpus(texts, eval_texts(), args.max_chars) if corpus_part(c) == args.part]
    if not corpus:
        raise SystemExit("empty corpus")
    rng = random.Random(args.seed)
    rng.shuffle(corpus)
    print(f"{len(corpus)} {args.part} lines from {len(texts)} labels; rendering {args.count}")

    args.out.mkdir(parents=True, exist_ok=True)
    rows, shard, made, tried = defaultdict(list), 0, 0, 0
    while made < args.count:
        if tried > 20 * args.count:
            raise SystemExit(f"only {made} renderable lines after {tried} tries; check fonts/corpus")
        text = corpus[tried % len(corpus)]
        tried += 1
        sample = render_sample(text, fonts, rng)
        if sample is None:
            continue
        image, meta = sample
        if image.width / image.height > args.max_aspect:
            continue
        rows["image"].append(image)
        rows["text"].append(text)
        for key in ("font", "kind", "style", "italic"):
            rows[key].append(meta[key])
        made += 1
        if len(rows["text"]) == args.shard_size or made == args.count:
            Dataset.from_dict(rows, features=FEATURES).to_parquet(str(args.out / f"shard-{shard:05d}.parquet"))
            print(f"  shard {shard}: {made}/{args.count}")
            shard += 1
            rows = defaultdict(list)


if __name__ == "__main__":
    main()
