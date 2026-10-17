#!/usr/bin/env python3
"""Copy a hub OCR dataset to local parquet shards minus rows whose label shares a 16-char
shingle with any eval text (frozen fixtures, UI strings, screen fixtures), so it can be trained on
without leaking eval text. Registered afterwards as a parquet alias (data_sources.py).

    python scripts/filter_source.py LazyGreed/khmer_img2txt_200K --out cache/lazygreed/train
"""
import argparse
import json
from pathlib import Path

from datasets import load_dataset

EVAL = Path(__file__).resolve().parents[2] / "eval"


def shingles(s: str, n: int = 16) -> set[str]:
    s = s.replace("​", "")
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}


def eval_shingles() -> set[str]:
    texts = [json.loads(l)["text"] for l in (EVAL / "fixtures/labels.jsonl").read_text(encoding="utf-8").splitlines() if l]
    for name in ("ui_strings.txt", "en_ui_strings.txt"):
        texts += (EVAL / name).read_text(encoding="utf-8").splitlines()
    screen = EVAL / "screen_text.jsonl"
    if not screen.is_file():
        raise SystemExit(f"{screen} missing: build the screen text first (build_screen_fixtures.py text)")
    texts += [l for row in screen.read_text(encoding="utf-8").splitlines() if row for l in json.loads(row)["lines"]]
    return set().union(*(shingles(t) for t in texts if t.strip()))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("hub_id")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--text_col", default="text")
    args = p.parse_args()
    banned = eval_shingles()
    ds = load_dataset(args.hub_id, split=args.split)
    kept = ds.filter(lambda t: not (shingles(t) & banned), input_columns=[args.text_col], num_proc=8)
    print(f"kept {len(kept)} / {len(ds)} rows")
    args.out.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(kept), 20_000):
        kept.select(range(i, min(i + 20_000, len(kept)))).to_parquet(str(args.out / f"shard-{i // 20_000:05d}.parquet"))


if __name__ == "__main__":
    main()
