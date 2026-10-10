#!/usr/bin/env python3
"""Report label-length and aspect-ratio outliers per source before mixing it into training.

    python scripts/data_stats.py soyvitou sokheng hanuman kiteaether mrrtmob handwriting darayut siemreap
"""
import argparse
import random

from trocr_khmer import MAX_LABEL_LEN, load_ocr_datasets, load_trocr_processor


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("datasets", nargs="+")
    p.add_argument("--model_dir", default="experiments/checkpoint-3/final", help="Source of the Khmer tokenizer")
    p.add_argument("--sample", type=int, default=2000)
    p.add_argument("--aspects", type=float, nargs="+", default=[16, 20, 24, 32])
    args = p.parse_args()

    tokenizer = load_trocr_processor(args.model_dir).tokenizer
    for name in args.datasets:
        ds = load_ocr_datasets([name])
        sub = ds.select(random.Random(0).sample(range(len(ds)), min(args.sample, len(ds))))
        too_long = sum(len(tokenizer(t).input_ids) > MAX_LABEL_LEN for t in sub["text"]) / len(sub)
        aspects = sorted(im.width / im.height for im in sub["image"])
        drops = " ".join(f">{a:g}:{sum(x > a for x in aspects) / len(aspects):.1%}" for a in args.aspects)
        print(
            f"{name}\tn={len(ds)}\ttoo_long={too_long:.1%}\t"
            f"aspect p50={aspects[len(aspects) // 2]:.1f} p95={aspects[int(len(aspects) * 0.95)]:.1f}\t{drops}"
        )


if __name__ == "__main__":
    main()
