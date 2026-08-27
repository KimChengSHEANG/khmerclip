#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from khmerclip_eval.cer import character_error_rate
from khmerclip_eval.fixtures import load_fixtures
from khmerclip_eval.runners import easyocr as easyocr_runner
from khmerclip_eval.runners import tesseract as tess_runner
from khmerclip_eval.summary import aggregate, passes_gate

REPO = ROOT.parents[1]
DEFAULT_FIXTURES = ROOT / "fixtures"
DEFAULT_TESS = REPO / "apps/mac/Vendor/tesseract"
DEFAULT_EASYOCR = REPO / "apps/mac/Vendor/easyocr"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    p.add_argument(
        "--engine",
        action="append",
        required=True,
        choices=["tesseract", "easyocr", "trocr-hf", "trocr-coreml"],
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, default=Path("bench_results.csv"))
    p.add_argument("--tesseract-root", type=Path, default=DEFAULT_TESS)
    p.add_argument("--easyocr-root", type=Path, default=DEFAULT_EASYOCR)
    p.add_argument(
        "--easyocr-python",
        type=Path,
        default=DEFAULT_EASYOCR / "venv/bin/python3",
    )
    p.add_argument("--trocr-model-dir", type=Path, default=None)
    p.add_argument("--trocr-coreml-bin", type=Path, default=None)
    p.add_argument("--compare-to", default="tesseract")
    p.add_argument("--candidate", default=None)
    args = p.parse_args()

    items = load_fixtures(args.fixtures)
    if args.limit is not None:
        items = items[: args.limit]

    rows: list[dict] = []
    for engine in args.engine:
        for item in items:
            t0 = time.perf_counter()
            if engine == "tesseract":
                pred = tess_runner.recognize(item.image_path, args.tesseract_root)
            elif engine == "easyocr":
                pred = easyocr_runner.recognize(
                    item.image_path, args.easyocr_root, args.easyocr_python
                )
            elif engine == "trocr-hf":
                from khmerclip_eval.runners import trocr_hf

                if not args.trocr_model_dir:
                    raise SystemExit("--trocr-model-dir required")
                pred = trocr_hf.recognize(item.image_path, args.trocr_model_dir)
            else:
                from khmerclip_eval.runners import trocr_coreml

                if not args.trocr_coreml_bin:
                    raise SystemExit("--trocr-coreml-bin required")
                pred = trocr_coreml.recognize(item.image_path, args.trocr_coreml_bin)
            ms = (time.perf_counter() - t0) * 1000.0
            rows.append(
                {
                    "engine": engine,
                    "id": item.id,
                    "subset": item.subset,
                    "cer": character_error_rate(pred, item.text),
                    "ms": ms,
                    "pred": pred,
                    "ref": item.text,
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["engine", "id", "subset", "cer", "ms", "pred", "ref"]
        )
        w.writeheader()
        w.writerows(rows)

    by = aggregate(rows)
    for engine, stats in by.items():
        print(
            f"{engine}: n={stats['n']} cer={stats['cer']:.4f} "
            f"median_ms={stats['median_ms']:.1f} mixed_cer={stats['mixed_cer']:.4f}"
        )
    if args.candidate and args.compare_to in by and args.candidate in by:
        ok, reasons = passes_gate(by[args.candidate], by[args.compare_to])
        print("PASS" if ok else "FAIL")
        for r in reasons:
            print(f"  {r}")


if __name__ == "__main__":
    main()
