#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from khmerclipper_eval.cer import character_error_rate
from khmerclipper_eval.fixtures import load_fixtures
from khmerclipper_eval.runners import tesseract as tess_runner
from khmerclipper_eval.summary import aggregate, passes_gate

REPO = ROOT.parents[1]
DEFAULT_FIXTURES = ROOT / "fixtures"
DEFAULT_TESS = REPO / "apps/mac/Vendor/tesseract"


def parse_variant(spec: str) -> dict:
    """"split_wide_at=10,crop=1" -> {"split_wide_at": 10.0, "crop": True}."""
    out: dict = {}
    for part in filter(None, spec.split(",")):
        key, value = part.split("=")
        out[key] = bool(int(value)) if key in ("crop", "normalize", "split_touching", "cycle_cap") else float(value)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    p.add_argument(
        "--engine",
        action="append",
        required=True,
        choices=["tesseract", "tesseract-ft", "trocr-hf", "trocr-coreml", "trocr-pipeline", "trocr-desktop"],
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, default=Path("bench_results.csv"))
    p.add_argument("--tesseract-root", type=Path, default=DEFAULT_TESS)
    p.add_argument("--tesseract-ft-root", type=Path, default=None)
    p.add_argument("--tesseract-ft-lang", default="khm+eng")
    p.add_argument("--trocr-model-dir", type=Path, default=None)
    p.add_argument("--trocr-coreml-bin", type=Path, default=None)
    p.add_argument(
        "--variant",
        default="",
        help="trocr-pipeline flags, e.g. split_wide_at=10,beam_gate=0.5,crop=1,normalize=1",
    )
    p.add_argument("--compare-to", default="tesseract")
    p.add_argument("--candidate", default=None)
    args = p.parse_args()

    items = load_fixtures(args.fixtures)
    if args.limit is not None:
        items = items[: args.limit]

    rows: list[dict] = []
    for engine in args.engine:
        if engine == "trocr-coreml":
            from khmerclipper_eval.runners import trocr_coreml

            if not args.trocr_coreml_bin:
                raise SystemExit("--trocr-coreml-bin required")
            t0 = time.perf_counter()
            predictions = trocr_coreml.recognize_many(
                [item.image_path for item in items], args.trocr_coreml_bin
            )
            batch_wall_ms = (time.perf_counter() - t0) * 1000.0
            fallback_ms = batch_wall_ms / len(items) if items else 0.0
            for item, (pred, measured_ms) in zip(items, predictions, strict=True):
                rows.append(
                    {
                        "engine": engine,
                        "id": item.id,
                        "subset": item.subset,
                        "cer": character_error_rate(pred, item.text),
                        "ms": measured_ms if measured_ms is not None else fallback_ms,
                        "pred": pred,
                        "ref": item.text,
                    }
                )
            continue

        if engine == "trocr-desktop":
            # The shipped desktop ONNX path (apps/desktop), CPU provider, Latin fallback included.
            from PIL import Image

            sys.path.insert(0, str(REPO / "apps/desktop/src"))
            from khmerclipper.ocr import trocr as desktop_trocr

            if not args.trocr_model_dir:
                raise SystemExit("--trocr-model-dir required")
            session = desktop_trocr.load_session(args.trocr_model_dir)
            if items:  # warm-up: first ONNX run allocates arenas
                desktop_trocr.recognize_with_session(session, Image.open(items[0].image_path))

        for item in items:
            t0 = time.perf_counter()
            if engine == "tesseract":
                pred = tess_runner.recognize(item.image_path, args.tesseract_root)
            elif engine == "tesseract-ft":
                if not args.tesseract_ft_root:
                    raise SystemExit("--tesseract-ft-root required")
                pred = tess_runner.recognize(
                    item.image_path, args.tesseract_ft_root, args.tesseract_ft_lang
                )
            elif engine == "trocr-pipeline":
                from PIL import Image

                from khmerclipper_eval import pipeline

                if not args.trocr_model_dir:
                    raise SystemExit("--trocr-model-dir required")
                pred = pipeline.recognize(
                    Image.open(item.image_path), args.trocr_model_dir, **parse_variant(args.variant)
                )
            elif engine == "trocr-hf":
                from khmerclipper_eval.runners import trocr_hf

                if not args.trocr_model_dir:
                    raise SystemExit("--trocr-model-dir required")
                pred = trocr_hf.recognize(item.image_path, args.trocr_model_dir)
            elif engine == "trocr-desktop":
                # The app puts one line per row; labels join lines with a space, as OCRBench does.
                pred = desktop_trocr.recognize_with_session(session, Image.open(item.image_path)).replace("\n", " ")
            else:
                raise AssertionError(f"unhandled engine: {engine}")
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
            f"median_ms={stats['median_ms']:.1f} mixed_cer={stats['mixed_cer']:.4f} "
            f"ui_cer={stats['ui_cer']:.4f}"
        )
    if args.candidate and args.compare_to in by and args.candidate in by:
        ok, reasons = passes_gate(by[args.candidate], by[args.compare_to])
        print("PASS" if ok else "FAIL")
        for r in reasons:
            print(f"  {r}")


if __name__ == "__main__":
    main()
