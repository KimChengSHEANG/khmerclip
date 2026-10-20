#!/usr/bin/env python3
"""Add real screen captures to fixtures_real/ with TrOCR-prefilled draft labels.

    python scripts/draft_real_fixtures.py ~/Desktop/khmer-captures

Each new image (deduplicated by content) is copied to fixtures_real/real/real-NNN.png
and appended to fixtures_real/labels.jsonl with "verified": false. Correct the text,
then set "verified": true; the loader skips unverified rows. fixtures_real/review.md
shows every image above its current label (open it in a Markdown preview).

Multi-line captures are fine: the label is the lines joined with single spaces,
matching what OCRBench prints.

The app's "Flag as Wrong" folder works as src too (Mac: ~/Library/Application Support/
KhmerClipper/feedback, desktop: History > Feedback Folder). Each flagged <id>.png has an
<id>.json with what the app read; that text becomes the draft (no OCRBench needed) and
the engine/app version are kept on the row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REPO = ROOT.parents[1]
DEFAULT_OUT = ROOT / "fixtures_real"
DEFAULT_BIN = REPO / "apps/mac/build/Build/Products/Release/OCRBench"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".heic", ".tif", ".tiff", ".bmp"}

Recognizer = Callable[[Sequence[Path]], list[str]]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_rows(labels: Path) -> list[dict]:
    if not labels.is_file():
        return []
    return [json.loads(line) for line in labels.read_text(encoding="utf-8").splitlines() if line.strip()]


def _app_capture(path: Path) -> dict | None:
    """The app's sidecar for a flagged capture, if there is one with OCR text."""
    try:
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) and isinstance(meta.get("ocr_text"), str) else None


def _space_join(text: str) -> str:
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def add_images(src: Path, out: Path, recognize: Recognizer) -> list[dict]:
    """Copy new images from src into out/real/, prefill labels, return the new rows."""
    labels = out / "labels.jsonl"
    rows = _read_rows(labels)
    seen = {row["sha256"] for row in rows}
    (out / "real").mkdir(parents=True, exist_ok=True)

    sources = sorted(p for p in Path(src).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    new_rows, new_paths = [], []
    n = len(rows)
    for path in sources:
        digest = _sha(path)
        if digest in seen:
            continue
        seen.add(digest)
        n += 1
        rel = f"real/real-{n:03d}.png"
        Image.open(path).convert("RGB").save(out / rel)
        row = {"id": f"real-{n:03d}", "image": rel, "text": "", "subset": "real",
               "source": path.name, "sha256": digest, "verified": False}
        if meta := _app_capture(path):
            row["text"] = _space_join(meta["ocr_text"])
            row["app"] = {k: meta[k] for k in ("engine", "model_version", "app_version") if k in meta}
        else:
            new_paths.append(out / rel)
        new_rows.append(row)

    if new_paths:
        to_fill = [r for r in new_rows if "app" not in r]
        for row, text in zip(to_fill, recognize(new_paths), strict=True):
            row["text"] = text
    if new_rows:
        with labels.open("a", encoding="utf-8") as f:
            for row in new_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_review(out, rows + new_rows)
    return new_rows


def write_review(out: Path, rows: list[dict]) -> None:
    lines = ["# Real fixtures review", "",
             "Edit the text in `labels.jsonl`, then set `\"verified\": true`. "
             "Rerun the draft script to refresh this page.", ""]
    for row in rows:
        mark = "✅" if row.get("verified") else "⬜ draft"
        app = " · app: " + "/".join(str(v) for v in row["app"].values()) if row.get("app") else ""
        lines += [f"## {row['id']} {mark}{app}", "", f"![{row['id']}]({row['image']})", "",
                  "```text", row["text"], "```", ""]
    (out / "review.md").write_text("\n".join(lines), encoding="utf-8")


def coreml_recognizer(binary: Path) -> Recognizer:
    from khmerclipper_eval.runners import trocr_coreml

    os.environ.setdefault("KHMERCLIPPER_VENDOR", str(REPO / "apps/mac/Vendor"))

    def recognize(paths: Sequence[Path]) -> list[str]:
        return [text for text, _ms in trocr_coreml.recognize_many(paths, binary)]

    return recognize


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", type=Path, help="folder of screenshots")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--trocr-coreml-bin", type=Path, default=DEFAULT_BIN)
    args = p.parse_args()

    def recognize(paths: Sequence[Path]) -> list[str]:  # only needed for images without an app sidecar
        if not args.trocr_coreml_bin.is_file():
            raise SystemExit(f"OCRBench not found at {args.trocr_coreml_bin}; run `make -C apps/mac ocrbench`")
        return coreml_recognizer(args.trocr_coreml_bin)(paths)

    new = add_images(args.src, args.out, recognize)
    total = len(_read_rows(args.out / "labels.jsonl"))
    print(f"added {len(new)} images ({total} total) -> {args.out / 'labels.jsonl'}")
    print(f"review: {args.out / 'review.md'}")


if __name__ == "__main__":
    main()
