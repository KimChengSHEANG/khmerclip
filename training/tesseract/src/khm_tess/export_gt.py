"""Export cached Hugging Face Khmer line datasets to tesstrain ground truth.

Writes ``<out>/<alias>-<index:08d>.png`` + ``.gt.txt`` pairs and ``export_stats.json``.
Reads parquet straight from the hub cache so test splits can be excluded by file name:
SoyVitou's dataset card lumps testset.parquet (our eval fixtures' source) into ``train``.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import re
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

REPO = Path(__file__).resolve().parents[4]

# alias -> (hub cache dir, glob relative to the snapshot root, text column).
# Test splits are deliberately absent.
SOURCES = {
    "soyvitou-train": ("datasets--SoyVitou--62k-images-khmer-printed-dataset", "trainset.parquet", "text"),
    "soyvitou-valid": ("datasets--SoyVitou--62k-images-khmer-printed-dataset", "validset.parquet", "text"),
    "sokheng": ("datasets--Sokheng--khmer-synthetic-ocr-v1-100k", "data/train-*.parquet", "text"),
    "hanuman": ("datasets--seanghay--khmer-hanuman-100k", "data/train-*.parquet", "text"),
    "kiteaether": ("datasets--KiteAether--c_khmer_gemma_ocr_train", "data/train-*.parquet", "text"),
    "mrrtmob": ("datasets--mrrtmob--khmer_english_ocr_image_line", "data/train-*.parquet", "text"),
    "darayut": ("datasets--Darayut--khmer-document-synthetic-low-res", "data/train-*.parquet", "label"),
    "siemreap": ("datasets--KimkosalYon--khmer-ocr-200k-siemreap-arial", "data/train-*.parquet", "text"),
}
DEFAULT_SOURCES = [
    "soyvitou-train",
    "soyvitou-valid:2000",
    "sokheng",
    "hanuman",
    "kiteaether",
    "darayut",
    "siemreap:100000",
    "mrrtmob:200000",
]
LATIN_RE = re.compile(r"[A-Za-z]")
MIN_SIDE = 8


def clean_label(text: str | None) -> str | None:
    # ponytail: strip only; CER is on raw text, so no NFC / space collapsing.
    if text is None:
        return None
    label = text.strip()
    if not label or any(c in label for c in "\n\r\t"):
        return None
    return label


def load_blocklist(labels_jsonl: Path, ui_strings: Path) -> set[str]:
    block = set()
    for line in labels_jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            block.add(json.loads(line)["text"].strip())
    for line in ui_strings.read_text(encoding="utf-8").splitlines():
        if line.strip():
            block.add(line.strip())
    return block


def source_files(hub: Path, alias: str) -> list[Path]:
    repo, pattern, _ = SOURCES[alias]
    files = sorted(Path(p) for p in glob.glob(str(hub / repo / "snapshots" / "*" / pattern)))
    if not files:
        raise FileNotFoundError(f"no parquet for {alias} under {hub / repo}")
    return files


def iter_rows(
    files: list[Path], limit: int | None, text_col: str = "text"
) -> Iterator[tuple[int, bytes | None, str | None]]:
    """Yield (global_index, image_bytes, text); with ``limit``, every Nth row spread over all files."""
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    stride = 1 if limit is None else max(1, total // limit)
    index = taken = 0
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=256, columns=["image", text_col]):
            for image, text in zip(batch.column("image").to_pylist(), batch.column(text_col).to_pylist()):
                if index % stride == 0:
                    yield index, (image or {}).get("bytes"), text
                    taken += 1
                    if limit is not None and taken >= limit:
                        return
                index += 1


def _write_image(data: bytes | None, path: Path) -> bool:
    if not data:
        return False
    try:
        with Image.open(io.BytesIO(data)) as im:
            if min(im.size) < MIN_SIDE:
                return False
            if im.mode in ("RGBA", "LA", "P") or "transparency" in im.info:
                # Flatten onto white: a bare convert("L") turns transparent pixels black,
                # which would silently blank out black-on-transparent text lines.
                rgba = im.convert("RGBA")
                im = Image.new("RGBA", rgba.size, "white")
                im.alpha_composite(rgba)
            im.convert("L").save(path)
    except OSError:  # PIL.UnidentifiedImageError subclasses OSError
        path.unlink(missing_ok=True)
        return False
    return True


def export_source(
    alias: str, files: list[Path], out_dir: Path, blocklist: set[str], limit: int | None = None
) -> dict[str, int]:
    text_col = SOURCES[alias][2] if alias in SOURCES else "text"
    stats = dict(kept=0, dropped_label=0, dropped_blocklist=0, dropped_image=0, with_space=0, with_latin=0)
    for index, data, text in iter_rows(files, limit, text_col):
        label = clean_label(text)
        if label is None:
            stats["dropped_label"] += 1
            continue
        if label in blocklist:
            stats["dropped_blocklist"] += 1
            continue
        png = out_dir / f"{alias}-{index:08d}.png"
        gt = out_dir / f"{alias}-{index:08d}.gt.txt"
        if not (png.exists() and gt.exists()):  # gt is written last, so a pair implies a full write
            if not _write_image(data, png):
                stats["dropped_image"] += 1
                continue
            gt.write_text(label + "\n", encoding="utf-8")
        stats["kept"] += 1
        stats["with_space"] += " " in label
        stats["with_latin"] += bool(LATIN_RE.search(label))
    return stats


def parse_source(spec: str) -> tuple[str, int | None]:
    alias, _, limit = spec.partition(":")
    if alias not in SOURCES:
        raise SystemExit(f"unknown source {alias!r}; choose from {sorted(SOURCES)}")
    return alias, int(limit) if limit else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hub", type=Path, default=Path.home() / ".cache/huggingface/hub")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fixtures-labels", type=Path, default=REPO / "training/eval/fixtures/labels.jsonl")
    p.add_argument("--ui-strings", type=Path, default=REPO / "training/eval/ui_strings.txt")
    p.add_argument("--source", action="append", help="alias[:limit]; repeatable")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    blocklist = load_blocklist(args.fixtures_labels, args.ui_strings)
    all_stats = {}
    for spec in args.source or DEFAULT_SOURCES:
        alias, limit = parse_source(spec)
        all_stats[alias] = export_source(alias, source_files(args.hub, alias), args.out, blocklist, limit)
        print(alias, json.dumps(all_stats[alias]), flush=True)
    (args.out / "export_stats.json").write_text(json.dumps(all_stats, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
