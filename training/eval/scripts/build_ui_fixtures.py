#!/usr/bin/env python3
"""Render deterministic Khmer UI strings on alternating light/dark canvases."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"
DEFAULT_FONT = ROOT / "fonts/NotoSansKhmer-Regular.ttf"
DEFAULT_STRINGS = ROOT / "ui_strings.txt"
MIXED_RE = re.compile(r"[A-Za-z0-9]")


def resolve_font() -> Path:
    font = Path(os.environ.get("KHMER_UI_FONT", DEFAULT_FONT)).expanduser()
    if not font.is_file():
        raise FileNotFoundError(
            f"Khmer UI font not found at {font}; set KHMER_UI_FONT or see README.md"
        )
    return font


def load_strings(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if not lines or any(not line for line in lines):
        raise ValueError(f"{path} must contain non-empty lines")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    parser.add_argument("--strings", type=Path, default=DEFAULT_STRINGS)
    parser.add_argument("--font-size", type=int, default=40)
    args = parser.parse_args()

    strings = load_strings(args.strings)
    font_path = resolve_font()
    font = ImageFont.truetype(str(font_path), args.font_size)
    ui_dir = args.fixtures / "ui"
    if ui_dir.exists():
        shutil.rmtree(ui_dir)
    ui_dir.mkdir(parents=True)

    rows = []
    for number, text in enumerate(strings, 1):
        dark = number % 2 == 0
        background = "#17191c" if dark else "#f3f5f7"
        panel = "#24272b" if dark else "#ffffff"
        foreground = "#f4f5f6" if dark else "#17191c"
        left, top = 30, 25
        probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        box = probe.textbbox((0, 0), text, font=font)
        text_width = box[2] - box[0]
        text_height = box[3] - box[1]
        width = max(360, text_width + 100)
        height = max(112, text_height + 64)

        image = Image.new("RGB", (width, height), background)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle(
            (16, 12, width - 16, height - 12), radius=14, fill=panel
        )
        draw.text(
            (left + 18, top - box[1] + 5),
            text,
            font=font,
            fill=foreground,
        )

        fixture_id = f"ui-{number:03d}"
        relative_image = f"ui/{fixture_id}.png"
        image.save(args.fixtures / relative_image, optimize=True)
        rows.append(
            {
                "id": fixture_id,
                "image": relative_image,
                "text": text,
                "subset": "mixed" if MIXED_RE.search(text) else "ui",
                "source": "ui_strings.txt",
                "theme": "dark" if dark else "light",
                "font": font_path.name,
            }
        )

    labels_path = args.fixtures / "labels.jsonl"
    existing = []
    if labels_path.is_file():
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if not row["image"].startswith("ui/"):
                    existing.append(row)
    with labels_path.open("w", encoding="utf-8") as labels:
        for row in [*existing, *rows]:
            labels.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} UI fixtures to {ui_dir}")


if __name__ == "__main__":
    main()
