# training/eval/src/khmerclip_eval/fixtures.py
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

# "real" = app captures of real screens (fixtures_real/), "screen"/"screen_long" =
# Chrome-rendered clips (fixtures_screen/), each kept in a separate root
# so the frozen 200-image set stays comparable across runs.
ALLOWED_SUBSETS = frozenset({"printed", "ui", "mixed", "real", "screen", "screen_long"})


@dataclass(frozen=True)
class FixtureItem:
    id: str
    image_path: Path
    text: str
    subset: str


def load_fixtures(root: Path) -> list[FixtureItem]:
    root = Path(root)
    labels_path = root / "labels.jsonl"
    if not labels_path.is_file():
        raise FileNotFoundError(f"missing labels.jsonl under {root}")
    items: list[FixtureItem] = []
    unverified = 0
    with labels_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in ("id", "image", "text", "subset"):
                if key not in row:
                    raise ValueError(f"labels.jsonl:{line_no} missing {key}")
            if row["subset"] not in ALLOWED_SUBSETS:
                raise ValueError(f"labels.jsonl:{line_no} bad subset {row['subset']!r}")
            # Draft rows hold the model's own prefill; scoring them would report a fake 0 CER.
            if row.get("verified") is False:
                unverified += 1
                continue
            image_path = root / row["image"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            items.append(
                FixtureItem(
                    id=str(row["id"]),
                    image_path=image_path,
                    text=str(row["text"]),
                    subset=str(row["subset"]),
                )
            )
    if unverified:
        warnings.warn(f"skipped {unverified} unverified rows in {labels_path}", stacklevel=2)
    if not items:
        raise ValueError(f"no verified fixtures in {labels_path}")
    return items
