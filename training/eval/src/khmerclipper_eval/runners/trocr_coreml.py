from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Sequence


_TIMING_LINE = re.compile(r"^ms=(\d+(?:\.\d+)?)$")


def recognize(image_path: Path, binary: Path) -> str:
    image_path = Path(image_path)
    binary = Path(binary)
    result = subprocess.run(
        [str(binary), "--engine", "trocr", str(image_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip() or f"OCRBench exited with {result.returncode}"
        )
    return result.stdout.strip()


def recognize_many(
    image_paths: Sequence[Path], binary: Path
) -> list[tuple[str, float | None]]:
    paths = [Path(path) for path in image_paths]
    if not paths:
        return []

    binary = Path(binary)
    result = subprocess.run(
        [str(binary), "--engine", "trocr", *(str(path) for path in paths)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip() or f"OCRBench exited with {result.returncode}"
        )

    texts = [line.strip() for line in result.stdout.splitlines()]
    timings = [
        float(match.group(1))
        for line in result.stderr.splitlines()
        if (match := _TIMING_LINE.fullmatch(line.strip()))
    ]
    if len(texts) != len(paths):
        raise RuntimeError(
            f"OCRBench returned {len(texts)} texts for {len(paths)} images"
        )

    return [
        (text, timings[index] if index < len(timings) else None)
        for index, text in enumerate(texts)
    ]
