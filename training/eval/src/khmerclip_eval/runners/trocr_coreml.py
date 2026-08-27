from __future__ import annotations

import subprocess
from pathlib import Path


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
