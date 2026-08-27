from __future__ import annotations

import subprocess
from pathlib import Path


def recognize(image_path: Path, resource_root: Path, python_bin: Path) -> str:
    resource_root = Path(resource_root)
    image_path = Path(image_path)
    python_bin = Path(python_bin)
    script = resource_root / "recognize.py"
    result = subprocess.run(
        [
            str(python_bin),
            str(script),
            "--resource-root",
            str(resource_root),
            "--model",
            "khmer",
            str(image_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip() or f"easyocr exited with {result.returncode}"
        )
    return result.stdout.strip()
