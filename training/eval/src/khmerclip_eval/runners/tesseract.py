from __future__ import annotations

import subprocess
from pathlib import Path


def recognize(image_path: Path, vendor_root: Path) -> str:
    vendor_root = Path(vendor_root)
    image_path = Path(image_path)
    binary = vendor_root / "bin/tesseract"
    tessdata = vendor_root / "tessdata"
    result = subprocess.run(
        [
            str(binary),
            str(image_path),
            "stdout",
            "--tessdata-dir",
            str(tessdata),
            "-l",
            "khm+eng",
            "--psm",
            "6",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip() or f"tesseract exited with {result.returncode}"
        )
    return result.stdout.strip()
