#!/usr/bin/env python3
"""Write deterministic SHA-256 checksums for the frozen fixture set."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    args = parser.parse_args()

    manifest = args.fixtures / "SHA256SUMS"
    files = sorted(
        path
        for path in args.fixtures.rglob("*")
        if path.is_file() and path != manifest
    )
    content = "".join(
        f"{sha256(path)}  {path.relative_to(args.fixtures).as_posix()}\n"
        for path in files
    )
    manifest.write_text(content, encoding="utf-8")
    print(f"Wrote {len(files)} checksums to {manifest}")


if __name__ == "__main__":
    main()
