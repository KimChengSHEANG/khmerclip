#!/usr/bin/env python3
"""Download the OFL Google Fonts Khmer files used by the synthetic renderer into cache/fonts.

    python scripts/fetch_fonts.py
"""
import argparse
import urllib.parse
import urllib.request
from pathlib import Path

from trocr_khmer.synth import GOOGLE_KHMER_FONTS

BASE = "https://raw.githubusercontent.com/google/fonts/main/"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("cache/fonts"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for repo_path in GOOGLE_KHMER_FONTS:
        dest = args.out / Path(repo_path).name
        if dest.is_file():
            continue
        urllib.request.urlretrieve(BASE + urllib.parse.quote(repo_path), dest)
        print(f"fetched {dest.name}")
    print(f"{len(GOOGLE_KHMER_FONTS)} fonts in {args.out}")


if __name__ == "__main__":
    main()
