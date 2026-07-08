#!/usr/bin/env python3
"""Copy a trained Khmer EasyOCR recognition model into Vendor/easyocr/khmer/.

EasyOCR custom models need three files sharing the same base name (khmer):
  khmer.pth  -> models/
  khmer.yaml -> khmer/user_network/
  khmer.py   -> khmer/user_network/

charset.txt (one character per line) must match the trained weights; when
present it is copied and used to regenerate khmer.yaml's character_list.

Usage:
  python3 install_khmer_model.py /path/to/dir/containing/khmer.{pth,yaml,py}
  python3 install_khmer_model.py khmer.pth khmer.yaml khmer.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_NAME = "khmer"
WEIGHTS_DIR = ROOT / "models"
NETWORK_DIR = ROOT / MODEL_NAME / "user_network"
CHARSET_FILE = ROOT / MODEL_NAME / "charset.txt"


def yaml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def sync_yaml_from_charset(charset_path: Path, yaml_path: Path) -> None:
    """Rewrite khmer.yaml character_list from charset.txt (one char per line)."""
    chars = charset_path.read_text(encoding="utf-8").splitlines()
    char_list = "".join(chars)
    header = yaml_path.read_text(encoding="utf-8").split("character_list:", 1)[0]
    yaml_path.write_text(
        header.rstrip()
        + f'\ncharacter_list: "{yaml_escape(char_list)}"\n',
        encoding="utf-8",
    )


def install(pth: Path, yaml: Path, py: Path) -> None:
    for path in (pth, yaml, py):
        if not path.is_file():
            sys.exit(f"Not found: {path}")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    NETWORK_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pth, WEIGHTS_DIR / f"{MODEL_NAME}.pth")
    shutil.copy2(yaml, NETWORK_DIR / f"{MODEL_NAME}.yaml")
    shutil.copy2(py, NETWORK_DIR / f"{MODEL_NAME}.py")
    charset_src = yaml.parent / "charset.txt"
    if not charset_src.is_file():
        charset_src = yaml.parent.parent / "charset.txt"
    if charset_src.is_file():
        shutil.copy2(charset_src, CHARSET_FILE)
    if CHARSET_FILE.is_file():
        sync_yaml_from_charset(CHARSET_FILE, NETWORK_DIR / f"{MODEL_NAME}.yaml")
    print(f"Installed Khmer EasyOCR model into {ROOT / MODEL_NAME}/")


def install_from_dir(src: Path) -> None:
    pth = src / f"{MODEL_NAME}.pth"
    yaml = src / f"{MODEL_NAME}.yaml"
    py = src / f"{MODEL_NAME}.py"
    # Legacy layout: weights + yaml dropped directly under khmer/ (no user_network/).
    if not py.is_file():
        py = ROOT / MODEL_NAME / "user_network" / f"{MODEL_NAME}.py"
    if not yaml.is_file():
        yaml = ROOT / MODEL_NAME / "user_network" / f"{MODEL_NAME}.yaml"
    missing = [p.name for p in (pth, yaml, py) if not p.is_file()]
    if missing:
        sys.exit(f"Missing in {src}: {', '.join(missing)}")
    install(pth, yaml, py)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        help="Source directory, or three explicit paths: khmer.pth khmer.yaml khmer.py",
    )
    args = parser.parse_args()
    if len(args.paths) == 1:
        install_from_dir(Path(args.paths[0]))
    elif len(args.paths) == 3:
        install(Path(args.paths[0]), Path(args.paths[1]), Path(args.paths[2]))
    else:
        parser.error("Pass one source directory or exactly three file paths")


if __name__ == "__main__":
    main()
