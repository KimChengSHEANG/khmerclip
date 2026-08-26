#!/usr/bin/env python3
"""Run EasyOCR on an image and print recognized text to stdout.

Invoked by EasyOCREngine.swift. Models live under --resource-root:
  models/                  CRAFT detection + khmer.pth recognition weights
  khmer/user_network/      khmer.yaml + khmer.py matching the weights
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MODELS: dict[str, str] = {
    "khmer": "khmer",
}


def model_paths(resource_root: Path, model: str) -> tuple[Path, Path, Path, Path | None]:
    network_name = MODELS[model]
    user_network = resource_root / network_name / "user_network"
    weights = resource_root / "models" / f"{network_name}.pth"
    yaml = user_network / f"{network_name}.yaml"
    py = user_network / f"{network_name}.py"
    storage = resource_root / "models"
    return storage, user_network, weights, yaml if yaml.is_file() else None


def sync_yaml_from_charset(charset_path: Path, yaml_path: Path) -> None:
    chars = charset_path.read_text(encoding="utf-8").splitlines()
    char_list = "".join(chars)
    escaped = char_list.replace("\\", "\\\\").replace('"', '\\"')
    header = yaml_path.read_text(encoding="utf-8").split("character_list:", 1)[0]
    yaml_path.write_text(
        header.rstrip() + f'\ncharacter_list: "{escaped}"\n',
        encoding="utf-8",
    )


def migrate_legacy_layout(resource_root: Path, model: str) -> None:
    """Copy khmer/{pth,yaml} into models/ + user_network/ when only legacy paths exist."""
    network_name = MODELS[model]
    legacy_dir = resource_root / network_name
    legacy_pth = legacy_dir / f"{network_name}.pth"
    legacy_yaml = legacy_dir / f"{network_name}.yaml"
    storage, user_network, weights, yaml = model_paths(resource_root, model)
    py = user_network / f"{network_name}.py"
    if weights.is_file() and yaml is not None and py.is_file():
        return
    if legacy_pth.is_file():
        storage.mkdir(parents=True, exist_ok=True)
        if not weights.is_file():
            shutil.copy2(legacy_pth, weights)
    if legacy_yaml.is_file() and yaml is None:
        user_network.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy_yaml, user_network / f"{network_name}.yaml")
    charset = legacy_dir / "charset.txt"
    yaml_path = user_network / f"{network_name}.yaml"
    if charset.is_file() and yaml_path.is_file():
        sync_yaml_from_charset(charset, yaml_path)


def require_khmer_model(resource_root: Path, model: str) -> tuple[Path, Path]:
    migrate_legacy_layout(resource_root, model)
    storage, user_network, weights, yaml = model_paths(resource_root, model)
    missing: list[str] = []
    if yaml is None:
        missing.append(str(user_network / f"{MODELS[model]}.yaml"))
    if not (user_network / f"{MODELS[model]}.py").is_file():
        missing.append(str(user_network / f"{MODELS[model]}.py"))
    if not weights.is_file():
        missing.append(str(weights))
    if missing:
        msg = (
            "Khmer EasyOCR model is not installed. Add khmer.pth, khmer.yaml, and "
            "khmer.py, then run:\n"
            f"  python3 {resource_root / 'install_khmer_model.py'} /path/to/files\n"
            "Missing:\n  " + "\n  ".join(missing)
        )
        print(msg, file=sys.stderr)
        sys.exit(2)
    storage.mkdir(parents=True, exist_ok=True)
    user_network.mkdir(parents=True, exist_ok=True)
    return storage, user_network


def build_reader(resource_root: Path, model: str):
    import easyocr

    network_name = MODELS[model]
    storage, user_network = require_khmer_model(resource_root, model)
    # EasyOCR has no built-in km_char.txt; Khmer glyphs come from khmer.yaml
    # character_list. Reader lang must be a supported code (e.g. en).
    return easyocr.Reader(
        ["en"],
        gpu=False,
        recog_network=network_name,
        model_storage_directory=str(storage),
        user_network_directory=str(user_network),
        download_enabled=True,
        verbose=False,
    )


def recognize_image(resource_root: Path, model: str, image_path: Path) -> str:
    reader = build_reader(resource_root, model)
    lines = reader.readtext(str(image_path), detail=0, paragraph=True)
    if isinstance(lines, str):
        return lines.strip()
    return "\n".join(line.strip() for line in lines if line and line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, nargs="?", help="PNG/JPEG image path")
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(MODELS), default="khmer")
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Load models and exit (used by EasyOCREngine.preload)",
    )
    args = parser.parse_args()

    if args.warmup:
        build_reader(args.resource_root.resolve(), args.model)
        return

    if args.image is None:
        parser.error("image is required unless --warmup is set")

    text = recognize_image(args.resource_root.resolve(), args.model, args.image.resolve())
    sys.stdout.write(text)
    if text and not text.endswith("\n"):
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
