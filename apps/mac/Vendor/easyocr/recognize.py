#!/usr/bin/env python3
"""Run EasyOCR on an image and print recognized text to stdout.

Invoked by EasyOCREngine.swift. Models live under --resource-root:
  models/                  CRAFT detection + khmer.pth recognition weights
  khmer/user_network/      khmer.yaml + khmer.py matching the weights
"""
from __future__ import annotations

import argparse
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


def require_khmer_model(resource_root: Path, model: str) -> tuple[Path, Path]:
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
    return easyocr.Reader(
        ["km"],
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
