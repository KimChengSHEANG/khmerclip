"""Entry point: `khmerclipper` (tray app), `khmerclipper --capture`, `khmerclipper ocr IMAGE`."""

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="khmerclipper", description="Khmer OCR screen capture")
    parser.add_argument("--capture", action="store_true",
                        help="start a capture (signals the running instance; bind this to a desktop shortcut on Wayland)")
    sub = parser.add_subparsers(dest="command")
    ocr = sub.add_parser("ocr", help="OCR an image file and print the text")
    ocr.add_argument("image", type=Path)
    ocr.add_argument("--engine", choices=["tesseract", "trocr"], default="tesseract")
    ocr.add_argument("--flatten", action="store_true", help="join lines into one paragraph")
    ocr.add_argument("-o", "--output", type=Path, help="write UTF-8 text here instead of stdout")
    return parser


def _err(message: object) -> None:
    if sys.stderr:  # None in the windowed (no console) Windows build
        print(f"khmerclipper: {message}", file=sys.stderr)


def _ocr(args: argparse.Namespace) -> int:
    from PIL import Image

    import khmerclipper.ocr as ocr
    from khmerclipper.errors import KhmerClipperError
    from khmerclipper.formatter import apply

    try:
        with Image.open(args.image) as img:
            img.load()
            text = ocr.recognize(img, args.engine)
    except (OSError, KhmerClipperError) as e:
        _err(e)
        return 1
    text = apply(text, preserve_linebreaks=not args.flatten)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    elif sys.stdout:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to a legacy code page
        print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "ocr":
        return _ocr(args)
    from khmerclipper.ui.app import run_gui  # Qt is imported only for the GUI

    return run_gui(capture=args.capture)


if __name__ == "__main__":
    raise SystemExit(main())
