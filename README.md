# KhmerClip

macOS menu-bar OCR for Khmer (and mixed Khmer/English) text — screen capture, file OCR, and on-device engines.

## Download

Builds for macOS, Windows and Linux are on the [Releases](https://github.com/KimChengSHEANG/khmerclip/releases) page. They are unsigned:

- **macOS:** unzip, move KhmerClip to Applications, then right-click → Open the first time (or run `xattr -dr com.apple.quarantine /Applications/KhmerClip.app`).
- **Windows:** SmartScreen warns on first run: More info → Run anyway.
- **Linux:** `chmod +x KhmerClip-*.AppImage`, then run it.

## Layout

| Path | What it is |
|------|------------|
| [`apps/mac`](apps/mac) | KhmerClip macOS app (Swift / Xcode) |
| [`apps/desktop`](apps/desktop) | KhmerClip for Windows + Linux (Python / PySide6) |
| [`training/trocr`](training/trocr) | TrOCR Khmer fine-tuning |
| [`training/tesseract`](training/tesseract) | Tesseract `khm` LSTM fine-tune (CPU, Docker) |

Training outputs feed the app’s TrOCR and Tesseract engines. Large build products and model weights are **not** in git — follow each subproject README / `Makefile` to generate them locally.

## Quick start (app)

See [`apps/mac/README.md`](apps/mac/README.md): install `xcodegen`, then `make run` from `apps/mac`.

Windows/Linux: see [`apps/desktop/README.md`](apps/desktop/README.md) (`pip install -e '.[dev]'`, then `python -m khmerclip`).

## Training

- TrOCR: [`training/trocr/README.md`](training/trocr/README.md)
- Tesseract: [`training/tesseract/README.md`](training/tesseract/README.md)

## Project guide

Full overview — models, training, datasets, fonts, build/run, known issues: [`docs/PROJECT.md`](docs/PROJECT.md)

## License

MIT, see [LICENSE](LICENSE). Bundled third-party components keep their own licenses (Tesseract: Apache-2.0; Google Fonts: OFL).
