# KhmerClip

macOS menu-bar OCR for Khmer (and mixed Khmer/English) text — screen capture, file OCR, and on-device engines.

## Layout

| Path | What it is |
|------|------------|
| [`apps/mac`](apps/mac) | KhmerClip macOS app (Swift / Xcode) |
| [`training/easyocr`](training/easyocr) | EasyOCR-compatible Khmer recognition training |
| [`training/trocr`](training/trocr) | TrOCR Khmer fine-tuning |
| [`training/tesseract`](training/tesseract) | Tesseract `khm` LSTM fine-tune (CPU, Docker) |
| [`training/tesseract`](training/tesseract) | Tesseract `khm` LSTM fine-tune (CPU, Docker) |

Training outputs feed the app’s EasyOCR and TrOCR engines. Large build products and model weights are **not** in git — follow each subproject README / `Makefile` to generate them locally.

## Quick start (app)

See [`apps/mac/README.md`](apps/mac/README.md): install `xcodegen`, then `make run` from `apps/mac`.

## Training

- EasyOCR: [`training/easyocr/README.md`](training/easyocr/README.md)
- TrOCR: [`training/trocr/README.md`](training/trocr/README.md)
- Tesseract: [`training/tesseract/README.md`](training/tesseract/README.md)
