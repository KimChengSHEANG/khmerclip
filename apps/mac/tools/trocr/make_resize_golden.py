"""Write the PIL-bilinear 384x384 RGB bytes that TrOCRPreprocessTests compares
Swift's resizedRGB against. Training (trocr_khmer.dataset.OCRDataset) resizes
with exactly this call, so matching it removes the train/inference pixel gap.

Usage: .trocr-venv/bin/python3 tools/trocr/make_resize_golden.py
"""
from pathlib import Path

from PIL import Image

MAC = Path(__file__).resolve().parents[2]
SRC = MAC.parents[1] / "training/eval/fixtures/printed/printed-001.png"
OUT = MAC / "Tests/Fixtures/ocr/printed-001.pil384.rgb"

img = Image.open(SRC).convert("RGB").resize((384, 384), Image.BILINEAR)
OUT.write_bytes(img.tobytes())
print("wrote", OUT, OUT.stat().st_size, "bytes")
