"""Khmer EasyOCR: train & evaluate an EasyOCR-compatible recognizer for Khmer.

Public modules:
    khmer_ocr.config      - config + charset (de)serialisation helpers
    khmer_ocr.data        - Hugging Face dataset -> LMDB conversion
    khmer_ocr.evaluate    - CER / WER / accuracy evaluation
    khmer_ocr.packaging   - export a checkpoint for the `easyocr` package
    khmer_ocr.trainer     - vendored EasyOCR-style recognition trainer
"""

__version__ = "0.1.0"
