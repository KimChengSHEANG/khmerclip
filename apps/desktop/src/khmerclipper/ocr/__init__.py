"""OCR engine selection. Engines are imported lazily (onnxruntime is slow to import)."""

import threading
from collections.abc import Callable
from enum import StrEnum

from PIL import Image


class Engine(StrEnum):
    TESSERACT = "tesseract"
    TROCR = "trocr"


def recognize(image: Image.Image, engine: "Engine | str", progress: Callable[[str], None] | None = None,
              cancel: threading.Event | None = None) -> str:
    engine = Engine(engine)
    if engine is Engine.TROCR:
        from khmerclip.ocr import trocr

        return trocr.recognize(image, progress=progress, cancel=cancel)
    from khmerclip.ocr import tesseract

    return tesseract.recognize(image)
