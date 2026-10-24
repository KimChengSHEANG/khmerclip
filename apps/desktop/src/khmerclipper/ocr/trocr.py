"""TrOCR via onnxruntime: greedy, no-KV-cache decode per text line (same loop, caps and
Latin-line Tesseract fallback as apps/mac/Sources/OCR/TrOCREngine.swift)."""

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from khmerclipper.errors import Cancelled, EngineUnavailable, KhmerClipperError
from khmerclipper.ocr import trocr_models
from khmerclipper.ocr.preprocess import content_crop, line_strips, pixel_values, split_wide
from khmerclipper.ocr.tokenizer import TrOCRTokenizer

# Total decoded ids incl. the start token; matches training MAX_LABEL_LEN and the mac
# TrOCREngine.maxLength (64 cut off long Khmer lines).
MAX_LENGTH = 128
# Greedy decode can lock onto one token; forbid extending a run past MAX_RUN identical ids
# (mac TrOCREngine.maxRun: greedy CER 0.172 -> 0.056). Labels never repeat a token > 3x.
MAX_RUN = 4


def blocked_token(ids: list[int], max_run: int = MAX_RUN) -> int | None:
    tail = ids[-max_run:]
    return tail[-1] if len(ids) > max_run and len(set(tail)) == 1 else None

# Lowest Tesseract word confidence for it to replace a Latin-only TrOCR line.
# Measured (apps/mac TrOCREngine.tesseractMinConfidence): clean UI strips >= 75.9,
# all correct; scanned eval lines Tesseract gets wrong <= 62.7.
TESSERACT_MIN_CONFIDENCE = 70.0


def contains_khmer(text: str) -> bool:
    """Khmer (U+1780-17FF) or Khmer Symbols (U+19E0-19FF)."""
    return any("\u1780" <= c <= "\u17ff" or "\u19e0" <= c <= "\u19ff" for c in text)


def latin_fallback(strip: Image.Image, line: str) -> str:
    """The Khmer fine-tune misspells clean English UI text ("Warrantly", "PriceSnbox")
    that Tesseract reads perfectly, but Tesseract is worse on low-res scanned English.
    So a Latin-only line switches to Tesseract only when every word is confident.
    Lines with any Khmer stay with TrOCR. ponytail: one tesseract run per Latin-only line."""
    if contains_khmer(line):
        return line
    from khmerclipper.ocr import tesseract

    try:
        text, min_conf = tesseract.recognize_line(strip)
    except KhmerClipperError:
        return line
    return text if text and min_conf >= TESSERACT_MIN_CONFIDENCE else line


@dataclass
class Session:
    encoder: Any
    decoder: Any
    tokenizer: TrOCRTokenizer


_cache: dict[Path, Session] = {}
_lock = threading.Lock()


def load_session(model_dir: Path) -> Session:
    with _lock:
        if model_dir not in _cache:
            import onnxruntime as ort

            try:
                _cache[model_dir] = Session(
                    ort.InferenceSession(str(model_dir / "encoder.onnx"), providers=["CPUExecutionProvider"]),
                    ort.InferenceSession(str(model_dir / "decoder.onnx"), providers=["CPUExecutionProvider"]),
                    TrOCRTokenizer.from_file(model_dir / "tokenizer.json"),
                )
            except Exception as e:  # onnxruntime raises several unrelated types for bad files
                raise EngineUnavailable(
                    f"TrOCR model files are unreadable ({e}). Delete {model_dir} and download again.") from e
        return _cache[model_dir]


def installed_model_dir() -> Path:
    if override := os.environ.get("KHMERCLIPPER_TROCR_DIR"):
        return Path(override)
    manifest = trocr_models.load_manifest()
    if not trocr_models.is_installed(manifest):
        raise EngineUnavailable("The TrOCR model isn't downloaded yet.")
    return trocr_models.model_dir(manifest)


def preload() -> None:
    """Warm the ONNX sessions while the user drags a selection. Errors surface on recognize()."""
    try:
        load_session(installed_model_dir())
    except KhmerClipperError:
        pass


def recognize(image: Image.Image, progress: Callable[[str], None] | None = None,
              cancel: threading.Event | None = None) -> str:
    if progress:
        progress("Loading model…")
    return recognize_with_session(load_session(installed_model_dir()), image, progress, cancel)


def recognize_with_session(session, image: Image.Image, progress: Callable[[str], None] | None = None,
                           cancel: threading.Event | None = None) -> str:
    tok = session.tokenizer
    lines: list[str] = []
    done = 0
    for strip in line_strips(image):
        # A long line squashed into 384x384 loses its thin marks; decode it in pieces.
        text = ""
        for sep, piece in split_wide(strip):
            # Cross-attention K/V once per piece; self-attention K/V carried step to step.
            cross_k, cross_v = session.encoder.run(None, {"pixel_values": pixel_values(piece)})
            past_k = past_v = np.zeros((*cross_k.shape[:3], 0, cross_k.shape[4]), np.float32)
            ids = [tok.decoder_start_id]
            while len(ids) < MAX_LENGTH:
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                logits, past_k, past_v = session.decoder.run(None, {
                    "input_ids": np.array([ids[-1:]], dtype=np.int64),
                    "position": np.array([len(ids) - 1], dtype=np.int64),
                    "cross_k": cross_k, "cross_v": cross_v, "past_k": past_k, "past_v": past_v})
                step = logits[0, -1].copy()
                if (blocked := blocked_token(ids)) is not None:
                    step[blocked] = -np.inf
                ids.append(int(np.argmax(step)))
                if progress:
                    progress(f"Reading text… ({done + len(ids)})")
                if ids[-1] == tok.eos_id:
                    break
            done += len(ids)
            if piece_text := tok.decode(ids).strip():
                text += (sep if text else "") + piece_text
        # Tesseract gets the noise-free crop the model saw (avatars, borders, icons left out).
        if line := latin_fallback(content_crop(strip), text):
            lines.append(line)
    return "\n".join(lines)
