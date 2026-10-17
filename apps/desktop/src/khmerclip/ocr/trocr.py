"""TrOCR via onnxruntime: greedy, no-KV-cache decode per text line (same loop as
apps/mac/Sources/OCR/TrOCREngine.swift)."""

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from khmerclip.errors import Cancelled, EngineUnavailable, KhmerClipError
from khmerclip.ocr import trocr_models
from khmerclip.ocr.preprocess import line_strips, pixel_values
from khmerclip.ocr.tokenizer import TrOCRTokenizer

MAX_LENGTH = 64


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
    if override := os.environ.get("KHMERCLIP_TROCR_DIR"):
        return Path(override)
    manifest = trocr_models.load_manifest()
    if not trocr_models.is_installed(manifest):
        raise EngineUnavailable("The TrOCR model isn't downloaded yet.")
    return trocr_models.model_dir(manifest)


def preload() -> None:
    """Warm the ONNX sessions while the user drags a selection. Errors surface on recognize()."""
    try:
        load_session(installed_model_dir())
    except KhmerClipError:
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
        hidden = session.encoder.run(["encoder_hidden_states"], {"pixel_values": pixel_values(strip)})[0]
        ids = [tok.decoder_start_id]
        while len(ids) < MAX_LENGTH:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            logits = session.decoder.run(
                ["logits"], {"input_ids": np.array([ids], dtype=np.int64), "encoder_hidden_states": hidden})[0]
            ids.append(int(np.argmax(logits[0, -1])))
            if progress:
                progress(f"Reading text… ({done + len(ids)})")
            if ids[-1] == tok.eos_id:
                break
        done += len(ids)
        if line := tok.decode(ids):
            lines.append(line)
    return "\n".join(lines)
