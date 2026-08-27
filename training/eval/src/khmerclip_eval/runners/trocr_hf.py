from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel

from trocr_khmer import configure_generation, load_trocr_processor
from trocr_khmer.dataset import MAX_LABEL_LEN


@lru_cache(maxsize=4)
def _load_model(model_dir: str) -> tuple:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = load_trocr_processor(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device).eval()
    configure_generation(model, processor, MAX_LABEL_LEN)
    return processor, model, device


def recognize(image_path: Path, model_dir: Path) -> str:
    model_dir = Path(model_dir)
    image_path = Path(image_path)
    processor, model, device = _load_model(str(model_dir.resolve()))
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_length=128,
            num_beams=4,
            length_penalty=1.0,
            no_repeat_ngram_size=0,
        )
    texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return texts[0].strip()
