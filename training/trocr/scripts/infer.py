"""Run the fine-tuned model on one or more images.

    python scripts/infer.py experiments/checkpoint-1/final path/to/image.jpg [more images...]
"""
import sys

import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel

from trocr_khmer import load_trocr_processor


def main():
    if len(sys.argv) < 3:
        sys.exit(f"usage: python {sys.argv[0]} <model_dir> <image_path> [image_path ...]")

    model_dir, image_paths = sys.argv[1], sys.argv[2:]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = load_trocr_processor(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device).eval()

    images = [Image.open(p).convert("RGB") for p in image_paths]
    pixel_values = processor(images, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        # no_repeat_ngram_size must stay 0: Khmer byte-level tokens legitimately repeat
        # trigrams (shared UTF-8 lead bytes), and blocking them wrecks transcription.
        generated_ids = model.generate(
            pixel_values,
            max_length=128,
            num_beams=4,
            length_penalty=1.0,
            no_repeat_ngram_size=0,
        )
    texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    for path, text in zip(image_paths, texts):
        print(f"{path}: {text}")


if __name__ == "__main__":
    main()
