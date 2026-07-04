"""Run the fine-tuned model on one or more images.

    python scripts/infer.py trocr-khmer-final path/to/image.jpg [more images...]
"""
import sys

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def main():
    if len(sys.argv) < 3:
        sys.exit(f"usage: python {sys.argv[0]} <model_dir> <image_path> [image_path ...]")

    model_dir, image_paths = sys.argv[1], sys.argv[2:]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = TrOCRProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device).eval()

    images = [Image.open(p).convert("RGB") for p in image_paths]
    pixel_values = processor(images, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_length=128, num_beams=4)
    texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    for path, text in zip(image_paths, texts):
        print(f"{path}: {text}")


if __name__ == "__main__":
    main()
