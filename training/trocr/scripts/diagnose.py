"""Diagnose the Khmer garbage-output bug. Run in the working env:

    python scripts/diagnose.py experiments/checkpoint-3/final [optional_image.png]

It isolates each stage: (1) tokenizer round-trip, (2) label ids for a Khmer
sample, (3) actual model generation internals. Whichever stage first shows the
E1-9E-xx -> E1-xx-9E byte transposition is the culprit.
"""
import sys

from trocr_khmer import load_trocr_processor

SAMPLE = "ព្រះរាជាណាចក្រកម្ពុជា"


def show_bytes(label, s):
    print(f"  {label}: {s!r}")
    print(f"      utf-8: {s.encode('utf-8', 'surrogatepass').hex(' ')}")


def main():
    model_dir = sys.argv[1]
    image_path = sys.argv[2] if len(sys.argv) > 2 else None

    processor = load_trocr_processor(model_dir)
    tok = processor.tokenizer

    print("=== STAGE 1: tokenizer round-trip (no model involved) ===")
    show_bytes("original ", SAMPLE)
    enc = tok(SAMPLE)
    ids = enc.input_ids
    print(f"  input_ids ({len(ids)}): {ids}")
    print(f"  tokens: {tok.convert_ids_to_tokens(ids)}")
    decoded = tok.decode(ids, skip_special_tokens=True)
    show_bytes("roundtrip", decoded)
    print(f"  ROUND-TRIP OK? {decoded == SAMPLE}")
    print(f"  bos={tok.bos_token_id} eos={tok.eos_token_id} pad={tok.pad_token_id} cls={tok.cls_token_id} sep={tok.sep_token_id}")

    if image_path is None:
        print("\n(no image given — skipping model generation stage)")
        return

    import torch
    from PIL import Image
    from transformers import VisionEncoderDecoderModel

    print("\n=== STAGE 2: model generation internals ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device).eval()
    print(f"  config: decoder_start={model.config.decoder_start_token_id} eos={model.config.eos_token_id} "
          f"pad={model.config.pad_token_id} vocab={model.config.vocab_size}")
    gc = model.generation_config
    print(f"  gen_config: decoder_start={gc.decoder_start_token_id} eos={gc.eos_token_id} "
          f"no_repeat_ngram={gc.no_repeat_ngram_size} num_beams={gc.num_beams} length_penalty={gc.length_penalty}")

    img = Image.open(image_path).convert("RGB")
    pixel_values = processor(img, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        out = model.generate(pixel_values, max_length=64, num_beams=1, do_sample=False)
    gen_ids = out[0].tolist()
    print(f"  generated ids ({len(gen_ids)}): {gen_ids}")
    print(f"  generated tokens: {tok.convert_ids_to_tokens(gen_ids)}")
    show_bytes("gen text ", tok.decode(gen_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
