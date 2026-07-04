# Exports id->token decode data for TrOCRTokenizer.swift (decode-only; this
# app never tokenizes text, only the image is model input).
# Usage: python3 export_tokenizer.py <size> <out_path> [checkpoint]
import sys, json
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

size = sys.argv[1]
out_path = sys.argv[2]
ckpt = sys.argv[3] if len(sys.argv) > 3 else f"microsoft/trocr-{size}-printed"

processor = TrOCRProcessor.from_pretrained(ckpt)
tok = processor.tokenizer
config = VisionEncoderDecoderModel.from_pretrained(ckpt, low_cpu_mem_usage=False).config

vocab_size = config.decoder.vocab_size
tokenizer_kind = "sentencepiece" if tok.__class__.__name__.lower().startswith("xlmroberta") else "bpe"

id_to_token = []
for i in range(vocab_size):
    if i < tok.vocab_size:
        id_to_token.append(tok.convert_ids_to_tokens(i))
    else:
        id_to_token.append("")

data = {
    "kind": tokenizer_kind,
    "bos_id": tok.bos_token_id,
    "eos_id": tok.eos_token_id,
    "pad_id": tok.pad_token_id,
    "unk_id": tok.unk_token_id,
    "id_to_token": id_to_token,
}
with open(out_path, "w") as f:
    json.dump(data, f, ensure_ascii=False)
print("wrote", out_path, "kind", tokenizer_kind, "vocab", len(id_to_token))
