# Exports id->token decode data for TrOCRTokenizer.swift (decode-only; this
# app never tokenizes text, only the image is model input).
# Usage: python3 export_tokenizer.py <size> <out_path> [checkpoint]
import sys, json
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

DEFAULT_CHECKPOINTS = {
    "small": "microsoft/trocr-small-printed",
    "base": "microsoft/trocr-base-printed",
    "large": "microsoft/trocr-large-printed",
    "khmer": "KiteAether/khmer-trcor-06-10",
}

size = sys.argv[1]
out_path = sys.argv[2]
ckpt = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_CHECKPOINTS[size]

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

# decoder_start_token_id is NOT always eos_token_id -- true for the official
# microsoft/trocr-*-printed checkpoints (a documented quirk: the top-level
# config doesn't set it, so it falls back to the decoder sub-config, which
# happens to equal eos there). Fine-tunes can and do set it independently --
# KiteAether/khmer-trcor-06-10 has a top-level decoder_start_token_id=0 (bos)
# that overrides a *stale* decoder.decoder_start_token_id=2 left over from
# its base model's sub-config. Verified against actual model.generate()
# output (which token id position 0 of the generated sequence actually is)
# rather than trusting either config field blindly. Top-level wins when set.
decoder_start_id = getattr(config, "decoder_start_token_id", None)
if decoder_start_id is None:
    decoder_start_id = config.decoder.decoder_start_token_id

data = {
    "kind": tokenizer_kind,
    "bos_id": tok.bos_token_id,
    "eos_id": tok.eos_token_id,
    "pad_id": tok.pad_token_id,
    "unk_id": tok.unk_token_id,
    "decoder_start_id": decoder_start_id,
    "id_to_token": id_to_token,
}
with open(out_path, "w") as f:
    json.dump(data, f, ensure_ascii=False)
print("wrote", out_path, "kind", tokenizer_kind, "vocab", len(id_to_token))
