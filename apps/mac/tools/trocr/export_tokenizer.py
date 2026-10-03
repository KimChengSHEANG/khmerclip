# Exports id->token decode data for TrOCRTokenizer.swift (decode-only; this
# app never tokenizes text, only the image is model input).
# Usage: python3 export_tokenizer.py <size> <out_path> [checkpoint]
import sys, json
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

DEFAULT_CHECKPOINTS = {
    "small": "microsoft/trocr-small-printed",
    "base": "microsoft/trocr-base-printed",
    "large": "microsoft/trocr-large-printed",
    "khmer": "lkhapple/Khmer-TrOCR-OCR",
}

size = sys.argv[1]
out_path = sys.argv[2]
ckpt = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_CHECKPOINTS[size]

processor = TrOCRProcessor.from_pretrained(ckpt)
tok = processor.tokenizer
config = VisionEncoderDecoderModel.from_pretrained(ckpt, low_cpu_mem_usage=False).config

vocab_size = config.decoder.vocab_size
tokenizer_kind = "sentencepiece" if tok.__class__.__name__.lower().startswith("xlmroberta") else "bpe"

# convert_ids_to_tokens resolves both base-vocab pieces AND added tokens, so
# call it for every id -- the khmer fine-tune appends ~91 literal Khmer tokens
# past tok.vocab_size (50265) that the model actually emits; dropping them left
# decode producing empty/U+FFFD output. Fall back to "" only if it returns None.
id_to_token = [tok.convert_ids_to_tokens(i) or "" for i in range(vocab_size)]

# Ids of *added* tokens whose piece is literal UTF-8 text (e.g. Khmer glyphs),
# not GPT-2 byte-BPE. The bpe decoder must pass these through verbatim rather
# than byte-mapping their scalars (which would drop them). Empty for the pure
# RoBERTa/GPT-2 English checkpoints. Specials are excluded (handled separately).
special_ids = {tok.bos_token_id, tok.eos_token_id, tok.pad_token_id, tok.unk_token_id}
literal_ids = sorted(i for i in tok.get_added_vocab().values()
                     if i < vocab_size and i not in special_ids)

# decoder_start_token_id is NOT always eos_token_id -- true for the official
# microsoft/trocr-*-printed checkpoints (a documented quirk: the top-level
# config doesn't set it, so it falls back to the decoder sub-config, which
# happens to equal eos there). Fine-tunes can and do set it independently --
# the khmer fine-tune has a top-level decoder_start_token_id=0 (bos)
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
    "literal_ids": literal_ids,
}
with open(out_path, "w") as f:
    json.dump(data, f, ensure_ascii=False)
print("wrote", out_path, "kind", tokenizer_kind, "vocab", len(id_to_token))
