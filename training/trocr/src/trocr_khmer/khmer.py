"""Extend a TrOCR tokenizer with dedicated Khmer-script tokens.

The decoder uses a byte-level BPE (RoBERTa) tokenizer — natively for ``trocr-base-printed``,
and grafted on for ``trocr-small-printed`` (whose own SentencePiece tokenizer can't round-trip
Khmer; see ``processor.BYTE_LEVEL_TOKENIZER``). A byte-level BPE *can* encode Khmer, but only
through the UTF-8 byte fallback: it has no merges for the script, so every Khmer character is
spelled out as its 3 raw byte tokens. That has two costs for fine-tuning:

* **Length.** A line of Khmer becomes ~3x more tokens than characters, so ``MAX_LABEL_LEN``
  fills up ~3x faster and long lines get silently truncated during training.
* **Difficulty.** The decoder has to learn Khmer as sequences of shared UTF-8 lead bytes
  (every consonant starts ``E1 9E`` / ``E1 9F``) instead of as whole characters.

Adding each Khmer character as its own token fixes both: one character -> one token, exact
round-trip preserved, and the decoder gets a dedicated embedding per character. The model's
decoder embedding matrix (and its tied LM head) must be resized to match the tokenizer's vocab
(``resize_token_embeddings`` grows or, when grafting a smaller byte-level vocab onto the small
model, shrinks it); HF seeds any added rows from the existing embeddings' mean/covariance.

Because the added Khmer entries are *regular* tokens (not special tokens), they survive
``batch_decode(..., skip_special_tokens=True)`` and appear in the output as normal text.
"""
import unicodedata

# Unicode ranges that make up printed Khmer text.
#   U+1780-U+17FF  Khmer block: consonants, independent/dependent vowels, the COENG
#                  subscript joiner (U+17D2), diacritic signs, digits (U+17E0-U+17E9)
#                  and the lek attak numerals (U+17F0-U+17F9).
#   U+19E0-U+19FF  Khmer Symbols block: lunar-date and astrological signs.
# Unassigned code points inside these ranges are skipped via unicodedata.
_KHMER_RANGES = ((0x1780, 0x17FF), (0x19E0, 0x19FF))

# Zero-width characters that occur in real Khmer text. Khmer has no space between words;
# U+200B (zero-width space) is the conventional word separator, and ZWNJ/ZWJ (U+200C/U+200D)
# control how clusters render. They are invisible but semantically present in the labels, so
# they need their own tokens too.
_KHMER_FORMAT_CHARS = ("​", "‌", "‍")


def khmer_token_chars() -> list[str]:
    """Return the list of Khmer characters to register as tokenizer tokens.

    Only code points that Unicode actually assigns are included, so the gaps in the Khmer
    block don't turn into junk tokens.
    """
    chars: list[str] = []
    for lo, hi in _KHMER_RANGES:
        for cp in range(lo, hi + 1):
            ch = chr(cp)
            try:
                unicodedata.name(ch)  # raises ValueError for unassigned code points
            except ValueError:
                continue
            chars.append(ch)
    chars.extend(_KHMER_FORMAT_CHARS)
    return chars


def add_khmer_tokens(processor, model=None) -> int:
    """Add Khmer characters to ``processor.tokenizer`` and resize ``model`` to match.

    Only characters the tokenizer doesn't already represent as a single token are added, so
    calling this on an already-extended checkpoint is a no-op. Returns the number of tokens
    added. Pass ``model`` during training/first extension so its decoder embeddings + tied LM
    head grow to the new vocab size; loading an already-extended checkpoint needs no resize.
    """
    tokenizer = processor.tokenizer
    num_added = tokenizer.add_tokens(khmer_token_chars())

    if model is not None:
        # Resize the *decoder* (the text side of the VisionEncoderDecoder); this grows the
        # input embeddings and the tied output/LM-head. mean_resizing (HF default) seeds the
        # new rows from the existing embedding distribution.
        model.decoder.resize_token_embeddings(len(tokenizer))
        model.config.decoder.vocab_size = len(tokenizer)
        model.config.vocab_size = len(tokenizer)

    return num_added
