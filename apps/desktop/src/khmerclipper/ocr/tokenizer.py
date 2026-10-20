"""Decode-only TrOCR tokenizer. Port of apps/mac/Sources/OCR/TrOCRTokenizer.swift.

Two families: GPT-2 byte-level BPE ('Ġ' = leading space, non-ASCII bytes mapped to
printable scalars) and SentencePiece ('▁' = leading space). The Khmer fine-tune is BPE
plus literal Khmer added tokens (listed in literal_ids) that decode verbatim.
"""

import json
from collections.abc import Sequence
from pathlib import Path


def _gpt2_byte_encoder() -> dict[int, str]:
    printable = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    encoder = {b: chr(b) for b in printable}
    n = 0
    for b in range(256):
        if b not in printable:
            encoder[b] = chr(256 + n)
            n += 1
    return encoder


_BYTE_ENCODER = _gpt2_byte_encoder()
_BYTE_DECODER = {ch: b for b, ch in _BYTE_ENCODER.items()}


class TrOCRTokenizer:
    def __init__(self, payload: dict):
        self.kind: str = payload["kind"]
        if self.kind not in ("bpe", "sentencepiece"):
            raise ValueError(f"unknown tokenizer kind {self.kind!r}")
        self.eos_id: int = payload["eos_id"]
        # NOT always eos: fine-tunes may start from bos. Use whatever was exported.
        self.decoder_start_id: int = payload["decoder_start_id"]
        self._id_to_token: list[str] = payload["id_to_token"]
        self._special = {payload["bos_id"], payload["eos_id"], payload["pad_id"], payload["unk_id"]}
        self._literal = set(payload.get("literal_ids") or [])

    @classmethod
    def from_file(cls, path: Path) -> "TrOCRTokenizer":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def decode(self, ids: Sequence[int]) -> str:
        kept = [i for i in ids if i not in self._special and 0 <= i < len(self._id_to_token)]
        if self.kind == "sentencepiece":
            return "".join(self._id_to_token[i] for i in kept).replace("▁", " ").strip(" \t")
        out: list[str] = []
        buf = bytearray()
        for i in kept:
            token = self._id_to_token[i]
            if i in self._literal:
                if buf:
                    out.append(buf.decode("utf-8", "replace"))
                    buf.clear()
                out.append(token)
            else:
                buf.extend(_BYTE_DECODER[ch] for ch in token if ch in _BYTE_DECODER)
        if buf:
            out.append(buf.decode("utf-8", "replace"))
        return "".join(out)
