import json

import pytest

from khmerclipper.ocr.tokenizer import _BYTE_ENCODER, TrOCRTokenizer

SPECIALS = ["<s>", "<pad>", "</s>", "<unk>"]


def tok(kind="bpe", start=2, extra=(), literal=None):
    payload = {"kind": kind, "bos_id": 0, "eos_id": 2, "pad_id": 1, "unk_id": 3,
               "decoder_start_id": start, "id_to_token": SPECIALS + list(extra)}
    if literal is not None:
        payload["literal_ids"] = literal
    return TrOCRTokenizer(payload)


def test_bpe_reconstructs_bytes_and_keeps_leading_space():
    t = tok(extra=[".", "Ġthe", ",", "Ġto"])
    assert t.decode([0, 5, 7, 2]) == " the to"


def test_bpe_passes_literal_added_tokens_through_verbatim():
    t = tok(extra=["a", "b", "ក"], literal=[6])
    assert t.decode([0, 4, 5, 6, 2]) == "abក"


def test_bpe_byte_mapped_khmer_decodes_utf8():
    # A Khmer char spelled as GPT-2 byte-mapped pieces (the base vocab's way) must decode to UTF-8.
    piece = "".join(_BYTE_ENCODER[b] for b in "ក".encode("utf-8"))
    assert tok(extra=[piece]).decode([4]) == "ក"


def test_sentencepiece_replaces_marker_and_trims_leading_space():
    t = tok(kind="sentencepiece", extra=["▁the", ".", ",", "▁to"])
    assert t.decode([0, 4, 7, 2]) == "the to"


def test_decoder_start_id_reflects_exported_value():
    assert tok(start=2).decoder_start_id == 2
    assert tok(start=0).decoder_start_id == 0


def test_out_of_range_and_special_ids_are_dropped():
    assert tok(extra=["a"]).decode([0, 1, 2, 3, 4, 99, -1]) == "a"


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        tok(kind="wordpiece")


def test_from_file_reads_utf8(tmp_path):
    p = tmp_path / "tokenizer.json"
    p.write_text(json.dumps({"kind": "bpe", "bos_id": 0, "eos_id": 2, "pad_id": 1, "unk_id": 3,
                             "decoder_start_id": 0, "id_to_token": SPECIALS + ["ខ"], "literal_ids": [4]},
                            ensure_ascii=False), encoding="utf-8")
    assert TrOCRTokenizer.from_file(p).decode([4]) == "ខ"
