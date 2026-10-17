import os
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from helpers import FIXTURES
from khmerclip.errors import Cancelled
from khmerclip.ocr import trocr
from khmerclip.ocr.tokenizer import TrOCRTokenizer

TOKENIZER = TrOCRTokenizer({"kind": "bpe", "bos_id": 0, "eos_id": 2, "pad_id": 1, "unk_id": 3,
                            "decoder_start_id": 0, "literal_ids": [4, 5],
                            "id_to_token": ["<s>", "<pad>", "</s>", "<unk>", "ក", "ខ"]})


class FakeEncoder:
    def run(self, names, feeds):
        assert feeds["pixel_values"].shape == (1, 3, 384, 384)
        return [np.zeros((1, 4, 8), np.float32)]


class ScriptedDecoder:
    """Emits script[seq_len - 1] as the next token (last entry repeats)."""

    def __init__(self, script):
        self.script, self.calls = script, 0

    def run(self, names, feeds):
        self.calls += 1
        ids = feeds["input_ids"]
        assert ids.dtype == np.int64
        seq = ids.shape[1]
        logits = np.zeros((1, seq, 6), np.float32)
        logits[0, -1, self.script[min(seq - 1, len(self.script) - 1)]] = 1.0
        return [logits]


def session(script):
    return SimpleNamespace(encoder=FakeEncoder(), decoder=ScriptedDecoder(script), tokenizer=TOKENIZER)


def lines_image(n):
    img = Image.new("RGB", (400, 100 * n), "white")
    for k in range(n):
        ImageDraw.Draw(img).rectangle((20, 100 * k + 30, 380, 100 * k + 60), fill="black")
    return img


def test_greedy_decode_until_eos():
    assert trocr.recognize_with_session(session([4, 5, 2]), lines_image(1)) == "កខ"


def test_each_line_decoded_and_joined_with_newline():
    s = session([4, 5, 2])
    assert trocr.recognize_with_session(s, lines_image(2)) == "កខ\nកខ"
    assert s.decoder.calls == 6


def test_decode_stops_at_max_length_without_eos():
    s = session([4])
    assert trocr.recognize_with_session(s, lines_image(1)) == "ក" * (trocr.MAX_LENGTH - 1)
    assert s.decoder.calls == trocr.MAX_LENGTH - 1


def test_cancel_raises():
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        trocr.recognize_with_session(session([4, 2]), lines_image(1), cancel=cancel)


def test_progress_reports_token_count():
    seen = []
    trocr.recognize_with_session(session([4, 5, 2]), lines_image(1), progress=seen.append)
    assert seen[-1] == "Reading text… (4)"


@pytest.mark.skipif(not os.environ.get("KHMERCLIP_TROCR_DIR"), reason="set KHMERCLIP_TROCR_DIR to an exported model")
def test_real_model_reads_fixture():
    with Image.open(FIXTURES / "1.png") as img:
        text = trocr.recognize(img)
    assert text.strip()
