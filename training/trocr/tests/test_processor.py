from types import SimpleNamespace

import pytest

from trocr_khmer import processor as processor_mod
from trocr_khmer.khmer import add_khmer_tokens
from trocr_khmer.processor import BYTE_LEVEL_TOKENIZER, load_model_and_processor, load_trocr_processor

KHMER_SAMPLE = ["ក", "ស", "្", "ា"]


def _fake_model(monkeypatch, vocab_size):
    model = SimpleNamespace(config=SimpleNamespace(decoder=SimpleNamespace(vocab_size=vocab_size)))
    monkeypatch.setattr(
        processor_mod.VisionEncoderDecoderModel, "from_pretrained", staticmethod(lambda path: model)
    )
    return model


@pytest.fixture
def extended_dir(tmp_path):
    proc = load_trocr_processor(BYTE_LEVEL_TOKENIZER, tokenizer_name=BYTE_LEVEL_TOKENIZER)
    add_khmer_tokens(proc)
    proc.save_pretrained(tmp_path)
    return tmp_path, proc


def test_local_checkpoint_reuses_saved_tokenizer(monkeypatch, extended_dir):
    path, saved = extended_dir
    model = _fake_model(monkeypatch, len(saved.tokenizer))
    got_model, proc, added = load_model_and_processor(str(path))
    assert got_model is model and added == 0
    assert len(proc.tokenizer) == len(saved.tokenizer)
    assert proc.tokenizer.convert_tokens_to_ids(KHMER_SAMPLE) == saved.tokenizer.convert_tokens_to_ids(KHMER_SAMPLE)


def test_local_checkpoint_vocab_mismatch_raises(monkeypatch, extended_dir):
    path, saved = extended_dir
    _fake_model(monkeypatch, len(saved.tokenizer) + 1)
    with pytest.raises(ValueError, match="decoder vocab"):
        load_model_and_processor(str(path))


def test_local_checkpoint_missing_khmer_tokens_raises(monkeypatch, tmp_path):
    proc = load_trocr_processor(BYTE_LEVEL_TOKENIZER, tokenizer_name=BYTE_LEVEL_TOKENIZER)
    proc.save_pretrained(tmp_path)
    _fake_model(monkeypatch, len(proc.tokenizer))
    with pytest.raises(ValueError, match="missing"):
        load_model_and_processor(str(tmp_path))
