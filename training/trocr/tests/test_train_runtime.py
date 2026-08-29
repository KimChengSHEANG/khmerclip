"""Train-time eval must stay small and greedy so the GPU spends wall time training."""

from types import SimpleNamespace

from trocr_khmer.train_runtime import build_generation_config, limit_eval_dataset


class _FakeDataset:
    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n

    def select(self, indices):
        return _FakeDataset(len(list(indices)))


def test_limit_eval_dataset_caps_rows():
    ds = _FakeDataset(350_000)
    limited = limit_eval_dataset(ds, max_samples=4096)
    assert len(limited) == 4096


def test_limit_eval_dataset_none_keeps_all():
    ds = _FakeDataset(100)
    assert limit_eval_dataset(ds, max_samples=None) is ds


def test_limit_eval_dataset_does_not_grow_small_sets():
    ds = _FakeDataset(50)
    assert len(limit_eval_dataset(ds, max_samples=4096)) == 50


def test_train_generation_config_defaults_to_greedy():
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(cls_token_id=0, pad_token_id=1, sep_token_id=2)
    )
    cfg = build_generation_config(processor, max_length=128, num_beams=1)
    assert cfg.num_beams == 1
    assert cfg.no_repeat_ngram_size == 0
    assert cfg.length_penalty == 1.0
    assert cfg.use_cache is True


def test_generation_config_honors_beam_count():
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(cls_token_id=0, pad_token_id=1, sep_token_id=2)
    )
    cfg = build_generation_config(processor, max_length=64, num_beams=4)
    assert cfg.num_beams == 4
    assert cfg.early_stopping is True
