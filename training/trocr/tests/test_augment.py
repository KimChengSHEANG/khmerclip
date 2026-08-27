import random

from PIL import Image

from trocr_khmer.augment import maybe_augment


def test_maybe_augment_keeps_size_and_mode():
    image = Image.new("RGB", (120, 40), color=(200, 100, 50))
    rng = random.Random(42)

    for _ in range(20):
        out = maybe_augment(image, rng=rng)
        assert out.size == image.size
        assert out.mode == "RGB"


def test_maybe_augment_is_deterministic_with_fixed_rng():
    image = Image.new("RGB", (64, 32), color=(128, 64, 32))
    rng1 = random.Random(7)
    rng2 = random.Random(7)

    out1 = maybe_augment(image, rng=rng1)
    out2 = maybe_augment(image, rng=rng2)

    assert list(out1.getdata()) == list(out2.getdata())
