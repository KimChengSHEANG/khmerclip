import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import coremltools as ct  # noqa: E402

from convert import convert_decoder, convert_encoder  # noqa: E402
from kv_decoder import KVDecoderStep  # noqa: E402
from test_kv_decoder import masks, tiny_model  # noqa: E402

MAX_LEN = 16
pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="CoreML predict needs macOS")


@torch.no_grad()
def test_stateful_decoder_matches_eager_over_steps() -> None:
    model = tiny_model(learned_pos=True)
    pixels = torch.rand(1, 3, 32, 32)
    enc = model.encoder(pixel_values=pixels).last_hidden_state

    eager = KVDecoderStep(model, MAX_LEN).eval()
    ck, cv = eager.cross_kv(enc)
    enc_ml = convert_encoder(model, 32, ct.precision.FLOAT16)
    enc_out = enc_ml.predict({"pixel_values": pixels.numpy()})
    np.testing.assert_allclose(enc_out["cross_k"], ck.numpy(), atol=2e-2)
    np.testing.assert_allclose(enc_out["cross_v"], cv.numpy(), atol=2e-2)

    dec_ml = convert_decoder(model, tuple(enc.shape), MAX_LEN, ct.precision.FLOAT16)
    state = dec_ml.make_state()
    for pos, tok in enumerate([0, 7, 9, 3, 11]):
        attn, write = masks(pos, MAX_LEN)
        ids = torch.tensor([[tok]], dtype=torch.int32)
        p = torch.tensor([pos], dtype=torch.int32)
        want = eager(ids, p, attn, write, ck, cv)[0, 0].numpy()
        got = dec_ml.predict(
            {"input_ids": ids.numpy(), "position": p.numpy(), "attn_mask": attn.numpy(),
             "write_mask": write.numpy(), "cross_k": enc_out["cross_k"], "cross_v": enc_out["cross_v"]},
            state=state,
        )["logits"][0, 0]
        np.testing.assert_allclose(got, want, atol=3e-2)
        assert int(got.argmax()) == int(want.argmax())
