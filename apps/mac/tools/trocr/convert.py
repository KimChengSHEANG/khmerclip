# Converts a TrOCR (VisionEncoderDecoderModel) checkpoint to a CoreML
# encoder.mlpackage + decoder.mlpackage pair for TrOCREngine.swift.
#
# Usage: python3 tools/trocr/convert.py <size> <out_dir> [checkpoint]
#   size        small|base|large|khmer -- picks the default checkpoint below
#   out_dir     e.g. Vendor/trocr/base
#   checkpoint  optional override, e.g. a local path or HF repo id (use this
#               to point at a newer Khmer fine-tune once one lands)
#
# After converting, regenerate the matching tokenizer.json with:
#   python3 tools/trocr/export_tokenizer.py <size> <out_dir>/tokenizer.json [checkpoint]
#
# Requires: pip install torch transformers coremltools
import argparse

import coremltools as ct
import numpy as np
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from kv_decoder import KVDecoderStep

DEFAULT_CHECKPOINTS = {
    "small": "microsoft/trocr-small-printed",
    "base": "microsoft/trocr-base-printed",
    "large": "microsoft/trocr-large-printed",
    # Khmer printed-text fine-tune that actually reads the image (byte-BPE).
    # Two earlier candidates were rejected after measuring them: the byte-BPE
    # KiteAether/khmer-trcor-06-10 emitted broken 2-byte UTF-8 (U+FFFD), and a
    # SentencePiece one hallucinated fluent-but-unrelated Khmer. This one reads
    # printed Khmer correctly when the text fills the frame -- TrOCREngine
    # feeds the same PIL-bilinear 384x384 preprocessing as training (no
    # content crop). Swap via the checkpoint override arg if a better one lands.
    "khmer": "lkhapple/Khmer-TrOCR-OCR",
}

MAX_LEN = 128  # total decoded ids incl. start; == training MAX_LABEL_LEN


def load(ckpt):
    model = VisionEncoderDecoderModel.from_pretrained(ckpt, low_cpu_mem_usage=False)
    # trocr-large-* sinusoidal position tables are left on the meta device by
    # HF fast init (plain attribute, not a buffer); recompute them for real.
    from transformers.models.trocr.modeling_trocr import TrOCRSinusoidalPositionalEmbedding
    for mod in model.modules():
        if isinstance(mod, TrOCRSinusoidalPositionalEmbedding) and mod.weights is not None and mod.weights.is_meta:
            mod.weights = mod.get_embedding(mod.weights.shape[0], mod.embedding_dim, mod.padding_idx)
    return model.eval()


class Encoder(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.encoder = m.encoder
        self.proj = getattr(m, "enc_to_dec_proj", None)  # only when hidden sizes differ

    def forward(self, pixel_values):
        h = self.encoder(pixel_values=pixel_values).last_hidden_state
        return self.proj(h) if self.proj is not None else h


def convert_encoder(model, img_size, precision):
    example = torch.rand(1, 3, img_size, img_size)
    traced = torch.jit.trace(Encoder(model).eval(), (example,))
    return ct.convert(
        traced,
        inputs=[ct.TensorType(name="pixel_values", shape=example.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="encoder_hidden_states", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=precision,
        minimum_deployment_target=ct.target.macOS15,
    )


def convert_decoder(model, enc_shape, max_len, precision):
    step = KVDecoderStep(model, max_len, cache_dtype=torch.float16).eval()
    attn = torch.full((1, 1, 1, max_len), -1e4)
    attn[..., 0] = 0
    write = torch.zeros(1, 1, max_len, 1)
    write[:, :, 0, 0] = 1
    example = (torch.zeros(1, 1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32),
               attn, write, torch.rand(*enc_shape))
    traced = torch.jit.trace(step, example)
    cache = ct.TensorType(shape=tuple(step.k_cache.shape), dtype=np.float16)
    return ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, 1), dtype=np.int32),
            ct.TensorType(name="position", shape=(1,), dtype=np.int32),
            ct.TensorType(name="attn_mask", shape=(1, 1, 1, max_len), dtype=np.float32),
            ct.TensorType(name="write_mask", shape=(1, 1, max_len, 1), dtype=np.float32),
            ct.TensorType(name="encoder_hidden_states", shape=tuple(enc_shape), dtype=np.float32),
        ],
        states=[ct.StateType(wrapped_type=cache, name="k_cache"),
                ct.StateType(wrapped_type=cache, name="v_cache")],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=precision,
        minimum_deployment_target=ct.target.macOS15,
    )


def int8(mlmodel):
    import coremltools.optimize.coreml as cto
    cfg = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric"))
    return cto.linear_quantize_weights(mlmodel, cfg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("size", choices=sorted(DEFAULT_CHECKPOINTS))
    p.add_argument("out_dir")
    p.add_argument("checkpoint", nargs="?")
    p.add_argument("--encoder-fp32", action="store_true", help="keep the encoder FP32 (FP16 CER fallback)")
    p.add_argument("--int8", action="store_true", help="int8 weight quantization (latency fallback)")
    a = p.parse_args()

    ckpt = a.checkpoint or DEFAULT_CHECKPOINTS[a.size]
    model = load(ckpt)
    sz = TrOCRProcessor.from_pretrained(ckpt).image_processor.size
    img_size = sz["height"] if isinstance(sz, dict) else getattr(sz, "height", sz)

    with torch.no_grad():
        enc_shape = tuple(Encoder(model)(torch.rand(1, 3, img_size, img_size)).shape)
    print("img_size", img_size, "encoder out", enc_shape)

    enc_prec = ct.precision.FLOAT32 if a.encoder_fp32 else ct.precision.FLOAT16
    enc_ml = convert_encoder(model, img_size, enc_prec)
    dec_ml = convert_decoder(model, enc_shape, MAX_LEN, ct.precision.FLOAT16)  # states must be fp16
    if a.int8:
        enc_ml, dec_ml = int8(enc_ml), int8(dec_ml)
    enc_ml.save(f"{a.out_dir}/encoder.mlpackage")
    dec_ml.save(f"{a.out_dir}/decoder.mlpackage")
    print("saved to", a.out_dir)


if __name__ == "__main__":
    main()
