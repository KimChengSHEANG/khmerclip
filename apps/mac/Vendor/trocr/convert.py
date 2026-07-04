# Converts a TrOCR (VisionEncoderDecoderModel) checkpoint to a CoreML
# encoder.mlpackage + decoder.mlpackage pair for TrOCREngine.swift.
#
# Usage: python3 convert.py <size> <out_dir> [checkpoint]
#   size        small|base|large -- only used to pick the default checkpoint
#   out_dir     e.g. Vendor/trocr/base
#   checkpoint  optional override, e.g. a local path or HF repo id for a
#               fine-tuned model (defaults to microsoft/trocr-<size>-printed)
#
# After converting, regenerate the matching tokenizer.json with:
#   python3 export_tokenizer.py <size> <out_dir>/tokenizer.json [checkpoint]
#
# Requires: pip install torch transformers coremltools
import sys, torch, coremltools as ct
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

size = sys.argv[1]  # small|base|large
out_dir = sys.argv[2]
ckpt = sys.argv[3] if len(sys.argv) > 3 else f"microsoft/trocr-{size}-printed"

model = VisionEncoderDecoderModel.from_pretrained(ckpt, low_cpu_mem_usage=False)

# Some checkpoints (trocr-large-*) use sinusoidal (non-learned) decoder position
# embeddings whose `.weights` is a plain attribute, not a registered buffer/param.
# HF's fast/meta init computes it under the meta device during __init__ and never
# re-materializes it (materialization only touches real params/buffers), leaving
# it permanently meta. Recompute it for real now that we're off the meta context.
from transformers.models.trocr.modeling_trocr import TrOCRSinusoidalPositionalEmbedding
for mod in model.modules():
    if isinstance(mod, TrOCRSinusoidalPositionalEmbedding) and mod.weights is not None and mod.weights.is_meta:
        n = mod.weights.shape[0]
        mod.weights = mod.get_embedding(n, mod.embedding_dim, mod.padding_idx)

model.eval()
processor = TrOCRProcessor.from_pretrained(ckpt)

sz = processor.image_processor.size
img_size = sz["height"] if isinstance(sz, dict) else getattr(sz, "height", sz)
print("img_size", img_size, "d_model(decoder hidden)", model.config.decoder.hidden_size if hasattr(model.config.decoder,'hidden_size') else model.config.decoder.d_model)

class Encoder(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.encoder = m.encoder
    def forward(self, pixel_values):
        return self.encoder(pixel_values=pixel_values).last_hidden_state

class DecoderStep(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.decoder = m.decoder
    def forward(self, input_ids, encoder_hidden_states):
        out = self.decoder(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states, use_cache=False)
        return out.logits

enc = Encoder(model)
dec = DecoderStep(model)

example_pixel = torch.rand(1, 3, img_size, img_size)
enc_out = enc(example_pixel)
print("encoder out shape", enc_out.shape)

example_ids = torch.tensor([[model.config.decoder.decoder_start_token_id, 100]])
dec_out = dec(example_ids, enc_out)
print("decoder out shape", dec_out.shape)

traced_enc = torch.jit.trace(enc, (example_pixel,))
traced_dec = torch.jit.trace(dec, (example_ids, enc_out))

enc_mlmodel = ct.convert(
    traced_enc,
    inputs=[ct.TensorType(name="pixel_values", shape=example_pixel.shape)],
    outputs=[ct.TensorType(name="encoder_hidden_states")],
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT32,
    minimum_deployment_target=ct.target.macOS13,
)
enc_mlmodel.save(f"{out_dir}/encoder.mlpackage")

dec_mlmodel = ct.convert(
    traced_dec,
    inputs=[
        ct.TensorType(name="input_ids", shape=ct.Shape(shape=(1, ct.RangeDim(1, 64))), dtype=ct.converters.mil.input_types.np.int32),
        ct.TensorType(name="encoder_hidden_states", shape=enc_out.shape),
    ],
    outputs=[ct.TensorType(name="logits")],
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT32,
    minimum_deployment_target=ct.target.macOS13,
)
dec_mlmodel.save(f"{out_dir}/decoder.mlpackage")
print("saved to", out_dir)
