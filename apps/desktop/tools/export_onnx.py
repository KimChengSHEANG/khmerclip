"""Export a TrOCR checkpoint to encoder.onnx + decoder.onnx + tokenizer.json.

Mirrors the mac app's Vendor/trocr/convert.py (CoreML) and export_tokenizer.py, but emits
ONNX for khmerclip.ocr.trocr. Needs the `tools` extra:
    .venv/bin/pip install -e '.[tools]'
    .venv/bin/python tools/export_onnx.py <checkpoint dir or HF id> exported/khmer
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers.models.trocr.modeling_trocr import TrOCRSinusoidalPositionalEmbedding

PARITY_TOLERANCE = 1e-3


class Encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, pixel_values):
        return self.encoder(pixel_values=pixel_values).last_hidden_state


class DecoderStep(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, input_ids, encoder_hidden_states):
        return self.decoder(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states, use_cache=False).logits


def load_model(ckpt: str) -> VisionEncoderDecoderModel:
    model = VisionEncoderDecoderModel.from_pretrained(ckpt, low_cpu_mem_usage=False)
    # trocr-large-style sinusoidal position tables can be left on the meta device; rebuild them.
    for mod in model.modules():
        if isinstance(mod, TrOCRSinusoidalPositionalEmbedding) and mod.weights is not None and mod.weights.is_meta:
            mod.weights = mod.get_embedding(mod.weights.shape[0], mod.embedding_dim, mod.padding_idx)
    return model.eval()


def decoder_start_id(model) -> int:
    # Top-level config wins: the Khmer fine-tune sets it to bos (0) over a stale sub-config value.
    start = getattr(model.config, "decoder_start_token_id", None)
    return start if start is not None else model.config.decoder.decoder_start_token_id


def export_tokenizer(processor, model, out_path: Path) -> None:
    tok = processor.tokenizer
    vocab_size = model.config.decoder.vocab_size
    kind = "sentencepiece" if tok.__class__.__name__.lower().startswith("xlmroberta") else "bpe"
    specials = {tok.bos_token_id, tok.eos_token_id, tok.pad_token_id, tok.unk_token_id}
    data = {
        "kind": kind,
        "bos_id": tok.bos_token_id,
        "eos_id": tok.eos_token_id,
        "pad_id": tok.pad_token_id,
        "unk_id": tok.unk_token_id,
        "decoder_start_id": decoder_start_id(model),
        "id_to_token": [tok.convert_ids_to_tokens(i) or "" for i in range(vocab_size)],
        "literal_ids": sorted(i for i in tok.get_added_vocab().values() if i < vocab_size and i not in specials),
    }
    out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint)
    processor = TrOCRProcessor.from_pretrained(args.checkpoint)
    size = processor.image_processor.size
    img = size["height"] if isinstance(size, dict) else size
    enc, dec = Encoder(model), DecoderStep(model)

    pixel = torch.rand(1, 3, img, img)
    with torch.no_grad():
        hidden = enc(pixel)
    start = decoder_start_id(model)
    ids = torch.tensor([[start, 100]], dtype=torch.int64)
    # Torch reference before export: tracing leaves the in-memory model producing different
    # logits afterwards, so a post-export reference reports a bogus divergence.
    ids3 = torch.tensor([[start, 100, 200]], dtype=torch.int64)
    with torch.no_grad():
        ref = dec(ids3, hidden).numpy()

    torch.onnx.export(enc, (pixel,), out / "encoder.onnx", input_names=["pixel_values"],
                      output_names=["encoder_hidden_states"], opset_version=17, dynamo=False)
    torch.onnx.export(dec, (ids, hidden), out / "decoder.onnx",
                      input_names=["input_ids", "encoder_hidden_states"], output_names=["logits"],
                      dynamic_axes={"input_ids": {1: "seq"}, "logits": {1: "seq"}},
                      opset_version=17, dynamo=False)
    export_tokenizer(processor, model, out / "tokenizer.json")

    # Parity: onnxruntime vs torch, with a longer sequence than traced to prove the dynamic axis.
    ort_hidden = ort.InferenceSession(str(out / "encoder.onnx")).run(None, {"pixel_values": pixel.numpy()})[0]
    got = ort.InferenceSession(str(out / "decoder.onnx")).run(
        None, {"input_ids": ids3.numpy(), "encoder_hidden_states": ort_hidden})[0]
    diff = float(np.abs(got - ref).max())
    print(f"parity max abs diff: {diff:.2e}")
    if diff > PARITY_TOLERANCE:
        raise SystemExit(f"ONNX output diverges from torch ({diff:.2e} > {PARITY_TOLERANCE})")
    print("exported to", out)


if __name__ == "__main__":
    main()
