# TrOCR fine-tune — Khmer printed text

Fine-tunes `microsoft/trocr-base-printed` on [SoyVitou/62k-images-khmer-printed-dataset](https://huggingface.co/datasets/SoyVitou/62k-images-khmer-printed-dataset) (62.7k image/text pairs, mixed Khmer + English printed text).

Target machine: Ubuntu, i9-8950HK, 32GB RAM, RTX 3090 (24GB VRAM). **Requires CUDA** — the script asserts `torch.cuda.is_available()` and will refuse to run on CPU-only/Mac.

TrOCR's decoder tokenizer is a byte-level BPE (RoBERTa) trained on English. It *can* encode Khmer, but only through its UTF-8 byte fallback — it has no merges for the script, so every Khmer character becomes 3 raw byte tokens. Before training we therefore **extend the tokenizer with a dedicated token per Khmer character** (the Khmer block `U+1780–U+17FF`, the Khmer Symbols block `U+19E0–U+19FF`, and the zero-width word-separator characters) and resize the decoder's embeddings + tied LM head to match. This collapses one line of Khmer from ~3× its length in byte tokens down to one token per character — shorter sequences (less truncation under `MAX_LABEL_LEN`), and a decoder that learns whole characters instead of shared UTF-8 lead bytes. Fine-tuning the whole model then teaches it to read Khmer; this is the standard approach for adapting TrOCR to a non-Latin script.

The character set and the resize live in [`src/trocr_khmer/khmer.py`](src/trocr_khmer/khmer.py) (`add_khmer_tokens`), invoked automatically by `scripts/train.py`. The added characters are regular (not special) tokens, so they round-trip through `decode(..., skip_special_tokens=True)` as normal text, and the extended tokenizer is saved into each run's `final/` folder for inference/eval. We fine-tune `microsoft/trocr-base-printed`, whose byte-level tokenizer round-trips Khmer cleanly (`microsoft/trocr-small-printed` ships a SentencePiece tokenizer that fails to load on current `transformers`).

## Project layout

```
TrOCR-KhmerClip/
├── src/trocr_khmer/     # shared library (dataset, metrics)
├── scripts/             # CLI entry points (train, eval, infer)
├── docs/                # reference notes (dataset links)
├── requirements.txt
└── pyproject.toml
```



## Setup (Ubuntu, RTX 3090)

```bash
python3 -m venv venv
source venv/bin/activate

# CUDA build of torch — check your driver's CUDA version with `nvidia-smi` first (cu121 works for most recent drivers)
pip install uv
uv pip install torch torchvision

uv pip install -r requirements.txt
uv pip install -e .
```

Verify GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```



## Train

```bash
# 1. Smoke test first — 200 samples (streamed, no full download), 1 epoch. Confirms the pipeline runs before committing hours to the full run.
python scripts/train.py --smoke_test

# 2. Full fine-tune. Default: 8 epochs, batch size 32, bf16, microsoft/trocr-base-printed.
#    Override the base model with --model_name if needed.
python scripts/train.py
```

Each run gets its own auto-incrementing folder under `experiments/` (`experiments/checkpoint-1/`, `experiments/checkpoint-2/`, …). Inside it, HF keeps the best-CER checkpoint plus the last 2, and the final model + processor are saved to `experiments/checkpoint-N/final/`.

Resume an interrupted run:

```bash
python scripts/train.py --resume_from_checkpoint experiments/checkpoint-1/checkpoint-1000
```

Watch progress:

```bash
tensorboard --logdir experiments/checkpoint-1
```

Defaults target full RTX 3090 utilization (~18–20GB VRAM at batch 16). If GPU util stays low, try `--train_batch_size 20`. If you hit OOM, drop `--train_batch_size` or raise `--grad_accum` to keep the same effective batch.

## Evaluation

Run evaluation independently against any saved checkpoint or fine-tuned model:

```bash
python scripts/evaluate_model.py experiments/checkpoint-1/final
```

Optional quick evaluation with a smaller sample set:

```bash
python scripts/evaluate_model.py experiments/checkpoint-1/final --max_eval_samples 200
```



## Inference

```bash
python scripts/infer.py experiments/checkpoint-1/final some_image.jpg
```

