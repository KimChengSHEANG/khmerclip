# TrOCR fine-tune — Khmer printed text

Fine-tunes `microsoft/trocr-base-printed` on Khmer image/text datasets. Pass `--model_name microsoft/trocr-small-printed` to fine-tune the smaller/faster checkpoint instead.

### Datasets

Train (and evaluate) on one dataset or a combination via `--datasets`. Five are pre-registered by short alias; any other value is treated as a raw Hugging Face hub id (assumed to expose `image`/`text` columns). All are concatenated when several are given, and a reproducible 3% validation split is carved with `seed=42`.

| alias | dataset | size |
| --- | --- | --- |
| `soyvitou` (default) | [SoyVitou/62k-images-khmer-printed-dataset](https://huggingface.co/datasets/SoyVitou/62k-images-khmer-printed-dataset) | 62.7k printed, mixed Khmer + English |
| `sokheng` | [Sokheng/khmer-synthetic-ocr-v1-100k](https://huggingface.co/datasets/Sokheng/khmer-synthetic-ocr-v1-100k) | 100k synthetic |
| `hanuman` | [seanghay/khmer-hanuman-100k](https://huggingface.co/datasets/seanghay/khmer-hanuman-100k) | 100k |
| `mrrtmob` | [mrrtmob/khmer_english_ocr_image_line](https://huggingface.co/datasets/mrrtmob/khmer_english_ocr_image_line) | 12M lines, ~40.2GB, Khmer + English (gated — accept its terms on the hub first) |
| `kiteaether` | [KiteAether/c_khmer_gemma_ocr_train](https://huggingface.co/datasets/KiteAether/c_khmer_gemma_ocr_train) | Khmer OCR |

```bash
python scripts/train.py --datasets soyvitou                              # one (the default)
python scripts/train.py --datasets sokheng hanuman                       # combined
python scripts/train.py --datasets soyvitou sokheng hanuman mrrtmob kiteaether  # all five
python scripts/train.py --datasets seanghay/khmer-hanuman-100k           # raw hub id
```

The registry and loader live in [`src/trocr_khmer/data_sources.py`](src/trocr_khmer/data_sources.py).

Target machine: Ubuntu, i9-8950HK, 32GB RAM, RTX 3090 (24GB VRAM). **Requires CUDA** — the script asserts `torch.cuda.is_available()` and will refuse to run on CPU-only/Mac.

### Tokenizer

`trocr-base-printed`'s decoder natively uses a byte-level BPE (RoBERTa) tokenizer that round-trips any UTF-8 — including Khmer — exactly, so the default run pairs the model with its own tokenizer. (For consistency the code always sources the decoder tokenizer from `BYTE_LEVEL_TOKENIZER = microsoft/trocr-base-printed`; with the base model this is a no-op graft.) The smaller `trocr-small-printed` checkpoint instead ships an XLM-RoBERTa **SentencePiece** decoder tokenizer that can't handle per-character Khmer: it maps most Khmer code points to `<unk>`, and once we add per-character tokens its fast implementation crashes (`AddedVocabulary bad split`) while the slow one inserts spurious spaces between characters — so labels don't round-trip. When you fine-tune small, we therefore **graft the byte-level BPE tokenizer from `microsoft/trocr-base-printed`** onto the small checkpoint's image processor and resize the small decoder's token-embedding table + tied LM head to that vocab (the transformer weights stay the small model's; only the decoder's token embeddings/LM head are rebuilt and retrained).

A byte-level BPE round-trips any UTF-8, but encodes each Khmer character as its 3 raw UTF-8 byte tokens. So before training we also **extend the tokenizer with a dedicated token per Khmer character** (the Khmer block `U+1780–U+17FF`, the Khmer Symbols block `U+19E0–U+19FF`, and the zero-width word-separator characters). This gives each character its own token — shorter sequences (less truncation under `MAX_LABEL_LEN`) and a decoder that learns whole characters instead of shared UTF-8 lead bytes. The decoder's token embeddings/LM head are resized for these added tokens on both checkpoints. Fine-tuning the whole model then teaches it to read Khmer; this is the standard approach for adapting TrOCR to a non-Latin script.

The tokenizer sourcing lives in [`src/trocr_khmer/processor.py`](src/trocr_khmer/processor.py) (`BYTE_LEVEL_TOKENIZER`, `load_trocr_processor`); the Khmer character set and resize live in [`src/trocr_khmer/khmer.py`](src/trocr_khmer/khmer.py) (`add_khmer_tokens`). Both are invoked automatically by `scripts/train.py`. The added characters are regular (not special) tokens, so they round-trip through `decode(..., skip_special_tokens=True)` as normal text, and the extended tokenizer is saved into each run's `final/` folder for inference/eval. `sentencepiece` is a required dependency (the small checkpoint's original tokenizer needs it to load at all).

## Project layout

```
training/trocr/
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

# 2. Full fine-tune. Default: 8 epochs, batch size 8 x grad_accum 4 (effective 32), bf16,
#    microsoft/trocr-base-printed, on the soyvitou dataset. Override the base model with
#    --model_name and the training data with --datasets (see Datasets above); with
#    trocr-small-printed you can raise --train_batch_size (e.g. 32 --grad_accum 1).
python scripts/train.py
python scripts/train.py --datasets soyvitou sokheng hanuman mrrtmob kiteaether   # train on all five combined
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

By default it evaluates on the `soyvitou` validation split. Pass the same `--datasets` you trained with to reproduce training's held-out eval set (same seed / split), or point it at any dataset to test cross-dataset generalization:

```bash
python scripts/evaluate_model.py experiments/checkpoint-1/final --datasets sokheng hanuman
```

Optional quick evaluation with a smaller sample set:

```bash
python scripts/evaluate_model.py experiments/checkpoint-1/final --max_eval_samples 200
```



## Inference

```bash
python scripts/infer.py experiments/checkpoint-1/final some_image.jpg
```

