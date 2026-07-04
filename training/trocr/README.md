# TrOCR fine-tune — Khmer printed text

Fine-tunes `microsoft/trocr-base-printed` on [SoyVitou/62k-images-khmer-printed-dataset](https://huggingface.co/datasets/SoyVitou/62k-images-khmer-printed-dataset) (62.7k image/text pairs, mixed Khmer + English printed text).

Target machine: Ubuntu, i9-8950HK, 32GB RAM, RTX 3090 (24GB VRAM). **Requires CUDA** — the script asserts `torch.cuda.is_available()` and will refuse to run on CPU-only/Mac.

TrOCR's decoder tokenizer is a byte-level BPE (RoBERTa), which encodes any UTF-8 text — including Khmer — as byte tokens. Fine-tuning the whole model (not just a head) teaches it to produce fluent Khmer byte sequences; this is the standard approach used for adapting TrOCR to non-Latin scripts.

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

# 2. Full fine-tune. Default: 8 epochs, batch size 16, bf16.
python scripts/train.py
```

Checkpoints land in `trocr-khmer-checkpoints/` (best-CER checkpoint kept, plus last 2). Final model + processor saved to `trocr-khmer-final/`.

Resume an interrupted run:

```bash
python scripts/train.py --resume_from_checkpoint trocr-khmer-checkpoints/checkpoint-1000
```

Watch progress:

```bash
tensorboard --logdir trocr-khmer-checkpoints
```

Defaults target full RTX 3090 utilization (~18–20GB VRAM at batch 16). If GPU util stays low, try `--train_batch_size 20`. If you hit OOM, drop `--train_batch_size` or raise `--grad_accum` to keep the same effective batch.

## Evaluation

Run evaluation independently against any saved checkpoint or fine-tuned model:

```bash
python scripts/evaluate_model.py trocr-khmer-final
```

Optional quick evaluation with a smaller sample set:

```bash
python scripts/evaluate_model.py trocr-khmer-final --max_eval_samples 200
```



## Inference

```bash
python scripts/infer.py trocr-khmer-final some_image.jpg
```

