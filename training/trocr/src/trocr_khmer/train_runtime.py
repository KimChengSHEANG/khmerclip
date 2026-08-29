"""Runtime helpers that keep OCR train/eval GPU-bound instead of eval-bound."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import GenerationConfig


def limit_eval_dataset(dataset, max_samples: int | None):
    """Cap the held-out set used for CER so eval stays minutes, not hours.

    Full multi-dataset 3% splits can be hundreds of thousands of rows. Beam-search
    generate over that set monopolizes wall time and leaves the GPU under-fed vs
    teacher-forced training. Ranking checkpoints only needs a fixed subsample.
    """
    if max_samples is None:
        return dataset
    n = min(max_samples, len(dataset))
    if n == len(dataset):
        return dataset
    return dataset.select(range(n))


def build_generation_config(processor, max_length: int, num_beams: int = 1) -> GenerationConfig:
    """Khmer-safe generation settings.

    Mid-train / throughput-oriented eval should use ``num_beams=1`` (greedy): much
    higher GPU util than beam search while still ranking by CER. Pass ``num_beams=4``
    for final offline scoring when quality matters more than speed.
    """
    return GenerationConfig(
        decoder_start_token_id=processor.tokenizer.cls_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.sep_token_id,
        num_beams=num_beams,
        no_repeat_ngram_size=0,
        length_penalty=1.0,
        early_stopping=num_beams > 1,
        max_length=max_length,
        use_cache=True,
    )


@torch.inference_mode()
def run_cer_eval(
    model,
    dataset,
    data_collator,
    processor,
    cer_metric,
    *,
    batch_size: int,
    generation_config: GenerationConfig,
    num_workers: int = 8,
    prefetch_factor: int = 4,
) -> dict:
    """Batched greedy/beam CER eval without Seq2SeqTrainer overhead.

    Uses bf16 autocast + pinned DataLoader workers so generate() stays closer to
    compute-bound on a 3090 (Trainer eval often leaves VRAM and SMs half-idle).
    """
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=data_collator,
    )

    pad_token_id = processor.tokenizer.pad_token_id
    pred_texts: list[str] = []
    label_texts: list[str] = []

    device = next(model.parameters()).device
    # Autocast even if weights are already bf16 — generate path still benefits from TF32 matmuls.
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            generated = model.generate(pixel_values, generation_config=generation_config)

        pred_ids = generated.detach().cpu().numpy()
        label_ids = labels.numpy()
        label_ids = np.where(label_ids != -100, label_ids, pad_token_id)
        pred_texts.extend(processor.batch_decode(pred_ids, skip_special_tokens=True))
        label_texts.extend(processor.batch_decode(label_ids, skip_special_tokens=True))

    return {
        "eval_cer": float(cer_metric.compute(predictions=pred_texts, references=label_texts)),
        "eval_samples": len(pred_texts),
    }
