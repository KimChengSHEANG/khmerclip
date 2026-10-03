"""Evaluate a TrOCR checkpoint on a Khmer OCR validation split.

Faster than Seq2SeqTrainer.evaluate: bf16 autocast, large batches, KV cache, and a
plain DataLoader generate loop so the 3090 stays closer to compute-bound.
"""
import argparse
import time

import evaluate
import torch
from transformers import VisionEncoderDecoderModel

from trocr_khmer import (
    DATASET_REGISTRY,
    DEFAULT_DATASETS,
    MAX_LABEL_LEN,
    OCRDataCollator,
    OCRDataset,
    build_generation_config,
    configure_generation,
    limit_eval_dataset,
    load_ocr_datasets,
    load_ocr_split,
    load_trocr_processor,
    run_cer_eval,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a TrOCR checkpoint or fine-tuned model on a Khmer OCR validation split"
    )
    parser.add_argument("model_dir", help="Directory containing the model and processor")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        metavar="NAME",
        help=(
            "Dataset(s) to evaluate on; the validation split is carved with the same seed as "
            "training, so pass the same --datasets you trained with to reproduce that eval set. "
            "Known aliases: " + ", ".join(DATASET_REGISTRY) + "."
        ),
    )
    # Greedy + batch 64 fills ~18–20GB and lifts util vs the old batch-16 / beam-4 Trainer path.
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="1=greedy (fast, default); 4=higher-quality CER (slower, lower GPU util)",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Optional limit for a quick evaluation run",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Evaluate every row of --datasets (for dedicated test sets like handwriting_test, synth_val)",
    )
    parser.add_argument("--max_eval_per_source", type=int, default=None)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU not found — evaluation expects a GPU-enabled environment"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.full:
        eval_hf = load_ocr_datasets(args.datasets)
    else:
        eval_hf = load_ocr_split(
            args.datasets, test_size=0.03, seed=42, max_eval_per_source=args.max_eval_per_source
        )["test"]
    eval_hf = limit_eval_dataset(eval_hf, args.max_eval_samples)

    processor = load_trocr_processor(args.model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_dir)
    model.cuda()
    if torch.cuda.is_bf16_supported():
        model = model.to(dtype=torch.bfloat16)

    configure_generation(model, processor, MAX_LABEL_LEN)
    generation_config = build_generation_config(
        processor, MAX_LABEL_LEN, num_beams=args.num_beams
    )

    data_collator = OCRDataCollator(processor)
    eval_dataset = OCRDataset(eval_hf, processor)
    cer_metric = evaluate.load("cer")

    print(
        f"Evaluating {len(eval_dataset)} samples "
        f"(batch={args.eval_batch_size}, beams={args.num_beams}, bf16={torch.cuda.is_bf16_supported()})"
    )
    t0 = time.perf_counter()
    metrics = run_cer_eval(
        model,
        eval_dataset,
        data_collator,
        processor,
        cer_metric,
        batch_size=args.eval_batch_size,
        generation_config=generation_config,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    elapsed = time.perf_counter() - t0
    metrics["eval_runtime"] = elapsed
    metrics["eval_samples_per_second"] = metrics["eval_samples"] / elapsed if elapsed > 0 else 0.0
    print(metrics)


if __name__ == "__main__":
    main()
