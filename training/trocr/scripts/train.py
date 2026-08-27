"""Fine-tune TrOCR on one or more Khmer OCR datasets.

Train on a single dataset or combine several (see trocr_khmer.data_sources for the registry):
    python scripts/train.py --datasets soyvitou
    python scripts/train.py --datasets sokheng hanuman
    python scripts/train.py --datasets soyvitou sokheng hanuman

Each run gets its own auto-incrementing folder under experiments/:
    experiments/checkpoint-1/          <- first run (HF checkpoints + final/ model)
    experiments/checkpoint-2/          <- next run, and so on

Run on Ubuntu + CUDA GPU (tested target: RTX 3090, 24GB VRAM):
    python scripts/train.py
    python scripts/train.py --smoke_test          # 200-sample sanity run before the real thing
    python scripts/train.py --resume_from_checkpoint experiments/checkpoint-1/checkpoint-1000
"""
import argparse
import re
from pathlib import Path

import evaluate
import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    VisionEncoderDecoderModel,
)

from trocr_khmer import (
    BYTE_LEVEL_TOKENIZER,
    DATASET_REGISTRY,
    DEFAULT_DATASETS,
    MAX_LABEL_LEN,
    OCRDataCollator,
    OCRDataset,
    add_khmer_tokens,
    build_compute_metrics,
    build_generation_config,
    configure_generation,
    limit_eval_dataset,
    load_ocr_split,
    load_trocr_processor,
    patch_ocr_loss,
)

# trocr-base-printed: its decoder natively uses the byte-level BPE (RoBERTa) tokenizer that
# round-trips per-character Khmer, so grafting BYTE_LEVEL_TOKENIZER (also base's) is a no-op
# here — we only resize the decoder for the added Khmer tokens (see add_khmer_tokens).
# Override with --model_name microsoft/trocr-small-printed to fine-tune the smaller/faster
# checkpoint instead; its own XLM-RoBERTa SentencePiece tokenizer can't round-trip Khmer, so
# the graft in load_trocr_processor replaces it with the byte-level BPE.
MODEL_NAME = "microsoft/trocr-base-printed"


def next_experiment_dir(experiments_dir: str) -> Path:
    """Return experiments/checkpoint-N for the next unused, auto-incrementing N."""
    base = Path(experiments_dir)
    base.mkdir(parents=True, exist_ok=True)
    existing = [
        int(m.group(1))
        for p in base.iterdir()
        if p.is_dir() and (m := re.fullmatch(r"checkpoint-(\d+)", p.name))
    ]
    return base / f"checkpoint-{max(existing, default=0) + 1}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=MODEL_NAME, help="Base TrOCR checkpoint to fine-tune")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        metavar="NAME",
        help=(
            "One or more datasets to train on (concatenated if several). Known aliases: "
            + ", ".join(DATASET_REGISTRY) + ". Any other value is treated as a raw hub id."
        ),
    )
    parser.add_argument("--experiments_dir", default="experiments")
    parser.add_argument("--output_dir", default=None, help="Run folder; defaults to the next experiments/checkpoint-N")
    parser.add_argument("--epochs", type=int, default=10)
    # Batch 32 fills ~18GB on a 3090 for training; mid-train CER uses greedy so eval can go higher.
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Run mid-train CER every N optimizer steps (default: 50 smoke / 2000 full)",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=4096,
        help="Cap mid-train CER eval size (full 3%% splits of multi-dataset runs are huge). Pass 0 for no cap.",
    )
    parser.add_argument(
        "--eval_num_beams",
        type=int,
        default=1,
        help="Beams for mid-train CER only (default greedy=1). Use scripts/evaluate_model.py --num_beams 4 for final CER.",
    )
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--smoke_test", action="store_true", help="200 samples, 1 epoch, verifies the pipeline runs end to end")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply light train-time image augmentations (blur, contrast, JPEG, small rotate)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else next_experiment_dir(args.experiments_dir)
    final_dir = output_dir / "final"
    print(f"Writing this run to {output_dir}")

    assert torch.cuda.is_available(), "CUDA GPU not found — this script expects the RTX 3090 box, not CPU/Mac"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    split = load_ocr_split(args.datasets, test_size=0.03, seed=42, smoke_test=args.smoke_test)

    # Graft the byte-level BPE tokenizer onto the checkpoint's image processor (the small
    # model's own SentencePiece tokenizer can't round-trip Khmer); the decoder is resized to
    # this tokenizer's vocab below.
    processor = load_trocr_processor(args.model_name, tokenizer_name=BYTE_LEVEL_TOKENIZER)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name)

    # Add a dedicated token per Khmer character (one token per character instead of 3 UTF-8
    # byte tokens) and resize the decoder's embeddings + tied LM head to the grafted
    # tokenizer's vocab. Must run before configure_generation, which reads the resized decoder
    # vocab size into the config.
    num_added = add_khmer_tokens(processor, model)
    print(f"Added {num_added} Khmer tokens; decoder vocab is now {model.config.decoder.vocab_size}")

    patch_ocr_loss(model)
    configure_generation(model, processor, MAX_LABEL_LEN)

    data_collator = OCRDataCollator(processor)
    train_dataset = OCRDataset(split["train"], processor, augment=args.augment)
    # Cap mid-train CER: beam-4 generate over a huge holdout leaves the GPU ~60% util and
    # can burn hours per eval. Greedy + capped subset keeps ranking signal and GPU busy training.
    max_eval = None if args.max_eval_samples == 0 else args.max_eval_samples
    eval_hf = limit_eval_dataset(split["test"], max_eval)
    print(
        f"Mid-train CER eval on {len(eval_hf)} / {len(split['test'])} held-out rows "
        f"(beams={args.eval_num_beams}, batch={args.eval_batch_size})"
    )
    eval_dataset = OCRDataset(eval_hf, processor)

    cer_metric = evaluate.load("cer")

    # Greedy mid-train CER; saved final/ model still gets beam-4 via configure_generation.
    train_generation = build_generation_config(
        processor, MAX_LABEL_LEN, num_beams=args.eval_num_beams
    )
    eval_steps = args.eval_steps if args.eval_steps is not None else (50 if args.smoke_test else 2000)
    save_steps = eval_steps

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=1 if args.smoke_test else args.epochs,
        learning_rate=args.lr,
        warmup_steps=500,
        bf16=True,
        predict_with_generate=True,
        generation_max_length=MAX_LABEL_LEN,
        generation_num_beams=args.eval_num_beams,
        generation_config=train_generation,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=3,
        logging_steps=10 if args.smoke_test else 50,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=args.num_workers > 0,
        dataloader_prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        report_to=["tensorboard"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(processor, cer_metric),
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    configure_generation(trainer.model, processor, MAX_LABEL_LEN)
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"Saved fine-tuned model to {final_dir}")


if __name__ == "__main__":
    main()
