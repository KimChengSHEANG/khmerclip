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
    GenerationConfig,
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
    configure_generation,
    load_ocr_split,
    load_trocr_processor,
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
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=2)  # effective batch = 16 * 2 = 32, ~fills base model in 24GB
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--smoke_test", action="store_true", help="200 samples, 1 epoch, verifies the pipeline runs end to end")
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

    model.config.decoder.loss_type = "ForCausalLMLoss"  
    
    configure_generation(model, processor, MAX_LABEL_LEN)

    data_collator = OCRDataCollator(processor)
    train_dataset = OCRDataset(split["train"], processor)
    eval_dataset = OCRDataset(split["test"], processor)

    cer_metric = evaluate.load("cer")

    khmer_generation = GenerationConfig(
        decoder_start_token_id=processor.tokenizer.cls_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.sep_token_id,
        num_beams=4,
        no_repeat_ngram_size=0,
        length_penalty=1.0,
        early_stopping=True,
    )

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
        generation_num_beams=4,
        generation_config=khmer_generation,
        eval_strategy="steps",
        eval_steps=50 if args.smoke_test else 1000,
        save_steps=50 if args.smoke_test else 1000,
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
