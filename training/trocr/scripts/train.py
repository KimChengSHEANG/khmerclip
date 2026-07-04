"""Fine-tune TrOCR-base on the SoyVitou/62k-images-khmer-printed-dataset.

Run on Ubuntu + CUDA GPU (tested target: RTX 3090, 24GB VRAM):
    python scripts/train.py
    python scripts/train.py --smoke_test          # 200-sample sanity run before the real thing
    python scripts/train.py --resume_from_checkpoint trocr-khmer-checkpoints/checkpoint-1000
"""
import argparse

import evaluate
import torch
from datasets import load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    VisionEncoderDecoderModel,
)

from trocr_khmer import MAX_LABEL_LEN, OCRDataCollator, OCRDataset, build_compute_metrics, load_trocr_processor

MODEL_NAME = "microsoft/trocr-base-printed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="trocr-khmer-checkpoints")
    parser.add_argument("--final_dir", default="trocr-khmer-final")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--smoke_test", action="store_true", help="200 samples, 1 epoch, verifies the pipeline runs end to end")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU not found — this script expects the RTX 3090 box, not CPU/Mac"
    torch.backends.cudnn.benchmark = True

    if args.smoke_test:
        stream = load_dataset("SoyVitou/62k-images-khmer-printed-dataset", split="train", streaming=True)
        from datasets import Dataset
        dataset = Dataset.from_list(list(stream.take(200)))
    else:
        dataset = load_dataset("SoyVitou/62k-images-khmer-printed-dataset", split="train")
    split = dataset.train_test_split(test_size=0.03, seed=42)

    processor = load_trocr_processor(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = MAX_LABEL_LEN
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    data_collator = OCRDataCollator(processor)
    train_dataset = OCRDataset(split["train"], processor)
    eval_dataset = OCRDataset(split["test"], processor)

    cer_metric = evaluate.load("cer")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=1 if args.smoke_test else args.epochs,
        learning_rate=args.lr,
        warmup_steps=500,
        bf16=True,
        predict_with_generate=True,
        generation_max_length=MAX_LABEL_LEN,
        generation_num_beams=1,
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

    trainer.save_model(args.final_dir)
    processor.save_pretrained(args.final_dir)
    print(f"Saved fine-tuned model to {args.final_dir}")


if __name__ == "__main__":
    main()
