import argparse

import evaluate
import torch
from datasets import load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, VisionEncoderDecoderModel

from trocr_khmer import (
    MAX_LABEL_LEN,
    OCRDataCollator,
    OCRDataset,
    build_compute_metrics,
    configure_generation,
    load_trocr_processor,
)

MODEL_NAME = "microsoft/trocr-base-printed"


def main():
    parser = argparse.ArgumentParser(description="Evaluate a TrOCR checkpoint or fine-tuned model on the Khmer OCR validation split")
    parser.add_argument("model_dir", help="Directory containing the model and processor")
    parser.add_argument("--dataset_name", default="SoyVitou/62k-images-khmer-printed-dataset")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Optional limit for a quick evaluation run")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU not found — evaluation expects a GPU-enabled environment"

    dataset = load_dataset(args.dataset_name, split="train")
    split = dataset.train_test_split(test_size=0.03, seed=42)

    if args.max_eval_samples is not None:
        split["test"] = split["test"].select(range(min(args.max_eval_samples, len(split["test"]))))

    processor = load_trocr_processor(args.model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_dir)

    configure_generation(model, processor, MAX_LABEL_LEN)

    data_collator = OCRDataCollator(processor)
    eval_dataset = OCRDataset(split["test"], processor)
    cer_metric = evaluate.load("cer")

    training_args = Seq2SeqTrainingArguments(
        output_dir="./eval_out",
        per_device_eval_batch_size=args.eval_batch_size,
        predict_with_generate=True,
        generation_max_length=MAX_LABEL_LEN,
        generation_num_beams=1,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=args.num_workers > 0,
        dataloader_prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        report_to=[]
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(processor, cer_metric),
    )

    metrics = trainer.evaluate()
    print(metrics)


if __name__ == "__main__":
    main()
