import argparse

import evaluate
import torch
from transformers import GenerationConfig, Seq2SeqTrainer, Seq2SeqTrainingArguments, VisionEncoderDecoderModel

from trocr_khmer import (
    DATASET_REGISTRY,
    DEFAULT_DATASETS,
    MAX_LABEL_LEN,
    OCRDataCollator,
    OCRDataset,
    build_compute_metrics,
    configure_generation,
    load_ocr_split,
    load_trocr_processor,
)

MODEL_NAME = "microsoft/trocr-small-printed"


def main():
    parser = argparse.ArgumentParser(description="Evaluate a TrOCR checkpoint or fine-tuned model on a Khmer OCR validation split")
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
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Optional limit for a quick evaluation run")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU not found — evaluation expects a GPU-enabled environment"

    split = load_ocr_split(args.datasets, test_size=0.03, seed=42)

    if args.max_eval_samples is not None:
        split["test"] = split["test"].select(range(min(args.max_eval_samples, len(split["test"]))))

    processor = load_trocr_processor(args.model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_dir)

    configure_generation(model, processor, MAX_LABEL_LEN)

    data_collator = OCRDataCollator(processor)
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
        output_dir="./eval_out",
        per_device_eval_batch_size=args.eval_batch_size,
        predict_with_generate=True,
        generation_max_length=MAX_LABEL_LEN,
        generation_num_beams=4,
        generation_config=khmer_generation,
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
