"""Load TrOCRProcessor reliably across transformers versions."""

from transformers import RobertaTokenizer, TrOCRProcessor, ViTImageProcessor


def load_trocr_processor(model_name_or_path: str) -> TrOCRProcessor:
    """Load TrOCRProcessor from a hub id or local checkpoint directory.

    transformers 5.x routes AutoTokenizer through TokenizersBackend, which fails for
    microsoft/trocr-base-printed because the hub checkpoint only ships slow Roberta
    vocab files (no tokenizer.json). Building the processor from its parts works.
    """
    try:
        return TrOCRProcessor.from_pretrained(model_name_or_path)
    except ValueError as exc:
        if "Couldn't instantiate the backend tokenizer" not in str(exc):
            raise

    image_processor = ViTImageProcessor.from_pretrained(model_name_or_path)
    tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)
    return TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
