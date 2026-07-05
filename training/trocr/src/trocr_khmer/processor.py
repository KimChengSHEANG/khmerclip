"""Load TrOCRProcessor reliably across transformers versions."""

from transformers import RobertaTokenizer, TrOCRProcessor, ViTImageProcessor


def configure_generation(model, processor, max_length):
    """Set the special-token ids and generation params for Khmer OCR.

    IMPORTANT: microsoft/trocr-base-printed ships with ``no_repeat_ngram_size=3`` and
    ``length_penalty=2.0``. Those defaults are tuned for English *word-level* BPE. This
    pipeline feeds Khmer through the byte-level BPE tokenizer, where every Khmer character
    shares the same UTF-8 lead bytes (E1 9E / E1 9F), so byte-trigrams repeat constantly.
    A ``no_repeat_ngram_size`` constraint then forbids re-emitting those trigrams and
    strangles the decoder a few characters into every sentence — the model learns Khmer
    fine (low teacher-forced loss) but generation collapses to garbage (~0.82 CER).
    Disable the n-gram block and use a neutral length penalty for exact transcription.
    """
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.max_length = max_length
    model.config.no_repeat_ngram_size = 0  # never n-gram-block byte-level Khmer
    model.config.length_penalty = 1.0
    model.config.num_beams = 4
    model.config.early_stopping = True
    # Keep generation_config (saved to disk, read by generate()) in sync with model.config.
    if getattr(model, "generation_config", None) is not None:
        gc = model.generation_config
        gc.decoder_start_token_id = model.config.decoder_start_token_id
        gc.pad_token_id = model.config.pad_token_id
        gc.eos_token_id = model.config.eos_token_id
        gc.max_length = max_length
        gc.no_repeat_ngram_size = 0
        gc.length_penalty = 1.0
        gc.num_beams = 4
        gc.early_stopping = True
    return model


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
