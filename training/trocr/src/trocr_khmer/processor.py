"""Load TrOCRProcessor reliably across transformers versions."""

from transformers import AutoTokenizer, TrOCRProcessor, ViTImageProcessor

# Byte-level BPE (RoBERTa) tokenizer used for the decoder. trocr-small-printed's own
# XLM-RoBERTa SentencePiece tokenizer cannot round-trip per-character Khmer: the fast
# tokenizer panics on our added tokens ("AddedVocabulary bad split") and the slow one
# inserts spurious spaces between them (so labels don't round-trip). The byte-level BPE from
# trocr-base-printed round-trips arbitrary UTF-8 — including the added Khmer tokens — exactly,
# so we graft it onto the small checkpoint at train time (see load_trocr_processor).
BYTE_LEVEL_TOKENIZER = "microsoft/trocr-base-printed"


def configure_generation(model, processor, max_length):
    """Set the special-token ids and generation params for Khmer OCR.

    IMPORTANT: the printed TrOCR checkpoints ship generation defaults tuned for English —
    ``no_repeat_ngram_size=3`` and ``length_penalty=2.0``. Those are wrong for Khmer OCR.
    An n-gram block forbids re-emitting short repeated pieces (Khmer packs repeated
    consonant/vowel clusters, and the byte-level fallback makes every character share the
    same UTF-8 lead bytes), which strangles the decoder a few characters into every
    sentence — the model learns Khmer fine (low teacher-forced loss) but generation
    collapses to garbage (~0.82 CER). Disable the n-gram block and use a neutral length
    penalty for exact transcription.
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
        model.generation_config.update(
            decoder_start_token_id=model.config.decoder_start_token_id,
            pad_token_id=model.config.pad_token_id,
            eos_token_id=model.config.eos_token_id,
            max_length=max_length,
            no_repeat_ngram_size=0,
            length_penalty=1.0,
            num_beams=4,
            early_stopping=True,
        )
    return model


def load_trocr_processor(model_name_or_path: str, tokenizer_name: str | None = None) -> TrOCRProcessor:
    """Load TrOCRProcessor from a hub id or local checkpoint directory.

    Pass ``tokenizer_name`` to pair a *different* tokenizer with the checkpoint's image
    processor. Training does this to graft the byte-level ``BYTE_LEVEL_TOKENIZER`` onto
    trocr-small-printed (whose own SentencePiece tokenizer can't round-trip Khmer); the
    decoder embeddings are then resized to match by ``add_khmer_tokens``. Leave it ``None``
    to load the processor saved next to a checkpoint (eval/inference) — the grafted,
    Khmer-extended tokenizer is written into each run's ``final/`` folder.

    When ``tokenizer_name`` is ``None`` we try ``TrOCRProcessor.from_pretrained`` first;
    transformers 5.x can route the fast tokenizer through a TokenizersBackend that fails when
    a checkpoint ships only slow vocab files (no tokenizer.json), so we fall back to building
    the processor from its parts.
    """
    if tokenizer_name is None:
        try:
            return TrOCRProcessor.from_pretrained(model_name_or_path)
        except ValueError as exc:
            if "Couldn't instantiate the backend tokenizer" not in str(exc):
                raise
        tokenizer_name = model_name_or_path

    image_processor = ViTImageProcessor.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
