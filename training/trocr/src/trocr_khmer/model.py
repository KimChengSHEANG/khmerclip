"""VisionEncoderDecoder helpers for Khmer OCR training."""

from transformers import VisionEncoderDecoderModel
from transformers.loss.loss_utils import ForCausalLMLoss


def patch_ocr_loss(model: VisionEncoderDecoderModel) -> VisionEncoderDecoderModel:
    """Align teacher-forced training loss with autoregressive generation.

    ``encode_ocr_labels`` drops the tokenizer's leading ``<s>`` so
    ``shift_tokens_right`` builds ``[<s>, t0, t1, …]`` — the same prefix
    ``generate()`` uses.  The stock ``ForCausalLMLoss`` then shifts labels
    again (``logits[i]`` targets ``labels[i+1]``), so the first character is
    never trained and every position is off by one.  Pass ``shift_labels=labels``
    so ``logits[i]`` targets ``labels[i]`` directly.
    """

    def loss_function(logits, labels, vocab_size, num_items_in_batch=None, **kwargs):
        return ForCausalLMLoss(
            logits=logits,
            labels=labels,
            shift_labels=labels,
            vocab_size=vocab_size,
            num_items_in_batch=num_items_in_batch,
            **kwargs,
        )

    model.loss_function = loss_function
    model.config.decoder.loss_type = "ForCausalLMLoss"
    return model
