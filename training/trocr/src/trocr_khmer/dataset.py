import numpy as np
import torch
from PIL import Image

MAX_LABEL_LEN = 128


def encode_ocr_labels(tokenizer, text: str, max_label_len: int = MAX_LABEL_LEN) -> list[int]:
    """Tokenize OCR ground truth for VisionEncoderDecoder training.

    The RoBERTa decoder tokenizer adds a leading ``<s>`` (``cls_token_id``), but
    ``VisionEncoderDecoderModel`` also prepends ``decoder_start_token_id`` via
    ``shift_tokens_right``. Keeping both makes training see ``<s><s>text</s>`` while
    generation starts ``<s>text`` — teacher-forced loss looks fine but autoregressive
    generation. Drop the tokenizer's leading BOS; keep the trailing ``</s>``.
    """
    input_ids = tokenizer(
        text,
        max_length=max_label_len,
        truncation=True,
    ).input_ids
    if input_ids and input_ids[0] == tokenizer.cls_token_id:
        input_ids = input_ids[1:]
    return input_ids


def _load_rgb_image(image) -> Image.Image:
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    return image.convert("RGB")


class OCRDataset(torch.utils.data.Dataset):
    """Does all per-sample preprocessing (ViT resize/normalize + tokenization) inside the
    DataLoader worker, so it runs in parallel across workers and is prefetched ahead of the GPU.

    The ViT preprocessing replicates ``ViTImageProcessor`` (resize -> rescale -> normalize) with a
    PIL resize + torch path, which is far faster than HF's slow numpy image processor and keeps the
    CPU workers from starving the GPU. Params are read off the processor so they stay in sync.
    """

    def __init__(self, hf_dataset, processor, max_label_len=MAX_LABEL_LEN):
        self.dataset = hf_dataset
        self.processor = processor
        self.max_label_len = max_label_len

        ip = processor.image_processor
        self.size = (ip.size["width"], ip.size["height"])  # PIL wants (w, h)
        self.resample = ip.resample  # 2 == PIL bilinear
        self.rescale_factor = ip.rescale_factor
        # shape (3, 1, 1) so it broadcasts over CHW pixel tensors
        self.mean = torch.tensor(ip.image_mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(ip.image_std, dtype=torch.float32).view(-1, 1, 1)

    def __len__(self):
        return len(self.dataset)

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        image = image.resize(self.size, self.resample)
        arr = torch.from_numpy(np.asarray(image, dtype=np.float32))  # HWC
        pixel_values = arr.permute(2, 0, 1).mul_(self.rescale_factor)  # CHW, rescaled
        pixel_values.sub_(self.mean).div_(self.std)
        return pixel_values

    def __getitem__(self, idx):
        row = self.dataset[idx]
        image = _load_rgb_image(row["image"])
        pixel_values = self._preprocess_image(image)
        input_ids = encode_ocr_labels(
            self.processor.tokenizer,
            row["text"],
            max_label_len=self.max_label_len,
        )
        return {"pixel_values": pixel_values, "labels": input_ids}


class OCRDataCollator:
    """Stacks worker-preprocessed pixel values and dynamically pads labels to the batch max."""

    def __init__(self, processor, max_label_len=MAX_LABEL_LEN):
        self.processor = processor
        self.max_label_len = max_label_len
        self.pad_token_id = processor.tokenizer.pad_token_id

    def __call__(self, features):
        pixel_values = torch.stack([f["pixel_values"] for f in features])

        max_len = max(len(f["labels"]) for f in features)
        labels = torch.full((len(features), max_len), -100, dtype=torch.long)
        for i, f in enumerate(features):
            ids = f["labels"]
            labels[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

        return {"pixel_values": pixel_values, "labels": labels}


def build_compute_metrics(processor, cer_metric):
    def compute_metrics(eval_pred):
        pred_ids, label_ids = eval_pred.predictions, eval_pred.label_ids
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        return {"cer": cer_metric.compute(predictions=pred_str, references=label_str)}

    return compute_metrics
