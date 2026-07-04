import numpy as np
import torch
from PIL import Image

MAX_LABEL_LEN = 128


class OCRDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, processor, max_label_len=MAX_LABEL_LEN):
        self.dataset = hf_dataset
        self.processor = processor
        self.max_label_len = max_label_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        image = row["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            row["text"],
            padding="max_length",
            max_length=self.max_label_len,
            truncation=True,
        ).input_ids
        labels = [
            label if label != self.processor.tokenizer.pad_token_id else -100
            for label in labels
        ]
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}


def build_compute_metrics(processor, cer_metric):
    def compute_metrics(eval_pred):
        pred_ids, label_ids = eval_pred.predictions, eval_pred.label_ids
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        return {"cer": cer_metric.compute(predictions=pred_str, references=label_str)}

    return compute_metrics
