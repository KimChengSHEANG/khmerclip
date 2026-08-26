"""Evaluate a trained recognizer on an LMDB dataset.

Reports exact-match accuracy, CER, WER (via jiwer) and the normalized edit
distance used by the trainer.
"""
import csv

import torch
import torch.nn.functional as F
import torch.utils.data
import jiwer
from nltk.metrics.distance import edit_distance

from .config import load_charset
from .trainer.utils import CTCLabelConverter, AttnLabelConverter
from .trainer.dataset import hierarchical_dataset, AlignCollate
from .trainer.model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _strip_module_prefix(state_dict):
    if all(k.startswith("module.") for k in state_dict):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_model(opt, saved_model):
    """Build the model from `opt` and load weights (DataParallel-aware)."""
    if "CTC" in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    if opt.rgb:
        opt.input_channel = 3

    model = Model(opt).to(device)
    state = torch.load(saved_model, map_location=device)
    model.load_state_dict(_strip_module_prefix(state), strict=True)
    model.eval()
    return model, converter


def run_evaluation(opt, saved_model, eval_data, batch_size=128, workers=4,
                   save_preds=None, limit=None):
    """Run evaluation and return a metrics dict."""
    if not getattr(opt, "character", None):
        opt.character = load_charset(opt.character_file)
    opt.data_filtering_off = True  # never drop test samples
    if not hasattr(opt, "select_data"):
        opt.select_data = "/"
    opt.batch_size = batch_size
    opt.workers = workers

    model, converter = load_model(opt, saved_model)
    print(f"Loaded {saved_model}")

    align = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD,
                         contrast_adjust=opt.contrast_adjust)
    dataset, _ = hierarchical_dataset(root=eval_data, opt=opt)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.workers, collate_fn=align, pin_memory=True)
    print(f"Evaluating on {len(dataset)} samples from {eval_data}")

    gts, preds_all, confs_all = [], [], []
    with torch.no_grad():
        for bidx, (image_tensors, labels) in enumerate(loader):
            if limit is not None and bidx >= limit:
                break
            bs = image_tensors.size(0)
            image = image_tensors.to(device)
            text_for_pred = torch.LongTensor(bs, opt.batch_max_length + 1).fill_(0).to(device)

            if "CTC" in opt.Prediction:
                preds = model(image, text_for_pred)
                preds_size = torch.IntTensor([preds.size(1)] * bs)
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index.view(-1).data, preds_size.data)
            else:
                length_for_pred = torch.IntTensor([opt.batch_max_length] * bs).to(device)
                preds = model(image, text_for_pred, is_train=False)
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            for gt, pred, pmax in zip(labels, preds_str, preds_max_prob):
                if "Attn" in opt.Prediction:
                    pred = pred[: pred.find("[s]")]
                try:
                    conf = float(pmax.cumprod(dim=0)[-1])
                except IndexError:
                    conf = 0.0
                gts.append(gt)
                preds_all.append(pred)
                confs_all.append(conf)

    metrics = compute_metrics(gts, preds_all)

    print("\n================ Evaluation results ================")
    print(f"  Samples                : {metrics['n']}")
    print(f"  Exact-match accuracy   : {metrics['exact_acc']:.3f} %")
    print(f"  Character Error Rate   : {metrics['cer'] * 100:.3f} %")
    print(f"  Word Error Rate        : {metrics['wer'] * 100:.3f} %")
    print(f"  Normalized edit dist.  : {metrics['norm_ed']:.4f}  (1.0 = perfect)")
    print("====================================================")

    if save_preds:
        with open(save_preds, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ground_truth", "prediction", "confidence", "correct"])
            for g, p, c in zip(gts, preds_all, confs_all):
                w.writerow([g, p, f"{c:.4f}", int(g == p)])
        print(f"Wrote per-sample predictions to {save_preds}")

    return metrics


def compute_metrics(gts, preds):
    """Return dict with n, exact_acc, cer, wer, norm_ed."""
    n = len(gts)
    exact = sum(1 for g, p in zip(gts, preds) if g == p)
    exact_acc = exact / n * 100 if n else 0.0

    norm_ed = 0.0
    for g, p in zip(gts, preds):
        if len(g) == 0 or len(p) == 0:
            norm_ed += 0
        elif len(g) > len(p):
            norm_ed += 1 - edit_distance(p, g) / len(g)
        else:
            norm_ed += 1 - edit_distance(p, g) / len(p)
    norm_ed = norm_ed / n if n else 0.0

    pairs = [(g, p) for g, p in zip(gts, preds) if len(g) > 0]
    refs = [g for g, _ in pairs]
    hyps = [p for _, p in pairs]
    cer = jiwer.cer(refs, hyps) if refs else float("nan")
    wer = jiwer.wer(refs, hyps) if refs else float("nan")

    return {"n": n, "exact_acc": exact_acc, "cer": cer, "wer": wer, "norm_ed": norm_ed}
