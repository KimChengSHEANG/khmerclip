"""CLI: train a Khmer EasyOCR recognition model.

    python -m khmer_ocr.cli.train --config configs/khmer.yaml
    python -m khmer_ocr.cli.train --dataset-count 3
    python -m khmer_ocr.cli.train --datasets soyvitou,sokheng --num_iter 300
    khmer-ocr-train --config configs/khmer.yaml       # console script

Checkpoints and logs are written to experiments/model-{N}/ (N auto-increments
per new run). Pass --saved_model experiments/model-1/best_norm_ED.pth to resume
that run in the same directory.
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from khmer_ocr.config import load_config, load_charset
from khmer_ocr.data import DATASET_ORDER, equal_batch_ratios, resolve_dataset_names
from khmer_ocr.experiments import EXPERIMENTS_ROOT, resolve_experiment_dir
from khmer_ocr.trainer.engine import train


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/khmer.yaml")
    ap.add_argument("--num_iter", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--valInterval", type=int, default=None)
    ap.add_argument("--saved_model", type=str, default=None,
                    help="initial or resume checkpoint (.pth); resume keeps the same experiments/model-N/ dir")
    ap.add_argument("--experiment_name", type=str, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--amp", action="store_true", help="mixed-precision training")
    ds_group = ap.add_mutually_exclusive_group()
    ds_group.add_argument(
        "--datasets",
        default=None,
        help="comma-separated training dataset keys or 'all' "
             f"({', '.join(DATASET_ORDER)})",
    )
    ds_group.add_argument(
        "--dataset-count",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="train on the first N prepared datasets "
             f"({', '.join(DATASET_ORDER)})",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    overrides = {
        "num_iter": args.num_iter,
        "batch_size": args.batch_size,
        "valInterval": args.valInterval,
        "saved_model": args.saved_model,
        "experiment_name": args.experiment_name,
        "workers": args.workers,
    }
    opt = load_config(args.config, overrides)
    opt.character = load_charset(opt.character_file)

    if args.datasets is not None or args.dataset_count is not None:
        names = resolve_dataset_names(
            datasets=args.datasets, dataset_count=args.dataset_count,
        )
        opt.select_data = names
        opt.batch_ratio = equal_batch_ratios(len(names))
    elif not hasattr(opt, "select_data"):
        opt.select_data = ["/"]
    elif isinstance(opt.select_data, str):
        opt.select_data = [opt.select_data]

    if isinstance(opt.batch_ratio, str):
        opt.batch_ratio = [r.strip() for r in opt.batch_ratio.split(",") if r.strip()]
    elif not hasattr(opt, "batch_ratio"):
        opt.batch_ratio = ["1"]

    experiments_root = getattr(opt, "experiments_root", None) or EXPERIMENTS_ROOT
    opt.experiment_dir = resolve_experiment_dir(opt.saved_model or "", root=experiments_root)
    os.makedirs(opt.experiment_dir, exist_ok=True)

    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)
    cudnn.benchmark = True
    cudnn.deterministic = False

    opt.num_gpu = torch.cuda.device_count()
    ds_label = ", ".join(opt.select_data)
    print(f"Training '{opt.experiment_name}' -> {opt.experiment_dir} | "
          f"datasets: {ds_label} | "
          f"{len(opt.character)} characters | "
          f"{'GPU x' + str(opt.num_gpu) if opt.num_gpu else 'CPU'}")
    train(opt, amp=args.amp)


if __name__ == "__main__":
    main()
