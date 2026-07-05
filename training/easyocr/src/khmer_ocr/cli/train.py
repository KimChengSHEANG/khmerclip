"""CLI: train a Khmer EasyOCR recognition model.

    python -m khmer_ocr.cli.train --config configs/khmer.yaml
    python -m khmer_ocr.cli.train --config configs/khmer.yaml --num_iter 300 --batch_size 32
    khmer-ocr-train --config configs/khmer.yaml       # console script
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from khmer_ocr.config import load_config, load_charset
from khmer_ocr.trainer.engine import train


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/khmer.yaml")
    ap.add_argument("--num_iter", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--valInterval", type=int, default=None)
    ap.add_argument("--saved_model", type=str, default=None)
    ap.add_argument("--experiment_name", type=str, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--amp", action="store_true", help="mixed-precision training")
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

    if not hasattr(opt, "select_data"):
        opt.select_data = "/"
    if not hasattr(opt, "batch_ratio"):
        opt.batch_ratio = "1"

    os.makedirs(f"./saved_models/{opt.experiment_name}", exist_ok=True)

    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)
    cudnn.benchmark = True
    cudnn.deterministic = False

    opt.num_gpu = torch.cuda.device_count()
    print(f"Training '{opt.experiment_name}' | {len(opt.character)} characters | "
          f"{'GPU x' + str(opt.num_gpu) if opt.num_gpu else 'CPU'}")
    train(opt, amp=args.amp)


if __name__ == "__main__":
    main()
