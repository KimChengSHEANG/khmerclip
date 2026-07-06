"""CLI: evaluate a trained model on an LMDB test set (CER / WER / accuracy).

    python -m khmer_ocr.cli.evaluate --config configs/khmer.yaml \\
        --saved_model experiments/model-1/best_norm_ED.pth \\
        --eval_data data/lmdb/test/soyvitou --save_preds preds_test.csv
    khmer-ocr-eval --saved_model ...    # console script
"""
import argparse
import os

from khmer_ocr.config import load_config, load_charset
from khmer_ocr.evaluate import run_evaluation


def default_eval_data(valid_data):
    """Map a validation LMDB path to the matching test split."""
    if "/val/" in valid_data:
        return valid_data.replace("/val/", "/test/", 1)
    return os.path.join(os.path.dirname(valid_data), "test")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/khmer.yaml")
    ap.add_argument("--saved_model", required=True,
                    help="path to a trained checkpoint, e.g. experiments/model-1/best_norm_ED.pth")
    ap.add_argument("--eval_data", default=None,
                    help="LMDB dir (default: sibling 'test' of valid_data)")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save_preds", type=str, default=None, help="optional CSV output")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only first N batches")
    return ap.parse_args()


def main():
    args = parse_args()
    opt = load_config(args.config)
    opt.character = load_charset(opt.character_file)
    eval_data = args.eval_data or default_eval_data(opt.valid_data)

    run_evaluation(
        opt, saved_model=args.saved_model, eval_data=eval_data,
        batch_size=args.batch_size, workers=args.workers,
        save_preds=args.save_preds, limit=args.limit,
    )


if __name__ == "__main__":
    main()
