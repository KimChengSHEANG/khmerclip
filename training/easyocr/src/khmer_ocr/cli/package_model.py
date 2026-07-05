"""CLI: export a checkpoint for use with the official `easyocr` package.

    python -m khmer_ocr.cli.package_model --config configs/khmer.yaml \\
        --saved_model saved_models/khmer_g2/best_norm_ED.pth --name khmer_g2
    khmer-ocr-package --saved_model ...   # console script

Then:
    import easyocr
    reader = easyocr.Reader(['en'], recog_network='khmer_g2')
    reader.readtext('line.jpg', detail=0)
"""
import argparse
import os

from khmer_ocr.config import load_config
from khmer_ocr.packaging import package_model


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/khmer.yaml")
    ap.add_argument("--saved_model", required=True)
    ap.add_argument("--name", default=None, help="custom recog_network name")
    ap.add_argument("--easyocr-dir", default=os.path.expanduser("~/.EasyOCR"))
    args = ap.parse_args()

    opt = load_config(args.config)
    name = args.name or opt.experiment_name
    # Directory that contains the `khmer_ocr` package (this file is .../khmer_ocr/cli/).
    src_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    dst_pth, py_path, yaml_path = package_model(
        opt, saved_model=args.saved_model, name=name,
        easyocr_dir=args.easyocr_dir, src_path=src_path,
    )

    print(f"Installed custom EasyOCR model '{name}':")
    print(f"  {dst_pth}")
    print(f"  {py_path}")
    print(f"  {yaml_path}")
    print(f"\nUse it with:\n  reader = easyocr.Reader(['en'], recog_network='{name}')")


if __name__ == "__main__":
    main()
