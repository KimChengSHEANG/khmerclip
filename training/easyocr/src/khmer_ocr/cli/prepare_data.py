"""CLI: download the Khmer dataset from the Hub and convert it to LMDB.

    python -m khmer_ocr.cli.prepare_data                    # full dataset
    python -m khmer_ocr.cli.prepare_data --max-samples 2000 # quick subset
    khmer-ocr-prepare --out-dir data                        # console script
"""
import argparse

from khmer_ocr.data import build_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="data", help="output root directory")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="cap samples per split (for quick tests)")
    ap.add_argument("--map-size-gb", type=float, default=6.0,
                    help="LMDB map size per split in GB")
    args = ap.parse_args()

    build_dataset(out_dir=args.out_dir, max_samples=args.max_samples,
                  map_size_gb=args.map_size_gb)
    print("Done.")


if __name__ == "__main__":
    main()
