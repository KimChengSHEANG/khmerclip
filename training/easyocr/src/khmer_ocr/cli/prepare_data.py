"""CLI: download Khmer OCR datasets from the Hub and convert them to LMDB.

    python -m khmer_ocr.cli.prepare_data                         # soyvitou only
    python -m khmer_ocr.cli.prepare_data --dataset-count 3       # first 3 sources
    python -m khmer_ocr.cli.prepare_data --datasets all          # every source
    khmer-ocr-prepare --datasets sokheng,hanuman --max-samples 2000
"""
import argparse

from khmer_ocr.data import (
    DATASET_ORDER,
    DATASETS,
    build_dataset,
    default_workers,
    repair_lmdb,
    resolve_dataset_names,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="data", help="output root directory")
    ds_group = ap.add_mutually_exclusive_group()
    ds_group.add_argument(
        "--datasets",
        default=None,
        help="comma-separated keys or 'all' "
             f"({', '.join(DATASET_ORDER)})",
    )
    ds_group.add_argument(
        "--dataset-count",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="prepare the first N datasets from the canonical order "
             f"({', '.join(DATASET_ORDER)})",
    )
    ap.add_argument("--max-samples", type=int, default=None,
                    help="cap samples per HF split (for quick tests)")
    ap.add_argument("--map-size-gb", type=float, default=6.0,
                    help="default LMDB map size per split in GB (km_en_line uses 45)")
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"parallel encode processes (default: {default_workers()})",
    )
    ap.add_argument(
        "--list-datasets",
        action="store_true",
        help="print registered datasets and exit",
    )
    ap.add_argument(
        "--repair-lmdb",
        nargs="+",
        metavar="PATH",
        help="write missing num-samples keys into existing LMDB dirs and exit",
    )
    args = ap.parse_args()

    if args.repair_lmdb:
        for path in args.repair_lmdb:
            repair_lmdb(path)
        return

    if args.list_datasets:
        for key in DATASET_ORDER:
            spec = DATASETS[key]
            splits = ", ".join(spec["splits"])
            extra = f" [{spec['description']}]" if spec.get("description") else ""
            print(f"{key:12}  {spec['repo_id']}  splits: {splits}{extra}")
        return

    names = resolve_dataset_names(datasets=args.datasets, dataset_count=args.dataset_count)
    print(f"Preparing: {', '.join(names)}")

    workers = args.workers if args.workers is not None else default_workers()
    print(f"Using {workers} worker process(es)")

    build_dataset(
        out_dir=args.out_dir,
        dataset_names=names,
        max_samples=args.max_samples,
        map_size_gb=args.map_size_gb,
        workers=workers,
    )
    print("Done.")


if __name__ == "__main__":
    main()
