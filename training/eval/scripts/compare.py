#!/usr/bin/env python3
"""Paired comparison of two bench CSVs (same fixtures): character errors, fixtures improved /
worsened, CER and median latency per subset. This is the keep/drop gate of accuracy_v3.md.

    python scripts/compare.py results/base.csv results/cand.csv
"""
from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def char_errors(row: dict) -> int:
    return round(float(row["cer"]) * max(1, len(row["ref"])))


def paired(base: list[dict], cand: list[dict]) -> dict[str, dict]:
    by_id = {r["id"]: r for r in base}
    out: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    for c in cand:
        b = by_id[c["id"]]
        for subset in (c["subset"], "ALL"):
            s = out[subset]
            eb, ec = char_errors(b), char_errors(c)
            s["n"] += 1
            s["chars"] += max(1, len(c["ref"]))
            s["base_err"] += eb
            s["cand_err"] += ec
            s["improved"] += ec < eb
            s["worsened"] += ec > eb
            s.setdefault("base_ms", []).append(float(b["ms"]))
            s.setdefault("cand_ms", []).append(float(c["ms"]))
    return out


def main() -> None:
    base_path, cand_path = map(Path, sys.argv[1:3])
    load = lambda p: list(csv.DictReader(p.open(encoding="utf-8")))  # noqa: E731
    res = paired(load(base_path), load(cand_path))
    print(f"{'subset':12} {'n':>4} {'base_err':>8} {'cand_err':>8} {'delta%':>7} {'+':>4} {'-':>4} "
          f"{'base_cer':>8} {'cand_cer':>8} {'base_ms':>8} {'cand_ms':>8}")
    for subset in sorted(res, key=lambda k: (k == "ALL", k)):
        s = res[subset]
        delta = 100 * (s["cand_err"] - s["base_err"]) / max(1, s["base_err"])
        print(f"{subset:12} {int(s['n']):4d} {int(s['base_err']):8d} {int(s['cand_err']):8d} {delta:7.1f} "
              f"{int(s['improved']):4d} {int(s['worsened']):4d} {s['base_err'] / s['chars']:8.4f} "
              f"{s['cand_err'] / s['chars']:8.4f} {statistics.median(s['base_ms']):8.0f} "
              f"{statistics.median(s['cand_ms']):8.0f}")


if __name__ == "__main__":
    main()
