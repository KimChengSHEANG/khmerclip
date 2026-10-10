#!/usr/bin/env python3
"""Print mean CER per subset (printed / ui / mixed / all) for each engine in a bench.py CSV."""
import csv
import sys
from collections import defaultdict

by = defaultdict(list)
with open(sys.argv[1], encoding="utf-8") as f:
    for row in csv.DictReader(f):
        cer = float(row["cer"])
        by[(row["engine"], row["subset"])].append(cer)
        by[(row["engine"], "all")].append(cer)
for (engine, subset), cers in sorted(by.items()):
    print(f"{engine}\t{subset}\tn={len(cers)}\tcer={sum(cers) / len(cers):.4f}")
