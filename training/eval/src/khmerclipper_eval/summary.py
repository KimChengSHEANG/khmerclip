from __future__ import annotations

import statistics
from collections import defaultdict


def aggregate(rows: list[dict]) -> dict[str, dict]:
    by_engine: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_engine[row["engine"]].append(row)
    out: dict[str, dict] = {}
    for engine, items in by_engine.items():
        cers = [float(r["cer"]) for r in items]
        mss = [float(r["ms"]) for r in items]
        mixed = [float(r["cer"]) for r in items if r.get("subset") == "mixed"]
        ui = [float(r["cer"]) for r in items if r.get("subset") == "ui"]
        out[engine] = {
            "n": len(items),
            "cer": sum(cers) / len(cers),
            "median_ms": statistics.median(mss),
            "mixed_cer": (sum(mixed) / len(mixed)) if mixed else 0.0,
            "ui_cer": (sum(ui) / len(ui)) if ui else 0.0,
        }
    return out


def passes_gate(candidate: dict, tesseract: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not (candidate["cer"] < tesseract["cer"]):
        reasons.append(
            f"CER {candidate['cer']:.4f} not < Tess {tesseract['cer']:.4f}"
        )
    if not (candidate["median_ms"] <= 2.0 * tesseract["median_ms"]):
        reasons.append(
            f"median_ms {candidate['median_ms']:.1f} > 2× Tess {tesseract['median_ms']:.1f}"
        )
    if not (candidate["mixed_cer"] <= tesseract["mixed_cer"]):
        reasons.append(
            f"mixed_cer {candidate['mixed_cer']:.4f} > Tess {tesseract['mixed_cer']:.4f}"
        )
    if candidate.get("ui_cer", 0.0) > tesseract.get("ui_cer", 0.0):
        reasons.append(
            f"ui_cer {candidate.get('ui_cer', 0.0):.4f} > Tess {tesseract.get('ui_cer', 0.0):.4f}"
        )
    return (len(reasons) == 0, reasons)
