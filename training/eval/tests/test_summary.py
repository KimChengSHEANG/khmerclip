import pytest

from khmerclip_eval.summary import aggregate, passes_gate


def test_aggregate_median_and_cer():
    rows = [
        {"engine": "tesseract", "cer": 0.2, "ms": 10.0, "subset": "printed"},
        {"engine": "tesseract", "cer": 0.0, "ms": 30.0, "subset": "mixed"},
        {"engine": "trocr", "cer": 0.1, "ms": 20.0, "subset": "printed"},
        {"engine": "trocr", "cer": 0.0, "ms": 40.0, "subset": "mixed"},
    ]
    by = aggregate(rows)
    assert by["tesseract"]["n"] == 2
    assert by["tesseract"]["cer"] == 0.1
    assert by["tesseract"]["median_ms"] == 20.0
    assert by["tesseract"]["mixed_cer"] == 0.0


def test_passes_gate():
    tess = {"cer": 0.2, "median_ms": 50.0, "mixed_cer": 0.1}
    ok = {"cer": 0.1, "median_ms": 90.0, "mixed_cer": 0.1}
    bad_latency = {"cer": 0.1, "median_ms": 120.0, "mixed_cer": 0.1}
    assert passes_gate(ok, tess)[0] is True
    assert passes_gate(bad_latency, tess)[0] is False


def test_aggregate_ui_cer():
    rows = [
        {"engine": "t", "cer": 0.2, "ms": 1.0, "subset": "ui"},
        {"engine": "t", "cer": 0.4, "ms": 1.0, "subset": "ui"},
        {"engine": "t", "cer": 1.0, "ms": 1.0, "subset": "printed"},
    ]
    assert aggregate(rows)["t"]["ui_cer"] == pytest.approx(0.3)


def test_passes_gate_rejects_ui_regression():
    tess = {"cer": 0.25, "median_ms": 75.0, "mixed_cer": 0.09, "ui_cer": 0.02}
    cand = {"cer": 0.15, "median_ms": 75.0, "mixed_cer": 0.09, "ui_cer": 0.05}
    ok, reasons = passes_gate(cand, tess)
    assert ok is False
    assert any("ui_cer" in r for r in reasons)
