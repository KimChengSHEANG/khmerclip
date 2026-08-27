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
