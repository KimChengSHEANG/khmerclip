import base64
from pathlib import Path

from PIL import Image

from khmerclip import reports

URL = "https://relay.test/report"
UNSET = "https://khmerclip-report-relay.REPLACE_ME.workers.dev/report"


def test_enqueue_and_flush_deletes_on_success(tmp_path):
    p = reports.enqueue(reports.feedback_payload("សួស្តី", "", "tesseract"), tmp_path)
    sent = []
    assert reports.flush(tmp_path, post=lambda d: sent.append(d) or 200, url=URL) == 1
    assert not p.exists() and sent[0]["message"] == "សួស្តី" and sent[0]["type"] == "feedback"


def test_flush_keeps_file_on_network_error(tmp_path):
    p = reports.enqueue(reports.feedback_payload("x", "", "tesseract"), tmp_path)
    assert reports.flush(tmp_path, post=lambda d: 0, url=URL) == 0 and p.exists()
    reports.flush(tmp_path, post=lambda d: 503, url=URL)
    assert p.exists()


def test_flush_drops_rejected_reports(tmp_path):
    p = reports.enqueue(reports.feedback_payload("x", "", "tesseract"), tmp_path)
    reports.flush(tmp_path, post=lambda d: 413, url=URL)
    assert not p.exists()


def test_flush_does_nothing_until_url_configured(tmp_path):
    p = reports.enqueue(reports.feedback_payload("x", "", "tesseract"), tmp_path)
    called = []
    assert reports.flush(tmp_path, post=called.append, url=UNSET) == 0 and p.exists() and called == []


def test_flush_deletes_invalid_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert reports.flush(tmp_path, post=lambda d: 200, url=URL) == 0
    assert not bad.exists()


def test_drop_type_keeps_others(tmp_path):
    reports.enqueue(reports.crash_payload("E", "d", "tesseract"), tmp_path)
    keep = reports.enqueue(reports.feedback_payload("x", "", "tesseract"), tmp_path)
    reports.drop_type("crash", tmp_path)
    assert list(tmp_path.glob("*.json")) == [keep]


def test_crash_payload_scrubs_home():
    d = reports.crash_payload("E at a.py:f", f'File "{Path.home()}/x.py"', "trocr")
    assert str(Path.home()) not in d["details"] and '"~/x.py"' in d["details"]


def test_base_fields_shape():
    d = reports.base_fields("tesseract")
    assert len(d["id"]) == 32 and d["engine"] == "tesseract" and d["app_version"] and d["os"]


def test_capture_payload_downscales_large_png(tmp_path):
    big = tmp_path / "big.png"
    Image.effect_noise((2400, 1600), 100).convert("RGB").save(big)  # well over 2 MB as PNG
    d = reports.capture_payload(big, {"ocr_text": "ក"}, "ក", "", "tesseract")
    png = base64.b64decode(d["image_png_b64"])
    assert len(png) <= reports.MAX_PNG_BYTES and png.startswith(b"\x89PNG")
    assert d["type"] == "capture" and d["meta"] == {"ocr_text": "ក"} and d["correction"] == ""
