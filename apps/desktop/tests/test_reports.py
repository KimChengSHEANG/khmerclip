import base64
import http.client
import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

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
    assert len(png) <= reports.TARGET_PNG_BYTES < reports.MAX_PNG_BYTES and png.startswith(b"\x89PNG")
    assert d["type"] == "capture" and d["meta"] == {"ocr_text": "ក"} and d["correction"] == ""


@pytest.mark.parametrize("status", [200, 400, 401, 413])
def test_flush_deletes_on_final_statuses(tmp_path, status):
    p = reports.enqueue(reports.feedback_payload("x", "", "tesseract"), tmp_path)
    n = reports.flush(tmp_path, post=lambda d: status, url=URL)
    assert not p.exists() and n == (1 if status == 200 else 0)


@pytest.mark.parametrize("status", [0, 429, 500, 503])
def test_flush_keeps_on_retryable_statuses(tmp_path, status):
    p = reports.enqueue(reports.feedback_payload("x", "", "tesseract"), tmp_path)
    assert reports.flush(tmp_path, post=lambda d: status, url=URL) == 0 and p.exists()


def test_post_json_http_error_returns_code(monkeypatch):
    def boom(*a, **k):
        raise urllib.error.HTTPError(URL, 429, "slow down", {}, None)
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert reports.post_json({"id": "a"}, URL) == 429


@pytest.mark.parametrize("exc", [urllib.error.URLError("down"), http.client.RemoteDisconnected("x"),
                                 ValueError("bad url")])
def test_post_json_never_raises(monkeypatch, exc):
    def boom(*a, **k):
        raise exc
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert reports.post_json({"id": "a"}, URL) == 0


def test_post_json_sends_headers_and_body(monkeypatch):
    seen = {}

    class Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake(request, timeout):
        seen["req"] = request
        return Resp()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    assert reports.post_json({"id": "a", "m": "ក"}, URL) == 200
    req = seen["req"]
    assert req.get_header("X-khmerclip-key") == reports.APP_KEY
    assert req.get_header("Content-type") == "application/json"
    assert json.loads(req.data.decode("utf-8")) == {"id": "a", "m": "ក"}


def test_base_fields_normalized(monkeypatch):
    monkeypatch.setattr(reports.platform, "platform", lambda: "Win\n dows  " + "x" * 200)
    d = reports.base_fields("a\nb")
    assert d["engine"] == "a b" and "\n" not in d["os"] and len(d["os"]) == 100
    assert reports.base_fields("")["engine"] == "unknown"
    assert reports.base_fields("  ")["engine"] == "unknown"


def test_free_text_truncated():
    long = "ក" * 30_000
    assert len(reports.feedback_payload(long, "", "t")["message"]) == reports.MAX_TEXT_CHARS
    assert len(reports.crash_payload("E", long, "t")["details"]) == reports.MAX_TEXT_CHARS


def test_capture_text_truncated(tmp_path):
    img = tmp_path / "a.png"
    Image.new("RGB", (10, 10)).save(img)
    d = reports.capture_payload(img, {}, "ក" * 30_000, "ខ" * 30_000, "t")
    assert len(d["ocr_text"]) == len(d["correction"]) == reports.MAX_TEXT_CHARS


def test_feedback_email_trimmed_and_blanked_when_too_long():
    assert reports.feedback_payload("m", "  a@b.co \n", "t")["email"] == "a@b.co"
    ok = "a" * 249 + "@b.co"  # exactly 254
    assert reports.feedback_payload("m", ok, "t")["email"] == ok
    assert reports.feedback_payload("m", "a" * 250 + "@b.co", "t")["email"] == ""
    # The relay counts UTF-16 units: 128 astral chars are 256 units, so over the limit.
    assert reports.feedback_payload("m", "\U0001F600" * 128, "t")["email"] == ""
