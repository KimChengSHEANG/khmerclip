import faulthandler
import json
import sys
import threading
from pathlib import Path

import pytest

from khmerclip import crash
from khmerclip.prefs import Prefs


@pytest.fixture(autouse=True)
def _restore_hooks(monkeypatch):
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    yield
    faulthandler.disable()
    if crash._fault_file is not None:
        crash._fault_file.close()
        monkeypatch.setattr(crash, "_fault_file", None)


def test_record_writes_scrubbed_crash_without_scanned_text(tmp_path):
    try:
        raise ValueError(f"bad {Path.home()}/secret.png")
    except ValueError as e:
        path = crash.record(e, "trocr", tmp_path)
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["type"] == "crash" and d["signature"].startswith("ValueError at ")
    assert str(Path.home()) not in d["details"] and "image_png_b64" not in d and "ocr_text" not in d


def test_signature_without_traceback_is_type_name():
    assert crash.signature(RuntimeError("x")) == "RuntimeError"


def test_signature_is_relative_to_package():
    with pytest.raises(AttributeError) as info:
        crash.signature(5)  # raises inside khmerclip/crash.py:signature
    assert crash.signature(info.value) == "AttributeError at crash.py:signature"


def test_signature_outside_package_uses_basename(tmp_path):
    src = tmp_path / "elsewhere.py"
    src.write_text("def boom():\n    raise KeyError('k')\n", encoding="utf-8")
    ns: dict = {}
    exec(compile(src.read_text(encoding="utf-8"), str(src), "exec"), ns)
    with pytest.raises(KeyError) as info:
        ns["boom"]()
    assert crash.signature(info.value) == "KeyError at elsewhere.py:boom"


def test_excepthook_respects_pref_and_chains(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(sys, "excepthook", lambda *a: seen.append(a))
    monkeypatch.setattr(threading, "excepthook", lambda args: None)
    enabled = [False]
    crash.install(lambda: enabled[0], lambda: "tesseract", tmp_path / "outbox")
    err = RuntimeError("x")
    sys.excepthook(RuntimeError, err, None)
    assert list((tmp_path / "outbox").glob("*.json")) == [] and len(seen) == 1
    enabled[0] = True
    sys.excepthook(RuntimeError, err, None)
    assert len(list((tmp_path / "outbox").glob("*.json"))) == 1 and len(seen) == 2


def test_hook_never_masks_original_when_recording_fails(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(sys, "excepthook", lambda *a: seen.append(a))
    monkeypatch.setattr(threading, "excepthook", lambda args: None)
    crash.install(lambda: 1 / 0, lambda: "tesseract", tmp_path / "outbox")
    sys.excepthook(RuntimeError, RuntimeError("x"), None)
    assert len(seen) == 1


def test_fault_log_becomes_one_crash(tmp_path):
    (tmp_path / "fault.log").write_text("Fatal Python error: Segmentation fault", encoding="utf-8")
    crash.collect_fault_log("tesseract", tmp_path / "outbox")
    [p] = (tmp_path / "outbox").glob("*.json")
    assert json.loads(p.read_text(encoding="utf-8"))["signature"] == "Native crash"
    assert (tmp_path / "fault.log").read_text(encoding="utf-8") == ""


def test_collect_fault_log_survives_enqueue_failure(tmp_path, monkeypatch):
    (tmp_path / "fault.log").write_text("Fatal Python error", encoding="utf-8")
    monkeypatch.setattr(crash.reports, "enqueue", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    crash.collect_fault_log("tesseract", tmp_path / "outbox")  # must not raise


def test_empty_fault_log_reports_nothing(tmp_path):
    (tmp_path / "fault.log").write_text("", encoding="utf-8")
    crash.collect_fault_log("tesseract", tmp_path / "outbox")
    assert not (tmp_path / "outbox").exists() or list((tmp_path / "outbox").glob("*.json")) == []


def test_prefs_defaults():
    p = Prefs()
    assert p.keep_captures is True and p.send_crash_reports is True
