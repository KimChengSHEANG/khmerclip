import faulthandler
import json
import sys
import threading
from pathlib import Path

import pytest

from khmerclipper import crash
from khmerclipper.prefs import Prefs


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
        crash.signature(5)  # raises inside khmerclipper/crash.py:signature
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


def _fault_text(current_file: str, other_file: str = "/usr/lib/python3.12/threading.py") -> str:
    # Real faulthandler layout (captured from Python 3.12): the faulting thread is not listed first.
    return (
        "Fatal Python error: Segmentation fault\n\n"
        "Thread 0x000000016c5b7000 (most recent call first):\n"
        f'  File "{other_file}", line 5 in idle\n'
        '  File "/usr/lib/python3.12/threading.py", line 1012 in run\n\n'
        "Current thread 0x00000001f800ff80 (most recent call first):\n"
        f'  File "{current_file}", line 8 in run\n'
        '  File "/somewhere/main.py", line 10 in <module>\n'
    )


def test_native_signature_uses_crashing_threads_top_frame():
    inside = crash._PACKAGE_DIR / "ocr" / "tesseract.py"
    assert crash.native_signature(_fault_text(str(inside))) == "Native crash at ocr/tesseract.py:run"
    assert crash.native_signature(_fault_text("/opt/lib/native.py")) == "Native crash at native.py:run"


def test_native_signature_never_borrows_another_threads_frame():
    text = ("Fatal Python error: Segmentation fault\n\n"
            "Thread 0x01 (most recent call first):\n"
            '  File "/x/idle.py", line 5 in idle\n\n'
            "Current thread 0x02 (most recent call first):\n"
            "  <no Python frame>\n")
    assert crash.native_signature(text) == "Native crash"


def test_native_signature_without_thread_header_uses_first_frame():
    text = 'Fatal Python error: Aborted\n\nStack (most recent call first):\n  File "/x/a.py", line 3 in go\n'
    assert crash.native_signature(text) == "Native crash at a.py:go"
    assert crash.native_signature("Fatal Python error: Segmentation fault") == "Native crash"


def test_fault_log_signature_names_the_frame(tmp_path):
    (tmp_path / "fault.log").write_text(_fault_text("/opt/lib/native.py"), encoding="utf-8")
    crash.collect_fault_log("tesseract", tmp_path / "outbox")
    [p] = (tmp_path / "outbox").glob("*.json")
    assert json.loads(p.read_text(encoding="utf-8"))["signature"] == "Native crash at native.py:run"


def test_record_skips_a_crash_already_queued(tmp_path):
    for _ in range(3):
        try:
            raise ValueError("same place")
        except ValueError as e:
            crash.record(e, "trocr", tmp_path)
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_duplicate_fault_log_is_skipped_and_cleared(tmp_path):
    for _ in range(2):
        (tmp_path / "fault.log").write_text(_fault_text("/opt/lib/native.py"), encoding="utf-8")
        crash.collect_fault_log("tesseract", tmp_path / "outbox")
        assert (tmp_path / "fault.log").read_text(encoding="utf-8") == ""
    assert len(list((tmp_path / "outbox").glob("*.json"))) == 1
