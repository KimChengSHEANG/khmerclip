import threading

import pytest
from PIL import Image

from khmerclip.errors import Cancelled, KhmerClipError
from khmerclip.history import HistoryStore
from khmerclip.ocr import trocr_models
from khmerclip.prefs import Prefs
from khmerclip.ui import app as app_module
from khmerclip.ui.app import Controller, TrayApp, preferred_qpa


@pytest.fixture
def controller(qtbot, tmp_path):
    copied = []
    c = Controller(Prefs.load(tmp_path / "prefs.json"), HistoryStore(tmp_path / "h.json"), set_clipboard=copied.append)
    c.copied = copied
    qtbot.addWidget(c.popup)
    return c


def test_handle_text_copies_records_and_shows(controller):
    controller.handle_text("ក  \nខ")
    assert controller.copied == ["ក  \nខ"]
    assert controller.history.records[0].text == "ក  \nខ"
    assert controller.popup.isVisible() and controller.popup.label.text() == "ក  \nខ"


def test_handle_text_respects_prefs(controller):
    controller.prefs.preserve_linebreaks = False
    controller.prefs.auto_copy = False
    controller.prefs.show_result = False
    controller.handle_text("ក\nខ")
    assert controller.copied == []
    assert controller.history.records[0].text == "ក ខ"
    assert not controller.popup.isVisible()


def test_empty_result_is_not_recorded(controller):
    controller.handle_text("  \n ")
    assert controller.history.records == []
    assert controller.popup.label.text() == "No text found"


def test_errors(controller):
    controller.handle_error(Cancelled())
    assert not controller.popup.isVisible()
    controller.handle_error(KhmerClipError("Tesseract was not found."))
    assert "Tesseract was not found." in controller.popup.label.text()


def test_run_ocr_end_to_end_off_thread(controller, qtbot, monkeypatch):
    monkeypatch.setattr(app_module, "recognize", lambda image, engine, progress=None, cancel=None: "សួស្តី")
    controller.run_ocr(Image.new("RGB", (20, 20), "white"))
    qtbot.waitUntil(lambda: bool(controller.history.records), timeout=3000)
    assert controller.history.records[0].text == "សួស្តី"
    assert controller.copied == ["សួស្តី"]


def test_submit_delivers_results_on_gui_thread(controller, qtbot):
    # Callbacks touch widgets, so they must run on the GUI thread, never the pool thread.
    seen = []
    controller.submit(lambda progress, cancel: (progress("p"), 1)[1],
                      on_done=lambda r: seen.append(threading.current_thread() is threading.main_thread()))
    qtbot.waitUntil(lambda: bool(seen), timeout=3000)
    assert seen == [True]


def test_preferred_qpa():
    assert preferred_qpa("linux", {"XDG_SESSION_TYPE": "wayland", "DISPLAY": ":0"}) == "xcb"
    assert preferred_qpa("linux", {"DISPLAY": ":0", "QT_QPA_PLATFORM": "wayland"}) is None  # user choice wins
    assert preferred_qpa("linux", {"WAYLAND_DISPLAY": "wayland-0"}) is None  # no XWayland available
    assert preferred_qpa("win32", {"DISPLAY": ":0"}) is None


def test_trocr_unpublished_explains_instead_of_crashing(controller, monkeypatch):
    monkeypatch.delenv("KHMERCLIP_TROCR_DIR", raising=False)
    monkeypatch.setattr(trocr_models, "load_manifest", lambda path=None: trocr_models.Manifest("khmer-v1", ()))
    controller.prefs.engine = "trocr"
    controller.run_ocr(Image.new("RGB", (20, 20), "white"))
    assert "isn't published" in controller.popup.label.text()
    assert controller.history.records == []


def test_tray_menu_and_commands(controller, monkeypatch):
    monkeypatch.setattr(app_module.HotkeyListener, "start", lambda self, *a, **k: False)
    controller.prefs.check_updates = False
    controller.prefs.welcomed = True
    captures = []
    monkeypatch.setattr(controller, "start_capture", lambda: captures.append(1))
    tray = TrayApp(controller)
    texts = [a.text() for a in tray.menu.actions() if a.text()]
    assert texts == ["Capture Text", "OCR Image File…", "History…", "Preferences…",
                     "Check for Updates…", "Quit KhmerClip"]
    tray.on_command("capture")
    assert captures == [1]
    tray.on_command("show")
    assert tray.prefs_window is not None and tray.prefs_window.isVisible()
