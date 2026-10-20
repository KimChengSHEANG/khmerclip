import sys

from khmerclipper.hotkey import HotkeyListener, pynput_to_qt, qt_to_pynput


def test_qt_to_pynput_basic_and_shifted_digit():
    assert qt_to_pynput("Ctrl+Shift+2") == "<ctrl>+<shift>+2"
    assert qt_to_pynput("Ctrl+Shift+@") == "<ctrl>+<shift>+2"   # Qt records the shifted symbol
    assert qt_to_pynput("Meta+Alt+K") == "<cmd>+<alt>+k"
    assert qt_to_pynput("F9") == "<f9>"
    assert qt_to_pynput("Ctrl+Space") == "<ctrl>+<space>"


def test_qt_to_pynput_rejects_unsupported():
    assert qt_to_pynput("") is None
    assert qt_to_pynput("K") is None                    # no modifier
    assert qt_to_pynput("Ctrl+Hyper+K") is None
    assert qt_to_pynput("Ctrl+K, Ctrl+L") is None       # multi-chord
    assert qt_to_pynput("Ctrl+PgUp") is None


def test_pynput_to_qt_roundtrip():
    assert pynput_to_qt("<ctrl>+<shift>+2") == "Ctrl+Shift+2"
    assert pynput_to_qt("<cmd>+<f5>") == "Meta+F5"
    assert qt_to_pynput(pynput_to_qt("<alt>+<space>")) == "<alt>+<space>"


def test_listener_refuses_on_wayland():
    assert HotkeyListener().start("<ctrl>+k", lambda: None, backend="portal") is False


def test_listener_reports_missing_pynput(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynput", None)
    assert HotkeyListener().start("<ctrl>+k", lambda: None, backend="qt") is False
