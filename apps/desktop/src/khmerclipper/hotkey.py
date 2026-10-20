"""Global capture hotkey. pynput on Windows and X11; Wayland has no app-level global
hotkeys, so users bind `khmerclipper --capture` in their desktop settings instead."""

import re
from collections.abc import Callable

from khmerclipper.capture.geometry import capture_backend

_MODS = {"ctrl": "<ctrl>", "shift": "<shift>", "alt": "<alt>", "meta": "<cmd>"}
_NAMED = {"space": "<space>", "print": "<print_screen>"}
_SHIFTED = dict(zip("!@#$%^&*()", "1234567890"))  # US layout: Qt reports Shift+2 as "@"
_FKEY = re.compile(r"f([1-9]|1[0-9]|2[0-4])")


def qt_to_pynput(seq: str) -> str | None:
    if not seq or "," in seq or seq.endswith("++"):
        return None
    *mods, key = [p.strip() for p in seq.split("+")]
    mods = [m.lower() for m in mods]
    if any(m not in _MODS for m in mods):
        return None
    lower = key.lower()
    if _FKEY.fullmatch(lower):
        key_out = f"<{lower}>"
    elif lower in _NAMED:
        key_out = _NAMED[lower]
    elif len(key) == 1:
        key_out = _SHIFTED.get(key, key.lower()) if "shift" in mods else key.lower()
    else:
        return None
    if not mods and not (key_out.startswith("<f") or key_out == "<print_screen>"):
        return None  # a bare letter would hijack normal typing
    return "+".join([_MODS[m] for m in mods] + [key_out])


def pynput_to_qt(hotkey: str) -> str:
    qt_mods = {v: k.capitalize() for k, v in _MODS.items()}
    qt_named = {v: k.capitalize() for k, v in _NAMED.items()}
    out = []
    for part in hotkey.split("+"):
        if part in qt_mods:
            out.append(qt_mods[part])
        elif part in qt_named:
            out.append(qt_named[part])
        elif re.fullmatch(r"<f\d+>", part):
            out.append(part[1:-1].upper())
        else:
            out.append(part.upper())
    return "+".join(out)


class HotkeyListener:
    def __init__(self):
        self._listener = None

    def start(self, hotkey: str, callback: Callable[[], None], backend: str | None = None) -> bool:
        """callback runs on pynput's thread: pass a Qt signal's emit, never touch widgets."""
        self.stop()
        if (backend or capture_backend()) != "qt":
            return False  # Wayland: pynput would only see XWayland windows -- worse than nothing
        try:
            from pynput import keyboard

            listener = keyboard.GlobalHotKeys({hotkey: callback})
            listener.start()
        except Exception:  # pynput missing (macOS dev), no X display, invalid combo
            return False
        self._listener = listener
        return True

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
