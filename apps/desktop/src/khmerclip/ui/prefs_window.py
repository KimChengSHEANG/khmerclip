from functools import partial

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QKeySequenceEdit, QLabel, QMessageBox, QWidget

from khmerclip.hotkey import pynput_to_qt, qt_to_pynput
from khmerclip.prefs import Prefs

WAYLAND_HINT = ("Global shortcuts aren't available to apps on Wayland. Add a custom shortcut in your "
                "desktop's keyboard settings that runs:  khmerclip --capture")
CHECKS = (
    ("auto_copy", "Copy result to clipboard"),
    ("show_result", "Show result popup"),
    ("preserve_linebreaks", "Preserve line breaks"),
    ("launch_at_login", "Launch at login"),
    ("check_updates", "Check for updates daily"),
)


class PrefsWindow(QWidget):
    changed = Signal(str)  # name of the Prefs field that changed (already saved)

    def __init__(self, prefs: Prefs, trocr_available: bool, hotkey_supported: bool):
        super().__init__()
        self.setWindowTitle("KhmerClip Preferences")
        self.prefs = prefs
        form = QFormLayout(self)

        self.engine = QComboBox()
        self.engine.addItem("Tesseract", "tesseract")
        self.engine.addItem("TrOCR (neural; one-time model download)", "trocr")
        if not trocr_available:
            item = self.engine.model().item(self.engine.findData("trocr"))
            item.setEnabled(False)
            item.setToolTip("The TrOCR model hasn't been published yet.")
        self.engine.setCurrentIndex(max(0, self.engine.findData(prefs.engine)))
        self.engine.currentIndexChanged.connect(lambda _i: self._set("engine", self.engine.currentData()))
        form.addRow("OCR engine", self.engine)

        self.hotkey = QKeySequenceEdit(QKeySequence(pynput_to_qt(prefs.hotkey)))
        self.hotkey.editingFinished.connect(self._hotkey_edited)
        form.addRow("Capture shortcut", self.hotkey)
        if not hotkey_supported:
            self.hotkey.setEnabled(False)
            hint = QLabel(WAYLAND_HINT)
            hint.setWordWrap(True)
            hint.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            form.addRow(hint)

        self.checks: dict[str, QCheckBox] = {}
        for name, label in CHECKS:
            box = QCheckBox(label)
            box.setChecked(getattr(prefs, name))
            box.toggled.connect(partial(self._set, name))
            form.addRow(box)
            self.checks[name] = box

    def _set(self, name: str, value) -> None:
        setattr(self.prefs, name, value)
        self.prefs.save()
        self.changed.emit(name)

    def _hotkey_edited(self) -> None:
        seq = self.hotkey.keySequence().toString(QKeySequence.SequenceFormat.PortableText)
        hotkey = qt_to_pynput(seq)
        if hotkey is None:
            QMessageBox.warning(self, "Capture shortcut",
                                f"“{seq}” can't be used. Combine Ctrl/Alt/Shift/Meta with a letter, digit, "
                                "F-key or Space.")
            self.hotkey.setKeySequence(QKeySequence(pynput_to_qt(self.prefs.hotkey)))
            return
        self._set("hotkey", hotkey)
