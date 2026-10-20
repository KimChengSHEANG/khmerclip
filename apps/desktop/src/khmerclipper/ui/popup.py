from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from khmerclipper.ui.fonts import khmer_font

PREVIEW_CHARS = 1500


class ResultPopup(QWidget):
    """Transient bottom-right popup with the OCR result (the mac HUD's counterpart)."""

    copy_requested = Signal(str)
    TIMEOUT_MS = 5000

    def __init__(self):
        super().__init__(None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self._text = ""
        self.label = QLabel()
        self.label.setTextFormat(Qt.TextFormat.PlainText)  # OCR output must never render as HTML
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.label.setFont(khmer_font(13))
        self.label.setMaximumWidth(420)
        self.copy_button = QPushButton("Copy")
        self.copy_button.clicked.connect(lambda: self.copy_requested.emit(self._text))
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.hide)
        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(self.copy_button)
        buttons.addWidget(close_button)
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addLayout(buttons)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.hide)

    def show_text(self, text: str) -> None:
        self._show(text, copyable=True)

    def show_message(self, message: str) -> None:
        self._show(message, copyable=False)

    def _show(self, text: str, copyable: bool) -> None:
        self._text = text
        self.label.setText(text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS] + "…")
        self.copy_button.setVisible(copyable)
        self.adjustSize()
        area = QGuiApplication.primaryScreen().availableGeometry()
        self.move(area.right() - self.width() - 16, area.bottom() - self.height() - 16)
        self.show()
        self.raise_()
        self.timer.start(self.TIMEOUT_MS)

    def enterEvent(self, event):
        self.timer.stop()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.timer.start(self.TIMEOUT_MS)
        super().leaveEvent(event)
