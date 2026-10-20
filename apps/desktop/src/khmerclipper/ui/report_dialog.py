from pathlib import Path

from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QPlainTextEdit, QVBoxLayout

from khmerclipper.ui.fonts import khmer_font

MAX_IMAGE_WIDTH = 480


class ReportDialog(QDialog):
    """Shows what will be sent for a capture report and lets the user fix the OCR text."""

    def __init__(self, image_path: Path, ocr_text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Report a wrong result")
        self._ocr_text = ocr_text
        layout = QVBoxLayout(self)
        image = QLabel()
        pixmap = QPixmap(str(image_path))
        if pixmap.width() > MAX_IMAGE_WIDTH:
            pixmap = pixmap.scaledToWidth(MAX_IMAGE_WIDTH)
        image.setPixmap(pixmap)
        layout.addWidget(image)
        layout.addWidget(QLabel("KhmerClipper read"))
        read = QLabel(ocr_text)
        read.setFont(khmer_font(12))
        read.setWordWrap(True)
        layout.addWidget(read)
        layout.addWidget(QLabel("What it should say"))
        self.correction = QPlainTextEdit(ocr_text)
        self.correction.setFont(khmer_font(12))
        layout.addWidget(self.correction)
        layout.addWidget(QLabel("This image and text will be sent to the KhmerClipper developer."))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.send_button = buttons.addButton("Send", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def correction_text(self) -> str:
        text = self.correction.toPlainText().strip()
        return "" if text == self._ocr_text.strip() else text
