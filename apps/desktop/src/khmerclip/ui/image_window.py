from pathlib import Path

from PIL import Image
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFileDialog, QLabel, QPushButton, QVBoxLayout, QWidget

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.webp)"


class ImageOCRWindow(QWidget):
    """"OCR Image File…": drop an image or choose one."""

    image_chosen = Signal(object)  # PIL.Image.Image (RGB)
    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCR Image File")
        self.setAcceptDrops(True)
        self.resize(420, 260)
        drop = QLabel("Drop an image here")
        drop.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop.setStyleSheet("border: 2px dashed palette(mid); border-radius: 8px; padding: 40px;")
        choose = QPushButton("Choose File…")
        choose.clicked.connect(self.choose_file)
        layout = QVBoxLayout(self)
        layout.addWidget(drop, 1)
        layout.addWidget(choose)

    def choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose an image", "", IMAGE_FILTER)
        if path:
            self.open_path(path)

    def open_path(self, path: str) -> None:
        try:
            with Image.open(path) as img:
                image = img.convert("RGB")
        except (OSError, Image.DecompressionBombError) as e:  # missing, unreadable, not an image, too large
            self.failed.emit(f"Can't open {Path(path).name}: {e}")
            return
        self.image_chosen.emit(image)

    def dragEnterEvent(self, event):
        if any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            if url.isLocalFile():
                self.open_path(url.toLocalFile())
                break
