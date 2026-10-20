"""Frozen-screenshot region selector: grab every screen first, then show each grab in a
fullscreen overlay and crop from the grab. Cropping from the grabbed image (not a second
capture) keeps the selection exact under display scaling and on any monitor layout."""

import io
from collections.abc import Callable
from functools import partial

from PIL import Image
from PySide6.QtCore import QBuffer, QIODevice, QObject, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter, QPen, QPixmap, QScreen
from PySide6.QtWidgets import QWidget

from khmerclipper.capture.geometry import normalize_drag, to_physical


def qimage_to_pil(image: QImage) -> Image.Image:
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buf, "PNG")
    return Image.open(io.BytesIO(bytes(buf.data()))).convert("RGB")


def grab_screen(screen: QScreen) -> QPixmap:
    return screen.grabWindow(0)


class _Overlay(QWidget):
    finished = Signal(object)  # (x, y, w, h) in widget coords, or None

    def __init__(self, screen: QScreen, pixmap: QPixmap):
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.Tool)
        self.pixmap = pixmap
        self._origin = None
        self._current = None
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)
        self.setGeometry(screen.geometry())

    def _selection(self) -> QRect | None:
        if self._origin is None or self._current is None:
            return None
        x, y, w, h = normalize_drag(self._origin.x(), self._origin.y(), self._current.x(), self._current.y())
        return QRect(int(x), int(y), int(w), int(h))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.pixmap)
        selection = self._selection()
        if not selection:
            return
        # Match the macOS overlay: screen stays undimmed, only the selection is shaded.
        painter.fillRect(selection, QColor(0, 0, 0, 77))
        painter.setPen(QPen(QColor("white"), 1.5))
        painter.drawRect(selection)
        self._draw_label(painter, f"{selection.width()} × {selection.height()}")

    def _draw_label(self, painter: QPainter, text: str) -> None:
        font = QFont()
        font.setPointSize(12)
        font.setWeight(QFont.Weight.Medium)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        box = QRectF(0, 0, metrics.horizontalAdvance(text) + 16, metrics.height() + 8)
        box.moveBottomLeft(self._current + QPointF(14, -14))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 191))
        painter.drawRoundedRect(box, 4, 4)
        painter.setPen(QColor("white"))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._origin = self._current = event.position()
            self.update()

    def mouseMoveEvent(self, event):
        if self._origin is not None:
            self._current = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or self._origin is None:
            return
        self._current = event.position()
        self.finished.emit(normalize_drag(self._origin.x(), self._origin.y(), self._current.x(), self._current.y()))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.finished.emit(None)


class RegionSelector(QObject):
    selected = Signal(object)  # PIL.Image.Image
    cancelled = Signal()

    def __init__(self, grab: Callable[[QScreen], QPixmap] = grab_screen, parent: QObject | None = None):
        super().__init__(parent)
        self._grab = grab
        self.overlays: list[_Overlay] = []

    def start(self) -> None:
        if self.overlays:
            return
        # Grab every screen before showing any overlay so no overlay lands in a capture.
        shots = [(screen, self._grab(screen)) for screen in QGuiApplication.screens()]
        for screen, pixmap in shots:
            overlay = _Overlay(screen, pixmap)
            overlay.finished.connect(partial(self._finish, overlay))
            overlay.showFullScreen()
            overlay.raise_()
            overlay.activateWindow()
            self.overlays.append(overlay)

    def _finish(self, overlay: _Overlay, rect) -> None:
        # Derive the scale from the actual grab vs. overlay size instead of trusting
        # devicePixelRatio: some platforms return physical-size grabs with dpr 1.
        ratio = overlay.pixmap.width() / max(1, overlay.width())
        image = overlay.pixmap.toImage()
        for ov in self.overlays:
            ov.close()
            ov.deleteLater()
        self.overlays = []
        physical = to_physical(rect, ratio, image.width(), image.height()) if rect else None
        if physical is None:
            self.cancelled.emit()
            return
        self.selected.emit(qimage_to_pil(image.copy(QRect(*physical))))
