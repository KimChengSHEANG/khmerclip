from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QPixmap

from khmerclip.capture.overlay import RegionSelector


def fake_grab_at(ratio):
    """Pretend every screen is captured at `ratio` device pixels per logical pixel."""
    def grab(screen):
        g = screen.geometry()
        pixmap = QPixmap(int(g.width() * ratio), int(g.height() * ratio))
        pixmap.fill(QColor("red"))
        return pixmap
    return grab


def test_overlay_crop_uses_pixmap_to_widget_ratio(qtbot):
    selector = RegionSelector(grab=fake_grab_at(2.0))
    selector.start()
    overlay = selector.overlays[0]
    qtbot.waitExposed(overlay)
    with qtbot.waitSignal(selector.selected, timeout=2000) as blocker:
        qtbot.mousePress(overlay, Qt.MouseButton.LeftButton, pos=QPoint(10, 10))
        qtbot.mouseRelease(overlay, Qt.MouseButton.LeftButton, pos=QPoint(110, 60))
    image = blocker.args[0]
    assert image.size == (200, 100)
    assert image.getpixel((5, 5)) == (255, 0, 0)
    assert selector.overlays == []


def test_escape_cancels(qtbot):
    selector = RegionSelector(grab=fake_grab_at(1.0))
    selector.start()
    with qtbot.waitSignal(selector.cancelled, timeout=2000):
        qtbot.keyClick(selector.overlays[0], Qt.Key.Key_Escape)
    assert selector.overlays == []


def test_click_without_drag_cancels(qtbot):
    selector = RegionSelector(grab=fake_grab_at(1.0))
    selector.start()
    overlay = selector.overlays[0]
    with qtbot.waitSignal(selector.cancelled, timeout=2000):
        qtbot.mousePress(overlay, Qt.MouseButton.LeftButton, pos=QPoint(10, 10))
        qtbot.mouseRelease(overlay, Qt.MouseButton.LeftButton, pos=QPoint(11, 11))


def test_start_twice_does_not_stack_overlays(qtbot):
    selector = RegionSelector(grab=fake_grab_at(1.0))
    selector.start()
    count = len(selector.overlays)
    selector.start()
    assert len(selector.overlays) == count
    qtbot.keyClick(selector.overlays[0], Qt.Key.Key_Escape)
