from collections.abc import Callable

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox,
                               QPushButton, QVBoxLayout, QWidget)

from khmerclip.history import HistoryStore, ScanRecord
from khmerclip.ui.fonts import khmer_font
from khmerclip.ui.report_dialog import ReportDialog

PREVIEW_CHARS = 300
THUMB_SIZE = QSize(72, 40)


class HistoryWindow(QWidget):
    def __init__(self, store: HistoryStore, copy: Callable[[str], None],
                 report: Callable[[ScanRecord, str], None] | None = None):
        super().__init__()
        self.setWindowTitle("KhmerClip History")
        self.resize(480, 420)
        self.store, self._copy, self._report = store, copy, report
        self.list = QListWidget()
        self.list.setFont(khmer_font(12))
        self.list.setWordWrap(True)
        self.list.itemDoubleClicked.connect(lambda _item: self.copy_selected())
        buttons = QHBoxLayout()
        for label, slot in (("Copy", self.copy_selected), ("Delete", self.delete_selected), ("Clear All", self.clear_all)):
            button = QPushButton(label)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        layout = QVBoxLayout(self)
        layout.addWidget(self.list)
        layout.addLayout(buttons)

    def refresh(self) -> None:
        self.list.clear()
        for record in self.store.records:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, record.id)
            item.setToolTip(record.date)
            row = self._row_widget(record)
            item.setSizeHint(row.sizeHint())
            self.list.addItem(item)
            self.list.setItemWidget(item, row)

    def _row_widget(self, record: ScanRecord) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        has_capture = self.store.has_capture(record.id)
        if has_capture:
            thumb = QLabel()
            thumb.setFixedSize(THUMB_SIZE)
            pixmap = QPixmap(str(self.store.captures_dir / f"{record.id}.png"))
            thumb.setPixmap(pixmap.scaled(THUMB_SIZE, Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation))
            thumb.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            layout.addWidget(thumb)
        preview = record.text if len(record.text) <= PREVIEW_CHARS else record.text[:PREVIEW_CHARS] + "…"
        text = QLabel(preview)
        text.setFont(khmer_font(12))
        text.setWordWrap(True)
        text.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)  # clicks select the row
        layout.addWidget(text, 1)
        if has_capture:
            reported = self.store.is_flagged(record.id)
            button = QPushButton("Reported" if reported else "Report")
            button.setObjectName("report_button")
            button.setEnabled(not reported)
            button.clicked.connect(lambda _=False, r=record: self.report_record(r))
            layout.addWidget(button)
        return row

    def report_record(self, record: ScanRecord) -> None:
        dialog = ReportDialog(self.store.captures_dir / f"{record.id}.png",
                              self.store.capture_meta(record.id).get("ocr_text", record.text), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        correction = dialog.correction_text()
        if self._report is not None:
            try:
                self._report(record, correction)
            except Exception:
                return  # not queued: leave the row unreported so the user can retry
        self.store.flag(record.id, correction)
        self.refresh()

    def _selected(self) -> ScanRecord | None:
        item = self.list.currentItem()
        if item is None:
            return None
        record_id = item.data(Qt.ItemDataRole.UserRole)
        return next((r for r in self.store.records if r.id == record_id), None)

    def copy_selected(self) -> None:
        if record := self._selected():
            self._copy(record.text)

    def delete_selected(self) -> None:
        if record := self._selected():
            self.store.remove(record.id)
            self.refresh()

    def clear_all(self) -> None:
        if not self.store.records:
            return
        if QMessageBox.question(self, "Clear History", "Delete all saved scans?") == QMessageBox.StandardButton.Yes:
            self.store.clear()
            self.refresh()

    def showEvent(self, event):
        self.refresh()
        super().showEvent(event)
