from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QHBoxLayout, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
                               QVBoxLayout, QWidget)

from khmerclip.history import HistoryStore, ScanRecord
from khmerclip.ui.fonts import khmer_font

PREVIEW_CHARS = 300


class HistoryWindow(QWidget):
    def __init__(self, store: HistoryStore, copy: Callable[[str], None]):
        super().__init__()
        self.setWindowTitle("KhmerClip History")
        self.resize(480, 420)
        self.store, self._copy = store, copy
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
            preview = record.text if len(record.text) <= PREVIEW_CHARS else record.text[:PREVIEW_CHARS] + "…"
            item = QListWidgetItem(preview)
            item.setData(Qt.ItemDataRole.UserRole, record.id)
            item.setToolTip(record.date)
            self.list.addItem(item)

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
