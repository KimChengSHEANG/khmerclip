"""Run blocking work (OCR, downloads, portal, update check) off the GUI thread."""

import threading
from collections.abc import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


class TaskSignals(QObject):
    progress = Signal(object)
    done = Signal(object)
    failed = Signal(object)


class Task(QRunnable):
    """fn(progress, cancel) runs on the global pool; results arrive as queued signals."""

    def __init__(self, fn: Callable[[Callable[[object], None], threading.Event], object]):
        super().__init__()
        self.setAutoDelete(False)  # Python owns the object; signals must outlive run()
        self.fn = fn
        self.signals = TaskSignals()
        self.cancel = threading.Event()

    def run(self) -> None:
        try:
            result = self.fn(self.signals.progress.emit, self.cancel)
        except Exception as e:
            self.signals.failed.emit(e)
        else:
            self.signals.done.emit(result)


def submit(fn) -> Task:
    task = Task(fn)
    QThreadPool.globalInstance().start(task)
    return task
