"""Automatic crash capture: Python exceptions become queued crash reports, and native crashes
(segfaults) are captured by faulthandler into a log that is turned into a report on next launch."""

import faulthandler
import sys
import threading
import traceback
from collections.abc import Callable
from pathlib import Path

from khmerclip import reports

_PACKAGE_DIR = Path(__file__).resolve().parent
_fault_file = None  # faulthandler needs the file object to stay open for the whole process


def signature(exc: BaseException) -> str:
    frames = traceback.extract_tb(exc.__traceback__)
    if not frames:
        return type(exc).__name__
    last = frames[-1]
    try:
        where = Path(last.filename).resolve().relative_to(_PACKAGE_DIR).as_posix()
    except (ValueError, OSError):
        where = Path(last.filename).name
    return f"{type(exc).__name__} at {where}:{last.name}"


def record(exc: BaseException, engine: str, outbox: Path | None = None) -> Path:
    details = "".join(traceback.format_exception(exc))
    return reports.enqueue(reports.crash_payload(signature(exc), details, engine), outbox)


def _fault_log(outbox: Path | None) -> Path:
    return (outbox or reports.outbox_dir()).parent / "fault.log"


def install(enabled: Callable[[], bool], engine: Callable[[], str], outbox: Path | None = None) -> None:
    global _fault_file
    previous_excepthook, previous_thread_hook = sys.excepthook, threading.excepthook

    def maybe_record(exc: BaseException | None) -> None:
        try:
            if exc is not None and enabled():
                record(exc, engine(), outbox)
        except Exception:  # reporting must never mask the original error
            pass

    def excepthook(exc_type, exc, tb) -> None:
        maybe_record(exc)
        previous_excepthook(exc_type, exc, tb)

    def thread_hook(args) -> None:
        maybe_record(args.exc_value)
        previous_thread_hook(args)

    sys.excepthook, threading.excepthook = excepthook, thread_hook
    try:
        log = _fault_log(outbox)
        log.parent.mkdir(parents=True, exist_ok=True)
        new_file = open(log, "a", encoding="utf-8")  # noqa: SIM115 - must outlive this function
        faulthandler.enable(file=new_file)
        if _fault_file is not None:
            _fault_file.close()
        _fault_file = new_file
    except (OSError, RuntimeError):
        pass  # no native-crash capture; Python exceptions are still recorded


def collect_fault_log(engine: str, outbox: Path | None = None) -> None:
    log = _fault_log(outbox)
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    if text.strip():
        reports.enqueue(reports.crash_payload("Native crash", text, engine), outbox)
    if text:
        try:
            log.write_text("", encoding="utf-8")
        except OSError:
            pass


def discard_fault_log(outbox: Path | None = None) -> None:
    try:
        _fault_log(outbox).write_text("", encoding="utf-8")
    except OSError:
        pass
