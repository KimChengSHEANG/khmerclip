import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from khmerclip import paths


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ScanRecord:
    text: str
    kind: str = "text"
    date: str = field(default_factory=_now)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


class HistoryStore:
    """Scan history, plus (opt-in) each capture's image in ``captures/<id>.png`` with a
    ``<id>.json`` of OCR details. Reporting a scan copies that pair into ``feedback/`` (with the user's
    correction), which also marks the scan "Reported"; it feeds the eval set
    (training/eval/scripts/draft_real_fixtures.py). Same layout as the Mac app."""

    MAX_RECORDS = 200

    def __init__(self, path: Path | None = None):
        self.path = path or paths.data_dir() / "history.json"
        self.captures_dir = self.path.parent / "captures"
        self.feedback_dir = self.path.parent / "feedback"
        self.records: list[ScanRecord] = self._load()

    def add(self, record: ScanRecord, image=None, meta: dict | None = None) -> None:
        """``image`` (PIL) is kept with ``meta`` only when given — the caller checks the pref."""
        if image is not None:
            self.captures_dir.mkdir(parents=True, exist_ok=True)
            image.save(self.captures_dir / f"{record.id}.png")
            paths.atomic_write_text(self.captures_dir / f"{record.id}.json",
                                    json.dumps(meta or {}, ensure_ascii=False, indent=1))
        self.records.insert(0, record)
        for dropped in self.records[self.MAX_RECORDS:]:
            self._delete_capture(dropped.id)
        del self.records[self.MAX_RECORDS:]
        self._save()

    def remove(self, record_id: str) -> None:
        self.records = [r for r in self.records if r.id != record_id]
        self._delete_capture(record_id)
        self._save()

    def clear(self) -> None:
        self.records = []
        self.purge_captures()
        self._save()

    def has_capture(self, record_id: str) -> bool:
        return (self.captures_dir / f"{record_id}.png").is_file()

    def is_flagged(self, record_id: str) -> bool:
        return (self.feedback_dir / f"{record_id}.png").is_file()

    def capture_meta(self, record_id: str) -> dict:
        try:
            meta = json.loads((self.captures_dir / f"{record_id}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return meta if isinstance(meta, dict) else {}

    def flag(self, record_id: str, correction: str = "") -> None:
        """Copy the capture pair into feedback/ with the correction. Reported clips outlive the history."""
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        png = self.captures_dir / f"{record_id}.png"
        if png.is_file():
            shutil.copy2(png, self.feedback_dir / png.name)
        meta = self.capture_meta(record_id)
        meta["correction"] = correction
        paths.atomic_write_text(self.feedback_dir / f"{record_id}.json",
                                json.dumps(meta, ensure_ascii=False, indent=1))

    def purge_captures(self) -> None:
        shutil.rmtree(self.captures_dir, ignore_errors=True)

    def _delete_capture(self, record_id: str) -> None:
        for suffix in (".png", ".json"):
            (self.captures_dir / f"{record_id}{suffix}").unlink(missing_ok=True)

    def _load(self) -> list[ScanRecord]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return [ScanRecord(text=r["text"], kind=r["kind"], date=r["date"], id=r["id"]) for r in raw]
        except FileNotFoundError:
            return []
        except (OSError, ValueError, TypeError, KeyError):
            # Keep the unreadable file for the user instead of overwriting it on the next add().
            try:
                os.replace(self.path, self.path.with_name(self.path.name + ".bak"))
            except OSError:
                pass
            return []

    def _save(self) -> None:
        data = [asdict(r) for r in self.records]
        paths.atomic_write_text(self.path, json.dumps(data, ensure_ascii=False, indent=1))
