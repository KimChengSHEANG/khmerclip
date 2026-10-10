import json
import os
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
    MAX_RECORDS = 200

    def __init__(self, path: Path | None = None):
        self.path = path or paths.data_dir() / "history.json"
        self.records: list[ScanRecord] = self._load()

    def add(self, record: ScanRecord) -> None:
        self.records.insert(0, record)
        del self.records[self.MAX_RECORDS:]
        self._save()

    def remove(self, record_id: str) -> None:
        self.records = [r for r in self.records if r.id != record_id]
        self._save()

    def clear(self) -> None:
        self.records = []
        self._save()

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
