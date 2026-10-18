import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from khmerclip import paths

ENGINES = ("tesseract", "trocr")


@dataclass
class Prefs:
    auto_copy: bool = True
    show_result: bool = True
    preserve_linebreaks: bool = True
    engine: str = "tesseract"
    hotkey: str = "<ctrl>+<shift>+2"
    launch_at_login: bool = False
    check_updates: bool = True
    last_update_check: float = 0.0
    welcomed: bool = False
    keep_captures: bool = True  # keep capture images so wrong results can be flagged
    send_crash_reports: bool = True
    path: Path | None = field(default=None, repr=False, compare=False)

    @classmethod
    def load(cls, path: Path | None = None) -> "Prefs":
        prefs = cls(path=path or paths.config_dir() / "prefs.json")
        try:
            raw = json.loads(prefs.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return prefs
        if not isinstance(raw, dict):
            return prefs
        for f in fields(cls):
            if f.name == "path" or f.name not in raw:
                continue
            value, default = raw[f.name], getattr(prefs, f.name)
            # Exact type match (bool is an int subclass, so isinstance would accept 1 for a bool).
            if type(value) is type(default) or (type(default) is float and type(value) is int):
                setattr(prefs, f.name, type(default)(value))
        if prefs.engine not in ENGINES:
            prefs.engine = "tesseract"
        return prefs

    def save(self) -> None:
        data = asdict(self)
        data.pop("path")
        paths.atomic_write_text(self.path, json.dumps(data, indent=2, ensure_ascii=False))
