# KhmerClip Desktop (Windows + Linux) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship KhmerClip for Windows and Linux: a tray app that captures a screen region, runs Khmer(+English) OCR (Tesseract or TrOCR/ONNX), and puts the text on the clipboard. Build and publish installers from GitHub Actions.

**Architecture:** New Python package `khmerclip` at `apps/desktop/`. The headless core (OCR engines, preprocessing, persistence, downloader, updater) never imports Qt, so it runs in pytest and the `khmerclip ocr` CLI on any OS. A thin PySide6 GUI layer adds the tray, capture overlay or Wayland portal, popups and windows. PyInstaller freezes it. Inno Setup (Windows) and an AppImage (Linux) package it. Both bundle Tesseract plus pinned `tessdata_fast`. TrOCR models download on first use from a pinned Hugging Face revision.

**Tech Stack:** Python 3.12, PySide6-Essentials 6.8, onnxruntime, numpy, Pillow, platformdirs, pynput (Win/X11 hotkeys), jeepney (Wayland portal), pytest + pytest-qt, PyInstaller, Inno Setup 6, linuxdeploy + appimagetool, GitHub Actions. Dev-only export tools: torch 2.7, transformers 4.57, onnx, huggingface_hub.

**Spec:** `docs/superpowers/specs/2026-10-03-khmerclip-desktop-windows-linux-design.md`

## Global Constraints

- All new code lives under `apps/desktop/` plus `.github/workflows/desktop.yml` and one row in the root `README.md`. Do not modify `apps/mac/` (read-only reference; its OCR fixtures at `apps/mac/Tests/Fixtures/ocr/*.png|.txt` are reused by tests).
- Python `>=3.12,<3.13`; `PySide6-Essentials~=6.8.0`; `onnxruntime>=1.20,<2`.
- App name `KhmerClip`. Version is single-sourced: `apps/desktop/src/khmerclip/__init__.py` → `__version__ = "0.1.0"`. Release tag in this repo: `desktop-v<version>`. Release tag in the releases repo: `v<version>`.
- Default engine `"tesseract"`. Default hotkey `"<ctrl>+<shift>+2"` (pynput syntax).
- Tessdata: `tessdata_fast` tag `4.1.0`. `khm.traineddata` sha256 `47f110575341b322052f3becbefee61a3ecf1ef549352b5d4d33d28afe30d099`, size 1446926. `eng.traineddata` sha256 `7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2`, size 4113088. Tesseract args: `-l khm+eng --psm 6`.
- Model repo (public HF): `KimChengSHEANG/khmerclip-trocr-khmer-onnx`. Releases repo (public GitHub): `KimChengSHEANG/khmerclip-releases`. Update feed: `https://api.github.com/repos/KimChengSHEANG/khmerclip-releases/releases/latest`.
- Every text file read or write passes `encoding="utf-8"` explicitly, because the Windows default code page mangles Khmer. JSON is written with `ensure_ascii=False`.
- Qt is imported only in `capture/overlay.py`, `single_instance.py` and `ui/*`. Core modules and `khmerclip ocr` must import without Qt.
- Run everything from `apps/desktop/` with the venv: `.venv/bin/python -m pytest` (Windows: `.venv\Scripts\python -m pytest`).
- **Git in this worktree session:** call `/usr/bin/git` (absolute path). The rtk hook rewrites plain `git`, and the worktree guard refuses the rewritten command. Run git from the worktree root. Do not chain git with `;` or `&&`.
- The mac app's TrOCR is the reference for behavior. `apps/mac/Sources/OCR/TrOCREngine.swift` and `TrOCRTokenizer.swift` are ported, not redesigned.

## Review Focus

1. **Windows non-ASCII paths and code pages.** A Khmer Windows username puts resources under a non-ASCII path, and Tesseract opens `--tessdata-dir` through the ANSI code page. Expect OCR to still work: use the 8.3 short path. Expect output to be UTF-8 regardless of the system code page. → Task 4 `test_ascii_path_for_non_ascii_dir` (Windows CI) and the Windows fixture test.
2. **Fractional display scaling (125%/150%) and multiple monitors.** Expect the cropped image to be exactly the region the user dragged on any screen. → Task 9 `test_to_physical_*` at 1.25 and 1.5, and Task 10 `test_overlay_crop_uses_pixmap_to_widget_ratio`.
3. **Wayland session (Ubuntu default).** Expect capture to go through the portal, no silent half-working X11 hotkey, and the prefs window to explain how to bind `khmerclip --capture`. → Task 9 `test_capture_backend_*` and Task 11 `test_listener_refuses_on_wayland`.
4. **Frozen Linux build spawning Tesseract.** PyInstaller's `LD_LIBRARY_PATH` leaks into the child process. Expect the bundled Tesseract to load its own libraries. → Task 4 `test_subprocess_env_*` and Task 17's AppImage smoke test run with system Tesseract removed.
5. **Interrupted, corrupt or disk-full model download (1.3 GB).** Expect no half-written model file, a clear error, and a retry that resumes cleanly. → Task 3 `test_download_*` (checksum mismatch, cancel, unreachable URL, skip-existing).

---

## File Structure

```
apps/desktop/
  pyproject.toml                    # deps, extras (dev/tools), pytest config, console script
  .gitignore                        # build/, dist/, resources/tesseract/, exported models
  README.md                         # dev setup, run, test, build, release, model publishing
  resources/
    icon.png                        # copy of mac icon_256x256.png
    fonts/NotoSansKhmer-Regular.ttf # bundled Khmer font (OFL)
    fonts/OFL.txt
    tesseract/                      # gitignored; tessdata/ fetched, Windows exe+dlls vendored in CI
  src/khmerclip/
    __init__.py                     # __version__
    __main__.py                     # CLI: GUI (default), --capture, `ocr IMAGE`
    paths.py                        # resource/data/config dirs, atomic_write_text
    errors.py                       # KhmerClipError, EngineUnavailable, Cancelled
    formatter.py                    # preserve/flatten line breaks
    prefs.py                        # Prefs dataclass <-> prefs.json
    history.py                      # ScanRecord, HistoryStore <-> history.json
    download.py                     # RemoteFile, sha256_file, is_present, download
    updater.py                      # GitHub releases/latest check
    autostart.py                    # launch-at-login (Windows registry / XDG autostart)
    hotkey.py                       # Qt<->pynput hotkey strings, HotkeyListener
    single_instance.py              # QLocalServer command channel (Qt)
    ocr/
      __init__.py                   # Engine enum, recognize() dispatcher
      tesseract.py                  # Tesseract subprocess engine
      tokenizer.py                  # TrOCR tokenizer decode (port of TrOCRTokenizer.swift)
      preprocess.py                 # ink mask, line strips, content crop, pixel values
      trocr_models.py               # manifest, install dir, install(), build_manifest()
      trocr_manifest.json           # committed; filled by tools/publish_models.py
      trocr.py                      # onnxruntime greedy decode
    capture/
      __init__.py
      geometry.py                   # drag normalization, logical->physical crop, backend choice
      portal.py                     # xdg-desktop-portal Screenshot (Wayland)
      overlay.py                    # Qt frozen-screenshot region selector (Win/X11)
    ui/
      __init__.py
      fonts.py                      # bundled Khmer font -> QFont
      worker.py                     # QRunnable task with progress/cancel
      popup.py                      # transient result popup
      history_window.py
      prefs_window.py
      image_window.py               # OCR Image File… drop zone
      app.py                        # Controller, tray, run_gui()
  tools/
    fetch_tessdata.py               # pinned tessdata download
    export_onnx.py                  # TrOCR checkpoint -> encoder/decoder ONNX + tokenizer.json
    publish_models.py               # upload to HF, write trocr_manifest.json
  packaging/
    entry.py                        # PyInstaller entry script
    khmerclip.spec                  # PyInstaller onedir spec
    windows/vendor_tesseract.ps1    # copy choco Tesseract exe+dlls into resources/tesseract
    windows/khmerclip.iss           # Inno Setup per-user installer
    linux/AppRun
    linux/khmerclip.desktop
    linux/build_appimage.sh
  tests/
    helpers.py                      # FIXTURES path, normalize(), cer()
    conftest.py                     # QT_QPA_PLATFORM=offscreen
    test_*.py
.github/workflows/desktop.yml       # test matrix, build installers, publish release
```

---

### Task 1: Package scaffold, paths, errors, formatter

**Files:**
- Create: `apps/desktop/pyproject.toml`
- Create: `apps/desktop/.gitignore`
- Create: `apps/desktop/src/khmerclip/__init__.py`
- Create: `apps/desktop/src/khmerclip/paths.py`
- Create: `apps/desktop/src/khmerclip/errors.py`
- Create: `apps/desktop/src/khmerclip/formatter.py`
- Create: `apps/desktop/resources/icon.png` (copied), `apps/desktop/resources/fonts/NotoSansKhmer-Regular.ttf`, `apps/desktop/resources/fonts/OFL.txt` (downloaded)
- Create: `apps/desktop/tests/helpers.py`, `apps/desktop/tests/conftest.py`
- Test: `apps/desktop/tests/test_formatter.py`, `apps/desktop/tests/test_paths.py`

**Interfaces:**
- Produces:
  - `khmerclip.__version__: str`
  - `khmerclip.paths.APP_NAME = "KhmerClip"`
  - `resource_dir() -> Path`
  - `data_dir() -> Path`
  - `config_dir() -> Path`
  - `atomic_write_text(path: Path, text: str) -> None`
  - `khmerclip.errors`: `KhmerClipError(Exception)`, `EngineUnavailable(KhmerClipError)`, `Cancelled(Exception)`
  - `khmerclip.formatter.apply(text: str, preserve_linebreaks: bool) -> str`
  - `tests/helpers.py`: `FIXTURES: Path`, `normalize(s: str) -> str`, `cer(hyp: str, ref: str) -> float`

- [ ] **Step 1: Create the project files and dev venv**

`apps/desktop/pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "khmerclip"
dynamic = ["version"]
description = "Khmer OCR screen capture for Windows and Linux"
requires-python = ">=3.12,<3.13"
dependencies = [
  "PySide6-Essentials~=6.8.0",
  "onnxruntime>=1.20,<2",
  "numpy>=1.26,<3",
  "pillow>=10.4",
  "platformdirs>=4.2",
  "pynput>=1.7.7; sys_platform != 'darwin'",
  "jeepney>=0.8; sys_platform == 'linux'",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-qt>=4.4", "pyinstaller>=6.10"]
tools = ["torch==2.7.0", "transformers==4.57.*", "onnx>=1.16", "huggingface_hub>=0.25"]

[project.scripts]
khmerclip = "khmerclip.__main__:main"

[tool.setuptools.dynamic]
version = { attr = "khmerclip.__version__" }

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
khmerclip = ["ocr/*.json"]

[tool.pytest.ini_options]
testpaths = ["tests"]
qt_api = "pyside6"
```

`apps/desktop/.gitignore`:
```gitignore
.venv/
build/
dist/
resources/tesseract/
exported/
*.AppImage
```

`apps/desktop/src/khmerclip/__init__.py`:
```python
"""KhmerClip for Windows and Linux."""

__version__ = "0.1.0"
```

Run from `apps/desktop/`:
```bash
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
mkdir -p resources/fonts
cp ../mac/Resources/Assets.xcassets/AppIcon.appiconset/icon_256x256.png resources/icon.png
curl -fsSL -o resources/fonts/NotoSansKhmer-Regular.ttf https://github.com/notofonts/notofonts.github.io/raw/main/fonts/NotoSansKhmer/hinted/ttf/NotoSansKhmer-Regular.ttf
curl -fsSL -o resources/fonts/OFL.txt https://raw.githubusercontent.com/notofonts/khmer/main/OFL.txt
file resources/fonts/NotoSansKhmer-Regular.ttf
```
Expected: `file` reports `TrueType Font data … Noto Sans Khmer`. If the `OFL.txt` URL 404s, copy the SIL OFL 1.1 text from https://openfontlicense.org/open-font-license-official-text/ into `resources/fonts/OFL.txt`.

- [ ] **Step 2: Write test helpers and failing tests**

`apps/desktop/tests/conftest.py`:
```python
import os

# Must be set before any Qt import so tests run headless on every OS and in CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
```

`apps/desktop/tests/helpers.py`:
```python
from pathlib import Path

# Shared OCR fixtures from the mac app (PNG + expected .txt pairs).
FIXTURES = Path(__file__).resolve().parents[2] / "mac" / "Tests" / "Fixtures" / "ocr"


def normalize(s: str) -> str:
    return " ".join(s.split())


def cer(hyp: str, ref: str) -> float:
    """Character error rate: Levenshtein distance / len(ref)."""
    prev = list(range(len(ref) + 1))
    for i, h in enumerate(hyp, 1):
        cur = [i]
        for j, r in enumerate(ref, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (h != r)))
        prev = cur
    return prev[-1] / max(1, len(ref))
```

`apps/desktop/tests/test_formatter.py`:
```python
from khmerclip.formatter import apply


def test_preserve_returns_text_unchanged():
    text = "ជួរទី១\nline two\n"
    assert apply(text, preserve_linebreaks=True) == text


def test_flatten_joins_trimmed_nonempty_lines_with_spaces():
    text = "  ជួរទី១  \r\n\n line two \nបី"
    assert apply(text, preserve_linebreaks=False) == "ជួរទី១ line two បី"


def test_flatten_of_blank_text_is_empty():
    assert apply("\n \n", preserve_linebreaks=False) == ""
```

`apps/desktop/tests/test_paths.py`:
```python
from khmerclip import paths


def test_resource_dir_contains_bundled_font_and_icon():
    root = paths.resource_dir()
    assert (root / "fonts" / "NotoSansKhmer-Regular.ttf").is_file()
    assert (root / "icon.png").is_file()


def test_atomic_write_text_writes_utf8_and_leaves_no_temp(tmp_path):
    target = tmp_path / "sub" / "f.json"
    paths.atomic_write_text(target, "ខ្មែរ")
    assert target.read_bytes().decode("utf-8") == "ខ្មែរ"
    assert [p.name for p in target.parent.iterdir()] == ["f.json"]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_formatter.py tests/test_paths.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'khmerclip.formatter'` (and `khmerclip.paths`).

- [ ] **Step 4: Implement paths, errors, formatter**

`apps/desktop/src/khmerclip/paths.py`:
```python
"""Where KhmerClip reads bundled resources and keeps per-user state."""

import os
import sys
from pathlib import Path

import platformdirs

APP_NAME = "KhmerClip"


def resource_dir() -> Path:
    """Read-only bundled resources: PyInstaller's extraction dir when frozen,
    else apps/desktop/resources in the source tree."""
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root) / "resources"
    return Path(__file__).resolve().parents[2] / "resources"


def data_dir() -> Path:
    path = Path(platformdirs.user_data_dir(APP_NAME, appauthor=False))
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_dir() -> Path:
    path = Path(platformdirs.user_config_dir(APP_NAME, appauthor=False))
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_text(path: Path, text: str) -> None:
    """Write via temp file + os.replace so a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
```

`apps/desktop/src/khmerclip/errors.py`:
```python
class KhmerClipError(Exception):
    """A failure whose str() is shown to the user as-is."""


class EngineUnavailable(KhmerClipError):
    """The chosen OCR engine can't run (binary/model missing)."""


class Cancelled(Exception):
    """The user cancelled a running OCR or download."""
```

`apps/desktop/src/khmerclip/formatter.py`:
```python
def apply(text: str, preserve_linebreaks: bool) -> str:
    """Port of OCRTextFormatter.swift: optionally flatten visual wrapping into one paragraph."""
    if preserve_linebreaks:
        return text
    return " ".join(s for line in text.splitlines() if (s := line.strip()))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_formatter.py tests/test_paths.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
/usr/bin/git add apps/desktop
/usr/bin/git commit -m "feat(desktop): scaffold Windows/Linux khmerclip package"
```

---

### Task 2: Preferences and scan history

**Files:**
- Create: `apps/desktop/src/khmerclip/prefs.py`
- Create: `apps/desktop/src/khmerclip/history.py`
- Test: `apps/desktop/tests/test_prefs.py`, `apps/desktop/tests/test_history.py`

**Interfaces:**
- Consumes: `paths.config_dir()`, `paths.data_dir()`, `paths.atomic_write_text()`
- Produces:
  - `Prefs` dataclass with fields `auto_copy: bool = True`, `show_result: bool = True`, `preserve_linebreaks: bool = True`, `engine: str = "tesseract"`, `hotkey: str = "<ctrl>+<shift>+2"`, `launch_at_login: bool = False`, `check_updates: bool = True`, `last_update_check: float = 0.0`, `welcomed: bool = False`, `path: Path | None` (not persisted)
  - `Prefs.load(path: Path | None = None) -> Prefs`
  - `Prefs.save() -> None`
  - `ScanRecord(text: str, kind: str = "text", date: str = <UTC ISO now>, id: str = <uuid hex>)` (frozen)
  - `HistoryStore(path: Path | None = None)` with `.records: list[ScanRecord]` (newest first), `.add(record)`, `.remove(record_id: str)`, `.clear()`, `MAX_RECORDS = 200`

- [ ] **Step 1: Write the failing tests**

`apps/desktop/tests/test_prefs.py`:
```python
import json

from khmerclip.prefs import Prefs


def test_missing_file_gives_defaults(tmp_path):
    prefs = Prefs.load(tmp_path / "prefs.json")
    assert prefs.engine == "tesseract"
    assert prefs.hotkey == "<ctrl>+<shift>+2"
    assert prefs.auto_copy is True


def test_save_then_load_roundtrips(tmp_path):
    path = tmp_path / "prefs.json"
    prefs = Prefs.load(path)
    prefs.engine = "trocr"
    prefs.auto_copy = False
    prefs.last_update_check = 123.5
    prefs.save()
    again = Prefs.load(path)
    assert (again.engine, again.auto_copy, again.last_update_check) == ("trocr", False, 123.5)
    assert "path" not in json.loads(path.read_text(encoding="utf-8"))


def test_corrupt_json_gives_defaults(tmp_path):
    path = tmp_path / "prefs.json"
    path.write_text("{not json", encoding="utf-8")
    assert Prefs.load(path).engine == "tesseract"


def test_wrong_typed_fields_fall_back_individually(tmp_path):
    path = tmp_path / "prefs.json"
    path.write_text(json.dumps({"auto_copy": "yes", "show_result": False, "last_update_check": 7,
                                "check_updates": 1, "engine": "easyocr", "bogus": 1}), encoding="utf-8")
    prefs = Prefs.load(path)
    assert prefs.auto_copy is True            # "yes" is not a bool -> default
    assert prefs.show_result is False         # valid -> kept
    assert prefs.last_update_check == 7.0     # int accepted for float
    assert prefs.check_updates is True        # 1 is not a bool -> default
    assert prefs.engine == "tesseract"        # unknown engine -> default
```

`apps/desktop/tests/test_history.py`:
```python
import json

from khmerclip.history import HistoryStore, ScanRecord


def test_add_puts_newest_first_and_persists(tmp_path):
    path = tmp_path / "history.json"
    store = HistoryStore(path)
    store.add(ScanRecord("first"))
    store.add(ScanRecord("ទីពីរ"))
    assert [r.text for r in store.records] == ["ទីពីរ", "first"]
    assert [r.text for r in HistoryStore(path).records] == ["ទីពីរ", "first"]
    assert "ទីពីរ" in path.read_bytes().decode("utf-8")  # raw UTF-8, not \u escapes


def test_caps_at_max_records(tmp_path):
    store = HistoryStore(tmp_path / "h.json")
    for i in range(HistoryStore.MAX_RECORDS + 5):
        store.add(ScanRecord(str(i)))
    assert len(store.records) == HistoryStore.MAX_RECORDS
    assert store.records[0].text == str(HistoryStore.MAX_RECORDS + 4)


def test_remove_and_clear(tmp_path):
    store = HistoryStore(tmp_path / "h.json")
    keep, drop = ScanRecord("keep"), ScanRecord("drop")
    store.add(keep)
    store.add(drop)
    store.remove(drop.id)
    assert [r.text for r in store.records] == ["keep"]
    store.clear()
    assert HistoryStore(tmp_path / "h.json").records == []


def test_corrupt_file_is_quarantined_not_lost(tmp_path):
    path = tmp_path / "history.json"
    path.write_text("[{broken", encoding="utf-8")
    store = HistoryStore(path)
    assert store.records == []
    assert (tmp_path / "history.json.bak").read_text(encoding="utf-8") == "[{broken"


def test_wrong_shape_is_quarantined(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(json.dumps({"text": "x"}), encoding="utf-8")
    assert HistoryStore(path).records == []
    assert (tmp_path / "history.json.bak").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_prefs.py tests/test_history.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'khmerclip.prefs'`.

- [ ] **Step 3: Implement**

`apps/desktop/src/khmerclip/prefs.py`:
```python
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
```

`apps/desktop/src/khmerclip/history.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_prefs.py tests/test_history.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add apps/desktop/src/khmerclip/prefs.py apps/desktop/src/khmerclip/history.py apps/desktop/tests/test_prefs.py apps/desktop/tests/test_history.py
/usr/bin/git commit -m "feat(desktop): add JSON-backed preferences and scan history"
```

---

### Task 3: Verified file downloader

**Files:**
- Create: `apps/desktop/src/khmerclip/download.py`
- Test: `apps/desktop/tests/test_download.py`

**Interfaces:**
- Consumes: `errors.KhmerClipError`, `errors.Cancelled`
- Produces:
  - `RemoteFile(name: str, url: str, sha256: str, size: int)` (frozen dataclass)
  - `sha256_file(path: Path) -> str`
  - `is_present(files: Sequence[RemoteFile], dest_dir: Path) -> bool`
  - `download(files: Sequence[RemoteFile], dest_dir: Path, progress: Callable[[int, int], None] | None = None, cancel: threading.Event | None = None, chunk_size: int = 1 << 20) -> None`. Raises `KhmerClipError` on network/disk/checksum failure. Raises `Cancelled` when `cancel` is set. Never leaves `*.part` or unverified files behind.

- [ ] **Step 1: Write the failing tests**

`apps/desktop/tests/test_download.py`:
```python
import hashlib
import threading

import pytest

from khmerclip.download import RemoteFile, download, is_present, sha256_file
from khmerclip.errors import Cancelled, KhmerClipError


def _source(tmp_path, name, payload: bytes, sha=None) -> RemoteFile:
    src = tmp_path / "src" / name
    src.parent.mkdir(exist_ok=True)
    src.write_bytes(payload)
    return RemoteFile(name, src.as_uri(), sha or hashlib.sha256(payload).hexdigest(), len(payload))


def test_download_verifies_and_installs(tmp_path):
    files = [_source(tmp_path, "a.bin", b"a" * 3000), _source(tmp_path, "b.bin", b"b" * 10)]
    seen = []
    download(files, tmp_path / "out", progress=lambda d, t: seen.append((d, t)), chunk_size=1024)
    assert (tmp_path / "out" / "a.bin").read_bytes() == b"a" * 3000
    assert is_present(files, tmp_path / "out")
    assert seen[-1] == (3010, 3010)


def test_download_checksum_mismatch_leaves_nothing(tmp_path):
    bad = _source(tmp_path, "m.bin", b"data", sha="0" * 64)
    with pytest.raises(KhmerClipError, match="Checksum mismatch"):
        download([bad], tmp_path / "out")
    assert list((tmp_path / "out").iterdir()) == []


def test_download_cancel_raises_and_cleans_part(tmp_path):
    f = _source(tmp_path, "c.bin", b"x" * 5000)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        download([f], tmp_path / "out", cancel=cancel, chunk_size=1024)
    assert list((tmp_path / "out").iterdir()) == []


def test_download_unreachable_url_is_user_facing_error(tmp_path):
    missing = RemoteFile("x.bin", (tmp_path / "nope.bin").as_uri(), "0" * 64, 1)
    with pytest.raises(KhmerClipError, match="x.bin"):
        download([missing], tmp_path / "out")
    assert list((tmp_path / "out").iterdir()) == []


def test_download_skips_files_already_present(tmp_path):
    f = _source(tmp_path, "s.bin", b"keep")
    out = tmp_path / "out"
    download([f], out)
    # Same name/size but an unreachable URL: a re-download would raise.
    again = RemoteFile("s.bin", (tmp_path / "gone").as_uri(), f.sha256, f.size)
    download([again], out)
    assert (out / "s.bin").read_bytes() == b"keep"


def test_sha256_file(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"abc")
    assert sha256_file(p) == hashlib.sha256(b"abc").hexdigest()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_download.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'khmerclip.download'`.

- [ ] **Step 3: Implement**

`apps/desktop/src/khmerclip/download.py`:
```python
"""Download pinned files with SHA-256 verification (tessdata, TrOCR models)."""

import hashlib
import os
import threading
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from khmerclip import __version__
from khmerclip.errors import Cancelled, KhmerClipError


@dataclass(frozen=True)
class RemoteFile:
    name: str
    url: str
    sha256: str
    size: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def is_present(files: Sequence[RemoteFile], dest_dir: Path) -> bool:
    # ponytail: size check only; files are SHA-verified before the rename that installs them,
    # so hashing ~1.3 GB again on every launch buys nothing.
    return all((dest_dir / f.name).is_file() and (dest_dir / f.name).stat().st_size == f.size for f in files)


def download(
    files: Sequence[RemoteFile],
    dest_dir: Path,
    progress: Callable[[int, int], None] | None = None,
    cancel: threading.Event | None = None,
    chunk_size: int = 1 << 20,
) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    total = sum(f.size for f in files)
    done = 0
    for f in files:
        final = dest_dir / f.name
        if final.is_file() and final.stat().st_size == f.size:
            done += f.size
            continue
        part = final.with_name(final.name + ".part")
        digest = hashlib.sha256()
        request = urllib.request.Request(f.url, headers={"User-Agent": f"KhmerClip/{__version__}"})
        try:
            with urllib.request.urlopen(request, timeout=30) as resp, open(part, "wb") as out:
                while block := resp.read(chunk_size):
                    if cancel is not None and cancel.is_set():
                        raise Cancelled()
                    digest.update(block)
                    out.write(block)
                    done += len(block)
                    if progress:
                        progress(done, total)
            if digest.hexdigest() != f.sha256:
                raise KhmerClipError(f"Checksum mismatch for {f.name}; the download was corrupted. Try again.")
            os.replace(part, final)
        except (Cancelled, KhmerClipError):
            raise
        except OSError as e:  # URLError, HTTPError, disk full, permission denied
            raise KhmerClipError(f"Downloading {f.name} failed: {e}") from e
        finally:
            part.unlink(missing_ok=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_download.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add apps/desktop/src/khmerclip/download.py apps/desktop/tests/test_download.py
/usr/bin/git commit -m "feat(desktop): add checksum-verified, cancellable downloader"
```

---

### Task 4: Tesseract engine and pinned tessdata

**Files:**
- Create: `apps/desktop/tools/fetch_tessdata.py`
- Create: `apps/desktop/src/khmerclip/ocr/__init__.py` (empty for now; Task 8 fills it)
- Create: `apps/desktop/src/khmerclip/ocr/tesseract.py`
- Test: `apps/desktop/tests/test_tesseract.py`

**Interfaces:**
- Consumes: `download.RemoteFile`, `download.download`, `paths.resource_dir()`, `errors.*`
- Produces:
  - `tesseract.find_tesseract(env: Mapping[str, str] | None = None) -> str | None` (env `KHMERCLIP_TESSERACT` > bundled `resources/tesseract/tesseract(.exe)` > `PATH`)
  - `tesseract.tessdata_dir() -> Path` (= `resource_dir()/"tesseract"/"tessdata"`)
  - `tesseract.subprocess_env(env: Mapping[str, str], frozen: bool) -> dict[str, str]`
  - `tesseract.ascii_path(path: Path) -> str`
  - `tesseract.recognize(image: PIL.Image.Image) -> str`. Raises `EngineUnavailable` or `KhmerClipError`.

- [ ] **Step 1: Write the tessdata fetch tool**

`apps/desktop/tools/fetch_tessdata.py`:
```python
"""Fetch pinned tessdata_fast models into resources/tesseract/tessdata.

Same files (and hashes) the mac app vendors. Run from apps/desktop:
    .venv/bin/python tools/fetch_tessdata.py
"""

from khmerclip.download import RemoteFile, download
from khmerclip.ocr.tesseract import tessdata_dir

BASE = "https://github.com/tesseract-ocr/tessdata_fast/raw/4.1.0/"
FILES = [
    RemoteFile("khm.traineddata", BASE + "khm.traineddata",
               "47f110575341b322052f3becbefee61a3ecf1ef549352b5d4d33d28afe30d099", 1446926),
    RemoteFile("eng.traineddata", BASE + "eng.traineddata",
               "7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2", 4113088),
]

if __name__ == "__main__":
    download(FILES, tessdata_dir())
    print("tessdata ready:", tessdata_dir())
```
(Run it after Step 4 creates `tesseract.py`.)

- [ ] **Step 2: Write the failing tests**

`apps/desktop/tests/test_tesseract.py`:
```python
import os
import sys
from pathlib import Path

import pytest
from PIL import Image

from helpers import FIXTURES, cer, normalize
from khmerclip.errors import EngineUnavailable
from khmerclip.ocr import tesseract

HAVE_TESSERACT = tesseract.find_tesseract() is not None and (tesseract.tessdata_dir() / "khm.traineddata").is_file()
needs_tesseract = pytest.mark.skipif(
    not HAVE_TESSERACT and not os.environ.get("KHMERCLIP_REQUIRE_TESSERACT"),
    reason="tesseract/tessdata not installed (CI sets KHMERCLIP_REQUIRE_TESSERACT=1)",
)


@needs_tesseract
@pytest.mark.parametrize("png", sorted(FIXTURES.glob("*.png")), ids=lambda p: p.name)
def test_fixture_matches_mac_baseline(png):
    expected = normalize(png.with_suffix(".txt").read_text(encoding="utf-8"))
    with Image.open(png) as img:
        got = normalize(tesseract.recognize(img))
    # Baselines came from Tesseract 5.5 on macOS. Allow small drift across builds of 5.x.
    assert cer(got, expected) <= 0.05, f"{png.name}: {got!r} vs {expected!r}"


def test_missing_binary_is_engine_unavailable(monkeypatch):
    monkeypatch.setattr(tesseract, "find_tesseract", lambda env=None: None)
    with pytest.raises(EngineUnavailable, match="Tesseract"):
        tesseract.recognize(Image.new("RGB", (10, 10), "white"))


def test_find_tesseract_prefers_env_override(tmp_path):
    exe = tmp_path / "my-tesseract"
    assert tesseract.find_tesseract({"KHMERCLIP_TESSERACT": str(exe), "PATH": ""}) == str(exe)


def test_subprocess_env_restores_original_ld_library_path_when_frozen():
    env = {"LD_LIBRARY_PATH": "/tmp/_MEIxyz", "LD_LIBRARY_PATH_ORIG": "/usr/lib/custom", "HOME": "/h"}
    out = tesseract.subprocess_env(env, frozen=True)
    assert out["LD_LIBRARY_PATH"] == "/usr/lib/custom"
    assert out["HOME"] == "/h"


def test_subprocess_env_drops_pyinstaller_path_without_original():
    env = {"LD_LIBRARY_PATH": "/tmp/_MEIxyz", "HOME": "/h"}
    assert tesseract.subprocess_env(env, frozen=True) == {"HOME": "/h"}


def test_subprocess_env_untouched_when_not_frozen():
    env = {"LD_LIBRARY_PATH": "/opt/lib"}
    assert tesseract.subprocess_env(env, frozen=False) == env


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only path handling")
def test_ascii_path_for_non_ascii_dir(tmp_path):
    d = tmp_path / "ខ្មែរ"
    d.mkdir()
    out = tesseract.ascii_path(d)
    if not out.isascii():
        pytest.skip("8.3 short names disabled on this volume")
    assert Path(out).resolve() == d.resolve()


def test_ascii_path_passthrough_for_ascii(tmp_path):
    assert tesseract.ascii_path(tmp_path) == str(tmp_path)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_tesseract.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'khmerclip.ocr'`.

- [ ] **Step 4: Implement**

`apps/desktop/src/khmerclip/ocr/__init__.py`: create as an empty file.

`apps/desktop/src/khmerclip/ocr/tesseract.py`:
```python
"""Tesseract subprocess engine (khm+eng), mirroring TesseractEngine.swift."""

import io
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from PIL import Image

from khmerclip.errors import EngineUnavailable, KhmerClipError
from khmerclip.paths import resource_dir

LANGS = "khm+eng"
TIMEOUT_S = 60


def find_tesseract(env: Mapping[str, str] | None = None) -> str | None:
    env = os.environ if env is None else env
    if override := env.get("KHMERCLIP_TESSERACT"):
        return override
    exe = "tesseract.exe" if sys.platform == "win32" else "tesseract"
    bundled = resource_dir() / "tesseract" / exe
    if bundled.is_file():
        return str(bundled)
    # Linux AppImage: AppRun puts the bundled binary first on PATH.
    return shutil.which("tesseract", path=env.get("PATH"))


def tessdata_dir() -> Path:
    return resource_dir() / "tesseract" / "tessdata"


def subprocess_env(env: Mapping[str, str], frozen: bool) -> dict[str, str]:
    """PyInstaller's Linux bootloader points LD_LIBRARY_PATH at its own bundled libs
    (saving the original in LD_LIBRARY_PATH_ORIG). Tesseract must not inherit that."""
    out = dict(env)
    if frozen:
        if "LD_LIBRARY_PATH_ORIG" in out:
            out["LD_LIBRARY_PATH"] = out["LD_LIBRARY_PATH_ORIG"]
        else:
            out.pop("LD_LIBRARY_PATH", None)
    return out


def ascii_path(path: Path) -> str:
    """Tesseract opens --tessdata-dir through the ANSI code page on Windows, so an
    install under a non-ASCII profile (e.g. a Khmer username) fails. Use the 8.3 short path."""
    s = str(path)
    if sys.platform != "win32" or s.isascii():
        return s
    import ctypes

    buf = ctypes.create_unicode_buffer(32768)
    n = ctypes.windll.kernel32.GetShortPathNameW(s, buf, len(buf))
    return buf.value if 0 < n < len(buf) else s


def recognize(image: Image.Image) -> str:
    exe = find_tesseract()
    if exe is None:
        raise EngineUnavailable("Tesseract was not found. Reinstall KhmerClip (developers: install tesseract-ocr).")
    tessdata = tessdata_dir()
    if not (tessdata / "khm.traineddata").is_file():
        raise EngineUnavailable("Khmer language data is missing. Developers: run tools/fetch_tessdata.py.")

    png = io.BytesIO()
    image.convert("RGB").save(png, "PNG")
    cmd = [exe, "stdin", "stdout", "--tessdata-dir", ascii_path(tessdata), "-l", LANGS, "--psm", "6"]
    try:
        proc = subprocess.run(
            cmd,
            input=png.getvalue(),
            capture_output=True,
            timeout=TIMEOUT_S,
            env=subprocess_env(os.environ, getattr(sys, "frozen", False)),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),  # no console flash on Windows
        )
    except subprocess.TimeoutExpired as e:
        raise KhmerClipError("Tesseract timed out.") from e
    except OSError as e:
        raise EngineUnavailable(f"Could not run Tesseract: {e}") from e
    if proc.returncode != 0:
        raise KhmerClipError("Tesseract failed: " + proc.stderr.decode("utf-8", "replace").strip())
    return proc.stdout.decode("utf-8", "replace").strip()
```

- [ ] **Step 5: Fetch tessdata, point at a local binary, run tests**

Run:
```bash
.venv/bin/python tools/fetch_tessdata.py
```
Expected: `tessdata ready: …/apps/desktop/resources/tesseract/tessdata`.

For a local Tesseract on macOS use Homebrew (`brew install tesseract`) or the mac app's vendored one from the main checkout, for example `export KHMERCLIP_TESSERACT=/Users/kim/Sync/workspace/khmerclip/apps/mac/Vendor/tesseract/bin/tesseract`.

Run: `.venv/bin/python -m pytest tests/test_tesseract.py -v`
Expected: 6 fixture tests plus 6 unit tests pass, and the Windows-only test is skipped. Without any Tesseract the fixture tests show as skipped. Windows and Linux CI run them for real in Task 17.

- [ ] **Step 6: Commit**

```bash
/usr/bin/git add apps/desktop/tools/fetch_tessdata.py apps/desktop/src/khmerclip/ocr apps/desktop/tests/test_tesseract.py
/usr/bin/git commit -m "feat(desktop): add Tesseract engine with pinned tessdata"
```

---

### Task 5: TrOCR tokenizer decode

**Files:**
- Create: `apps/desktop/src/khmerclip/ocr/tokenizer.py`
- Test: `apps/desktop/tests/test_tokenizer.py`

**Interfaces:**
- Produces: `TrOCRTokenizer(payload: dict)`, `TrOCRTokenizer.from_file(path: Path) -> TrOCRTokenizer`, attributes `kind: str`, `eos_id: int`, `decoder_start_id: int`, method `decode(ids: Sequence[int]) -> str`. Raises `ValueError` for an unknown `kind`. Reads the exact `tokenizer.json` format written by the mac app's `export_tokenizer.py` (keys `kind, bos_id, eos_id, pad_id, unk_id, decoder_start_id, id_to_token, literal_ids?`). Module constant `_BYTE_ENCODER: dict[int, str]` (GPT-2 byte → unicode char), used by tests.

- [ ] **Step 1: Write the failing tests** (port of `TrOCRTokenizerTests` in `apps/mac/Tests/TrOCRTests.swift`)

`apps/desktop/tests/test_tokenizer.py`:
```python
import json

import pytest

from khmerclip.ocr.tokenizer import _BYTE_ENCODER, TrOCRTokenizer

SPECIALS = ["<s>", "<pad>", "</s>", "<unk>"]


def tok(kind="bpe", start=2, extra=(), literal=None):
    payload = {"kind": kind, "bos_id": 0, "eos_id": 2, "pad_id": 1, "unk_id": 3,
               "decoder_start_id": start, "id_to_token": SPECIALS + list(extra)}
    if literal is not None:
        payload["literal_ids"] = literal
    return TrOCRTokenizer(payload)


def test_bpe_reconstructs_bytes_and_keeps_leading_space():
    t = tok(extra=[".", "Ġthe", ",", "Ġto"])
    assert t.decode([0, 5, 7, 2]) == " the to"


def test_bpe_passes_literal_added_tokens_through_verbatim():
    t = tok(extra=["a", "b", "ក"], literal=[6])
    assert t.decode([0, 4, 5, 6, 2]) == "abក"


def test_bpe_byte_mapped_khmer_decodes_utf8():
    # A Khmer char spelled as GPT-2 byte-mapped pieces (the base vocab's way) must decode to UTF-8.
    piece = "".join(_BYTE_ENCODER[b] for b in "ក".encode("utf-8"))
    assert tok(extra=[piece]).decode([4]) == "ក"


def test_sentencepiece_replaces_marker_and_trims_leading_space():
    t = tok(kind="sentencepiece", extra=["▁the", ".", ",", "▁to"])
    assert t.decode([0, 4, 7, 2]) == "the to"


def test_decoder_start_id_reflects_exported_value():
    assert tok(start=2).decoder_start_id == 2
    assert tok(start=0).decoder_start_id == 0


def test_out_of_range_and_special_ids_are_dropped():
    assert tok(extra=["a"]).decode([0, 1, 2, 3, 4, 99, -1]) == "a"


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        tok(kind="wordpiece")


def test_from_file_reads_utf8(tmp_path):
    p = tmp_path / "tokenizer.json"
    p.write_text(json.dumps({"kind": "bpe", "bos_id": 0, "eos_id": 2, "pad_id": 1, "unk_id": 3,
                             "decoder_start_id": 0, "id_to_token": SPECIALS + ["ខ"], "literal_ids": [4]},
                            ensure_ascii=False), encoding="utf-8")
    assert TrOCRTokenizer.from_file(p).decode([4]) == "ខ"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_tokenizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'khmerclip.ocr.tokenizer'`.

- [ ] **Step 3: Implement**

`apps/desktop/src/khmerclip/ocr/tokenizer.py`:
```python
"""Decode-only TrOCR tokenizer. Port of apps/mac/Sources/OCR/TrOCRTokenizer.swift.

Two families: GPT-2 byte-level BPE ('Ġ' = leading space, non-ASCII bytes mapped to
printable scalars) and SentencePiece ('▁' = leading space). The Khmer fine-tune is BPE
plus literal Khmer added tokens (listed in literal_ids) that decode verbatim.
"""

import json
from collections.abc import Sequence
from pathlib import Path


def _gpt2_byte_encoder() -> dict[int, str]:
    printable = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    encoder = {b: chr(b) for b in printable}
    n = 0
    for b in range(256):
        if b not in printable:
            encoder[b] = chr(256 + n)
            n += 1
    return encoder


_BYTE_ENCODER = _gpt2_byte_encoder()
_BYTE_DECODER = {ch: b for b, ch in _BYTE_ENCODER.items()}


class TrOCRTokenizer:
    def __init__(self, payload: dict):
        self.kind: str = payload["kind"]
        if self.kind not in ("bpe", "sentencepiece"):
            raise ValueError(f"unknown tokenizer kind {self.kind!r}")
        self.eos_id: int = payload["eos_id"]
        # NOT always eos: fine-tunes may start from bos. Use whatever was exported.
        self.decoder_start_id: int = payload["decoder_start_id"]
        self._id_to_token: list[str] = payload["id_to_token"]
        self._special = {payload["bos_id"], payload["eos_id"], payload["pad_id"], payload["unk_id"]}
        self._literal = set(payload.get("literal_ids") or [])

    @classmethod
    def from_file(cls, path: Path) -> "TrOCRTokenizer":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def decode(self, ids: Sequence[int]) -> str:
        kept = [i for i in ids if i not in self._special and 0 <= i < len(self._id_to_token)]
        if self.kind == "sentencepiece":
            return "".join(self._id_to_token[i] for i in kept).replace("▁", " ").strip(" \t")
        out: list[str] = []
        buf = bytearray()
        for i in kept:
            token = self._id_to_token[i]
            if i in self._literal:
                if buf:
                    out.append(buf.decode("utf-8", "replace"))
                    buf.clear()
                out.append(token)
            else:
                buf.extend(_BYTE_DECODER[ch] for ch in token if ch in _BYTE_DECODER)
        if buf:
            out.append(buf.decode("utf-8", "replace"))
        return "".join(out)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_tokenizer.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add apps/desktop/src/khmerclip/ocr/tokenizer.py apps/desktop/tests/test_tokenizer.py
/usr/bin/git commit -m "feat(desktop): port TrOCR tokenizer decode"
```

---

### Task 6: TrOCR image preprocessing

**Files:**
- Create: `apps/desktop/src/khmerclip/ocr/preprocess.py`
- Test: `apps/desktop/tests/test_preprocess.py`

**Interfaces:**
- Produces (port of `TrOCREngine.swift` `inkMask` / `lineStrips` / `contentCrop` / `preprocess`):
  - `InkMask(ink: np.ndarray[bool] shape (h, w), scale: int)`
  - `ink_mask(image: Image.Image, scan_max: int = 512) -> InkMask | None`
  - `line_strips(image: Image.Image) -> list[Image.Image]`. Returns `[image]` itself when there are 0 or 1 lines.
  - `content_crop(image: Image.Image) -> Image.Image`. Returns `image` itself when nothing to crop.
  - `pixel_values(image: Image.Image) -> np.ndarray` float32 C-contiguous, shape `(1, 3, 384, 384)`, values in [-1, 1]

- [ ] **Step 1: Write the failing tests**

`apps/desktop/tests/test_preprocess.py`:
```python
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from khmerclip.ocr.preprocess import content_crop, ink_mask, line_strips, pixel_values
from khmerclip.paths import resource_dir


def bars(size, rows, bg="white", fg="black"):
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    for top, bottom in rows:
        draw.rectangle((20, top, size[0] - 20, bottom), fill=fg)  # inclusive rows top..bottom
    return img


def test_ink_mask_none_for_degenerate_image():
    assert ink_mask(Image.new("RGB", (1, 50))) is None


def test_ink_mask_downscales_long_edge_to_scan_max():
    mask = ink_mask(Image.new("RGB", (2000, 100), "white"))
    assert mask.scale == 4
    assert mask.ink.shape == (25, 500)


def test_line_strips_splits_two_bars():
    strips = line_strips(bars((400, 200), [(30, 60), (120, 150)]))
    assert [s.size for s in strips] == [(400, 90), (400, 110)]  # boundary at (60 + 120 + 1) // 2


def test_line_strips_merges_small_mark_into_nearest_line():
    # 4-row dot (< 35% of the 31-row line) sits 5 rows below line 1, 42 above line 2.
    strips = line_strips(bars((400, 200), [(20, 50), (55, 58), (100, 130)]))
    assert [s.size for s in strips] == [(400, 79), (400, 121)]


def test_line_strips_single_line_returns_original():
    img = bars((400, 100), [(30, 60)])
    strips = line_strips(img)
    assert len(strips) == 1 and strips[0] is img


def test_line_strips_three_khmer_lines():
    font = ImageFont.truetype(str(resource_dir() / "fonts" / "NotoSansKhmer-Regular.ttf"), 32)
    img = Image.new("RGB", (600, 290), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(["ច្បាប់ភូមិបាល", "ផ្តល់សិទ្ធិអំណាចឲ្យរដ្ឋ", "Hello world"]):
        draw.text((20, 10 + i * 90), line, font=font, fill="black")
    assert len(line_strips(img)) == 3


def test_content_crop_pads_four_percent_around_ink():
    img = bars((400, 200), [])
    ImageDraw.Draw(img).rectangle((100, 80, 299, 119), fill="black")
    assert content_crop(img).size == (216, 42)  # x 92..308, y 79..121


def test_content_crop_works_in_dark_mode():
    img = Image.new("RGB", (400, 200), "black")
    ImageDraw.Draw(img).rectangle((100, 80, 299, 119), fill="white")
    assert content_crop(img).size == (216, 42)


def test_content_crop_keeps_blank_and_sliver_images():
    blank = Image.new("RGB", (400, 200), "white")
    assert content_crop(blank) is blank
    sliver = Image.new("RGB", (400, 200), "white")
    ImageDraw.Draw(sliver).rectangle((10, 10, 11, 11), fill="black")
    assert content_crop(sliver) is sliver


def test_pixel_values_shape_and_normalization():
    arr = pixel_values(Image.new("RGB", (50, 20), "white"))
    assert arr.shape == (1, 3, 384, 384) and arr.dtype == np.float32
    assert arr.flags["C_CONTIGUOUS"]
    assert np.allclose(arr, 1.0)
    rgba = Image.new("RGBA", (50, 20), (0, 0, 0, 255))
    assert np.allclose(pixel_values(rgba), -1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_preprocess.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'khmerclip.ocr.preprocess'`.

- [ ] **Step 3: Implement**

`apps/desktop/src/khmerclip/ocr/preprocess.py`:
```python
"""TrOCR preprocessing, ported from apps/mac/Sources/OCR/TrOCREngine.swift.

Keep the constants and integer arithmetic identical to the Swift version so both apps
split lines and crop the same way.
"""

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

SCAN_MAX = 512
THRESHOLD = 40          # per-channel difference from background that counts as ink
MERGE_PERCENT = 35      # runs shorter than this % of the tallest merge into a neighbour
SIZE = 384              # TrOCR ViT input


@dataclass
class InkMask:
    ink: np.ndarray  # bool, shape (h, w)
    scale: int       # mask coords * scale = full-res coords


def ink_mask(image: Image.Image, scan_max: int = SCAN_MAX) -> InkMask | None:
    w, h = image.size
    if w <= 1 or h <= 1:
        return None
    scale = max(1, -(-max(w, h) // scan_max))
    sw, sh = w // scale, h // scale
    if sw <= 1 or sh <= 1:
        return None
    px = np.asarray(image.convert("RGB").resize((sw, sh), Image.BILINEAR), dtype=np.int32)
    corners = np.stack([px[0, 0], px[0, -1], px[-1, 0], px[-1, -1]])
    bg = corners.sum(axis=0) // 4
    return InkMask((np.abs(px - bg) > THRESHOLD).any(axis=2), scale)


def line_strips(image: Image.Image) -> list[Image.Image]:
    """Split a clip into horizontal text-line strips by row ink projection."""
    mask = ink_mask(image)
    if mask is None:
        return [image]
    runs: list[list[int]] = []
    for y, inked in enumerate(mask.ink.any(axis=1)):
        if not inked:
            continue
        if runs and runs[-1][1] == y - 1:
            runs[-1][1] = y
        else:
            runs.append([y, y])
    # ponytail: fixed 35%-of-tallest merge cutoff (same as mac); tune if headings next to
    # small body text get merged.
    while len(runs) > 1:
        heights = [hi - lo + 1 for lo, hi in runs]
        tallest = max(heights)
        i = next((k for k, hgt in enumerate(heights) if hgt * 100 < tallest * MERGE_PERCENT), None)
        if i is None:
            break
        gap_above = runs[i][0] - runs[i - 1][1] if i > 0 else math.inf
        gap_below = runs[i + 1][0] - runs[i][1] if i < len(runs) - 1 else math.inf
        j = i - 1 if gap_above <= gap_below else i + 1
        runs[min(i, j)] = [min(runs[i][0], runs[j][0]), max(runs[i][1], runs[j][1])]
        del runs[max(i, j)]
    if len(runs) <= 1:
        return [image]

    s, mh = mask.scale, mask.ink.shape[0]
    strips = []
    for k in range(len(runs)):
        top = 0 if k == 0 else (runs[k - 1][1] + runs[k][0] + 1) // 2
        bottom = mh if k == len(runs) - 1 else (runs[k][1] + runs[k + 1][0] + 1) // 2
        strips.append(image.crop((0, top * s, image.width, min(image.height, bottom * s))))
    return strips


def content_crop(image: Image.Image) -> Image.Image:
    """Trim uniform background margin so the text fills the frame (TrOCR decodes garbage
    past the real text when given a loose crop)."""
    mask = ink_mask(image)
    if mask is None:
        return image
    ys, xs = np.nonzero(mask.ink)
    if xs.size == 0:
        return image
    sh, sw = mask.ink.shape
    min_x, max_x, min_y, max_y = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    # A mis-detected background (corner on ink) collapses the box to a sliver: don't trust it.
    if (max_x - min_x + 1) * 20 < sw or (max_y - min_y + 1) * 20 < sh:
        return image
    pad_x = max(1, (max_x - min_x + 1) * 4 // 100)
    pad_y = max(1, (max_y - min_y + 1) * 4 // 100)
    s = mask.scale
    x0, y0 = max(0, (min_x - pad_x) * s), max(0, (min_y - pad_y) * s)
    x1 = min(image.width, (max_x + 1 + pad_x) * s)
    y1 = min(image.height, (max_y + 1 + pad_y) * s)
    return image.crop((x0, y0, x1, y1))


def pixel_values(image: Image.Image) -> np.ndarray:
    """HF ViTImageProcessor equivalent: bilinear resize to 384x384, mean = std = 0.5."""
    img = content_crop(image).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - 0.5) / 0.5
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_preprocess.py -v`
Expected: 10 passed. If `test_line_strips_three_khmer_lines` fails, print the inked-row runs of `ink_mask(img).ink.any(axis=1)`. PIL lays out Khmer without shaping, so a mark can float into the inter-line gap. Increase the line spacing in the test (not the 35% constant) until lines separate, and say so in the commit message.

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add apps/desktop/src/khmerclip/ocr/preprocess.py apps/desktop/tests/test_preprocess.py
/usr/bin/git commit -m "feat(desktop): port TrOCR line split, content crop and normalization"
```

---

### Task 7: TrOCR ONNX export, model manifest and publishing

**Files:**
- Create: `apps/desktop/src/khmerclip/ocr/trocr_models.py`
- Create: `apps/desktop/src/khmerclip/ocr/trocr_manifest.json`
- Create: `apps/desktop/tools/export_onnx.py`
- Create: `apps/desktop/tools/publish_models.py`
- Test: `apps/desktop/tests/test_trocr_models.py`

**Interfaces:**
- Consumes: `download.RemoteFile`, `download.download`, `download.is_present`, `download.sha256_file`, `paths.data_dir()`, `errors.EngineUnavailable`
- Produces:
  - `MODEL_FILES = ("encoder.onnx", "decoder.onnx", "tokenizer.json")`
  - `MANIFEST_PATH: Path` (the committed `trocr_manifest.json`)
  - `Manifest(version: str, files: tuple[RemoteFile, ...])` with `.available: bool` and `.total_size: int`
  - `load_manifest(path: Path = MANIFEST_PATH) -> Manifest`
  - `model_dir(manifest: Manifest, root: Path | None = None) -> Path` (= `(root or data_dir()/"models"/"trocr") / manifest.version`)
  - `is_installed(manifest: Manifest, root: Path | None = None) -> bool`
  - `install(manifest: Manifest, root: Path | None = None, progress=None, cancel=None) -> Path`
  - `build_manifest(model_dir: Path, base_url: str, version: str) -> dict`
  - ONNX graph contract used by Task 8: `encoder.onnx` input `pixel_values` float32 `[1,3,384,384]`, output `encoder_hidden_states`. `decoder.onnx` inputs `input_ids` int64 `[1, seq]` (dynamic `seq`) and `encoder_hidden_states`, output `logits` `[1, seq, vocab]`.

- [ ] **Step 1: Write the failing tests**

`apps/desktop/tests/test_trocr_models.py`:
```python
import hashlib
import json

import pytest

from khmerclip.errors import EngineUnavailable
from khmerclip.ocr import trocr_models as tm


def make_model_dir(tmp_path):
    d = tmp_path / "exported"
    d.mkdir()
    for name in tm.MODEL_FILES:
        (d / name).write_bytes(name.encode())
    return d


def test_committed_manifest_parses():
    manifest = tm.load_manifest()
    assert manifest.version
    assert all(f.name in tm.MODEL_FILES for f in manifest.files)


def test_build_manifest_hashes_each_file(tmp_path):
    d = make_model_dir(tmp_path)
    data = tm.build_manifest(d, "https://huggingface.co/o/r/resolve/abc123", "khmer-v1")
    assert data["version"] == "khmer-v1"
    assert data["files"][0] == {
        "name": "encoder.onnx", "url": "https://huggingface.co/o/r/resolve/abc123/encoder.onnx",
        "sha256": hashlib.sha256(b"encoder.onnx").hexdigest(), "size": len(b"encoder.onnx")}


def test_load_manifest_roundtrip_and_install(tmp_path):
    d = make_model_dir(tmp_path)
    data = tm.build_manifest(d, d.as_uri(), "test-v1")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    manifest = tm.load_manifest(path)
    root = tmp_path / "models"
    assert manifest.available and not tm.is_installed(manifest, root)
    assert tm.install(manifest, root) == root / "test-v1"
    assert tm.is_installed(manifest, root)
    assert manifest.total_size == sum(len(n.encode()) for n in tm.MODEL_FILES)


def test_install_unpublished_manifest_raises(tmp_path):
    empty = tm.Manifest("khmer-v1", ())
    with pytest.raises(EngineUnavailable, match="published"):
        tm.install(empty, tmp_path)
    assert not tm.is_installed(empty, tmp_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_trocr_models.py -v`
Expected: FAIL with `ImportError: cannot import name 'trocr_models'`.

- [ ] **Step 3: Implement the manifest module and the empty committed manifest**

`apps/desktop/src/khmerclip/ocr/trocr_manifest.json` (stays empty until Step 6 publishes the models; while empty, the app reports TrOCR as "not published yet"):
```json
{
  "version": "khmer-v1",
  "files": []
}
```

`apps/desktop/src/khmerclip/ocr/trocr_models.py`:
```python
"""Which TrOCR ONNX files to download (pinned manifest) and where they live."""

import json
from dataclasses import dataclass
from pathlib import Path

from khmerclip import paths
from khmerclip.download import RemoteFile, download, is_present, sha256_file
from khmerclip.errors import EngineUnavailable

MANIFEST_PATH = Path(__file__).with_name("trocr_manifest.json")
MODEL_FILES = ("encoder.onnx", "decoder.onnx", "tokenizer.json")


@dataclass(frozen=True)
class Manifest:
    version: str
    files: tuple[RemoteFile, ...]

    @property
    def available(self) -> bool:
        return bool(self.files)

    @property
    def total_size(self) -> int:
        return sum(f.size for f in self.files)


def load_manifest(path: Path = MANIFEST_PATH) -> Manifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Manifest(raw["version"], tuple(RemoteFile(**f) for f in raw["files"]))


def model_dir(manifest: Manifest, root: Path | None = None) -> Path:
    return (root or paths.data_dir() / "models" / "trocr") / manifest.version


def is_installed(manifest: Manifest, root: Path | None = None) -> bool:
    return manifest.available and is_present(manifest.files, model_dir(manifest, root))


def install(manifest: Manifest, root: Path | None = None, progress=None, cancel=None) -> Path:
    if not manifest.available:
        raise EngineUnavailable("The TrOCR model hasn't been published yet. Use Tesseract for now.")
    dest = model_dir(manifest, root)
    download(manifest.files, dest, progress=progress, cancel=cancel)
    return dest


def build_manifest(model_dir: Path, base_url: str, version: str) -> dict:
    files = []
    for name in MODEL_FILES:
        p = model_dir / name
        files.append({"name": name, "url": f"{base_url}/{name}", "sha256": sha256_file(p), "size": p.stat().st_size})
    return {"version": version, "files": files}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_trocr_models.py -v`
Expected: 4 passed.

- [ ] **Step 5: Write the export and publish tools**

`apps/desktop/tools/export_onnx.py`:
```python
"""Export a TrOCR checkpoint to encoder.onnx + decoder.onnx + tokenizer.json.

Mirrors the mac app's Vendor/trocr/convert.py (CoreML) and export_tokenizer.py, but emits
ONNX for khmerclip.ocr.trocr. Needs the `tools` extra:
    .venv/bin/pip install -e '.[tools]'
    .venv/bin/python tools/export_onnx.py <checkpoint dir or HF id> exported/khmer
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers.models.trocr.modeling_trocr import TrOCRSinusoidalPositionalEmbedding

PARITY_TOLERANCE = 1e-3


class Encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, pixel_values):
        return self.encoder(pixel_values=pixel_values).last_hidden_state


class DecoderStep(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, input_ids, encoder_hidden_states):
        return self.decoder(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states, use_cache=False).logits


def load_model(ckpt: str) -> VisionEncoderDecoderModel:
    model = VisionEncoderDecoderModel.from_pretrained(ckpt, low_cpu_mem_usage=False)
    # trocr-large-style sinusoidal position tables can be left on the meta device; rebuild them.
    for mod in model.modules():
        if isinstance(mod, TrOCRSinusoidalPositionalEmbedding) and mod.weights is not None and mod.weights.is_meta:
            mod.weights = mod.get_embedding(mod.weights.shape[0], mod.embedding_dim, mod.padding_idx)
    return model.eval()


def decoder_start_id(model) -> int:
    # Top-level config wins: the Khmer fine-tune sets it to bos (0) over a stale sub-config value.
    start = getattr(model.config, "decoder_start_token_id", None)
    return start if start is not None else model.config.decoder.decoder_start_token_id


def export_tokenizer(processor, model, out_path: Path) -> None:
    tok = processor.tokenizer
    vocab_size = model.config.decoder.vocab_size
    kind = "sentencepiece" if tok.__class__.__name__.lower().startswith("xlmroberta") else "bpe"
    specials = {tok.bos_token_id, tok.eos_token_id, tok.pad_token_id, tok.unk_token_id}
    data = {
        "kind": kind,
        "bos_id": tok.bos_token_id,
        "eos_id": tok.eos_token_id,
        "pad_id": tok.pad_token_id,
        "unk_id": tok.unk_token_id,
        "decoder_start_id": decoder_start_id(model),
        "id_to_token": [tok.convert_ids_to_tokens(i) or "" for i in range(vocab_size)],
        "literal_ids": sorted(i for i in tok.get_added_vocab().values() if i < vocab_size and i not in specials),
    }
    out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint)
    processor = TrOCRProcessor.from_pretrained(args.checkpoint)
    size = processor.image_processor.size
    img = size["height"] if isinstance(size, dict) else size
    enc, dec = Encoder(model), DecoderStep(model)

    pixel = torch.rand(1, 3, img, img)
    with torch.no_grad():
        hidden = enc(pixel)
    start = decoder_start_id(model)
    ids = torch.tensor([[start, 100]], dtype=torch.int64)

    torch.onnx.export(enc, (pixel,), out / "encoder.onnx", input_names=["pixel_values"],
                      output_names=["encoder_hidden_states"], opset_version=17, dynamo=False)
    torch.onnx.export(dec, (ids, hidden), out / "decoder.onnx",
                      input_names=["input_ids", "encoder_hidden_states"], output_names=["logits"],
                      dynamic_axes={"input_ids": {1: "seq"}, "logits": {1: "seq"}},
                      opset_version=17, dynamo=False)
    export_tokenizer(processor, model, out / "tokenizer.json")

    # Parity: onnxruntime vs torch, with a longer sequence than traced to prove the dynamic axis.
    ids3 = torch.tensor([[start, 100, 200]], dtype=torch.int64)
    with torch.no_grad():
        ref = dec(ids3, hidden).numpy()
    ort_hidden = ort.InferenceSession(str(out / "encoder.onnx")).run(None, {"pixel_values": pixel.numpy()})[0]
    got = ort.InferenceSession(str(out / "decoder.onnx")).run(
        None, {"input_ids": ids3.numpy(), "encoder_hidden_states": ort_hidden})[0]
    diff = float(np.abs(got - ref).max())
    print(f"parity max abs diff: {diff:.2e}")
    if diff > PARITY_TOLERANCE:
        raise SystemExit(f"ONNX output diverges from torch ({diff:.2e} > {PARITY_TOLERANCE})")
    print("exported to", out)


if __name__ == "__main__":
    main()
```

`apps/desktop/tools/publish_models.py`:
```python
"""Upload exported TrOCR ONNX files to the public HF repo and pin them in trocr_manifest.json.

    .venv/bin/huggingface-cli login
    .venv/bin/python tools/publish_models.py exported/khmer --version khmer-v1
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

from khmerclip.ocr.trocr_models import MANIFEST_PATH, MODEL_FILES, build_manifest

REPO = "KimChengSHEANG/khmerclip-trocr-khmer-onnx"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--repo", default=REPO)
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=False, exist_ok=True)
    commit = api.upload_folder(repo_id=args.repo, folder_path=str(args.model_dir),
                               allow_patterns=list(MODEL_FILES),
                               commit_message=f"KhmerClip TrOCR ONNX {args.version}")
    # Pin by commit revision so later uploads never change what an installed app downloads.
    base_url = f"https://huggingface.co/{args.repo}/resolve/{commit.oid}"
    data = build_manifest(args.model_dir, base_url, args.version)
    MANIFEST_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", MANIFEST_PATH)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Export, verify, publish (needs the checkpoint; publishing needs HF login)**

The checkpoint is gitignored and lives only in the main checkout. Run from `apps/desktop/`:
```bash
.venv/bin/pip install -e '.[tools]'
.venv/bin/python tools/export_onnx.py /Users/kim/Sync/workspace/khmerclip/apps/mac/raw_models/TrOCR-khmer exported/khmer
```
Expected: `parity max abs diff: …` at or below `1.00e-03`, then `exported to exported/khmer`. If the diff exceeds tolerance the script exits non-zero. Do not publish; report the number.

**Human-gated:** publishing needs the user's Hugging Face write token (`.venv/bin/huggingface-cli login`). Without credentials, skip this step, leave `trocr_manifest.json` empty, and continue. The app stays correct, and TrOCR shows "not published yet". With credentials:
```bash
.venv/bin/python tools/publish_models.py exported/khmer --version khmer-v1
.venv/bin/python -m pytest tests/test_trocr_models.py -v
```
Expected: the manifest lists three files with `https://huggingface.co/KimChengSHEANG/khmerclip-trocr-khmer-onnx/resolve/<sha>/…` URLs, and 4 tests pass.

- [ ] **Step 7: Commit**

```bash
/usr/bin/git add apps/desktop/src/khmerclip/ocr/trocr_models.py apps/desktop/src/khmerclip/ocr/trocr_manifest.json apps/desktop/tools/export_onnx.py apps/desktop/tools/publish_models.py apps/desktop/tests/test_trocr_models.py
/usr/bin/git commit -m "feat(desktop): add TrOCR ONNX export, publishing and pinned model manifest"
```

---

### Task 8: TrOCR runtime, OCR dispatcher, `khmerclip ocr` CLI

**Files:**
- Create: `apps/desktop/src/khmerclip/ocr/trocr.py`
- Modify: `apps/desktop/src/khmerclip/ocr/__init__.py` (was empty)
- Create: `apps/desktop/src/khmerclip/__main__.py`
- Test: `apps/desktop/tests/test_trocr.py`, `apps/desktop/tests/test_dispatch_cli.py`

**Interfaces:**
- Consumes: `TrOCRTokenizer` (Task 5), `line_strips` and `pixel_values` (Task 6), `trocr_models.load_manifest/is_installed/model_dir` (Task 7), `tesseract.recognize` (Task 4), `formatter.apply`, `errors.*`
- Produces:
  - `trocr.MAX_LENGTH = 64`
  - `trocr.Session(encoder, decoder, tokenizer)` (dataclass; encoder/decoder expose onnxruntime's `.run(output_names, feeds)`)
  - `trocr.load_session(model_dir: Path) -> Session` (cached per dir)
  - `trocr.installed_model_dir() -> Path` (env `KHMERCLIP_TROCR_DIR` overrides)
  - `trocr.preload() -> None` (best effort)
  - `trocr.recognize(image, progress: Callable[[str], None] | None = None, cancel: threading.Event | None = None) -> str`
  - `trocr.recognize_with_session(session, image, progress=None, cancel=None) -> str`
  - `khmerclip.ocr.Engine` (`StrEnum`: `TESSERACT = "tesseract"`, `TROCR = "trocr"`)
  - `khmerclip.ocr.recognize(image, engine: Engine | str, progress=None, cancel=None) -> str`
  - `khmerclip.__main__.main(argv: list[str] | None = None) -> int`. `khmerclip ocr IMAGE [--engine tesseract|trocr] [--flatten] [-o OUT]`. With no subcommand it calls `khmerclip.ui.app.run_gui(capture: bool) -> int` (Task 15).

- [ ] **Step 1: Write the failing tests**

`apps/desktop/tests/test_trocr.py`:
```python
import os
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from helpers import FIXTURES
from khmerclip.errors import Cancelled
from khmerclip.ocr import trocr
from khmerclip.ocr.tokenizer import TrOCRTokenizer

TOKENIZER = TrOCRTokenizer({"kind": "bpe", "bos_id": 0, "eos_id": 2, "pad_id": 1, "unk_id": 3,
                            "decoder_start_id": 0, "literal_ids": [4, 5],
                            "id_to_token": ["<s>", "<pad>", "</s>", "<unk>", "ក", "ខ"]})


class FakeEncoder:
    def run(self, names, feeds):
        assert feeds["pixel_values"].shape == (1, 3, 384, 384)
        return [np.zeros((1, 4, 8), np.float32)]


class ScriptedDecoder:
    """Emits script[seq_len - 1] as the next token (last entry repeats)."""

    def __init__(self, script):
        self.script, self.calls = script, 0

    def run(self, names, feeds):
        self.calls += 1
        ids = feeds["input_ids"]
        assert ids.dtype == np.int64
        seq = ids.shape[1]
        logits = np.zeros((1, seq, 6), np.float32)
        logits[0, -1, self.script[min(seq - 1, len(self.script) - 1)]] = 1.0
        return [logits]


def session(script):
    return SimpleNamespace(encoder=FakeEncoder(), decoder=ScriptedDecoder(script), tokenizer=TOKENIZER)


def lines_image(n):
    img = Image.new("RGB", (400, 100 * n), "white")
    for k in range(n):
        ImageDraw.Draw(img).rectangle((20, 100 * k + 30, 380, 100 * k + 60), fill="black")
    return img


def test_greedy_decode_until_eos():
    assert trocr.recognize_with_session(session([4, 5, 2]), lines_image(1)) == "កខ"


def test_each_line_decoded_and_joined_with_newline():
    s = session([4, 5, 2])
    assert trocr.recognize_with_session(s, lines_image(2)) == "កខ\nកខ"
    assert s.decoder.calls == 6


def test_decode_stops_at_max_length_without_eos():
    s = session([4])
    assert trocr.recognize_with_session(s, lines_image(1)) == "ក" * (trocr.MAX_LENGTH - 1)
    assert s.decoder.calls == trocr.MAX_LENGTH - 1


def test_cancel_raises():
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        trocr.recognize_with_session(session([4, 2]), lines_image(1), cancel=cancel)


def test_progress_reports_token_count():
    seen = []
    trocr.recognize_with_session(session([4, 5, 2]), lines_image(1), progress=seen.append)
    assert seen[-1] == "Reading text… (4)"


@pytest.mark.skipif(not os.environ.get("KHMERCLIP_TROCR_DIR"), reason="set KHMERCLIP_TROCR_DIR to an exported model")
def test_real_model_reads_fixture():
    with Image.open(FIXTURES / "1.png") as img:
        text = trocr.recognize(img)
    assert text.strip()
```

`apps/desktop/tests/test_dispatch_cli.py`:
```python
import subprocess
import sys

import pytest
from PIL import Image

import khmerclip.ocr as ocr
from khmerclip.__main__ import main


def test_dispatch_routes_by_engine(monkeypatch):
    from khmerclip.ocr import tesseract, trocr
    monkeypatch.setattr(tesseract, "recognize", lambda image: "tess")
    monkeypatch.setattr(trocr, "recognize", lambda image, progress=None, cancel=None: "tr")
    img = Image.new("RGB", (4, 4))
    assert ocr.recognize(img, "tesseract") == "tess"
    assert ocr.recognize(img, ocr.Engine.TROCR) == "tr"
    with pytest.raises(ValueError):
        ocr.recognize(img, "easyocr")


def test_cli_writes_output_file_utf8(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr, "recognize", lambda image, engine, progress=None, cancel=None: "ក\nខ")
    png = tmp_path / "in.png"
    Image.new("RGB", (8, 8), "white").save(png)
    out = tmp_path / "out.txt"
    assert main(["ocr", str(png), "-o", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == "ក\nខ\n"
    assert main(["ocr", str(png), "--flatten", "-o", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == "ក ខ\n"


def test_cli_prints_to_stdout(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ocr, "recognize", lambda image, engine, progress=None, cancel=None: "សួស្តី")
    png = tmp_path / "in.png"
    Image.new("RGB", (8, 8), "white").save(png)
    assert main(["ocr", str(png)]) == 0
    assert capsys.readouterr().out == "សួស្តី\n"


def test_cli_missing_file_exits_1(tmp_path, capsys):
    assert main(["ocr", str(tmp_path / "nope.png")]) == 1
    assert "khmerclip:" in capsys.readouterr().err


def test_cli_ocr_path_does_not_import_qt():
    code = "import sys, khmerclip.__main__, khmerclip.ocr.tesseract, khmerclip.ocr.trocr; print('PySide6' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_trocr.py tests/test_dispatch_cli.py -v`
Expected: FAIL with `ImportError: cannot import name 'trocr' from 'khmerclip.ocr'` and `No module named 'khmerclip.__main__'`.

- [ ] **Step 3: Implement the TrOCR runtime**

`apps/desktop/src/khmerclip/ocr/trocr.py`:
```python
"""TrOCR via onnxruntime: greedy, no-KV-cache decode per text line (same loop as
apps/mac/Sources/OCR/TrOCREngine.swift)."""

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from khmerclip.errors import Cancelled, EngineUnavailable, KhmerClipError
from khmerclip.ocr import trocr_models
from khmerclip.ocr.preprocess import line_strips, pixel_values
from khmerclip.ocr.tokenizer import TrOCRTokenizer

MAX_LENGTH = 64


@dataclass
class Session:
    encoder: Any
    decoder: Any
    tokenizer: TrOCRTokenizer


_cache: dict[Path, Session] = {}
_lock = threading.Lock()


def load_session(model_dir: Path) -> Session:
    with _lock:
        if model_dir not in _cache:
            import onnxruntime as ort

            try:
                _cache[model_dir] = Session(
                    ort.InferenceSession(str(model_dir / "encoder.onnx"), providers=["CPUExecutionProvider"]),
                    ort.InferenceSession(str(model_dir / "decoder.onnx"), providers=["CPUExecutionProvider"]),
                    TrOCRTokenizer.from_file(model_dir / "tokenizer.json"),
                )
            except Exception as e:  # onnxruntime raises several unrelated types for bad files
                raise EngineUnavailable(
                    f"TrOCR model files are unreadable ({e}). Delete {model_dir} and download again.") from e
        return _cache[model_dir]


def installed_model_dir() -> Path:
    if override := os.environ.get("KHMERCLIP_TROCR_DIR"):
        return Path(override)
    manifest = trocr_models.load_manifest()
    if not trocr_models.is_installed(manifest):
        raise EngineUnavailable("The TrOCR model isn't downloaded yet.")
    return trocr_models.model_dir(manifest)


def preload() -> None:
    """Warm the ONNX sessions while the user drags a selection. Errors surface on recognize()."""
    try:
        load_session(installed_model_dir())
    except KhmerClipError:
        pass


def recognize(image: Image.Image, progress: Callable[[str], None] | None = None,
              cancel: threading.Event | None = None) -> str:
    if progress:
        progress("Loading model…")
    return recognize_with_session(load_session(installed_model_dir()), image, progress, cancel)


def recognize_with_session(session, image: Image.Image, progress: Callable[[str], None] | None = None,
                           cancel: threading.Event | None = None) -> str:
    tok = session.tokenizer
    lines: list[str] = []
    done = 0
    for strip in line_strips(image):
        hidden = session.encoder.run(["encoder_hidden_states"], {"pixel_values": pixel_values(strip)})[0]
        ids = [tok.decoder_start_id]
        while len(ids) < MAX_LENGTH:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            logits = session.decoder.run(
                ["logits"], {"input_ids": np.array([ids], dtype=np.int64), "encoder_hidden_states": hidden})[0]
            ids.append(int(np.argmax(logits[0, -1])))
            if progress:
                progress(f"Reading text… ({done + len(ids)})")
            if ids[-1] == tok.eos_id:
                break
        done += len(ids)
        if line := tok.decode(ids):
            lines.append(line)
    return "\n".join(lines)
```

- [ ] **Step 4: Implement the dispatcher and CLI**

`apps/desktop/src/khmerclip/ocr/__init__.py`:
```python
"""OCR engine selection. Engines are imported lazily (onnxruntime is slow to import)."""

import threading
from collections.abc import Callable
from enum import StrEnum

from PIL import Image


class Engine(StrEnum):
    TESSERACT = "tesseract"
    TROCR = "trocr"


def recognize(image: Image.Image, engine: "Engine | str", progress: Callable[[str], None] | None = None,
              cancel: threading.Event | None = None) -> str:
    engine = Engine(engine)
    if engine is Engine.TROCR:
        from khmerclip.ocr import trocr

        return trocr.recognize(image, progress=progress, cancel=cancel)
    from khmerclip.ocr import tesseract

    return tesseract.recognize(image)
```

`apps/desktop/src/khmerclip/__main__.py`:
```python
"""Entry point: `khmerclip` (tray app), `khmerclip --capture`, `khmerclip ocr IMAGE`."""

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="khmerclip", description="Khmer OCR screen capture")
    parser.add_argument("--capture", action="store_true",
                        help="start a capture (signals the running instance; bind this to a desktop shortcut on Wayland)")
    sub = parser.add_subparsers(dest="command")
    ocr = sub.add_parser("ocr", help="OCR an image file and print the text")
    ocr.add_argument("image", type=Path)
    ocr.add_argument("--engine", choices=["tesseract", "trocr"], default="tesseract")
    ocr.add_argument("--flatten", action="store_true", help="join lines into one paragraph")
    ocr.add_argument("-o", "--output", type=Path, help="write UTF-8 text here instead of stdout")
    return parser


def _err(message: object) -> None:
    if sys.stderr:  # None in the windowed (no console) Windows build
        print(f"khmerclip: {message}", file=sys.stderr)


def _ocr(args: argparse.Namespace) -> int:
    from PIL import Image

    import khmerclip.ocr as ocr
    from khmerclip.errors import KhmerClipError
    from khmerclip.formatter import apply

    try:
        with Image.open(args.image) as img:
            img.load()
            text = ocr.recognize(img, args.engine)
    except (OSError, KhmerClipError) as e:
        _err(e)
        return 1
    text = apply(text, preserve_linebreaks=not args.flatten)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    elif sys.stdout:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to a legacy code page
        print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "ocr":
        return _ocr(args)
    from khmerclip.ui.app import run_gui  # Qt is imported only for the GUI

    return run_gui(capture=args.capture)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_trocr.py tests/test_dispatch_cli.py -v`
Expected: 10 passed and 1 skipped (the real-model test).

If Task 7 Step 6 exported a model, also run `KHMERCLIP_TROCR_DIR=exported/khmer .venv/bin/python -m pytest tests/test_trocr.py -k real -v`. Expected: 1 passed. Then run `KHMERCLIP_TROCR_DIR=exported/khmer .venv/bin/python -m khmerclip ocr ../mac/Tests/Fixtures/ocr/1.png --engine trocr` and compare the Khmer text by eye against `1.txt`.

- [ ] **Step 6: Run the full suite and commit**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass (skips allowed for Tesseract/real-model tests).

```bash
/usr/bin/git add apps/desktop/src/khmerclip/ocr/trocr.py apps/desktop/src/khmerclip/ocr/__init__.py apps/desktop/src/khmerclip/__main__.py apps/desktop/tests/test_trocr.py apps/desktop/tests/test_dispatch_cli.py
/usr/bin/git commit -m "feat(desktop): add TrOCR onnxruntime engine, OCR dispatcher and ocr CLI"
```

---

### Task 9: Capture geometry, backend selection, Wayland portal

**Files:**
- Create: `apps/desktop/src/khmerclip/capture/__init__.py` (empty)
- Create: `apps/desktop/src/khmerclip/capture/geometry.py`
- Create: `apps/desktop/src/khmerclip/capture/portal.py`
- Test: `apps/desktop/tests/test_capture_geometry.py`, `apps/desktop/tests/test_portal.py`

**Interfaces:**
- Consumes: `errors.KhmerClipError`
- Produces:
  - `geometry.MIN_SIDE = 4` (logical px)
  - `geometry.normalize_drag(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float]` (x, y, w, h)
  - `geometry.to_physical(rect: tuple[float, float, float, float], ratio: float, image_w: int, image_h: int) -> tuple[int, int, int, int] | None`. Returns `None` when the selection is smaller than `MIN_SIDE` logical px or falls outside the image.
  - `geometry.capture_backend(platform: str | None = None, env: Mapping[str, str] | None = None) -> str`. Returns `"portal"` (Wayland) or `"qt"`.
  - `portal.request_path(unique_name: str, token: str) -> str`
  - `portal.uri_from_response(body: tuple) -> str | None`
  - `portal.uri_to_path(uri: str) -> Path`
  - `portal.screenshot() -> PIL.Image.Image | None` (blocking. `None` means the user cancelled. Raises `KhmerClipError` when the portal is unusable)

- [ ] **Step 1: Write the failing tests**

`apps/desktop/tests/test_capture_geometry.py`:
```python
from khmerclip.capture.geometry import capture_backend, normalize_drag, to_physical


def test_normalize_drag_any_direction():
    assert normalize_drag(50, 40, 10, 100) == (10, 40, 40, 60)


def test_to_physical_identity_at_100_percent():
    assert to_physical((10, 20, 100, 50), 1.0, 800, 600) == (10, 20, 100, 50)


def test_to_physical_150_percent():
    assert to_physical((10, 10, 100, 50), 1.5, 1920, 1080) == (15, 15, 150, 75)


def test_to_physical_125_percent_rounds_outward():
    # 3*1.25=3.75 -> floor 3 ; (3+10)*1.25=16.25 -> ceil 17
    assert to_physical((3, 3, 10, 10), 1.25, 1000, 1000) == (3, 3, 14, 14)


def test_to_physical_clamps_to_image():
    assert to_physical((790, 590, 50, 50), 1.0, 800, 600) == (790, 590, 10, 10)


def test_to_physical_rejects_tiny_or_outside_selection():
    assert to_physical((10, 10, 3, 100), 2.0, 800, 600) is None
    assert to_physical((900, 10, 50, 50), 1.0, 800, 600) is None


def test_capture_backend_wayland_session_uses_portal():
    assert capture_backend("linux", {"XDG_SESSION_TYPE": "wayland"}) == "portal"
    assert capture_backend("linux", {"WAYLAND_DISPLAY": "wayland-0"}) == "portal"


def test_capture_backend_x11_and_windows_use_qt():
    assert capture_backend("linux", {"XDG_SESSION_TYPE": "x11", "WAYLAND_DISPLAY": "wayland-0"}) == "qt"
    assert capture_backend("linux", {"DISPLAY": ":0"}) == "qt"
    assert capture_backend("win32", {"WAYLAND_DISPLAY": "x"}) == "qt"
```

`apps/desktop/tests/test_portal.py`:
```python
import sys
from pathlib import Path

import pytest

from khmerclip.capture import portal


def test_request_path_follows_portal_convention():
    assert portal.request_path(":1.42", "khmerclipab12") == "/org/freedesktop/portal/desktop/request/1_42/khmerclipab12"


def test_uri_from_response_success_and_cancel():
    assert portal.uri_from_response((0, {"uri": ("s", "file:///tmp/a%20b.png")})) == "file:///tmp/a%20b.png"
    assert portal.uri_from_response((1, {})) is None          # user cancelled
    assert portal.uri_from_response((0, {})) is None          # no uri: treat as cancelled


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file URIs")
def test_uri_to_path_unquotes():
    assert portal.uri_to_path("file:///tmp/a%20b%E1%9E%80.png") == Path("/tmp/a bក.png")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_capture_geometry.py tests/test_portal.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'khmerclip.capture'`.

- [ ] **Step 3: Implement**

`apps/desktop/src/khmerclip/capture/__init__.py`: create as an empty file.

`apps/desktop/src/khmerclip/capture/geometry.py`:
```python
import math
import os
import sys
from collections.abc import Mapping

MIN_SIDE = 4  # logical px; anything smaller is a click, not a selection


def normalize_drag(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float]:
    return min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)


def to_physical(rect: tuple[float, float, float, float], ratio: float, image_w: int, image_h: int
                ) -> tuple[int, int, int, int] | None:
    """Map a logical selection on an overlay to device pixels of the grabbed screenshot.
    `ratio` = screenshot pixels per overlay pixel (1.25/1.5/2.0 under display scaling).
    Rounds outward so no selected pixel is lost; clamps to the image."""
    x, y, w, h = rect
    if w < MIN_SIDE or h < MIN_SIDE:
        return None
    px0, py0 = max(0, math.floor(x * ratio)), max(0, math.floor(y * ratio))
    px1, py1 = min(image_w, math.ceil((x + w) * ratio)), min(image_h, math.ceil((y + h) * ratio))
    if px1 - px0 < 1 or py1 - py0 < 1:
        return None
    return px0, py0, px1 - px0, py1 - py0


def capture_backend(platform: str | None = None, env: Mapping[str, str] | None = None) -> str:
    """'portal' on Wayland (apps can't read the screen or grab global keys there), else 'qt'."""
    platform = sys.platform if platform is None else platform
    env = os.environ if env is None else env
    if not platform.startswith("linux"):
        return "qt"
    session = env.get("XDG_SESSION_TYPE")
    if session:
        return "portal" if session == "wayland" else "qt"
    return "portal" if env.get("WAYLAND_DISPLAY") else "qt"
```

`apps/desktop/src/khmerclip/capture/portal.py`:
```python
"""Wayland screen capture through org.freedesktop.portal.Screenshot (interactive mode shows
the desktop's own region picker). jeepney is imported lazily: it exists only on Linux."""

import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from PIL import Image

from khmerclip.errors import KhmerClipError


def request_path(unique_name: str, token: str) -> str:
    sender = unique_name.lstrip(":").replace(".", "_")
    return f"/org/freedesktop/portal/desktop/request/{sender}/{token}"


def uri_from_response(body: tuple) -> str | None:
    code, results = body
    if code != 0:
        return None
    uri = results.get("uri")
    return uri[1] if uri else None  # jeepney returns variants as (signature, value)


def uri_to_path(uri: str) -> Path:
    return Path(urllib.request.url2pathname(urllib.parse.urlparse(uri).path))


def screenshot() -> Image.Image | None:
    """Blocking; run off the GUI thread. None if the user cancelled the portal dialog."""
    try:
        from jeepney import DBusAddress, MatchRule, message_bus, new_method_call
        from jeepney.io.blocking import Proxy, open_dbus_connection

        token = "khmerclip" + uuid.uuid4().hex[:8]
        with open_dbus_connection(bus="SESSION") as conn:
            rule = MatchRule(type="signal", interface="org.freedesktop.portal.Request", member="Response",
                             path=request_path(conn.unique_name, token))
            Proxy(message_bus, conn).AddMatch(rule)
            screenshot_api = DBusAddress("/org/freedesktop/portal/desktop", bus_name="org.freedesktop.portal.Desktop",
                                         interface="org.freedesktop.portal.Screenshot")
            with conn.filter(rule) as queue:
                conn.send_and_get_reply(new_method_call(
                    screenshot_api, "Screenshot", "sa{sv}",
                    ("", {"handle_token": ("s", token), "interactive": ("b", True)})))
                response = conn.recv_until_filtered(queue)
        uri = uri_from_response(response.body)
        if uri is None:
            return None
        # ponytail: the portal may also save this file under ~/Pictures; we leave it (documented limitation).
        with Image.open(uri_to_path(uri)) as img:
            return img.convert("RGB")
    except Exception as e:  # no session bus, no portal, D-Bus error reply, unreadable file
        raise KhmerClipError(f"Screen capture through the desktop portal failed: {e}") from e
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_capture_geometry.py tests/test_portal.py -v`
Expected: 11 passed (10 on Windows, with the POSIX URI test skipped).

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add apps/desktop/src/khmerclip/capture apps/desktop/tests/test_capture_geometry.py apps/desktop/tests/test_portal.py
/usr/bin/git commit -m "feat(desktop): add capture geometry, backend choice and Wayland portal capture"
```

---

### Task 10: Qt region selector overlay (Windows / X11)

**Files:**
- Create: `apps/desktop/src/khmerclip/capture/overlay.py`
- Test: `apps/desktop/tests/test_overlay.py`

**Interfaces:**
- Consumes: `geometry.normalize_drag`, `geometry.to_physical`
- Produces:
  - `overlay.qimage_to_pil(image: QImage) -> PIL.Image.Image` (RGB)
  - `overlay.grab_screen(screen: QScreen) -> QPixmap`
  - `overlay.RegionSelector(grab: Callable[[QScreen], QPixmap] = grab_screen, parent: QObject | None = None)` with signals `selected(object)` (a `PIL.Image.Image`) and `cancelled()`, method `start() -> None`, attribute `overlays: list[QWidget]` (non-empty while selecting)

- [ ] **Step 1: Write the failing tests**

`apps/desktop/tests/test_overlay.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_overlay.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'khmerclip.capture.overlay'`.

- [ ] **Step 3: Implement**

`apps/desktop/src/khmerclip/capture/overlay.py`:
```python
"""Frozen-screenshot region selector: grab every screen first, then show each grab in a
fullscreen overlay and crop from the grab. Cropping from the grabbed image (not a second
capture) keeps the selection exact under display scaling and on any monitor layout."""

import io
from collections.abc import Callable
from functools import partial

from PIL import Image
from PySide6.QtCore import QBuffer, QIODevice, QObject, QRect, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPen, QPixmap, QRegion, QScreen
from PySide6.QtWidgets import QWidget

from khmerclip.capture.geometry import normalize_drag, to_physical


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
        dimmed = QRegion(self.rect())
        if selection:
            dimmed = dimmed.subtracted(QRegion(selection))
        painter.setClipRegion(dimmed)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 100))
        painter.setClipping(False)
        if selection:
            painter.setPen(QPen(QColor("#4da3ff"), 1))
            painter.drawRect(selection)

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_overlay.py -v`
Expected: 4 passed (offscreen Qt platform from `conftest.py`).

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add apps/desktop/src/khmerclip/capture/overlay.py apps/desktop/tests/test_overlay.py
/usr/bin/git commit -m "feat(desktop): add frozen-screenshot region selector overlay"
```

---

### Task 11: Global hotkey and single-instance command channel

**Files:**
- Create: `apps/desktop/src/khmerclip/hotkey.py`
- Create: `apps/desktop/src/khmerclip/single_instance.py`
- Test: `apps/desktop/tests/test_hotkey.py`, `apps/desktop/tests/test_single_instance.py`

**Interfaces:**
- Consumes: `geometry.capture_backend`
- Produces:
  - `hotkey.qt_to_pynput(seq: str) -> str | None` (`"Ctrl+Shift+2"` → `"<ctrl>+<shift>+2"`; None if unsupported)
  - `hotkey.pynput_to_qt(hotkey: str) -> str` (inverse, for display in `QKeySequenceEdit`)
  - `hotkey.HotkeyListener()` with `.start(hotkey: str, callback: Callable[[], None], backend: str | None = None) -> bool` and `.stop() -> None`. `start` returns False on Wayland, when pynput is missing, without an X display, or for an invalid combo.
  - `single_instance.server_name() -> str`
  - `single_instance.send_command(command: str, name: str | None = None, timeout_ms: int = 1000) -> bool`
  - `single_instance.CommandServer(name: str | None = None, parent=None)` with signal `command(str)` and methods `listen() -> bool`, `close() -> None`

- [ ] **Step 1: Write the failing tests**

`apps/desktop/tests/test_hotkey.py`:
```python
import sys

from khmerclip.hotkey import HotkeyListener, pynput_to_qt, qt_to_pynput


def test_qt_to_pynput_basic_and_shifted_digit():
    assert qt_to_pynput("Ctrl+Shift+2") == "<ctrl>+<shift>+2"
    assert qt_to_pynput("Ctrl+Shift+@") == "<ctrl>+<shift>+2"   # Qt records the shifted symbol
    assert qt_to_pynput("Meta+Alt+K") == "<cmd>+<alt>+k"
    assert qt_to_pynput("F9") == "<f9>"
    assert qt_to_pynput("Ctrl+Space") == "<ctrl>+<space>"


def test_qt_to_pynput_rejects_unsupported():
    assert qt_to_pynput("") is None
    assert qt_to_pynput("K") is None                    # no modifier
    assert qt_to_pynput("Ctrl+Hyper+K") is None
    assert qt_to_pynput("Ctrl+K, Ctrl+L") is None       # multi-chord
    assert qt_to_pynput("Ctrl+PgUp") is None


def test_pynput_to_qt_roundtrip():
    assert pynput_to_qt("<ctrl>+<shift>+2") == "Ctrl+Shift+2"
    assert pynput_to_qt("<cmd>+<f5>") == "Meta+F5"
    assert qt_to_pynput(pynput_to_qt("<alt>+<space>")) == "<alt>+<space>"


def test_listener_refuses_on_wayland():
    assert HotkeyListener().start("<ctrl>+k", lambda: None, backend="portal") is False


def test_listener_reports_missing_pynput(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynput", None)
    assert HotkeyListener().start("<ctrl>+k", lambda: None, backend="qt") is False
```

`apps/desktop/tests/test_single_instance.py`:
```python
import uuid

from khmerclip.single_instance import CommandServer, send_command, server_name


def test_command_roundtrip(qtbot):
    name = f"khmerclip-test-{uuid.uuid4().hex[:8]}"
    server = CommandServer(name)
    assert server.listen()
    with qtbot.waitSignal(server.command, timeout=3000) as blocker:
        assert send_command("capture", name)
    assert blocker.args == ["capture"]
    server.close()


def test_send_without_server_returns_false(qapp):
    assert send_command("capture", f"khmerclip-none-{uuid.uuid4().hex[:8]}", timeout_ms=200) is False


def test_server_name_is_ascii_and_stable():
    assert server_name() == server_name()
    assert server_name().isascii()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_hotkey.py tests/test_single_instance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'khmerclip.hotkey'`.

- [ ] **Step 3: Implement**

`apps/desktop/src/khmerclip/hotkey.py`:
```python
"""Global capture hotkey. pynput on Windows and X11; Wayland has no app-level global
hotkeys, so users bind `khmerclip --capture` in their desktop settings instead."""

import re
from collections.abc import Callable

from khmerclip.capture.geometry import capture_backend

_MODS = {"ctrl": "<ctrl>", "shift": "<shift>", "alt": "<alt>", "meta": "<cmd>"}
_NAMED = {"space": "<space>", "print": "<print_screen>"}
_SHIFTED = dict(zip("!@#$%^&*()", "1234567890"))  # US layout: Qt reports Shift+2 as "@"
_FKEY = re.compile(r"f([1-9]|1[0-9]|2[0-4])")


def qt_to_pynput(seq: str) -> str | None:
    if not seq or "," in seq or seq.endswith("++"):
        return None
    *mods, key = [p.strip() for p in seq.split("+")]
    mods = [m.lower() for m in mods]
    if any(m not in _MODS for m in mods):
        return None
    lower = key.lower()
    if _FKEY.fullmatch(lower):
        key_out = f"<{lower}>"
    elif lower in _NAMED:
        key_out = _NAMED[lower]
    elif len(key) == 1:
        key_out = _SHIFTED.get(key, key.lower()) if "shift" in mods else key.lower()
    else:
        return None
    if not mods and not (key_out.startswith("<f") or key_out == "<print_screen>"):
        return None  # a bare letter would hijack normal typing
    return "+".join([_MODS[m] for m in mods] + [key_out])


def pynput_to_qt(hotkey: str) -> str:
    qt_mods = {v: k.capitalize() for k, v in _MODS.items()}
    qt_named = {v: k.capitalize() for k, v in _NAMED.items()}
    out = []
    for part in hotkey.split("+"):
        if part in qt_mods:
            out.append(qt_mods[part])
        elif part in qt_named:
            out.append(qt_named[part])
        elif re.fullmatch(r"<f\d+>", part):
            out.append(part[1:-1].upper())
        else:
            out.append(part.upper())
    return "+".join(out)


class HotkeyListener:
    def __init__(self):
        self._listener = None

    def start(self, hotkey: str, callback: Callable[[], None], backend: str | None = None) -> bool:
        """callback runs on pynput's thread: pass a Qt signal's emit, never touch widgets."""
        self.stop()
        if (backend or capture_backend()) != "qt":
            return False  # Wayland: pynput would only see XWayland windows -- worse than nothing
        try:
            from pynput import keyboard

            listener = keyboard.GlobalHotKeys({hotkey: callback})
            listener.start()
        except Exception:  # pynput missing (macOS dev), no X display, invalid combo
            return False
        self._listener = listener
        return True

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
```

`apps/desktop/src/khmerclip/single_instance.py`:
```python
"""One KhmerClip per user. A second launch (`khmerclip --capture` bound to a desktop
shortcut on Wayland) forwards its command to the running instance over a local socket."""

import getpass
import hashlib
from functools import partial

from PySide6.QtCore import QObject, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket


def server_name() -> str:
    try:
        user = getpass.getuser()
    except Exception:  # no USER/USERNAME in a stripped environment
        user = "user"
    return "khmerclip-" + hashlib.sha1(user.encode("utf-8")).hexdigest()[:12]


def send_command(command: str, name: str | None = None, timeout_ms: int = 1000) -> bool:
    sock = QLocalSocket()
    sock.connectToServer(name or server_name())
    if not sock.waitForConnected(timeout_ms):
        return False
    sock.write(command.encode("utf-8") + b"\n")
    ok = sock.waitForBytesWritten(timeout_ms)
    sock.disconnectFromServer()
    return ok


class CommandServer(QObject):
    command = Signal(str)

    def __init__(self, name: str | None = None, parent: QObject | None = None):
        super().__init__(parent)
        self._name = name or server_name()
        self._server = QLocalServer(self)
        self._server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
        self._server.newConnection.connect(self._accept)

    def listen(self) -> bool:
        """Call only after send_command() failed: removes a stale socket left by a crash."""
        QLocalServer.removeServer(self._name)
        return self._server.listen(self._name)

    def close(self) -> None:
        self._server.close()

    def _accept(self) -> None:
        while (sock := self._server.nextPendingConnection()) is not None:
            sock.readyRead.connect(partial(self._read, sock))
            sock.disconnected.connect(sock.deleteLater)
            self._read(sock)  # data may already be buffered

    def _read(self, sock: QLocalSocket) -> None:
        while sock.canReadLine():
            line = bytes(sock.readLine()).decode("utf-8", "replace").strip()
            if line:
                self.command.emit(line)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_hotkey.py tests/test_single_instance.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add apps/desktop/src/khmerclip/hotkey.py apps/desktop/src/khmerclip/single_instance.py apps/desktop/tests/test_hotkey.py apps/desktop/tests/test_single_instance.py
/usr/bin/git commit -m "feat(desktop): add global hotkey listener and single-instance command channel"
```

---

### Task 12: Launch at login

**Files:**
- Create: `apps/desktop/src/khmerclip/autostart.py`
- Test: `apps/desktop/tests/test_autostart.py`

**Interfaces:**
- Consumes: `paths.atomic_write_text`
- Produces:
  - `launch_command(env: Mapping[str, str] | None = None, executable: str | None = None, frozen: bool | None = None) -> list[str]`
  - `desktop_entry(command: list[str]) -> str`
  - `autostart_file(config_home: Path | None = None) -> Path`
  - `set_enabled(enabled: bool, platform: str | None = None, config_home: Path | None = None, value_name: str = "KhmerClip", command: list[str] | None = None) -> None` (raises `OSError` on registry/file failure; `NotImplementedError` on other platforms)
  - `is_enabled(platform: str | None = None, config_home: Path | None = None, value_name: str = "KhmerClip") -> bool`

- [ ] **Step 1: Write the failing tests**

`apps/desktop/tests/test_autostart.py`:
```python
import sys

import pytest

from khmerclip import autostart


def test_launch_command_prefers_appimage_path():
    env = {"APPIMAGE": "/home/u/Apps/KhmerClip.AppImage"}
    assert autostart.launch_command(env, "/usr/bin/python3", frozen=True) == ["/home/u/Apps/KhmerClip.AppImage"]


def test_launch_command_frozen_and_dev():
    assert autostart.launch_command({}, r"C:\Apps\KhmerClip.exe", frozen=True) == [r"C:\Apps\KhmerClip.exe"]
    assert autostart.launch_command({}, "/venv/bin/python", frozen=False) == ["/venv/bin/python", "-m", "khmerclip"]


def test_desktop_entry_quotes_spaces_and_non_ascii():
    entry = autostart.desktop_entry(["/home/ស/My Apps/KhmerClip.AppImage"])
    assert 'Exec="/home/ស/My Apps/KhmerClip.AppImage"\n' in entry
    assert entry.startswith("[Desktop Entry]\n")


def test_desktop_entry_escapes_reserved_characters():
    entry = autostart.desktop_entry(['/tmp/a$b"c%d'])
    assert 'Exec="/tmp/a\\\\$b\\\\"c%%d"\n' in entry


def test_linux_enable_disable(tmp_path):
    cmd = ["/opt/KhmerClip.AppImage"]
    autostart.set_enabled(True, platform="linux", config_home=tmp_path, command=cmd)
    path = tmp_path / "autostart" / "khmerclip.desktop"
    assert path.read_text(encoding="utf-8") == autostart.desktop_entry(cmd)
    assert autostart.is_enabled(platform="linux", config_home=tmp_path)
    autostart.set_enabled(False, platform="linux", config_home=tmp_path)
    assert not path.exists()
    autostart.set_enabled(False, platform="linux", config_home=tmp_path)  # idempotent


def test_unsupported_platform_raises(tmp_path):
    with pytest.raises(NotImplementedError):
        autostart.set_enabled(True, platform="darwin", config_home=tmp_path, command=["x"])


@pytest.mark.skipif(sys.platform != "win32", reason="Windows registry")
def test_windows_run_key_roundtrip():
    name = "KhmerClipTest"
    cmd = [r"C:\Program Files\KhmerClip\KhmerClip.exe"]
    try:
        autostart.set_enabled(True, platform="win32", value_name=name, command=cmd)
        assert autostart.is_enabled(platform="win32", value_name=name)
    finally:
        autostart.set_enabled(False, platform="win32", value_name=name, command=cmd)
    assert not autostart.is_enabled(platform="win32", value_name=name)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_autostart.py -v`
Expected: FAIL with `ImportError: cannot import name 'autostart'`.

- [ ] **Step 3: Implement**

`apps/desktop/src/khmerclip/autostart.py`:
```python
"""Launch at login: HKCU Run key on Windows, XDG autostart .desktop file on Linux."""

import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from khmerclip.paths import atomic_write_text

APP_ID = "KhmerClip"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"


def launch_command(env: Mapping[str, str] | None = None, executable: str | None = None,
                   frozen: bool | None = None) -> list[str]:
    env = os.environ if env is None else env
    executable = sys.executable if executable is None else executable
    frozen = getattr(sys, "frozen", False) if frozen is None else frozen
    if env.get("APPIMAGE"):  # set by the AppImage runtime; sys.executable is a temp mount
        return [env["APPIMAGE"]]
    if frozen:
        return [executable]
    return [executable, "-m", "khmerclip"]


def _desktop_quote(arg: str) -> str:
    # Desktop Entry spec: inside quotes escape \ " ` $ with a backslash; the general string
    # escaping then doubles every backslash; % must be written %%.
    for ch in '\\"`$':  # backslash first
        arg = arg.replace(ch, "\\" + ch)
    arg = arg.replace("\\", "\\\\")
    return '"' + arg.replace("%", "%%") + '"'


def desktop_entry(command: list[str]) -> str:
    return ("[Desktop Entry]\nType=Application\nName=KhmerClip\n"
            f"Exec={' '.join(_desktop_quote(a) for a in command)}\n"
            "Terminal=false\nX-GNOME-Autostart-enabled=true\n")


def autostart_file(config_home: Path | None = None) -> Path:
    base = config_home or Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "autostart" / "khmerclip.desktop"


def set_enabled(enabled: bool, platform: str | None = None, config_home: Path | None = None,
                value_name: str = APP_ID, command: list[str] | None = None) -> None:
    platform = sys.platform if platform is None else platform
    command = command or launch_command()
    if platform == "win32":
        import winreg

        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
            if enabled:
                winreg.SetValueEx(key, value_name, 0, winreg.REG_SZ, subprocess.list2cmdline(command))
            else:
                try:
                    winreg.DeleteValue(key, value_name)
                except FileNotFoundError:
                    pass
    elif platform.startswith("linux"):
        path = autostart_file(config_home)
        if enabled:
            atomic_write_text(path, desktop_entry(command))
        else:
            path.unlink(missing_ok=True)
    else:
        raise NotImplementedError(f"launch at login is not supported on {platform}")


def is_enabled(platform: str | None = None, config_home: Path | None = None, value_name: str = APP_ID) -> bool:
    platform = sys.platform if platform is None else platform
    if platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
                winreg.QueryValueEx(key, value_name)
            return True
        except FileNotFoundError:
            return False
    if platform.startswith("linux"):
        return autostart_file(config_home).is_file()
    return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_autostart.py -v`
Expected: 6 passed, 1 skipped (Windows registry test runs on Windows CI).

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add apps/desktop/src/khmerclip/autostart.py apps/desktop/tests/test_autostart.py
/usr/bin/git commit -m "feat(desktop): add launch-at-login for Windows and Linux"
```

---

### Task 13: Update check

**Files:**
- Create: `apps/desktop/src/khmerclip/updater.py`
- Test: `apps/desktop/tests/test_updater.py`

**Interfaces:**
- Consumes: `khmerclip.__version__`
- Produces:
  - `FEED_URL = "https://api.github.com/repos/KimChengSHEANG/khmerclip-releases/releases/latest"`
  - `CHECK_INTERVAL_S = 86400`
  - `Release(version: str, page_url: str)` (frozen dataclass)
  - `parse_version(tag: str) -> tuple[int, int, int] | None`
  - `newer_release(payload: object, current: str) -> Release | None`
  - `fetch_latest(url: str = FEED_URL, timeout: float = 10) -> dict | None`
  - `check(current: str = __version__, fetch: Callable[[], dict | None] = fetch_latest) -> Release | None`
  - `due(last_check: float, now: float, interval: float = CHECK_INTERVAL_S) -> bool`

- [ ] **Step 1: Write the failing tests**

`apps/desktop/tests/test_updater.py`:
```python
from khmerclip import updater
from khmerclip.updater import Release, due, newer_release, parse_version

PAGE = "https://github.com/KimChengSHEANG/khmerclip-releases/releases/tag/"


def payload(tag="v0.2.0", **extra):
    return {"tag_name": tag, "html_url": PAGE + tag, "draft": False, "prerelease": False, **extra}


def test_parse_version():
    assert parse_version("v0.2.0") == (0, 2, 0)
    assert parse_version("0.10.1") == (0, 10, 1)
    assert parse_version("desktop-v1.0.0") is None
    assert parse_version("v1.0") is None


def test_newer_release_found():
    assert newer_release(payload(), "0.1.0") == Release("0.2.0", PAGE + "v0.2.0")


def test_same_or_older_is_not_offered():
    assert newer_release(payload("v0.1.0"), "0.1.0") is None
    assert newer_release(payload("v0.0.9"), "0.1.0") is None


def test_versions_compare_numerically():
    assert newer_release(payload("v0.10.0"), "0.9.0") is not None


def test_ignores_prerelease_draft_foreign_url_and_garbage():
    assert newer_release(payload(prerelease=True), "0.1.0") is None
    assert newer_release(payload(draft=True), "0.1.0") is None
    assert newer_release(payload(html_url="https://evil.example/x"), "0.1.0") is None
    assert newer_release([], "0.1.0") is None
    assert newer_release({"tag_name": 3}, "0.1.0") is None


def test_check_is_silent_on_network_failure():
    assert updater.check("0.1.0", fetch=lambda: None) is None
    assert updater.check("0.1.0", fetch=lambda: payload()) == Release("0.2.0", PAGE + "v0.2.0")


def test_fetch_latest_returns_none_for_unreachable(tmp_path):
    assert updater.fetch_latest((tmp_path / "missing.json").as_uri()) is None


def test_due():
    assert due(0, 86400)
    assert not due(1000, 1000 + 3600)
    assert due(5000, 100)  # clock moved backwards: check again
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_updater.py -v`
Expected: FAIL with `ImportError: cannot import name 'updater'`.

- [ ] **Step 3: Implement**

`apps/desktop/src/khmerclip/updater.py`:
```python
"""Daily check of the public releases repo. Never installs anything: the UI offers to
open the release page."""

import json
import re
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

from khmerclip import __version__

FEED_URL = "https://api.github.com/repos/KimChengSHEANG/khmerclip-releases/releases/latest"
CHECK_INTERVAL_S = 24 * 3600


@dataclass(frozen=True)
class Release:
    version: str
    page_url: str


def parse_version(tag: str) -> tuple[int, int, int] | None:
    m = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", tag.strip())
    return (int(m[1]), int(m[2]), int(m[3])) if m else None


def newer_release(payload: object, current: str) -> Release | None:
    if not isinstance(payload, dict) or payload.get("draft") or payload.get("prerelease"):
        return None
    tag, url = payload.get("tag_name"), payload.get("html_url")
    if not isinstance(tag, str) or not isinstance(url, str) or not url.startswith("https://github.com/"):
        return None
    latest, cur = parse_version(tag), parse_version(current)
    if latest is None or cur is None or latest <= cur:
        return None
    return Release(".".join(map(str, latest)), url)


def fetch_latest(url: str = FEED_URL, timeout: float = 10) -> dict | None:
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                                   "User-Agent": f"KhmerClip/{__version__}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (OSError, ValueError):  # offline, HTTP error, rate limit, bad JSON
        return None


def check(current: str = __version__, fetch: Callable[[], dict | None] = fetch_latest) -> Release | None:
    return newer_release(fetch(), current)


def due(last_check: float, now: float, interval: float = CHECK_INTERVAL_S) -> bool:
    return now < last_check or now - last_check >= interval
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_updater.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add apps/desktop/src/khmerclip/updater.py apps/desktop/tests/test_updater.py
/usr/bin/git commit -m "feat(desktop): add GitHub releases update check"
```

---

### Task 14: UI building blocks (font, worker, popup, history, preferences, image window)

**Files:**
- Create: `apps/desktop/src/khmerclip/ui/__init__.py` (empty)
- Create: `apps/desktop/src/khmerclip/ui/fonts.py`
- Create: `apps/desktop/src/khmerclip/ui/worker.py`
- Create: `apps/desktop/src/khmerclip/ui/popup.py`
- Create: `apps/desktop/src/khmerclip/ui/history_window.py`
- Create: `apps/desktop/src/khmerclip/ui/prefs_window.py`
- Create: `apps/desktop/src/khmerclip/ui/image_window.py`
- Test: `apps/desktop/tests/test_ui_widgets.py`

**Interfaces:**
- Consumes: `Prefs` (Task 2), `HistoryStore`/`ScanRecord` (Task 2), `hotkey.qt_to_pynput`/`pynput_to_qt` (Task 11), `paths.resource_dir`
- Produces:
  - `fonts.khmer_family() -> str`
  - `fonts.khmer_font(point_size: int = 12) -> QFont` (Noto Sans Khmer first, system fallback)
  - `worker.Task(fn: Callable[[Callable[[object], None], threading.Event], object])`. A `QRunnable` with `.signals.progress(object)`, `.signals.done(object)`, `.signals.failed(object)` and `.cancel: threading.Event`.
  - `worker.submit(fn) -> Task`
  - `popup.ResultPopup()` with signal `copy_requested(str)`, methods `show_text(text: str)` and `show_message(message: str)`, attributes `label: QLabel` and `copy_button: QPushButton`
  - `history_window.HistoryWindow(store: HistoryStore, copy: Callable[[str], None])` with `refresh()`, `copy_selected()`, `delete_selected()`, `clear_all()`, `list: QListWidget`
  - `prefs_window.PrefsWindow(prefs: Prefs, trocr_available: bool, hotkey_supported: bool)` with signal `changed(str)` (field name), attributes `engine: QComboBox`, `hotkey: QKeySequenceEdit`, `checks: dict[str, QCheckBox]`, method `_hotkey_edited()`. Constant `WAYLAND_HINT: str`.
  - `image_window.ImageOCRWindow()` with signals `image_chosen(object)` (PIL RGB image) and `failed(str)`, methods `open_path(path: str)` and `choose_file()`

- [ ] **Step 1: Write the failing tests**

`apps/desktop/tests/test_ui_widgets.py`:
```python
from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QLabel, QMessageBox

from helpers import FIXTURES
from khmerclip.history import HistoryStore, ScanRecord
from khmerclip.prefs import Prefs
from khmerclip.ui.fonts import khmer_font
from khmerclip.ui.history_window import HistoryWindow
from khmerclip.ui.image_window import ImageOCRWindow
from khmerclip.ui.popup import ResultPopup
from khmerclip.ui.prefs_window import WAYLAND_HINT, PrefsWindow
from khmerclip.ui.worker import Task


def test_khmer_font_is_bundled_noto(qapp):
    assert khmer_font().families()[0] == "Noto Sans Khmer"


def test_worker_reports_progress_and_result(qtbot):
    # Connect before starting: a queued cross-thread signal emitted with no receiver is lost.
    seen = []
    task = Task(lambda progress, cancel: (progress("step"), 42)[1])
    task.signals.progress.connect(seen.append)
    with qtbot.waitSignal(task.signals.done, timeout=3000) as blocker:
        QThreadPool.globalInstance().start(task)
    assert blocker.args == [42]
    assert seen == ["step"]


def test_worker_reports_failure(qtbot):
    def boom(progress, cancel):
        raise ValueError("bad")
    task = Task(boom)
    with qtbot.waitSignal(task.signals.failed, timeout=3000) as blocker:
        QThreadPool.globalInstance().start(task)
    assert isinstance(blocker.args[0], ValueError)


def test_popup_shows_plain_text_and_copies_full_text(qtbot):
    popup = ResultPopup()
    qtbot.addWidget(popup)
    popup.show_text("<b>ក</b>")
    assert popup.label.text() == "<b>ក</b>"
    assert popup.label.textFormat() == Qt.TextFormat.PlainText   # OCR text is never rendered as HTML
    long_text = "ក" * 2000
    popup.show_text(long_text)
    with qtbot.waitSignal(popup.copy_requested) as blocker:
        popup.copy_button.click()
    assert blocker.args == [long_text]


def test_popup_message_hides_copy(qtbot):
    popup = ResultPopup()
    qtbot.addWidget(popup)
    popup.show_message("No text found")
    assert popup.copy_button.isHidden()
    assert popup.isVisible()


def test_history_window_copy_delete_clear(qtbot, tmp_path, monkeypatch):
    store = HistoryStore(tmp_path / "h.json")
    store.add(ScanRecord("old"))
    store.add(ScanRecord("ថ្មី"))
    copied = []
    window = HistoryWindow(store, copied.append)
    qtbot.addWidget(window)
    window.refresh()
    assert window.list.count() == 2
    window.list.setCurrentRow(0)
    window.copy_selected()
    assert copied == ["ថ្មី"]
    window.delete_selected()
    assert [r.text for r in store.records] == ["old"]
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    window.clear_all()
    assert len(store.records) == 1
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    window.clear_all()
    assert store.records == [] and window.list.count() == 0


def test_prefs_window_saves_changes(qtbot, tmp_path):
    prefs = Prefs.load(tmp_path / "prefs.json")
    window = PrefsWindow(prefs, trocr_available=True, hotkey_supported=True)
    qtbot.addWidget(window)
    with qtbot.waitSignal(window.changed) as blocker:
        window.checks["auto_copy"].setChecked(False)
    assert blocker.args == ["auto_copy"]
    assert Prefs.load(tmp_path / "prefs.json").auto_copy is False
    window.engine.setCurrentIndex(window.engine.findData("trocr"))
    assert Prefs.load(tmp_path / "prefs.json").engine == "trocr"


def test_prefs_window_disables_unpublished_trocr_and_wayland_hotkey(qtbot, tmp_path):
    window = PrefsWindow(Prefs.load(tmp_path / "p.json"), trocr_available=False, hotkey_supported=False)
    qtbot.addWidget(window)
    assert not window.engine.model().item(window.engine.findData("trocr")).isEnabled()
    assert not window.hotkey.isEnabled()
    assert any(label.text() == WAYLAND_HINT for label in window.findChildren(QLabel))


def test_prefs_window_hotkey_validation(qtbot, tmp_path, monkeypatch):
    prefs = Prefs.load(tmp_path / "p.json")
    window = PrefsWindow(prefs, trocr_available=True, hotkey_supported=True)
    qtbot.addWidget(window)
    window.hotkey.setKeySequence(QKeySequence("Ctrl+Alt+K"))
    window._hotkey_edited()
    assert prefs.hotkey == "<ctrl>+<alt>+k"
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))
    window.hotkey.setKeySequence(QKeySequence("K"))
    window._hotkey_edited()
    assert prefs.hotkey == "<ctrl>+<alt>+k" and warned


def test_image_window_open_path(qtbot, tmp_path):
    window = ImageOCRWindow()
    qtbot.addWidget(window)
    with qtbot.waitSignal(window.image_chosen) as blocker:
        window.open_path(str(FIXTURES / "1.png"))
    assert blocker.args[0].mode == "RGB"
    bad = tmp_path / "notes.txt"
    bad.write_text("hello", encoding="utf-8")
    with qtbot.waitSignal(window.failed) as blocker:
        window.open_path(str(bad))
    assert "notes.txt" in blocker.args[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ui_widgets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'khmerclip.ui'`.

- [ ] **Step 3: Implement**

`apps/desktop/src/khmerclip/ui/__init__.py`: create as an empty file.

`apps/desktop/src/khmerclip/ui/fonts.py`:
```python
"""Stock Windows/Linux installs often lack a Khmer font (text renders as boxes), so every
text widget uses the bundled Noto Sans Khmer first."""

from PySide6.QtGui import QFont, QFontDatabase

from khmerclip.paths import resource_dir

FONT_FILE = "fonts/NotoSansKhmer-Regular.ttf"
_family: str | None = None


def khmer_family() -> str:
    global _family
    if _family is None:
        font_id = QFontDatabase.addApplicationFont(str(resource_dir() / FONT_FILE))
        families = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
        _family = families[0] if families else ""
    return _family


def khmer_font(point_size: int = 12) -> QFont:
    font = QFont()
    if family := khmer_family():
        font.setFamilies([family, *font.families()])  # Latin falls back to the system font
    font.setPointSize(point_size)
    return font
```

`apps/desktop/src/khmerclip/ui/worker.py`:
```python
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
```

`apps/desktop/src/khmerclip/ui/popup.py`:
```python
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from khmerclip.ui.fonts import khmer_font

PREVIEW_CHARS = 1500


class ResultPopup(QWidget):
    """Transient bottom-right popup with the OCR result (the mac HUD's counterpart)."""

    copy_requested = Signal(str)
    TIMEOUT_MS = 5000

    def __init__(self):
        super().__init__(None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self._text = ""
        self.label = QLabel()
        self.label.setTextFormat(Qt.TextFormat.PlainText)  # OCR output must never render as HTML
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.label.setFont(khmer_font(13))
        self.label.setMaximumWidth(420)
        self.copy_button = QPushButton("Copy")
        self.copy_button.clicked.connect(lambda: self.copy_requested.emit(self._text))
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.hide)
        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(self.copy_button)
        buttons.addWidget(close_button)
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addLayout(buttons)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.hide)

    def show_text(self, text: str) -> None:
        self._show(text, copyable=True)

    def show_message(self, message: str) -> None:
        self._show(message, copyable=False)

    def _show(self, text: str, copyable: bool) -> None:
        self._text = text
        self.label.setText(text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS] + "…")
        self.copy_button.setVisible(copyable)
        self.adjustSize()
        area = QGuiApplication.primaryScreen().availableGeometry()
        self.move(area.right() - self.width() - 16, area.bottom() - self.height() - 16)
        self.show()
        self.raise_()
        self.timer.start(self.TIMEOUT_MS)

    def enterEvent(self, event):
        self.timer.stop()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.timer.start(self.TIMEOUT_MS)
        super().leaveEvent(event)
```

`apps/desktop/src/khmerclip/ui/history_window.py`:
```python
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
```

`apps/desktop/src/khmerclip/ui/prefs_window.py`:
```python
from functools import partial

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QKeySequenceEdit, QLabel, QMessageBox, QWidget

from khmerclip.hotkey import pynput_to_qt, qt_to_pynput
from khmerclip.prefs import Prefs

WAYLAND_HINT = ("Global shortcuts aren't available to apps on Wayland. Add a custom shortcut in your "
                "desktop's keyboard settings that runs:  khmerclip --capture")
CHECKS = (
    ("auto_copy", "Copy result to clipboard"),
    ("show_result", "Show result popup"),
    ("preserve_linebreaks", "Preserve line breaks"),
    ("launch_at_login", "Launch at login"),
    ("check_updates", "Check for updates daily"),
)


class PrefsWindow(QWidget):
    changed = Signal(str)  # name of the Prefs field that changed (already saved)

    def __init__(self, prefs: Prefs, trocr_available: bool, hotkey_supported: bool):
        super().__init__()
        self.setWindowTitle("KhmerClip Preferences")
        self.prefs = prefs
        form = QFormLayout(self)

        self.engine = QComboBox()
        self.engine.addItem("Tesseract", "tesseract")
        self.engine.addItem("TrOCR (neural; one-time model download)", "trocr")
        if not trocr_available:
            item = self.engine.model().item(self.engine.findData("trocr"))
            item.setEnabled(False)
            item.setToolTip("The TrOCR model hasn't been published yet.")
        self.engine.setCurrentIndex(max(0, self.engine.findData(prefs.engine)))
        self.engine.currentIndexChanged.connect(lambda _i: self._set("engine", self.engine.currentData()))
        form.addRow("OCR engine", self.engine)

        self.hotkey = QKeySequenceEdit(QKeySequence(pynput_to_qt(prefs.hotkey)))
        self.hotkey.editingFinished.connect(self._hotkey_edited)
        form.addRow("Capture shortcut", self.hotkey)
        if not hotkey_supported:
            self.hotkey.setEnabled(False)
            hint = QLabel(WAYLAND_HINT)
            hint.setWordWrap(True)
            hint.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            form.addRow(hint)

        self.checks: dict[str, QCheckBox] = {}
        for name, label in CHECKS:
            box = QCheckBox(label)
            box.setChecked(getattr(prefs, name))
            box.toggled.connect(partial(self._set, name))
            form.addRow(box)
            self.checks[name] = box

    def _set(self, name: str, value) -> None:
        setattr(self.prefs, name, value)
        self.prefs.save()
        self.changed.emit(name)

    def _hotkey_edited(self) -> None:
        seq = self.hotkey.keySequence().toString(QKeySequence.SequenceFormat.PortableText)
        hotkey = qt_to_pynput(seq)
        if hotkey is None:
            QMessageBox.warning(self, "Capture shortcut",
                                f"“{seq}” can't be used. Combine Ctrl/Alt/Shift/Meta with a letter, digit, "
                                "F-key or Space.")
            self.hotkey.setKeySequence(QKeySequence(pynput_to_qt(self.prefs.hotkey)))
            return
        self._set("hotkey", hotkey)
```

`apps/desktop/src/khmerclip/ui/image_window.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ui_widgets.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add apps/desktop/src/khmerclip/ui apps/desktop/tests/test_ui_widgets.py
/usr/bin/git commit -m "feat(desktop): add Khmer font, worker, result popup, history, preferences and image windows"
```

---

### Task 15: Controller, tray app and `run_gui`

**Files:**
- Create: `apps/desktop/src/khmerclip/ui/app.py`
- Test: `apps/desktop/tests/test_app.py`

**Interfaces:**
- Consumes: everything above. Specifically: `Engine`/`recognize` (Task 8), `trocr.preload` (Task 8), `trocr_models.load_manifest/is_installed/install` (Task 7), `RegionSelector` (Task 10), `portal.screenshot` (Task 9), `capture_backend` (Task 9), `HotkeyListener`/`pynput_to_qt` (Task 11), `CommandServer`/`send_command` (Task 11), `autostart.set_enabled` (Task 12), `updater.fetch_latest/newer_release/due` (Task 13), `ResultPopup`/`HistoryWindow`/`PrefsWindow`/`ImageOCRWindow`/`Task` (Task 14), `formatter.apply`, `Prefs`, `HistoryStore`, `ScanRecord`
- Produces:
  - `Controller(prefs: Prefs, history: HistoryStore, set_clipboard: Callable[[str], None] | None = None)` with signal `capture_requested()` (safe to emit from any thread). Methods: `start_capture()`, `run_ocr(image)`, `handle_text(text: str, kind: str = "text")`, `handle_error(error: BaseException)`, `copy(text: str)`, `submit(fn, on_done=None, dialog=None) -> Task`. Attribute `popup: ResultPopup`.
  - `TrayApp(controller: Controller)` with `on_command(cmd: str)`, `show_history()`, `show_prefs()`, `show_image_window()`, `check_updates(manual: bool)`, attribute `menu: QMenu`
  - `preferred_qpa(platform: str, env: Mapping[str, str]) -> str | None`. Returns `"xcb"` on Linux when `DISPLAY` is set and `QT_QPA_PLATFORM` isn't. Under native Wayland a background app can't reliably set the clipboard or place its popup; XWayland can. Capture still uses the portal, because `capture_backend` reads `XDG_SESSION_TYPE`.
  - `run_gui(capture: bool = False) -> int` (called by `khmerclip.__main__.main`)

- [ ] **Step 1: Write the failing tests**

`apps/desktop/tests/test_app.py`:
```python
import threading

import pytest
from PIL import Image

from khmerclip.errors import Cancelled, KhmerClipError
from khmerclip.history import HistoryStore
from khmerclip.ocr import trocr_models
from khmerclip.prefs import Prefs
from khmerclip.ui import app as app_module
from khmerclip.ui.app import Controller, TrayApp, preferred_qpa


@pytest.fixture
def controller(qtbot, tmp_path):
    copied = []
    c = Controller(Prefs.load(tmp_path / "prefs.json"), HistoryStore(tmp_path / "h.json"), set_clipboard=copied.append)
    c.copied = copied
    qtbot.addWidget(c.popup)
    return c


def test_handle_text_copies_records_and_shows(controller):
    controller.handle_text("ក  \nខ")
    assert controller.copied == ["ក  \nខ"]
    assert controller.history.records[0].text == "ក  \nខ"
    assert controller.popup.isVisible() and controller.popup.label.text() == "ក  \nខ"


def test_handle_text_respects_prefs(controller):
    controller.prefs.preserve_linebreaks = False
    controller.prefs.auto_copy = False
    controller.prefs.show_result = False
    controller.handle_text("ក\nខ")
    assert controller.copied == []
    assert controller.history.records[0].text == "ក ខ"
    assert not controller.popup.isVisible()


def test_empty_result_is_not_recorded(controller):
    controller.handle_text("  \n ")
    assert controller.history.records == []
    assert controller.popup.label.text() == "No text found"


def test_errors(controller):
    controller.handle_error(Cancelled())
    assert not controller.popup.isVisible()
    controller.handle_error(KhmerClipError("Tesseract was not found."))
    assert "Tesseract was not found." in controller.popup.label.text()


def test_run_ocr_end_to_end_off_thread(controller, qtbot, monkeypatch):
    monkeypatch.setattr(app_module, "recognize", lambda image, engine, progress=None, cancel=None: "សួស្តី")
    controller.run_ocr(Image.new("RGB", (20, 20), "white"))
    qtbot.waitUntil(lambda: controller.history.records, timeout=3000)
    assert controller.history.records[0].text == "សួស្តី"
    assert controller.copied == ["សួស្តី"]


def test_submit_delivers_results_on_gui_thread(controller, qtbot):
    # Callbacks touch widgets, so they must run on the GUI thread, never the pool thread.
    seen = []
    controller.submit(lambda progress, cancel: (progress("p"), 1)[1],
                      on_done=lambda r: seen.append(threading.current_thread() is threading.main_thread()))
    qtbot.waitUntil(lambda: seen, timeout=3000)
    assert seen == [True]


def test_preferred_qpa():
    assert preferred_qpa("linux", {"XDG_SESSION_TYPE": "wayland", "DISPLAY": ":0"}) == "xcb"
    assert preferred_qpa("linux", {"DISPLAY": ":0", "QT_QPA_PLATFORM": "wayland"}) is None  # user choice wins
    assert preferred_qpa("linux", {"WAYLAND_DISPLAY": "wayland-0"}) is None  # no XWayland available
    assert preferred_qpa("win32", {"DISPLAY": ":0"}) is None


def test_trocr_unpublished_explains_instead_of_crashing(controller, monkeypatch):
    monkeypatch.delenv("KHMERCLIP_TROCR_DIR", raising=False)
    monkeypatch.setattr(trocr_models, "load_manifest", lambda path=None: trocr_models.Manifest("khmer-v1", ()))
    controller.prefs.engine = "trocr"
    controller.run_ocr(Image.new("RGB", (20, 20), "white"))
    assert "isn't published" in controller.popup.label.text()
    assert controller.history.records == []


def test_tray_menu_and_commands(controller, monkeypatch):
    monkeypatch.setattr(app_module.HotkeyListener, "start", lambda self, *a, **k: False)
    controller.prefs.check_updates = False
    controller.prefs.welcomed = True
    captures = []
    monkeypatch.setattr(controller, "start_capture", lambda: captures.append(1))
    tray = TrayApp(controller)
    texts = [a.text() for a in tray.menu.actions() if a.text()]
    assert texts == ["Capture Text", "OCR Image File…", "History…", "Preferences…",
                     "Check for Updates…", "Quit KhmerClip"]
    tray.on_command("capture")
    assert captures == [1]
    tray.on_command("show")
    assert tray.prefs_window is not None and tray.prefs_window.isVisible()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_app.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'khmerclip.ui.app'`.

- [ ] **Step 3: Implement**

`apps/desktop/src/khmerclip/ui/app.py`:
```python
"""Tray app wiring: capture -> OCR (worker thread) -> clipboard/history/popup."""

import os
import sys
import time
from collections.abc import Callable, Mapping
from functools import partial

from PySide6.QtCore import QObject, QThreadPool, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QAction, QDesktopServices, QGuiApplication, QIcon
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox, QProgressDialog, QSystemTrayIcon

from khmerclip import __version__, autostart, updater
from khmerclip.capture import portal
from khmerclip.capture.geometry import capture_backend
from khmerclip.capture.overlay import RegionSelector
from khmerclip.errors import Cancelled, KhmerClipError
from khmerclip.formatter import apply
from khmerclip.history import HistoryStore, ScanRecord
from khmerclip.hotkey import HotkeyListener, pynput_to_qt
from khmerclip.ocr import Engine, recognize, trocr_models
from khmerclip.paths import APP_NAME, resource_dir
from khmerclip.prefs import Prefs
from khmerclip.single_instance import CommandServer, send_command
from khmerclip.ui.history_window import HistoryWindow
from khmerclip.ui.image_window import ImageOCRWindow
from khmerclip.ui.popup import ResultPopup
from khmerclip.ui.prefs_window import PrefsWindow
from khmerclip.ui.worker import Task


def _update_progress(dialog: QProgressDialog, value: object) -> None:
    if isinstance(value, tuple):  # (bytes_done, bytes_total) from a download
        done, total = value
        dialog.setValue(int(done * 1000 / max(1, total)))
    else:
        dialog.setLabelText(str(value))


def _present(window) -> None:
    window.show()
    window.raise_()
    window.activateWindow()


def preferred_qpa(platform: str, env: Mapping[str, str]) -> str | None:
    """Run Qt under XWayland when available: a native-Wayland background app can't reliably
    set the clipboard (no focused surface) or position its popup. Capture still goes through
    the portal because capture_backend() reads XDG_SESSION_TYPE, not the Qt platform."""
    if platform.startswith("linux") and not env.get("QT_QPA_PLATFORM") and env.get("DISPLAY"):
        return "xcb"
    return None


class _Relay(QObject):
    """Receives one Task's signals on the GUI thread. Bound @Slot methods of a QObject that
    lives on the GUI thread are guaranteed queued delivery; bare closures are not."""

    def __init__(self, on_progress, on_done, on_failed):
        super().__init__()
        self._on_progress, self._on_done, self._on_failed = on_progress, on_done, on_failed

    @Slot(object)
    def progress(self, value) -> None:
        self._on_progress(value)

    @Slot(object)
    def done(self, result) -> None:
        self._on_done(result)

    @Slot(object)
    def failed(self, error) -> None:
        self._on_failed(error)


class Controller(QObject):
    capture_requested = Signal()  # emitted from the hotkey thread / IPC; delivered on the GUI thread

    def __init__(self, prefs: Prefs, history: HistoryStore, set_clipboard: Callable[[str], None] | None = None):
        super().__init__()
        self.prefs, self.history = prefs, history
        self._set_clipboard = set_clipboard or (lambda text: QGuiApplication.clipboard().setText(text))
        self.popup = ResultPopup()
        self.popup.copy_requested.connect(self.copy)
        self.selector = RegionSelector()
        self.selector.selected.connect(self.run_ocr)
        self._tasks: dict[Task, _Relay] = {}
        self.capture_requested.connect(self.start_capture)

    # --- capture -------------------------------------------------------------
    def start_capture(self) -> None:
        if self.prefs.engine == Engine.TROCR:
            from khmerclip.ocr import trocr

            self.submit(lambda progress, cancel: trocr.preload())  # warm up while the user selects
        if capture_backend() == "portal":
            self.submit(lambda progress, cancel: portal.screenshot(), on_done=self._portal_done)
        else:
            self.selector.start()

    def _portal_done(self, image) -> None:
        if image is not None:  # None = user cancelled the portal dialog
            self.run_ocr(image)

    # --- OCR -----------------------------------------------------------------
    def run_ocr(self, image) -> None:
        engine = Engine(self.prefs.engine)
        if engine is Engine.TROCR and not self._trocr_ready(image):
            return
        dialog = self._progress_dialog("Reading text…")
        self.submit(lambda progress, cancel: recognize(image, engine, progress, cancel),
                    on_done=self.handle_text, dialog=dialog)

    def _trocr_ready(self, image) -> bool:
        """True if TrOCR can run now. Otherwise explains or offers the one-time download
        (which re-runs this OCR when finished) and returns False."""
        manifest = trocr_models.load_manifest()
        if os.environ.get("KHMERCLIP_TROCR_DIR") or trocr_models.is_installed(manifest):
            return True
        if not manifest.available:
            self.popup.show_message("The TrOCR model isn't published yet. Switch to Tesseract in Preferences.")
            return False
        size_mb = max(1, manifest.total_size // 1_000_000)
        answer = QMessageBox.question(None, APP_NAME, f"TrOCR needs a one-time {size_mb} MB download. Download now?")
        if answer != QMessageBox.StandardButton.Yes:
            return False
        dialog = self._progress_dialog("Downloading TrOCR model…", maximum=1000)
        self.submit(lambda progress, cancel: trocr_models.install(
                        manifest, progress=lambda done, total: progress((done, total)), cancel=cancel),
                    on_done=lambda _dir: self.run_ocr(image), dialog=dialog)
        return False

    def handle_text(self, text: str, kind: str = "text") -> None:
        if not text.strip():
            self.popup.show_message("No text found")
            return
        formatted = apply(text, self.prefs.preserve_linebreaks)
        if self.prefs.auto_copy:
            self.copy(formatted)
        self.history.add(ScanRecord(formatted, kind))
        if self.prefs.show_result:
            self.popup.show_text(formatted)

    def handle_error(self, error: BaseException) -> None:
        if isinstance(error, Cancelled):
            return
        message = str(error) if isinstance(error, KhmerClipError) else f"Unexpected error: {error!r}"
        self.popup.show_message(f"Capture failed: {message}")

    def copy(self, text: str) -> None:
        self._set_clipboard(text)

    # --- background work -----------------------------------------------------
    def _progress_dialog(self, label: str, maximum: int = 0) -> QProgressDialog:
        dialog = QProgressDialog(label, "Cancel", 0, maximum)
        dialog.setWindowTitle(APP_NAME)
        dialog.setMinimumDuration(400)  # quick Tesseract runs never flash a dialog
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setValue(0)
        return dialog

    def submit(self, fn, on_done: Callable[[object], None] | None = None,
               dialog: QProgressDialog | None = None) -> Task:
        task = Task(fn)

        def finish() -> None:
            self._tasks.pop(task, None)
            if dialog is not None:
                dialog.close()
                dialog.deleteLater()

        def progress(value) -> None:
            if dialog is not None:
                _update_progress(dialog, value)

        def done(result) -> None:
            finish()
            if on_done is not None:
                on_done(result)

        def failed(error) -> None:
            finish()
            self.handle_error(error)

        relay = _Relay(progress, done, failed)
        self._tasks[task] = relay  # keep runnable, signals and relay alive until it reports back
        task.signals.progress.connect(relay.progress)
        task.signals.done.connect(relay.done)
        task.signals.failed.connect(relay.failed)
        if dialog is not None:
            dialog.canceled.connect(task.cancel.set)
        QThreadPool.globalInstance().start(task)  # start only after every connection exists
        return task


class TrayApp(QObject):
    def __init__(self, controller: Controller):
        super().__init__()
        self.c = controller
        prefs = controller.prefs
        self.hotkeys = HotkeyListener()
        self.history_window: HistoryWindow | None = None
        self.prefs_window: PrefsWindow | None = None
        self.image_window: ImageOCRWindow | None = None

        self.menu = QMenu()
        for text, slot in (("Capture Text", controller.start_capture),
                           ("OCR Image File…", self.show_image_window),
                           ("History…", self.show_history),
                           ("Preferences…", self.show_prefs),
                           ("Check for Updates…", partial(self.check_updates, True))):
            action = QAction(text, self.menu)
            action.triggered.connect(lambda _checked=False, s=slot: s())
            self.menu.addAction(action)
        self.menu.addSeparator()
        quit_action = QAction("Quit KhmerClip", self.menu)
        quit_action.triggered.connect(QApplication.quit)
        self.menu.addAction(quit_action)

        self.tray = QSystemTrayIcon(QIcon(str(resource_dir() / "icon.png")))
        self.tray.setToolTip(APP_NAME)
        self.tray.setContextMenu(self.menu)
        self.tray.activated.connect(self._on_activated)
        self.tray.show()

        self.hotkey_ok = self.register_hotkey()
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self.show_prefs()  # e.g. GNOME without the AppIndicator extension: keep the app reachable
        if not prefs.welcomed:
            how = (f"Press {pynput_to_qt(prefs.hotkey)}" if self.hotkey_ok
                   else "Bind “khmerclip --capture” to a keyboard shortcut")
            self.tray.showMessage(f"{APP_NAME} is running", f"{how} or use the tray icon to capture text.")
            prefs.welcomed = True
            prefs.save()
        if prefs.check_updates and updater.due(prefs.last_update_check, time.time()):
            self.check_updates(False)

    def register_hotkey(self) -> bool:
        return self.hotkeys.start(self.c.prefs.hotkey, self.c.capture_requested.emit)

    def on_command(self, command: str) -> None:
        if command == "capture":
            self.c.start_capture()
        elif command == "show":
            self.show_prefs()

    def _on_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.c.start_capture()

    def show_history(self) -> None:
        if self.history_window is None:
            self.history_window = HistoryWindow(self.c.history, self.c.copy)
        _present(self.history_window)

    def show_prefs(self) -> None:
        if self.prefs_window is None:
            self.prefs_window = PrefsWindow(self.c.prefs, trocr_available=trocr_models.load_manifest().available,
                                            hotkey_supported=capture_backend() == "qt")
            self.prefs_window.changed.connect(self._pref_changed)
        _present(self.prefs_window)

    def show_image_window(self) -> None:
        if self.image_window is None:
            self.image_window = ImageOCRWindow()
            self.image_window.image_chosen.connect(self.c.run_ocr)
            self.image_window.failed.connect(self.c.popup.show_message)
        _present(self.image_window)

    def _pref_changed(self, name: str) -> None:
        if name == "hotkey" and not self.register_hotkey():
            self.c.popup.show_message("That shortcut couldn't be registered. Try another one.")
        elif name == "launch_at_login":
            try:
                autostart.set_enabled(self.c.prefs.launch_at_login)
            except (OSError, NotImplementedError) as e:
                self.c.popup.show_message(f"Couldn't change launch at login: {e}")

    def check_updates(self, manual: bool) -> None:
        def fetch(progress, cancel):
            payload = updater.fetch_latest()
            return payload is not None, updater.newer_release(payload, __version__)

        def done(result) -> None:
            reachable, release = result
            if reachable:
                self.c.prefs.last_update_check = time.time()
                self.c.prefs.save()
            if release is None:
                if manual:
                    self.c.popup.show_message(f"{APP_NAME} {__version__} is up to date." if reachable
                                              else "Couldn't reach the update server. Try again later.")
                return
            answer = QMessageBox.question(None, f"{APP_NAME} update",
                                          f"{APP_NAME} {release.version} is available (you have {__version__}). "
                                          "Open the download page?")
            if answer == QMessageBox.StandardButton.Yes:
                QDesktopServices.openUrl(QUrl(release.page_url))

        self.c.submit(fetch, on_done=done)


def run_gui(capture: bool = False) -> int:
    if qpa := preferred_qpa(sys.platform, os.environ):
        os.environ["QT_QPA_PLATFORM"] = qpa  # must precede QApplication
    qapp = QApplication.instance() or QApplication(sys.argv[:1])
    qapp.setApplicationName(APP_NAME)
    qapp.setQuitOnLastWindowClosed(False)  # tray app: closing a window must not quit
    qapp.setWindowIcon(QIcon(str(resource_dir() / "icon.png")))

    if send_command("capture" if capture else "show"):
        return 0  # another instance is running and took the command
    server = CommandServer(parent=qapp)
    server.listen()

    controller = Controller(Prefs.load(), HistoryStore())
    tray = TrayApp(controller)
    server.command.connect(tray.on_command)
    qapp.aboutToQuit.connect(tray.hotkeys.stop)
    if capture:
        QTimer.singleShot(0, controller.start_capture)
    return qapp.exec()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_app.py -v`
Expected: 9 passed.

- [ ] **Step 5: Manual smoke run (dev machine, then Linux Wayland)**

Run: `.venv/bin/python -m khmerclip`
Expected: the tray icon appears and the welcome notification shows. On macOS dev, grant Screen Recording when asked. "Capture Text" shows the dimmed frozen screen. Dragging over Khmer text yields a popup with the text, which is also on the clipboard. History… lists it. Preferences… toggles persist across restarts. Run `.venv/bin/python -m khmerclip --capture` in a second terminal while the app runs. Expected: the running instance starts a capture, and the second process exits immediately.

On an Ubuntu **Wayland** session (the user's Ubuntu machine), run the same app. Then run `khmerclip --capture` from a second terminal.
Expected:
- The GNOME screenshot picker opens.
- After selecting Khmer text, the popup appears bottom-right.
- Pasting into another app (e.g. gedit) yields the text.

If pasting yields nothing, the XWayland override (`preferred_qpa`) isn't in effect. Check `echo $DISPLAY` and the `QT_QPA_PLATFORM` env, and report rather than shipping.

- [ ] **Step 6: Run the full suite and commit**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass (skips allowed).

```bash
/usr/bin/git add apps/desktop/src/khmerclip/ui/app.py apps/desktop/tests/test_app.py
/usr/bin/git commit -m "feat(desktop): wire tray app, capture-to-clipboard flow and model download prompt"
```

---

### Task 16: Packaging (PyInstaller, Inno Setup, AppImage)

**Files:**
- Create: `apps/desktop/packaging/entry.py`
- Create: `apps/desktop/packaging/khmerclip.spec`
- Create: `apps/desktop/packaging/windows/vendor_tesseract.ps1`
- Create: `apps/desktop/packaging/windows/khmerclip.iss`
- Create: `apps/desktop/packaging/linux/AppRun`
- Create: `apps/desktop/packaging/linux/khmerclip.desktop`
- Create: `apps/desktop/packaging/linux/build_appimage.sh`

**Interfaces:**
- Consumes: `khmerclip.__main__.main`, `resources/` (font, icon, `tesseract/tessdata`, Windows `tesseract.exe` + DLLs), `src/khmerclip/ocr/trocr_manifest.json`
- Produces (used by Task 17 CI):
  - `pyinstaller packaging/khmerclip.spec --noconfirm` → `dist/KhmerClip/KhmerClip(.exe)` (onedir)
  - Windows: `ISCC.exe packaging\windows\khmerclip.iss` with env `KHMERCLIP_VERSION` → `dist/KhmerClip-<ver>-windows-x64-setup.exe`
  - Linux: `packaging/linux/build_appimage.sh <ver>` → `dist/KhmerClip-<ver>-x86_64.AppImage`
  - Smoke contract: `<frozen app> ocr <png> -o <out.txt>` exits 0 and writes UTF-8 text

- [ ] **Step 1: Write the PyInstaller entry and spec**

`apps/desktop/packaging/entry.py`:
```python
from khmerclip.__main__ import main

raise SystemExit(main())
```

`apps/desktop/packaging/khmerclip.spec`:
```python
# PyInstaller onedir build. Run from apps/desktop:
#   pyinstaller packaging/khmerclip.spec --noconfirm
# Expects resources/tesseract/tessdata (tools/fetch_tessdata.py) and, on Windows,
# resources/tesseract/tesseract.exe + DLLs (packaging/windows/vendor_tesseract.ps1).
from pathlib import Path

ROOT = Path(SPECPATH).parent  # SPECPATH = apps/desktop/packaging

a = Analysis(
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT / "src")],
    datas=[
        (str(ROOT / "resources"), "resources"),
        (str(ROOT / "src" / "khmerclip" / "ocr" / "trocr_manifest.json"), "khmerclip/ocr"),
    ],
    hiddenimports=["pynput.keyboard._win32", "pynput.mouse._win32", "pynput.keyboard._xorg", "pynput.mouse._xorg"],
    excludes=["tkinter"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="KhmerClip", console=False,
          icon=str(ROOT / "resources" / "icon.png"))
coll = COLLECT(exe, a.binaries, a.datas, name="KhmerClip")
```

- [ ] **Step 2: Freeze locally and smoke-test the frozen CLI**

Run from `apps/desktop/` (the OS you're on; macOS works as a sanity check):
```bash
.venv/bin/pyinstaller packaging/khmerclip.spec --noconfirm
ls dist/KhmerClip/_internal/resources/fonts dist/KhmerClip/_internal/resources/tesseract/tessdata dist/KhmerClip/_internal/khmerclip/ocr/trocr_manifest.json
dist/KhmerClip/KhmerClip ocr ../mac/Tests/Fixtures/ocr/1.png -o /tmp/khmerclip-smoke.txt
cat /tmp/khmerclip-smoke.txt
```
Expected: the `ls` shows the font, `khm.traineddata`/`eng.traineddata` and the manifest. On macOS or Linux with Tesseract on `PATH` (or `KHMERCLIP_TESSERACT` exported), the smoke file contains Khmer text matching `1.txt`. PyInstaller warnings about the other OS's pynput backends are expected.

- [ ] **Step 3: Windows packaging files**

`apps/desktop/packaging/windows/vendor_tesseract.ps1`:
```powershell
# Copy the Chocolatey (UB Mannheim) Tesseract build -- exe + DLLs -- next to the fetched
# tessdata so PyInstaller bundles it under resources/tesseract. Run from apps/desktop.
$ErrorActionPreference = "Stop"
$src = "C:\Program Files\Tesseract-OCR"
$dst = "resources\tesseract"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
Copy-Item "$src\tesseract.exe" $dst
Copy-Item "$src\*.dll" $dst
& "$dst\tesseract.exe" --version
```

`apps/desktop/packaging/windows/khmerclip.iss`:
```ini
; Per-user installer (no admin). Build from apps/desktop after PyInstaller:
;   $env:KHMERCLIP_VERSION = "0.1.0"; & "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" packaging\windows\khmerclip.iss
#define AppVersion GetEnv("KHMERCLIP_VERSION")
#if AppVersion == ""
  #error KHMERCLIP_VERSION is not set
#endif

[Setup]
AppId={{6E0B3C52-6C4E-4C1B-9C7B-6B2E7A4F1D10}
AppName=KhmerClip
AppVersion={#AppVersion}
AppPublisher=KhmerClip
DefaultDirName={localappdata}\Programs\KhmerClip
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\..\dist
OutputBaseFilename=KhmerClip-{#AppVersion}-windows-x64-setup
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
UninstallDisplayIcon={app}\KhmerClip.exe

[Files]
Source: "..\..\dist\KhmerClip\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{userprograms}\KhmerClip"; Filename: "{app}\KhmerClip.exe"

[Registry]
; The app writes this value when "Launch at login" is on; remove it on uninstall.
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueName: "KhmerClip"; ValueType: none; Flags: uninsdeletevalue dontcreatekey

[Run]
Filename: "{app}\KhmerClip.exe"; Description: "Launch KhmerClip"; Flags: nowait postinstall skipifsilent
```

- [ ] **Step 4: Linux packaging files**

`apps/desktop/packaging/linux/AppRun`:
```sh
#!/bin/sh
# Bundled tesseract (deployed by linuxdeploy into usr/bin) wins over any system copy.
HERE="$(dirname "$(readlink -f "$0")")"
export PATH="$HERE/usr/bin:$PATH"
exec "$HERE/usr/lib/khmerclip/KhmerClip" "$@"
```

`apps/desktop/packaging/linux/khmerclip.desktop`:
```ini
[Desktop Entry]
Type=Application
Name=KhmerClip
Comment=Khmer OCR screen capture
Exec=KhmerClip
Icon=khmerclip
Categories=Utility;
Terminal=false
```

`apps/desktop/packaging/linux/build_appimage.sh`:
```bash
#!/usr/bin/env bash
# Package dist/KhmerClip (PyInstaller onedir) + the system Tesseract into an AppImage.
# Usage, from apps/desktop after pyinstaller:  packaging/linux/build_appimage.sh <version>
set -euo pipefail
VERSION="$1"
APPDIR="build/AppDir"
TOOLS="build/tools"
OUT="dist/KhmerClip-${VERSION}-x86_64.AppImage"
export APPIMAGE_EXTRACT_AND_RUN=1  # CI runners have no FUSE

mkdir -p "$TOOLS"
fetch() { [ -x "$TOOLS/$1" ] || { curl -fsSL -o "$TOOLS/$1" "$2"; chmod +x "$TOOLS/$1"; }; }
fetch linuxdeploy https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage
fetch appimagetool https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage

rm -rf "$APPDIR"
cp resources/icon.png build/khmerclip.png
# 1) Tesseract and its shared libs, rpath-patched by linuxdeploy. Only tesseract goes through
#    linuxdeploy: patching PyInstaller's binaries would corrupt its appended archive.
"$TOOLS/linuxdeploy" --appdir "$APPDIR" --executable "$(command -v tesseract)" \
  --desktop-file packaging/linux/khmerclip.desktop --icon-file build/khmerclip.png
# 2) The frozen app, copied untouched.
mkdir -p "$APPDIR/usr/lib"
cp -a dist/KhmerClip "$APPDIR/usr/lib/khmerclip"
rm -f "$APPDIR/AppRun"
install -m 755 packaging/linux/AppRun "$APPDIR/AppRun"
ARCH=x86_64 "$TOOLS/appimagetool" "$APPDIR" "$OUT"
echo "built $OUT"
```

Run: `chmod +x apps/desktop/packaging/linux/build_appimage.sh apps/desktop/packaging/linux/AppRun` (from the worktree root).

These two scripts run only on their target OS. Task 17's CI jobs execute and smoke-test them. Locally, check syntax: `bash -n apps/desktop/packaging/linux/build_appimage.sh && sh -n apps/desktop/packaging/linux/AppRun`. Expected: no output.

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add apps/desktop/packaging
/usr/bin/git commit -m "build(desktop): add PyInstaller, Inno Setup and AppImage packaging"
```

---

### Task 17: GitHub Actions (test, build, release) and docs

**Files:**
- Create: `.github/workflows/desktop.yml`
- Create: `apps/desktop/README.md`
- Modify: `README.md` (root: layout table row + quick start line)

**Interfaces:**
- Consumes: Task 16 build contracts, `tools/fetch_tessdata.py`, `pytest`, env `KHMERCLIP_REQUIRE_TESSERACT`, `KHMERCLIP_TESSERACT`
- Produces: workflow `desktop`. Jobs: `test` (matrix), `build` (matrix, uploads `khmerclip-Windows` / `khmerclip-Linux` artifacts), `release` (tag `desktop-v*` → GitHub release `v<ver>` on `KimChengSHEANG/khmerclip-releases` using secret `RELEASES_TOKEN`)

- [ ] **Step 1: Write the workflow**

`.github/workflows/desktop.yml`:
```yaml
name: desktop

on:
  push:
    branches: [main]
    tags: ["desktop-v*"]
    paths: ["apps/desktop/**", "apps/mac/Tests/Fixtures/**", ".github/workflows/desktop.yml"]
  pull_request:
    paths: ["apps/desktop/**", "apps/mac/Tests/Fixtures/**", ".github/workflows/desktop.yml"]
  workflow_dispatch:

defaults:
  run:
    working-directory: apps/desktop
    shell: bash

jobs:
  test:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-22.04, windows-latest]
    runs-on: ${{ matrix.os }}
    env:
      QT_QPA_PLATFORM: offscreen
      KHMERCLIP_REQUIRE_TESSERACT: "1"
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install Tesseract 5 + Qt runtime libs (Linux)
        if: runner.os == 'Linux'
        run: |
          sudo add-apt-repository -y ppa:alex-p/tesseract-ocr5
          sudo apt-get update
          sudo apt-get install -y tesseract-ocr libegl1 libgl1 libxkbcommon0 libxkbcommon-x11-0 libxcb-cursor0 libdbus-1-3 libfontconfig1
      - name: Install Tesseract 5 (Windows)
        if: runner.os == 'Windows'
        run: |
          choco install tesseract --no-progress -y
          echo "KHMERCLIP_TESSERACT=C:\Program Files\Tesseract-OCR\tesseract.exe" >> "$GITHUB_ENV"
      - run: python -m pip install -e ".[dev]"
      - run: python tools/fetch_tessdata.py
      - run: python -m pytest -v

  build:
    needs: test
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-22.04, windows-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install Tesseract 5 + Qt xcb runtime libs + Xvfb (Linux)
        if: runner.os == 'Linux'
        # PyInstaller can only bundle the Qt xcb plugin's deps (e.g. libxcb-cursor0) if they exist here.
        run: |
          sudo add-apt-repository -y ppa:alex-p/tesseract-ocr5
          sudo apt-get update
          sudo apt-get install -y tesseract-ocr libegl1 libgl1 libxkbcommon0 libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 libxcb-shape0 libdbus-1-3 libfontconfig1 xvfb
      - name: Install Tesseract 5 (Windows)
        if: runner.os == 'Windows'
        run: choco install tesseract --no-progress -y
      - run: python -m pip install -e ".[dev]"
      - run: python tools/fetch_tessdata.py
      - name: Vendor Tesseract (Windows)
        if: runner.os == 'Windows'
        shell: pwsh
        run: ./packaging/windows/vendor_tesseract.ps1
      - name: Version
        id: ver
        run: echo "version=$(python -c 'import khmerclip; print(khmerclip.__version__)')" >> "$GITHUB_OUTPUT"
      - run: pyinstaller packaging/khmerclip.spec --noconfirm
      - name: Installer + smoke test (Windows)
        if: runner.os == 'Windows'
        shell: pwsh
        env:
          KHMERCLIP_VERSION: ${{ steps.ver.outputs.version }}
        run: |
          # Smoke: the frozen app must OCR with its *bundled* tesseract (KHMERCLIP_TESSERACT unset here).
          Start-Process -Wait -FilePath dist\KhmerClip\KhmerClip.exe -ArgumentList 'ocr','..\mac\Tests\Fixtures\ocr\1.png','-o','smoke.txt'
          $text = Get-Content smoke.txt -Encoding utf8 -Raw
          if (-not $text.Trim()) { throw "frozen app produced no OCR text" }
          Write-Host $text
          # GUI smoke: the frozen tray app must load Qt and stay up (the ocr path never imports Qt).
          $p = Start-Process -PassThru -FilePath dist\KhmerClip\KhmerClip.exe
          Start-Sleep -Seconds 15
          if ($p.HasExited) { throw "frozen GUI exited early with code $($p.ExitCode)" }
          Stop-Process -Id $p.Id -Force
          $iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
          if (-not (Test-Path $iscc)) { choco install innosetup --no-progress -y }
          & $iscc packaging\windows\khmerclip.iss
      - name: AppImage + smoke test (Linux)
        if: runner.os == 'Linux'
        run: |
          packaging/linux/build_appimage.sh "${{ steps.ver.outputs.version }}"
          # Prove the AppImage uses its bundled tesseract, not the runner's.
          sudo apt-get remove -y tesseract-ocr
          APPIMAGE_EXTRACT_AND_RUN=1 "dist/KhmerClip-${{ steps.ver.outputs.version }}-x86_64.AppImage" \
            ocr ../mac/Tests/Fixtures/ocr/1.png -o smoke.txt
          test -s smoke.txt && cat smoke.txt
          # GUI smoke: the tray app must load the bundled Qt xcb plugin and stay up.
          # timeout's exit 124 = still running after 15 s = pass; anything else = crash.
          set +e
          APPIMAGE_EXTRACT_AND_RUN=1 xvfb-run -a timeout 15 "dist/KhmerClip-${{ steps.ver.outputs.version }}-x86_64.AppImage"
          code=$?
          set -e
          test "$code" -eq 124 || { echo "frozen GUI exited early with code $code"; exit 1; }
      - uses: actions/upload-artifact@v4
        with:
          name: khmerclip-${{ runner.os }}
          path: |
            apps/desktop/dist/KhmerClip-*-setup.exe
            apps/desktop/dist/KhmerClip-*.AppImage
          if-no-files-found: error

  release:
    if: startsWith(github.ref, 'refs/tags/desktop-v')
    needs: build
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: .
    steps:
      - uses: actions/checkout@v4
      - name: Tag must match __version__
        run: |
          ver=$(python3 -c "import re, pathlib; print(re.search(r'__version__ = \"(.+?)\"', pathlib.Path('apps/desktop/src/khmerclip/__init__.py').read_text()).group(1))")
          test "desktop-v$ver" = "$GITHUB_REF_NAME" || { echo "tag $GITHUB_REF_NAME != desktop-v$ver"; exit 1; }
          echo "VERSION=$ver" >> "$GITHUB_ENV"
      - uses: actions/download-artifact@v4
        with:
          path: artifacts
          merge-multiple: true
      - name: Publish to public releases repo
        env:
          GH_TOKEN: ${{ secrets.RELEASES_TOKEN }}
        run: |
          gh release create "v$VERSION" artifacts/* \
            --repo KimChengSHEANG/khmerclip-releases \
            --title "KhmerClip $VERSION" \
            --notes "KhmerClip $VERSION for Windows (installer) and Linux (AppImage)."
```

- [ ] **Step 2: Write `apps/desktop/README.md`**

````markdown
# KhmerClip for Windows and Linux

Tray app: select a screen region → Khmer (+English) OCR → text on the clipboard. Python +
PySide6. The macOS app lives in [`../mac`](../mac) and is separate.

## Features

- Region capture: Windows and X11 use a frozen-screenshot overlay. Wayland uses the desktop's own screenshot picker (xdg-desktop-portal).
- OCR engines (Preferences):
  - **Tesseract** (`khm+eng`, bundled): the default; works offline with no setup.
  - **TrOCR** (Khmer fine-tune, ONNX): about a 1.3 GB one-time download from Hugging Face on first use.
- Auto-copy, result popup, history (last 200), OCR Image File…, launch at login, daily update check.
- Global hotkey (default `Ctrl+Shift+2`) on Windows and X11. **Wayland:** add a custom shortcut in your desktop's keyboard settings that runs `khmerclip --capture` (for the AppImage: `/path/to/KhmerClip-x.y.z-x86_64.AppImage --capture`).

## Install

Download from [KhmerClip releases](https://github.com/KimChengSHEANG/khmerclip-releases/releases):

- **Windows:** `KhmerClip-<ver>-windows-x64-setup.exe`. Per-user install, no admin. Builds are unsigned, so SmartScreen warns: More info → Run anyway.
- **Linux:** `KhmerClip-<ver>-x86_64.AppImage`. Run `chmod +x` on it, then start it. GNOME needs the AppIndicator extension for the tray icon (on by default on Ubuntu).

## Develop

```bash
cd apps/desktop
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python tools/fetch_tessdata.py        # pinned khm/eng tessdata
.venv/bin/python -m khmerclip                   # tray app
.venv/bin/python -m khmerclip ocr image.png     # headless OCR
.venv/bin/python -m pytest
```

You need a `tesseract` 5.x on `PATH`, or `KHMERCLIP_TESSERACT=/path/to/tesseract`. Options: `apt install tesseract-ocr`, `brew install tesseract`, or the UB Mannheim build on Windows.

## TrOCR models

1. Export: `.venv/bin/pip install -e '.[tools]'`, then `.venv/bin/python tools/export_onnx.py <checkpoint> exported/khmer`. This checks ONNX against torch.
2. Publish (needs `huggingface-cli login`): `.venv/bin/python tools/publish_models.py exported/khmer --version khmer-v2`. This pins the HF commit and the SHA-256 values in `src/khmerclip/ocr/trocr_manifest.json`. Commit that file.
3. To test a local export without publishing, set `KHMERCLIP_TROCR_DIR=exported/khmer`.

## Release

One-time setup:

1. Create the public repo `KimChengSHEANG/khmerclip-releases`.
2. Create a fine-grained PAT with *Contents: read and write* on that repo.
3. Store the PAT as the `RELEASES_TOKEN` secret in this repo.

Each release:

1. Bump `__version__` in `src/khmerclip/__init__.py` and commit.
2. Run `git tag desktop-v<version>` and push the tag (`git push origin desktop-v<version>`).
3. CI tests, builds both installers, smoke-tests the frozen apps, and publishes release `v<version>` to the releases repo. Installed apps see it within a day.

Manual checklist before announcing a release:

- [ ] Windows 11 at 100% and 150% scaling, two monitors: capture on each monitor crops exactly the dragged region.
- [ ] Windows: the default `Ctrl+Shift+2` hotkey fires, and a custom hotkey set in Preferences fires after the change.
- [ ] Ubuntu X11: hotkey, capture, popup and clipboard.
- [ ] Ubuntu Wayland: `--capture` shortcut opens the portal picker, and text lands on the clipboard.
- [ ] TrOCR first-use download: progress shown, Cancel works, a retry resumes, and OCR runs after the download.
- [ ] Launch at login toggles and survives a reboot. Uninstall (Windows) removes the Run entry.
- [ ] "Check for Updates…" reports up to date, or offers the newer release.
- [ ] Khmer renders correctly (no boxes) in the popup and in History on a fresh install.
````

- [ ] **Step 3: Update the root README**

In `README.md`, add a row to the Layout table directly under the `apps/mac` row:
```markdown
| [`apps/desktop`](apps/desktop) | KhmerClip for Windows + Linux (Python / PySide6) |
```
Under "## Quick start (app)", append this line after the existing mac sentence:
```markdown
Windows/Linux: see [`apps/desktop/README.md`](apps/desktop/README.md) (`pip install -e '.[dev]'`, then `python -m khmerclip`).
```

- [ ] **Step 4: Validate the workflow locally, then on GitHub**

Run (from the worktree root): `python3 -c "import yaml, sys; yaml.safe_load(open('.github/workflows/desktop.yml')); print('ok')"`
Expected: `ok`. If PyYAML is missing, run `.venv/bin/pip install pyyaml` first, from `apps/desktop`.

**Human-gated (pushing is outward-facing):** after the user approves pushing the branch, run `gh workflow run desktop --ref <branch>`, or open a PR. Watch with `gh run watch`. Expected: `test` passes on both OSes, which includes the 6 Tesseract fixture tests and the Windows-only path and registry tests. `build` uploads `khmerclip-Windows` (setup exe) and `khmerclip-Linux` (AppImage), and both smoke steps print Khmer text. If a fixture's CER exceeds 0.05 on a runner's Tesseract build, report the diff. Do not loosen the threshold silently.

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add .github/workflows/desktop.yml apps/desktop/README.md README.md
/usr/bin/git commit -m "ci(desktop): test, build and publish Windows/Linux installers; add docs"
```

---

## Out of scope (v1)

QR/barcode, EasyOCR, text-to-speech, translation, window-picker capture, capture sound, code signing, int8 model quantization, macOS build of `apps/desktop`. Each needs a spec decision first, so don't add them while executing this plan.
