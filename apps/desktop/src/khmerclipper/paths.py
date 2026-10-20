"""Where KhmerClipper reads bundled resources and keeps per-user state."""

import os
import sys
from pathlib import Path

import platformdirs

APP_NAME = "KhmerClipper"


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
