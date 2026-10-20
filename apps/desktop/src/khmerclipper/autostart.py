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
