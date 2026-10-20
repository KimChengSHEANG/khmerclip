import sys

import pytest

from khmerclipper import autostart


def test_launch_command_prefers_appimage_path():
    env = {"APPIMAGE": "/home/u/Apps/KhmerClipper.AppImage"}
    assert autostart.launch_command(env, "/usr/bin/python3", frozen=True) == ["/home/u/Apps/KhmerClipper.AppImage"]


def test_launch_command_frozen_and_dev():
    assert autostart.launch_command({}, r"C:\Apps\KhmerClipper.exe", frozen=True) == [r"C:\Apps\KhmerClipper.exe"]
    assert autostart.launch_command({}, "/venv/bin/python", frozen=False) == ["/venv/bin/python", "-m", "khmerclipper"]


def test_desktop_entry_quotes_spaces_and_non_ascii():
    entry = autostart.desktop_entry(["/home/ស/My Apps/KhmerClipper.AppImage"])
    assert 'Exec="/home/ស/My Apps/KhmerClipper.AppImage"\n' in entry
    assert entry.startswith("[Desktop Entry]\n")


def test_desktop_entry_escapes_reserved_characters():
    entry = autostart.desktop_entry(['/tmp/a$b"c%d'])
    assert 'Exec="/tmp/a\\\\$b\\\\"c%%d"\n' in entry


def test_linux_enable_disable(tmp_path):
    cmd = ["/opt/KhmerClipper.AppImage"]
    autostart.set_enabled(True, platform="linux", config_home=tmp_path, command=cmd)
    path = tmp_path / "autostart" / "khmerclipper.desktop"
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
    name = "KhmerClipperTest"
    cmd = [r"C:\Program Files\KhmerClipper\KhmerClipper.exe"]
    try:
        autostart.set_enabled(True, platform="win32", value_name=name, command=cmd)
        assert autostart.is_enabled(platform="win32", value_name=name)
    finally:
        autostart.set_enabled(False, platform="win32", value_name=name, command=cmd)
    assert not autostart.is_enabled(platform="win32", value_name=name)
