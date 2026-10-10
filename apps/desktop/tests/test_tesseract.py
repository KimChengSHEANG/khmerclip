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
