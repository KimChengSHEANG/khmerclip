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
