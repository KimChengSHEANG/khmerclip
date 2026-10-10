import subprocess
from pathlib import Path

from khmerclip_eval.runners import tesseract


def _capture(monkeypatch):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="ក\n", stderr="")

    monkeypatch.setattr(tesseract.subprocess, "run", run)
    return calls


def test_recognize_defaults_to_khm_plus_eng(monkeypatch):
    calls = _capture(monkeypatch)
    assert tesseract.recognize(Path("x.png"), Path("/v")) == "ក"
    cmd = calls[0]
    assert cmd[cmd.index("-l") + 1] == "khm+eng"


def test_recognize_passes_lang(monkeypatch):
    calls = _capture(monkeypatch)
    tesseract.recognize(Path("x.png"), Path("/v"), lang="khm")
    cmd = calls[0]
    assert cmd[cmd.index("-l") + 1] == "khm"
