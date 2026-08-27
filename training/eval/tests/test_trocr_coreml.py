from pathlib import Path
from subprocess import CompletedProcess

import pytest

from khmerclip_eval.runners import trocr_coreml


def test_recognize_invokes_ocrbench_and_strips_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        assert command == [
            "/tmp/OCRBench",
            "--engine",
            "trocr",
            "/tmp/sample.png",
        ]
        assert kwargs == {"capture_output": True, "text": True, "check": False}
        return CompletedProcess(command, 0, "  សួស្តី \n", "ms=12.3\n")

    monkeypatch.setattr(trocr_coreml.subprocess, "run", fake_run)

    assert (
        trocr_coreml.recognize(Path("/tmp/sample.png"), Path("/tmp/OCRBench"))
        == "សួស្តី"
    )


def test_recognize_raises_with_ocrbench_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        trocr_coreml.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(args[0], 2, "", "model missing\n"),
    )

    with pytest.raises(RuntimeError, match="model missing"):
        trocr_coreml.recognize(Path("/tmp/sample.png"), Path("/tmp/OCRBench"))
