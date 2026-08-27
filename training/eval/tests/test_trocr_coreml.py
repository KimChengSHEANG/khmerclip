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


def test_recognize_many_uses_one_process_and_parses_timings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        assert command == [
            "/tmp/OCRBench",
            "--engine",
            "trocr",
            "/tmp/one.png",
            "/tmp/two.png",
        ]
        assert kwargs == {"capture_output": True, "text": True, "check": False}
        return CompletedProcess(
            command,
            0,
            " ទីមួយ \nទីពីរ\n",
            "CoreML ready\nms=20.5\nms=4.25\n",
        )

    monkeypatch.setattr(trocr_coreml.subprocess, "run", fake_run)

    assert trocr_coreml.recognize_many(
        [Path("/tmp/one.png"), Path("/tmp/two.png")], Path("/tmp/OCRBench")
    ) == [("ទីមួយ", 20.5), ("ទីពីរ", 4.25)]


def test_recognize_many_allows_missing_timings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        trocr_coreml.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(args[0], 0, "one\ntwo\n", "ms=3.0\n"),
    )

    assert trocr_coreml.recognize_many(
        [Path("/tmp/one.png"), Path("/tmp/two.png")], Path("/tmp/OCRBench")
    ) == [("one", 3.0), ("two", None)]
