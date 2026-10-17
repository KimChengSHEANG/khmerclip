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
