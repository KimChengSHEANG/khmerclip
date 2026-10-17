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
