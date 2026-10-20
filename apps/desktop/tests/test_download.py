import hashlib
import threading

import pytest

from khmerclipper.download import RemoteFile, download, is_present, sha256_file
from khmerclipper.errors import Cancelled, KhmerClipperError


def _source(tmp_path, name, payload: bytes, sha=None) -> RemoteFile:
    src = tmp_path / "src" / name
    src.parent.mkdir(exist_ok=True)
    src.write_bytes(payload)
    return RemoteFile(name, src.as_uri(), sha or hashlib.sha256(payload).hexdigest(), len(payload))


def test_download_verifies_and_installs(tmp_path):
    files = [_source(tmp_path, "a.bin", b"a" * 3000), _source(tmp_path, "b.bin", b"b" * 10)]
    seen = []
    download(files, tmp_path / "out", progress=lambda d, t: seen.append((d, t)), chunk_size=1024)
    assert (tmp_path / "out" / "a.bin").read_bytes() == b"a" * 3000
    assert is_present(files, tmp_path / "out")
    assert seen[-1] == (3010, 3010)


def test_download_checksum_mismatch_leaves_nothing(tmp_path):
    bad = _source(tmp_path, "m.bin", b"data", sha="0" * 64)
    with pytest.raises(KhmerClipperError, match="Checksum mismatch"):
        download([bad], tmp_path / "out")
    assert list((tmp_path / "out").iterdir()) == []


def test_download_cancel_raises_and_cleans_part(tmp_path):
    f = _source(tmp_path, "c.bin", b"x" * 5000)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        download([f], tmp_path / "out", cancel=cancel, chunk_size=1024)
    assert list((tmp_path / "out").iterdir()) == []


def test_download_unreachable_url_is_user_facing_error(tmp_path):
    missing = RemoteFile("x.bin", (tmp_path / "nope.bin").as_uri(), "0" * 64, 1)
    with pytest.raises(KhmerClipperError, match="x.bin"):
        download([missing], tmp_path / "out")
    assert list((tmp_path / "out").iterdir()) == []


def test_download_skips_files_already_present(tmp_path):
    f = _source(tmp_path, "s.bin", b"keep")
    out = tmp_path / "out"
    download([f], out)
    # Same name/size but an unreachable URL: a re-download would raise.
    again = RemoteFile("s.bin", (tmp_path / "gone").as_uri(), f.sha256, f.size)
    download([again], out)
    assert (out / "s.bin").read_bytes() == b"keep"


def test_sha256_file(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"abc")
    assert sha256_file(p) == hashlib.sha256(b"abc").hexdigest()
