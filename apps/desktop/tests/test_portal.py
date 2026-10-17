import sys
from pathlib import Path

import pytest

from khmerclip.capture import portal


def test_request_path_follows_portal_convention():
    assert portal.request_path(":1.42", "khmerclipab12") == "/org/freedesktop/portal/desktop/request/1_42/khmerclipab12"


def test_uri_from_response_success_and_cancel():
    assert portal.uri_from_response((0, {"uri": ("s", "file:///tmp/a%20b.png")})) == "file:///tmp/a%20b.png"
    assert portal.uri_from_response((1, {})) is None          # user cancelled
    assert portal.uri_from_response((0, {})) is None          # no uri: treat as cancelled


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file URIs")
def test_uri_to_path_unquotes():
    assert portal.uri_to_path("file:///tmp/a%20b%E1%9E%80.png") == Path("/tmp/a bក.png")
