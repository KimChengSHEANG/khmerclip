from khmerclip import updater
from khmerclip.updater import Release, due, newer_release, parse_version

PAGE = "https://github.com/KimChengSHEANG/khmerclip-releases/releases/tag/"


def payload(tag="v0.2.0", **extra):
    return {"tag_name": tag, "html_url": PAGE + tag, "draft": False, "prerelease": False, **extra}


def test_parse_version():
    assert parse_version("v0.2.0") == (0, 2, 0)
    assert parse_version("0.10.1") == (0, 10, 1)
    assert parse_version("desktop-v1.0.0") is None
    assert parse_version("v1.0") is None


def test_newer_release_found():
    assert newer_release(payload(), "0.1.0") == Release("0.2.0", PAGE + "v0.2.0")


def test_same_or_older_is_not_offered():
    assert newer_release(payload("v0.1.0"), "0.1.0") is None
    assert newer_release(payload("v0.0.9"), "0.1.0") is None


def test_versions_compare_numerically():
    assert newer_release(payload("v0.10.0"), "0.9.0") is not None


def test_ignores_prerelease_draft_foreign_url_and_garbage():
    assert newer_release(payload(prerelease=True), "0.1.0") is None
    assert newer_release(payload(draft=True), "0.1.0") is None
    assert newer_release(payload(html_url="https://evil.example/x"), "0.1.0") is None
    assert newer_release([], "0.1.0") is None
    assert newer_release({"tag_name": 3}, "0.1.0") is None


def test_check_is_silent_on_network_failure():
    assert updater.check("0.1.0", fetch=lambda: None) is None
    assert updater.check("0.1.0", fetch=lambda: payload()) == Release("0.2.0", PAGE + "v0.2.0")


def test_fetch_latest_returns_none_for_unreachable(tmp_path):
    assert updater.fetch_latest((tmp_path / "missing.json").as_uri()) is None


def test_due():
    assert due(0, 86400)
    assert not due(1000, 1000 + 3600)
    assert due(5000, 100)  # clock moved backwards: check again
