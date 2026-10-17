"""Daily check of the public releases repo. Never installs anything: the UI offers to
open the release page."""

import json
import re
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

from khmerclip import __version__

FEED_URL = "https://api.github.com/repos/KimChengSHEANG/khmerclip-releases/releases/latest"
CHECK_INTERVAL_S = 24 * 3600


@dataclass(frozen=True)
class Release:
    version: str
    page_url: str


def parse_version(tag: str) -> tuple[int, int, int] | None:
    m = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", tag.strip())
    return (int(m[1]), int(m[2]), int(m[3])) if m else None


def newer_release(payload: object, current: str) -> Release | None:
    if not isinstance(payload, dict) or payload.get("draft") or payload.get("prerelease"):
        return None
    tag, url = payload.get("tag_name"), payload.get("html_url")
    if not isinstance(tag, str) or not isinstance(url, str) or not url.startswith("https://github.com/"):
        return None
    latest, cur = parse_version(tag), parse_version(current)
    if latest is None or cur is None or latest <= cur:
        return None
    return Release(".".join(map(str, latest)), url)


def fetch_latest(url: str = FEED_URL, timeout: float = 10) -> dict | None:
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                                   "User-Agent": f"KhmerClip/{__version__}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (OSError, ValueError):  # offline, HTTP error, rate limit, bad JSON
        return None


def check(current: str = __version__, fetch: Callable[[], dict | None] = fetch_latest) -> Release | None:
    return newer_release(fetch(), current)


def due(last_check: float, now: float, interval: float = CHECK_INTERVAL_S) -> bool:
    return now < last_check or now - last_check >= interval
