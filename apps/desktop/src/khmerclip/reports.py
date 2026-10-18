"""Anonymous error/capture/feedback reports: build payloads, queue them as JSON files in an
outbox, and flush them to the report relay. Failures never raise into the app; unsent
reports stay queued and are retried on the next flush."""

import base64
import io
import json
import platform
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from khmerclip import __version__, paths

REPORT_URL = "https://khmerclip-report-relay.REPLACE_ME.workers.dev/report"
APP_KEY = "kc-reports-1"
MAX_PNG_BYTES = 2_000_000
_DONE_STATUSES = {200, 400, 401, 413}  # sent, or rejected for good: never retry


def outbox_dir() -> Path:
    return paths.data_dir() / "outbox"


def base_fields(engine: str) -> dict:
    return {"id": uuid.uuid4().hex, "app_version": __version__, "os": platform.platform(), "engine": engine}


def scrub_home(text: str) -> str:
    return text.replace(str(Path.home()), "~")


def _png_within_limit(image_path: Path) -> bytes:
    with Image.open(image_path) as opened:
        img = opened.copy()
    while True:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = buf.getvalue()
        if len(data) <= MAX_PNG_BYTES or min(img.size) <= 1:
            return data
        img = img.resize((max(1, img.width // 2), max(1, img.height // 2)), Image.LANCZOS)


def capture_payload(image_path: Path, meta: dict, ocr_text: str, correction: str, engine: str) -> dict:
    png = _png_within_limit(image_path)
    return {**base_fields(engine), "type": "capture", "image_png_b64": base64.b64encode(png).decode("ascii"),
            "ocr_text": ocr_text, "correction": correction, "meta": meta}


def feedback_payload(message: str, email: str, engine: str) -> dict:
    return {**base_fields(engine), "type": "feedback", "message": message, "email": email}


def crash_payload(signature: str, details: str, engine: str) -> dict:
    return {**base_fields(engine), "type": "crash", "signature": scrub_home(signature),
            "details": scrub_home(details)}


def enqueue(payload: dict, outbox: Path | None = None) -> Path:
    path = (outbox or outbox_dir()) / f"{payload['id']}.json"
    paths.atomic_write_text(path, json.dumps(payload, ensure_ascii=False))
    return path


def _load(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def drop_type(kind: str, outbox: Path | None = None) -> None:
    for path in (outbox or outbox_dir()).glob("*.json"):
        data = _load(path)
        if data is not None and data.get("type") == kind:
            path.unlink(missing_ok=True)


def post_json(payload: dict, url: str = REPORT_URL, timeout: float = 20) -> int:
    request = urllib.request.Request(
        url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", "X-KhmerClip-Key": APP_KEY,
                 "User-Agent": f"KhmerClip/{__version__}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as err:
        return err.code
    except OSError:
        return 0


def flush(outbox: Path | None = None, post: Callable[[dict], int] | None = None, url: str = REPORT_URL) -> int:
    """Send every queued report; return how many the relay accepted."""
    if "REPLACE_ME" in url:
        return 0
    post = post or (lambda d: post_json(d, url))
    sent = 0
    for path in sorted((outbox or outbox_dir()).glob("*.json")):
        data = _load(path)
        if data is None:  # corrupt: can never be sent
            path.unlink(missing_ok=True)
            continue
        status = post(data)
        if status == 200:
            sent += 1
        if status in _DONE_STATUSES:
            path.unlink(missing_ok=True)
    return sent
