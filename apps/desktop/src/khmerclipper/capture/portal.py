"""Wayland screen capture through org.freedesktop.portal.Screenshot (interactive mode shows
the desktop's own region picker). jeepney is imported lazily: it exists only on Linux."""

import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from PIL import Image

from khmerclip.errors import KhmerClipError


def request_path(unique_name: str, token: str) -> str:
    sender = unique_name.lstrip(":").replace(".", "_")
    return f"/org/freedesktop/portal/desktop/request/{sender}/{token}"


def uri_from_response(body: tuple) -> str | None:
    code, results = body
    if code != 0:
        return None
    uri = results.get("uri")
    return uri[1] if uri else None  # jeepney returns variants as (signature, value)


def uri_to_path(uri: str) -> Path:
    return Path(urllib.request.url2pathname(urllib.parse.urlparse(uri).path))


def screenshot() -> Image.Image | None:
    """Blocking; run off the GUI thread. None if the user cancelled the portal dialog."""
    try:
        from jeepney import DBusAddress, MatchRule, message_bus, new_method_call
        from jeepney.io.blocking import Proxy, open_dbus_connection

        token = "khmerclip" + uuid.uuid4().hex[:8]
        with open_dbus_connection(bus="SESSION") as conn:
            rule = MatchRule(type="signal", interface="org.freedesktop.portal.Request", member="Response",
                             path=request_path(conn.unique_name, token))
            Proxy(message_bus, conn).AddMatch(rule)
            screenshot_api = DBusAddress("/org/freedesktop/portal/desktop", bus_name="org.freedesktop.portal.Desktop",
                                         interface="org.freedesktop.portal.Screenshot")
            with conn.filter(rule) as queue:
                conn.send_and_get_reply(new_method_call(
                    screenshot_api, "Screenshot", "sa{sv}",
                    ("", {"handle_token": ("s", token), "interactive": ("b", True)})))
                response = conn.recv_until_filtered(queue)
        uri = uri_from_response(response.body)
        if uri is None:
            return None
        # ponytail: the portal may also save this file under ~/Pictures; we leave it (documented limitation).
        with Image.open(uri_to_path(uri)) as img:
            return img.convert("RGB")
    except Exception as e:  # no session bus, no portal, D-Bus error reply, unreadable file
        raise KhmerClipError(f"Screen capture through the desktop portal failed: {e}") from e
