from khmerclip.capture.geometry import capture_backend, normalize_drag, to_physical


def test_normalize_drag_any_direction():
    assert normalize_drag(50, 40, 10, 100) == (10, 40, 40, 60)


def test_to_physical_identity_at_100_percent():
    assert to_physical((10, 20, 100, 50), 1.0, 800, 600) == (10, 20, 100, 50)


def test_to_physical_150_percent():
    assert to_physical((10, 10, 100, 50), 1.5, 1920, 1080) == (15, 15, 150, 75)


def test_to_physical_125_percent_rounds_outward():
    # 3*1.25=3.75 -> floor 3 ; (3+10)*1.25=16.25 -> ceil 17
    assert to_physical((3, 3, 10, 10), 1.25, 1000, 1000) == (3, 3, 14, 14)


def test_to_physical_clamps_to_image():
    assert to_physical((790, 590, 50, 50), 1.0, 800, 600) == (790, 590, 10, 10)


def test_to_physical_rejects_tiny_or_outside_selection():
    assert to_physical((10, 10, 3, 100), 2.0, 800, 600) is None
    assert to_physical((900, 10, 50, 50), 1.0, 800, 600) is None


def test_capture_backend_wayland_session_uses_portal():
    assert capture_backend("linux", {"XDG_SESSION_TYPE": "wayland"}) == "portal"
    assert capture_backend("linux", {"WAYLAND_DISPLAY": "wayland-0"}) == "portal"


def test_capture_backend_x11_and_windows_use_qt():
    assert capture_backend("linux", {"XDG_SESSION_TYPE": "x11", "WAYLAND_DISPLAY": "wayland-0"}) == "qt"
    assert capture_backend("linux", {"DISPLAY": ":0"}) == "qt"
    assert capture_backend("win32", {"WAYLAND_DISPLAY": "x"}) == "qt"
