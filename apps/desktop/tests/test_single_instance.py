import uuid

from khmerclip.single_instance import CommandServer, send_command, server_name


def test_command_roundtrip(qtbot):
    name = f"khmerclip-test-{uuid.uuid4().hex[:8]}"
    server = CommandServer(name)
    assert server.listen()
    with qtbot.waitSignal(server.command, timeout=3000) as blocker:
        assert send_command("capture", name)
    assert blocker.args == ["capture"]
    server.close()


def test_send_without_server_returns_false(qapp):
    assert send_command("capture", f"khmerclip-none-{uuid.uuid4().hex[:8]}", timeout_ms=200) is False


def test_server_name_is_ascii_and_stable():
    assert server_name() == server_name()
    assert server_name().isascii()
