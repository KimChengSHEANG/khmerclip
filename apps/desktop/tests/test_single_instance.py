import threading
import uuid

from khmerclip.single_instance import CommandServer, send_command, server_name


def test_command_roundtrip(qtbot):
    name = f"khmerclip-test-{uuid.uuid4().hex[:8]}"
    server = CommandServer(name)
    assert server.listen()
    # Send from another thread, like a real second launch: on Windows a named-pipe write
    # only completes once the server reads it, which needs this thread's event loop.
    sent = []
    with qtbot.waitSignal(server.command, timeout=3000) as blocker:
        sender = threading.Thread(target=lambda: sent.append(send_command("capture", name)))
        sender.start()
    sender.join(3)
    assert blocker.args == ["capture"]
    assert sent == [True]
    server.close()


def test_send_without_server_returns_false(qapp):
    assert send_command("capture", f"khmerclip-none-{uuid.uuid4().hex[:8]}", timeout_ms=200) is False


def test_server_name_is_ascii_and_stable():
    assert server_name() == server_name()
    assert server_name().isascii()
