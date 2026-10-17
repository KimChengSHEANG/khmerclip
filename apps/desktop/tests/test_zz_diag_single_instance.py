# TEMPORARY Windows diagnostic for test_command_roundtrip; delete after.
import uuid

from PySide6.QtNetwork import QLocalServer, QLocalSocket

from khmerclip.single_instance import CommandServer


def test_diag(qtbot, capsys):
    for opt in (True, False):
        name = f"khmerclip-diag-{uuid.uuid4().hex[:8]}"
        server = CommandServer(name)
        if not opt:
            server._server.setSocketOptions(QLocalServer.SocketOption.NoOptions)
        ok = server.listen()
        print(f"\n[opt={opt}] listen={ok} full={server._server.fullServerName()} err={server._server.errorString()}")
        sock = QLocalSocket()
        sock.connectToServer(name)
        c = sock.waitForConnected(1000)
        print(f"[opt={opt}] connected={c} state={sock.state()} err={sock.error()} {sock.errorString()}")
        if c:
            n = sock.write(b"capture\n")
            print(f"[opt={opt}] wrote={n} pending={sock.bytesToWrite()}")
            w = sock.waitForBytesWritten(1000)
            print(f"[opt={opt}] waitWritten={w} pending={sock.bytesToWrite()} err={sock.errorString()}")
        sock.abort()
        server.close()
    assert False, "diag dump"
