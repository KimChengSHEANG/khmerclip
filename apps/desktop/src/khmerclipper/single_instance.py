"""One KhmerClipper per user. A second launch (`khmerclipper --capture` bound to a desktop
shortcut on Wayland) forwards its command to the running instance over a local socket."""

import getpass
import hashlib
import sys
from functools import partial

from PySide6.QtCore import QObject, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket


def server_name() -> str:
    try:
        user = getpass.getuser()
    except Exception:  # no USER/USERNAME in a stripped environment
        user = "user"
    return "khmerclipper-" + hashlib.sha1(user.encode("utf-8")).hexdigest()[:12]


def send_command(command: str, name: str | None = None, timeout_ms: int = 1000) -> bool:
    name = name or server_name()
    if sys.platform == "win32":
        return _send_windows_pipe(command, name)
    sock = QLocalSocket()
    sock.connectToServer(name)
    if not sock.waitForConnected(timeout_ms):
        return False
    sock.write(command.encode("utf-8") + b"\n")
    ok = sock.waitForBytesWritten(timeout_ms)
    sock.disconnectFromServer()
    return ok


def _send_windows_pipe(command: str, name: str) -> bool:
    """QLocalSocket's writes to a Windows named pipe stay pending when the sending thread has
    no Qt event loop (CI: waitForBytesWritten False, 8 bytes pending), so a second launch lost
    its command. A plain blocking write returns once the running instance has read it."""
    try:
        with open(rf"\\.\pipe\{name}", "wb", buffering=0) as pipe:
            pipe.write(command.encode("utf-8") + b"\n")
        return True
    except OSError:  # no instance listening (FileNotFoundError) or pipe busy
        return False


class CommandServer(QObject):
    command = Signal(str)

    def __init__(self, name: str | None = None, parent: QObject | None = None):
        super().__init__(parent)
        self._name = name or server_name()
        self._server = QLocalServer(self)
        self._server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
        self._server.newConnection.connect(self._accept)

    def listen(self) -> bool:
        """Call only after send_command() failed: removes a stale socket left by a crash."""
        QLocalServer.removeServer(self._name)
        return self._server.listen(self._name)

    def close(self) -> None:
        self._server.close()
        # Drop live client sockets now: if Qt destroyed them later (with this object), they'd
        # emit disconnected/readyRead into Python slots mid-destruction and crash.
        for sock in self._server.findChildren(QLocalSocket):
            sock.readyRead.disconnect()
            sock.disconnected.disconnect()
            sock.abort()
            sock.deleteLater()

    def _accept(self) -> None:
        while (sock := self._server.nextPendingConnection()) is not None:
            sock.readyRead.connect(partial(self._read, sock))
            sock.disconnected.connect(sock.deleteLater)
            self._read(sock)  # data may already be buffered

    def _read(self, sock: QLocalSocket) -> None:
        while sock.canReadLine():
            line = bytes(sock.readLine()).decode("utf-8", "replace").strip()
            if line:
                self.command.emit(line)
