class KhmerClipError(Exception):
    """A failure whose str() is shown to the user as-is."""


class EngineUnavailable(KhmerClipError):
    """The chosen OCR engine can't run (binary/model missing)."""


class Cancelled(Exception):
    """The user cancelled a running OCR or download."""
