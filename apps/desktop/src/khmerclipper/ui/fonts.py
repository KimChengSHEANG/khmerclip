"""Stock Windows/Linux installs often lack a Khmer font (text renders as boxes), so every
text widget uses the bundled Noto Sans Khmer first."""

from PySide6.QtGui import QFont, QFontDatabase

from khmerclipper.paths import resource_dir

FONT_FILE = "fonts/NotoSansKhmer-Regular.ttf"
_family: str | None = None


def khmer_family() -> str:
    global _family
    if _family is None:
        font_id = QFontDatabase.addApplicationFont(str(resource_dir() / FONT_FILE))
        families = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
        _family = families[0] if families else ""
    return _family


def khmer_font(point_size: int = 12) -> QFont:
    font = QFont()
    if family := khmer_family():
        font.setFamilies([family, *font.families()])  # Latin falls back to the system font
    font.setPointSize(point_size)
    return font
