# PyInstaller onedir build. Run from apps/desktop:
#   pyinstaller packaging/khmerclipper.spec --noconfirm
# Expects resources/tesseract/tessdata (tools/fetch_tessdata.py) and, on Windows,
# resources/tesseract/tesseract.exe + DLLs (packaging/windows/vendor_tesseract.ps1).
from pathlib import Path

ROOT = Path(SPECPATH).parent  # SPECPATH = apps/desktop/packaging

a = Analysis(
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT / "src")],
    datas=[
        (str(ROOT / "resources"), "resources"),
        (str(ROOT / "src" / "khmerclipper" / "ocr" / "trocr_manifest.json"), "khmerclipper/ocr"),
    ],
    hiddenimports=["pynput.keyboard._win32", "pynput.mouse._win32", "pynput.keyboard._xorg", "pynput.mouse._xorg"],
    # Dev-only export deps (tools extra) must never ship, even if installed in the build venv.
    excludes=["tkinter", "torch", "transformers", "onnx", "huggingface_hub"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="KhmerClipper", console=False,
          icon=str(ROOT / "resources" / "icon.png"))
coll = COLLECT(exe, a.binaries, a.datas, name="KhmerClipper")
