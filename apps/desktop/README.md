# KhmerClip for Windows and Linux

Tray app: select a screen region → Khmer (+English) OCR → text on the clipboard. Python +
PySide6. The macOS app lives in [`../mac`](../mac) and is separate.

## Features

- Region capture: Windows and X11 use a frozen-screenshot overlay. Wayland uses the desktop's own screenshot picker (xdg-desktop-portal).
- OCR engines (Preferences):
  - **Tesseract** (`khm+eng`, bundled): the default; works offline with no setup.
  - **TrOCR** (Khmer fine-tune, ONNX): about a 1.3 GB one-time download from Hugging Face on first use.
- Auto-copy, result popup, history (last 200), OCR Image File…, launch at login, daily update check.
- Global hotkey (default `Ctrl+Shift+2`) on Windows and X11. **Wayland:** add a custom shortcut in your desktop's keyboard settings that runs `khmerclip --capture` (for the AppImage: `/path/to/KhmerClip-x.y.z-x86_64.AppImage --capture`).

## Install

Download from [KhmerClip releases](https://github.com/KimChengSHEANG/khmerclip-releases/releases):

- **Windows:** `KhmerClip-<ver>-windows-x64-setup.exe`. Per-user install, no admin. Builds are unsigned, so SmartScreen warns: More info → Run anyway.
- **Linux:** `KhmerClip-<ver>-x86_64.AppImage`. Run `chmod +x` on it, then start it. GNOME needs the AppIndicator extension for the tray icon (on by default on Ubuntu).

## Develop

```bash
cd apps/desktop
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python tools/fetch_tessdata.py        # pinned khm/eng tessdata
.venv/bin/python -m khmerclip                   # tray app
.venv/bin/python -m khmerclip ocr image.png     # headless OCR
.venv/bin/python -m pytest
```

You need a `tesseract` 5.x on `PATH`, or `KHMERCLIP_TESSERACT=/path/to/tesseract`. Options: `apt install tesseract-ocr`, `brew install tesseract`, or the UB Mannheim build on Windows.

## TrOCR models

1. Export: `.venv/bin/pip install -e '.[tools]'`, then `.venv/bin/python tools/export_onnx.py <checkpoint> exported/khmer`. This checks ONNX against torch.
2. Publish (needs `huggingface-cli login`): `.venv/bin/python tools/publish_models.py exported/khmer --version khmer-v2`. This pins the HF commit and the SHA-256 values in `src/khmerclip/ocr/trocr_manifest.json`. Commit that file.
3. To test a local export without publishing, set `KHMERCLIP_TROCR_DIR=exported/khmer`.

## Release

One-time setup:

1. Create the public repo `KimChengSHEANG/khmerclip-releases`.
2. Create a fine-grained PAT with *Contents: read and write* on that repo.
3. Store the PAT as the `RELEASES_TOKEN` secret in this repo.

Each release:

1. Bump `__version__` in `src/khmerclip/__init__.py` and commit.
2. Run `git tag desktop-v<version>` and push the tag (`git push origin desktop-v<version>`).
3. CI tests, builds both installers, smoke-tests the frozen apps, and publishes release `v<version>` to the releases repo. Installed apps see it within a day.

Manual checklist before announcing a release:

- [ ] Windows 11 at 100% and 150% scaling, two monitors: capture on each monitor crops exactly the dragged region.
- [ ] Windows: the default `Ctrl+Shift+2` hotkey fires, and a custom hotkey set in Preferences fires after the change.
- [ ] Ubuntu X11: hotkey, capture, popup and clipboard.
- [ ] Ubuntu Wayland: `--capture` shortcut opens the portal picker, and text lands on the clipboard.
- [ ] TrOCR first-use download: progress shown, Cancel works, a retry resumes, and OCR runs after the download.
- [ ] Launch at login toggles and survives a reboot. Uninstall (Windows) removes the Run entry.
- [ ] "Check for Updates…" reports up to date, or offers the newer release.
- [ ] Khmer renders correctly (no boxes) in the popup and in History on a fresh install.
