# KhmerClipper Desktop (Windows + Linux) — Design

**Date:** 2026-10-03
**Status:** Approved decisions (stack, scope, hosting, CI) from user Q&A; implementation plan at
`docs/superpowers/plans/2026-10-03-khmerclipper-desktop-windows-linux.md`.

## Problem

KhmerClipper ships only as a native macOS app (`apps/mac`, Swift/AppKit/CoreML/ScreenCaptureKit).
None of that code runs on Windows or Linux. Users on those OSes need the same core loop:
select a screen region → Khmer(+English) OCR → text on clipboard.

## Decisions

| Topic | Decision |
|-------|----------|
| Stack | Python 3.12 + PySide6 (Qt 6) + onnxruntime, one codebase at `apps/desktop/` for Windows and Linux |
| mac app | `apps/mac` untouched |
| v1 features | Tray app, region capture, OCR, auto-copy, result popup, history, preferences, image-file OCR, launch at login, **TrOCR engine**, **auto-update check** |
| Out of v1 | QR/barcode, EasyOCR, text-to-speech, translate, window-picker capture, capture sound |
| OCR engines | Tesseract (bundled/required, default) and TrOCR (ONNX, downloaded on first use) |
| Model hosting | ONNX export of the Khmer TrOCR fine-tune on a **public Hugging Face repo** `KimChengSHEANG/khmerclipper-trocr-khmer-onnx`, pinned by commit revision + SHA-256 in a committed manifest |
| Installer + update hosting | **Public GitHub repo** `KimChengSHEANG/khmerclipper-releases`; CI publishes releases there; app reads its `releases/latest` API |
| Build infra | GitHub Actions in this (private) repo: test matrix `windows-latest` + `ubuntu-22.04`; build artifacts; publish on tag `desktop-v*` |
| Artifacts | Windows: Inno Setup installer `KhmerClipper-<ver>-windows-x64-setup.exe` (per-user, no admin). Linux: `KhmerClipper-<ver>-x86_64.AppImage` with Tesseract bundled |

## Architecture

```
apps/desktop/
  src/khmerclipper/
    core (headless, testable everywhere): paths, errors, formatter, prefs, history,
      download, updater, autostart, hotkey (pure parts), ocr/{tesseract,trocr,tokenizer,
      preprocess,trocr_models}, capture/geometry
    gui (Qt): capture/{overlay,portal}, single_instance, ui/{app,worker,popup,
      history_window,prefs_window,image_window,fonts}
  resources/  fonts (Noto Sans Khmer, OFL), icon, tesseract/ (fetched/bundled at build)
  tools/      fetch_tessdata.py, export_onnx.py, publish_models.py (dev-only)
  packaging/  PyInstaller spec, Inno Setup script, AppImage script
.github/workflows/desktop.yml
```

### OCR

- **Tesseract:** subprocess, PNG via stdin (`tesseract stdin stdout --tessdata-dir … -l khm+eng --psm 6`),
  stdout decoded as UTF-8 explicitly. Tessdata = `tessdata_fast` tag `4.1.0` (same files as the mac
  app, SHA-256 pinned). Binary: bundled next to resources on Windows; inside the AppImage on Linux;
  `PATH` fallback in dev.
- **TrOCR:** exact port of `apps/mac/Sources/OCR/TrOCREngine.swift`: ink mask (corner background,
  40/255 threshold, scan ≤ 512), `lineStrips` (35%-of-tallest merge), `contentCrop` (5% sliver guard,
  4% pad), 384×384 bilinear resize, mean=std=0.5, greedy decode ≤ 64 tokens per line, lines joined
  with `\n`. Tokenizer decode = port of `TrOCRTokenizer.swift` reading the same `tokenizer.json`
  format (byte-BPE + `literal_ids`, SentencePiece).
- **ONNX export** (dev tool, torch): encoder + decoder-step graphs mirroring `convert.py`, with a
  torch-vs-onnxruntime logits parity check (max abs diff < 1e-3).
- **Model download:** on first TrOCR use, after user confirmation: stream to `.part`, verify SHA-256,
  atomic rename into the per-user data dir; cancellable; no partial files left behind.

### Capture

- **Windows / X11:** grab every screen first (`QScreen.grabWindow(0)`), show frozen screenshot in a
  fullscreen overlay per screen, crop from the grabbed image using the device-pixel ratio. Selections
  smaller than 4 logical px cancel.
- **Wayland:** `org.freedesktop.portal.Screenshot` with `interactive=true` (the desktop's own region
  picker) via `jeepney`. The Qt UI itself runs under XWayland (`QT_QPA_PLATFORM=xcb` when `DISPLAY`
  is set and the user hasn't chosen a platform): a native-Wayland background app can't reliably set
  the clipboard or position its popup.

### Hotkey

- Windows / X11: global hotkey via `pynput`, default `<ctrl>+<shift>+2`, configurable.
- Wayland (no global hotkeys for apps): `khmerclipper --capture` CLI; a running instance receives the
  command over `QLocalSocket`. User binds it in their desktop's keyboard-shortcut settings.

### Persistence

`platformdirs` per-user dirs. `prefs.json` (config dir), `history.json` (data dir, max 200 records,
newest first), models under data dir. UTF-8 explicitly; atomic writes; corrupt files never crash the
app (corrupt history is moved to `history.json.bak`).

### Auto-update

Daily (pref, default on) GET `https://api.github.com/repos/KimChengSHEANG/khmerclipper-releases/releases/latest`.
If its tag (`vX.Y.Z`) is newer than `khmerclipper.__version__`, prompt and open the release page in the
browser. No silent install. Network errors are silent.

### Launch at login

Windows: `HKCU\Software\Microsoft\Windows\CurrentVersion\Run\KhmerClipper`. Linux:
`$XDG_CONFIG_HOME/autostart/khmerclipper.desktop` (uses `$APPIMAGE` path when running from AppImage).

## Constraints

- Python 3.12; PySide6-Essentials 6.8.x; onnxruntime 1.20+ (CPU).
- Khmer text must render on stock systems → bundle Noto Sans Khmer and set it on every text widget.
- Default engine Tesseract (works offline with zero setup). TrOCR option disabled while the manifest
  lists no files.
- App name `KhmerClipper`; version single-sourced in `khmerclipper/__init__.py`; release tag in this repo
  `desktop-v<version>`, in the releases repo `v<version>`.
- CI secret `RELEASES_TOKEN` (fine-grained PAT, contents:write on `khmerclipper-releases`).

## Known limitations (accepted for v1)

- Wayland: the portal may save its screenshot file in the user's Pictures folder; KhmerClipper doesn't
  delete it. GNOME needs the AppIndicator extension (default on Ubuntu) for the tray icon.
- Windows build is windowed (no console): `khmerclipper ocr …` CLI output is for dev/Linux use.
- TrOCR model download is fp32 (~1.3 GB). int8 quantization is a follow-up after a CER check.
- Installers are unsigned (SmartScreen warning on Windows).
