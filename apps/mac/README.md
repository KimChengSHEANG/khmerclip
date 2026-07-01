# Netra

Khmer OCR menu bar app for macOS — grab-select any text on screen, get instant Khmer (and mixed Khmer/English) text extraction. A TextSniper clone with Khmer support as the primary focus.

## Features (implemented)

- **Screen selection capture** — drag-select any screen region (`SelectionOverlayWindow`), like macOS's built-in screenshot tool.
- **OCR** — Vision (`VNRecognizeTextRequest`) text recognition, multi-language including Khmer, filtered against what the running OS build actually supports.
- **QR / barcode scanning** — `VNDetectBarcodesRequest`, separate hotkey from text capture.
- **Auto-copy to clipboard** on every successful capture.
- **Result HUD** — transient floating panel after capture, with Speak / Translate actions.
- **Scan history** — persisted to `~/Library/Application Support/Netra/history.json`, with copy/speak/translate/delete per entry.
- **Text-to-speech** — `AVSpeechSynthesizer`, Khmer voice preferred.
- **Translate** — Apple's on-device Translation framework, opened from the HUD or history.
- **Global hotkeys** — customizable (Carbon `RegisterEventHotKey`), one for text capture, one for QR/barcode.
- **Launch at login** — `SMAppService`.
- **Preferences window** — hotkey recorder, recognition language toggles, behavior toggles.
- **Menu bar only app** (`LSUIElement`), no dock icon.

## Not yet implemented

- Drag-and-drop OCR on image files.
- Capture-from-window (vs. drag-select region only).
- Onboarding / permission-request flow (Screen Recording permission is requested implicitly on first capture).
- App icon / asset catalog, code signing, notarization, distribution.
- Auto-update.

## Project layout

```
Netra/
  project.yml              # xcodegen spec — run `xcodegen generate` after editing
  Sources/
    App/                    # NetraApp entry point + AppDelegate (status item, wiring)
    Capture/                # selection overlay, ScreenCaptureKit capture, clipboard, result HUD
    OCR/                    # Vision text recognition
    Barcode/                # Vision barcode/QR detection
    Hotkey/                 # Carbon global hotkey registration + hotkey definition model
    History/                # scan history model, JSON-backed store, SwiftUI list
    Speech/                 # AVSpeechSynthesizer wrapper
    Translate/              # Translation framework view
    LaunchAtLogin/           # SMAppService wrapper
    Preferences/             # AppPreferences (UserDefaults-backed), preferences UI, hotkey recorder
```

## Build

Requires Xcode 16+, macOS 15+ deployment target, [XcodeGen](https://github.com/yonaskolb/XcodeGen).

```
cd Netra
xcodegen generate
open Netra.xcodeproj
```

First capture will prompt for Screen Recording permission (System Settings → Privacy & Security → Screen Recording).
