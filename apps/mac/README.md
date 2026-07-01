# Netra

Khmer OCR menu bar app for macOS — grab-select any text on screen, get instant Khmer (and mixed Khmer/English) text extraction. A TextSniper clone with Khmer support as the primary focus.

## Features (implemented)

- **Screen selection capture** — drag-select any screen region (`SelectionOverlayWindow`), like macOS's built-in screenshot tool.
- **OCR, dual-engine** — Vision (`VNRecognizeTextRequest`) for the ~30 languages Apple ships models for. **Vision does not support Khmer at all** (verified directly against `supportedRecognitionLanguages` on macOS 26 / Xcode 26.6 — not in the list, old or new API). Khmer (and Khmer+English mixed) capture is routed to a vendored, self-contained Tesseract 5 binary with `khm`/`eng` traineddata instead — see "Khmer OCR engine" below.
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

## Khmer OCR engine

Vision cannot recognize Khmer script, so `TextRecognizer` (`Netra/Sources/OCR/TextRecognizer.swift`) routes any capture where Khmer is a requested language to `TesseractEngine`, which shells out to a vendored Tesseract binary at `Netra/Vendor/tesseract/bin/tesseract` with `--tessdata-dir Netra/Vendor/tesseract/tessdata -l khm+eng --psm 6`.

That binary is **not** a reference to a Homebrew install — it and its ~14 dylibs (`libtesseract`, `libleptonica`, `libarchive`, and their transitive deps) were copied out of `/opt/homebrew` and rewritten with [`dylibbundler`](https://github.com/auriamg/macdylibbundler) so every `LC_LOAD_DYLIB` install name is `@executable_path/../lib/...`. Verified with `otool -L` that no `/opt/homebrew` paths remain, and confirmed the bundled copy runs correctly from inside a built `.app`'s `Contents/Resources/tesseract/`. Traineddata (`khm.traineddata`, `eng.traineddata`) came from the official [tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast) repo.

Accuracy was verified end-to-end: rendered `"សួស្តី​ពិភពលោក! Netra ២០២៦"` through CoreText (proper Khmer shaping, same rendering path a real screenshot would go through) and ran it through the bundled binary — exact match. A naive PIL-rendered test image (no complex-script shaping) produced garbage, which is expected and not representative of real on-screen text; CoreText-shaped input is the fair test and it passed.

To update the vendored binary (e.g. after a `brew upgrade tesseract`):
```
mkdir -p Netra/Vendor/tesseract/{bin,lib}
cp /opt/homebrew/bin/tesseract Netra/Vendor/tesseract/bin/tesseract
cd Netra/Vendor/tesseract
dylibbundler -od -b -x ./bin/tesseract -d ./lib -p '@executable_path/../lib/'
```

## Not yet implemented

- Drag-and-drop OCR on image files.
- Capture-from-window (vs. drag-select region only).
- Onboarding / permission-request flow (Screen Recording permission is requested implicitly on first capture).
- Vision-language coverage beyond Khmer+English is unverified at runtime (compiles, matches Apple's documented API, but no live capture test was run against non-Khmer languages).
- App icon / asset catalog, code signing, notarization, distribution.
- Auto-update.

## Known shortcuts / debt

- `SWIFT_VERSION` is pinned to `5.0` (language mode 5) instead of Swift 6, to sidestep strict-concurrency errors triggered by Apple's own Translation framework API (`TranslationSession.translate` sending across actor boundaries). Worth revisiting once that framework's concurrency annotations mature.
- The runtime capture→OCR→clipboard flow has not been exercised via the actual GUI (this environment has no WindowServer access); only the Tesseract half was verified end-to-end by invoking the bundled binary directly against a real CoreText-rendered image. The Vision path and full click-driven flow (overlay drag, hotkey trigger, HUD, history) are unverified.

## Project layout

```
Netra/
  project.yml              # xcodegen spec — run `xcodegen generate` after editing
  Vendor/tesseract/         # vendored, dylibbundler-relocated Tesseract 5 + khm/eng traineddata
  Sources/
    App/                    # NetraApp entry point + AppDelegate (status item, wiring)
    Capture/                # selection overlay, ScreenCaptureKit capture, clipboard, result HUD
    OCR/                    # Vision + Tesseract text recognition, TextRecognizer router
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
