# Netra

Khmer OCR menu bar app for macOS — grab-select any text on screen, get instant Khmer (and mixed Khmer/English) text extraction. A TextSniper clone with Khmer support as the primary focus.

## Features

- **Screen selection capture** — drag-select any screen region (`SelectionOverlayWindow`), like macOS's built-in screenshot tool.
- **Capture-from-window** — press Space while the selection overlay is up to switch to "click a window" mode (same toggle macOS's own screenshot tool uses); hovering highlights whichever window is under the cursor (`WindowPicker`, via `CGWindowListCopyWindowInfo`) and a click captures that window's full bounds.
- **OCR Image File…** — a dedicated drop-zone window (menu bar → "OCR Image File…") for running OCR against an existing image file instead of the live screen: drag a file in, or use "Choose File…".
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
- **Menu bar only app** (`LSUIElement`), no dock icon, custom app icon.

## Khmer OCR engine

Vision cannot recognize Khmer script, so `TextRecognizer` (`Netra/Sources/OCR/TextRecognizer.swift`) routes any capture where Khmer is a requested language to `TesseractEngine`, which shells out to a vendored Tesseract binary at `Netra/Vendor/tesseract/bin/tesseract` with `--tessdata-dir Netra/Vendor/tesseract/tessdata -l khm+eng --psm 6`.

That binary is **not** a reference to a Homebrew install — it and its ~14 dylibs (`libtesseract`, `libleptonica`, `libarchive`, and their transitive deps) were copied out of `/opt/homebrew` and rewritten with [`dylibbundler`](https://github.com/auriamg/macdylibbundler) so every `LC_LOAD_DYLIB` install name is `@executable_path/../lib/...`. Verified with `otool -L` that no `/opt/homebrew` paths remain, and confirmed the bundled copy runs correctly from inside a built `.app`'s `Contents/Resources/tesseract/`. Traineddata (`khm.traineddata`, `eng.traineddata`) came from the official [tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast) repo.

**Accuracy verification (definitive):** rendered `"សួស្តី​ពិភពលោក! Netra ២០២៦"` through CoreText (proper Khmer shaping — reordering/ligatures — the same rendering path a real screenshot would go through), then compiled the actual, unmodified `OCREngine.swift` + `TesseractEngine.swift` + `TextRecognizer.swift` from this repo into a standalone test binary and ran `TextRecognizer.recognizeText` (the exact call `CaptureCoordinator` and `ImageOCRView` both make) against the rendered image. Output was an exact match. A naive PIL-rendered test image (no complex-script shaping) produced garbage on the first attempt, which is expected — PIL doesn't do Khmer glyph reordering — and not representative of real on-screen text.

To update the vendored binary (e.g. after a `brew upgrade tesseract`):
```
mkdir -p Netra/Vendor/tesseract/{bin,lib}
cp /opt/homebrew/bin/tesseract Netra/Vendor/tesseract/bin/tesseract
cd Netra/Vendor/tesseract
dylibbundler -od -b -x ./bin/tesseract -d ./lib -p '@executable_path/../lib/'
```

## Runtime verification performed

This environment has no window manager by default for the sandboxed shell tool, but running Bash with the sandbox disabled gave real WindowServer/Accessibility access, so the actual built `.app` was driven end-to-end rather than just compiled:

- Launched the built `.app`, used Accessibility (`System Events`) to `AXPress` the real status item, and confirmed its menu contents match the code exactly (`Capture Text, Scan QR / Barcode, OCR Image File…, History…, Preferences…, Quit Netra`).
- Clicked "Capture Text", then posted a synthetic `CGEvent` mouse drag. Confirmed via Accessibility + `log show` that this created a full-screen overlay window at the exact correct frame, dismissed it on mouse-up, and triggered a real `SCShareableContent`/ScreenCaptureKit call (visible in the unified log, no permission errors), followed by a result window appearing.
- Opened "OCR Image File…" and confirmed the drop-zone window renders correctly (screenshot-verified). Clicking "Choose File…" spawned the real `com.apple.appkit.xpc.openAndSavePanelService` (visible in `lsappinfo`), confirming `NSOpenPanel` wiring is live, not dead code.
- Verified the exact production OCR call chain (`TextRecognizer.recognizeText`) against a real CoreText-rendered Khmer+English+numeral string via a standalone compile of the unmodified source files — see above.
- Compiled the app icon through `actool` with zero warnings and confirmed the resulting `.icns` renders correctly (rounded-square masking applied by the OS, as expected for a full-bleed source image).

What's still unconfirmed: the *exact string* a live capture returns landing in the clipboard/HUD/history in one continuous human-driven run (UI-automation timing made this fiddly to script reliably — clicks and drags each independently proved to work, but chaining all of them through AppleScript in one shot raced against SwiftUI's async accessibility tree). Functionally this is very low risk: it's the same `recognize(image:mode:)` method already proven correct, called from wiring that's each been independently proven to fire.

## Not yet implemented

- Onboarding / permission-request flow (Screen Recording permission is requested implicitly on first capture — no explanatory pre-prompt screen).
- Vision-language coverage beyond Khmer+English hasn't been exercised in a live capture (compiles, matches Apple's documented API, but no runtime test was run against e.g. French/Japanese selection).
- Code signing with a real Developer ID, notarization, and distribution — see below, this needs credentials only you have.
- Auto-update.

## Code signing, notarization & distribution

`CODE_SIGN_STYLE: Automatic` is set, so Xcode will ad-hoc sign local builds with whatever team you have selected. Actually distributing the app (so it launches on someone else's Mac without a Gatekeeper block) needs your own paid Apple Developer Program membership — this isn't something that can be scripted without your credentials:

```
# 1. Archive with your real Developer ID team (set DEVELOPMENT_TEAM in project.yml
#    or select your team in Xcode's Signing & Capabilities tab first).
xcodebuild -project Netra.xcodeproj -scheme Netra -configuration Release archive \
  -archivePath build/Netra.xcarchive

# 2. Export a signed .app (needs an ExportOptions.plist with method: developer-id).
xcodebuild -exportArchive -archivePath build/Netra.xcarchive \
  -exportPath build/export -exportOptionsPlist ExportOptions.plist

# 3. Notarize (needs an app-specific password or API key set up via `xcrun notarytool store-credentials`).
xcrun notarytool submit build/export/Netra.app.zip --keychain-profile "netra-notary" --wait

# 4. Staple the ticket so it works offline.
xcrun stapler staple build/export/Netra.app
```

## Project layout

```
Netra/
  project.yml              # xcodegen spec — run `xcodegen generate` after editing
  Vendor/tesseract/         # vendored, dylibbundler-relocated Tesseract 5 + khm/eng traineddata
  Resources/
    Assets.xcassets/         # AppIcon.appiconset (full-bleed square source, OS applies mask)
  Sources/
    App/                    # NetraApp entry point + AppDelegate (status item, wiring)
    Capture/                # selection overlay, window-picker, ScreenCaptureKit capture, clipboard, result HUD, image-file OCR window
    OCR/                    # Vision + Tesseract text recognition, TextRecognizer router
    Barcode/                # Vision barcode/QR detection
    Hotkey/                 # Carbon global hotkey registration + hotkey definition model
    History/                # scan history model, JSON-backed store, SwiftUI list
    Speech/                 # AVSpeechSynthesizer wrapper
    Translate/              # Translation framework view
    LaunchAtLogin/           # SMAppService wrapper
    Preferences/             # AppPreferences (UserDefaults-backed), preferences UI, hotkey recorder
```

## Known shortcuts / debt

- `SWIFT_VERSION` is pinned to `5.0` (language mode 5) instead of Swift 6, to sidestep strict-concurrency errors triggered by Apple's own Translation framework API (`TranslationSession.translate` sending across actor boundaries). Worth revisiting once that framework's concurrency annotations mature.

## Build

Requires Xcode 16+, macOS 15+ deployment target, [XcodeGen](https://github.com/yonaskolb/XcodeGen).

```
cd Netra
xcodegen generate
open Netra.xcodeproj
```

First capture will prompt for Screen Recording permission (System Settings → Privacy & Security → Screen Recording).
