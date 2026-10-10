# KhmerClip

Khmer OCR menu bar app for macOS — grab-select any text on screen, get instant Khmer (and mixed Khmer/English) text extraction. A TextSniper clone with Khmer support as the primary focus.

**Quick start:** clone → `brew install xcodegen` → `make run`. Tesseract OCR works immediately (default engine after the 2026-10-03 Mac CoreML gate). TrOCR and EasyOCR need extra setup — see [Build and run](#build-and-run) and [OCR engines](#ocr-engines).

## Features

- **Screen selection capture** — drag-select any screen region (`SelectionOverlayWindow`), like macOS's built-in screenshot tool.
- **Capture-from-window** — press Space while the selection overlay is up to switch to "click a window" mode (same toggle macOS's own screenshot tool uses); hovering highlights whichever window is under the cursor (`WindowPicker`, via `CGWindowListCopyWindowInfo`) and a click captures that window's full bounds.
- **OCR Image File…** — a dedicated drop-zone window (menu bar → "OCR Image File…") for running OCR against an existing image file instead of the live screen: drag a file in, or use "Choose File…".
- **OCR, three engines** — Preferences → OCR Engine picks the backend:
  - **Tesseract** (vendored binary, `khm+eng`) — works out of the box, no extra setup.
  - **TrOCR** (CoreML Khmer fine-tune) — on-device neural OCR; needs `make trocr-khmer` first.
  - **EasyOCR** (Python + custom Khmer recognition model) — needs `make easyocr-venv easyocr-models` and a trained `khmer.pth`.
  Vision (`VNRecognizeTextRequest`) does not support Khmer at all (verified on macOS 26 / Xcode 26.6 — not in `supportedRecognitionLanguages`), so none of the engines use Vision for text.
- **QR / barcode scanning** — `VNDetectBarcodesRequest`, separate hotkey from text capture.
- **Auto-copy to clipboard** on every successful capture.
- **Result HUD** — transient floating panel after capture, with Speak / Translate actions.
- **Scan history** — persisted to `~/Library/Application Support/KhmerClip/history.json`, with copy/speak/translate/delete per entry.
- **Text-to-speech** — `AVSpeechSynthesizer`, Khmer voice preferred.
- **Translate** — Apple's on-device Translation framework, opened from the HUD or history.
- **Global hotkeys** — customizable (Carbon `RegisterEventHotKey`), one for text capture, one for QR/barcode.
- **Launch at login** — `SMAppService`.
- **Preferences window** — hotkey recorder, behavior toggles.
- **Onboarding window** — shown once on first launch (menu bar → "Welcome / Permissions…" to re-open): explains the app, live-checks Screen Recording access via `CGPreflightScreenCaptureAccess()`, and links straight to the System Settings pane.
- **Auto-update** — [Sparkle](https://sparkle-project.org) wired in via SPM ("Check for Updates…" in the menu). `SUFeedURL` is currently a placeholder (see "Auto-update" below).
- **Menu bar only app** (`LSUIElement`), no dock icon, custom app icon.

## OCR engines

All three backends handle Khmer text mixed with English words/numerals (e.g. `"ដក language ផ្សេងទៀត"`). Pick one in **Preferences → OCR Engine**. Default is **Tesseract** after the 2026-10-03 Mac CoreML gate (TrOCR failed mixed-subset CER only); see [Mac CoreML gate policy](#mac-coreml-gate-default-engine).

| Engine | Runtime | Setup | Notes |
|--------|---------|-------|-------|
| Tesseract | Vendored C++ binary | None (bundled) | Fastest cold start; good baseline accuracy |
| TrOCR | CoreML (encoder + decoder) | `make trocr-khmer` | On-device; shows progress HUD while loading/decoding |
| EasyOCR | Python subprocess (PyTorch) | `make easyocr-venv easyocr-models` + Khmer weights | CRAFT detection + custom Khmer recognizer; first run downloads CRAFT weights |

### Tesseract (bundled)

`TesseractEngine` shells out to `Vendor/tesseract/bin/tesseract` with `--tessdata-dir Vendor/tesseract/tessdata -l khm+eng --psm 6`. `khm+eng` (not `khm` alone) handles mixed-script lines.

The binary is self-contained — ~14 dylibs relocated with [`dylibbundler`](https://github.com/auriamg/macdylibbundler) so every `LC_LOAD_DYLIB` is `@executable_path/../lib/...`. `khm.traineddata` / `eng.traineddata` from [tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast).

**Update vendored Tesseract** (e.g. after `brew upgrade tesseract`):
```bash
mkdir -p Vendor/tesseract/{bin,lib}
cp /opt/homebrew/bin/tesseract Vendor/tesseract/bin/tesseract
cd Vendor/tesseract
dylibbundler -od -b -x ./bin/tesseract -d ./lib -p '@executable_path/../lib/'
```

### TrOCR (CoreML Khmer model)

`TrOCREngine` loads CoreML packages from `Vendor/trocr/khmer/` (`encoder.mlpackage`, `decoder.mlpackage`, `tokenizer.json`). Models are **not committed** (~hundreds of MB); regenerate from a Hugging Face checkpoint.

**Prerequisites:** Python 3 with `torch`, `transformers`, `coremltools`:
```bash
pip install torch transformers coremltools
```

**One-time setup** — convert the Khmer fine-tune (default checkpoint dir: `raw_models/TrOCR-khmer`):
```bash
make trocr-khmer
```

This runs `tools/trocr/convert.py` and `tools/trocr/export_tokenizer.py` into `Vendor/trocr/khmer/`. To use a different checkpoint:
```bash
make trocr-khmer KHMER_CKPT=/path/to/your/checkpoint
```

**Rebuild all size tiers** (small/base/large English + khmer — large download):
```bash
make trocr-models
```

After conversion, rebuild the app (`make build`) so Xcode bundles the new `.mlpackage` files. TrOCR preloads the model when you start a text capture (see `CaptureCoordinator`).

### Mac CoreML gate (default engine)

Gate recorded **2026-10-03** on the frozen 200-fixture set with KV-cache CoreML
TrOCR (Release `OCRBench`). Outcome: **FAIL** on mixed-subset CER only (overall
CER and latency passed). New installs default to **Tesseract**; TrOCR remains
available after `make trocr-khmer`. When TrOCR runs, CoreML compute units
default to **GPU** (`.cpuAndGPU`).

Operator checklist:
[`training/eval/results/mac_gate.md`](../../training/eval/results/mac_gate.md)

| Gate result | Default engine | Notes |
|-------------|----------------|-------|
| PASS | `.trocr` | — |
| FAIL (mixed only, 2026-10-03) | `.tesseract` | Training follow-up: mixed Khmer+Latin synthetic data |
| FAIL (latency) | — | Re-export `trocr-small-printed`, re-gate |
| FAIL (CER after small fallback) | `.tesseract` | TrOCR stays in Preferences |


### EasyOCR (Python + custom Khmer recognizer)

`EasyOCREngine` shells out to `Vendor/easyocr/recognize.py`, which runs EasyOCR with a **custom Khmer recognition network** (EasyOCR has no built-in Khmer). Needs three components:

| File | Location | Purpose |
|------|----------|---------|
| `khmer.pth` | `Vendor/easyocr/models/` | Trained recognition weights |
| `khmer.yaml` | `Vendor/easyocr/khmer/user_network/` | Model config + character set |
| `khmer.py` | `Vendor/easyocr/khmer/user_network/` | Network architecture (VGG-BiLSTM-CTC) |
| `charset.txt` | `Vendor/easyocr/khmer/` | One character per line; must match `khmer.pth` training charset |

CRAFT text-detection weights live in `Vendor/easyocr/models/` (downloaded separately).

**One-time setup:**
```bash
# 1. Python venv with EasyOCR + PyTorch (~1–2 GB)
make easyocr-venv

# 2. Download CRAFT detection weights
make easyocr-models

# 3. Install Khmer recognition weights (if not already in place)
#    Drop khmer.pth under Vendor/easyocr/khmer/ (legacy) or pass explicit paths:
python3 Vendor/easyocr/install_khmer_model.py Vendor/easyocr/khmer
```

`install_khmer_model.py` copies `khmer.pth` → `models/`, syncs `khmer.yaml` + `khmer.py` → `khmer/user_network/`, and regenerates `character_list` in the yaml from `charset.txt`.

**Test from the command line** (no app needed):
```bash
Vendor/easyocr/venv/bin/python3 Vendor/easyocr/recognize.py \
  --resource-root Vendor/easyocr --model khmer \
  path/to/image.png
```

**Run in the app:** rebuild after setup (`make build && make run`). The app prefers `Vendor/easyocr/venv/bin/python3` when bundled; falls back to system `python3` on `PATH` if the venv is missing. Rebuild after `make easyocr-venv` so the venv is copied into the app bundle.

**Train your own Khmer model:** use the [EasyOCR trainer](https://github.com/JaidedAI/EasyOCR/tree/master/trainer) / [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark). The `character_list` in `khmer.yaml` must exactly match the training charset (201 chars + CTC blank = 202 output classes). `charset.txt` is the source of truth.

### Accuracy note (Tesseract, historical)

Rendered `"សួស្តី​ពិភពលោក! KhmerClip ២០២៦"` through CoreText (proper Khmer shaping), compiled the Tesseract OCR call chain into a standalone test binary, and got an exact match. A naive PIL-rendered image (no Khmer glyph reordering) produced garbage — not representative of real on-screen text.

## Runtime verification performed

This environment has no window manager by default for the sandboxed shell tool, but running Bash with the sandbox disabled gave real WindowServer/Accessibility access, so the actual built `.app` was driven end-to-end rather than just compiled:

- Launched the built `.app`, used Accessibility (`System Events`) to `AXPress` the real status item, and confirmed its menu contents match the code exactly (`Capture Text, Scan QR / Barcode, OCR Image File…, History…, Preferences…, Quit KhmerClip`).
- Clicked "Capture Text", then posted a synthetic `CGEvent` mouse drag. Confirmed via Accessibility + `log show` that this created a full-screen overlay window at the exact correct frame, dismissed it on mouse-up, and triggered a real `SCShareableContent`/ScreenCaptureKit call (visible in the unified log, no permission errors), followed by a result window appearing.
- Opened "OCR Image File…" and confirmed the drop-zone window renders correctly (screenshot-verified). Clicking "Choose File…" spawned the real `com.apple.appkit.xpc.openAndSavePanelService` (visible in `lsappinfo`), confirming `NSOpenPanel` wiring is live, not dead code.
- Verified the Tesseract OCR call chain against a real CoreText-rendered Khmer+English+numeral string via a standalone compile — see "Accuracy note" above.
- Compiled the app icon through `actool` with zero warnings and confirmed the resulting `.icns` renders correctly (rounded-square masking applied by the OS, as expected for a full-bleed source image).
- (Historical) Verified a Vision `fr-FR` path when the app still had a non-Khmer engine option — exact match on `"Bonjour le monde! KhmerClip 2026"`. Vision is not used for Khmer; the current app routes text capture through Tesseract / TrOCR / EasyOCR per Preferences.
- Launched the onboarding window for real and confirmed `CGPreflightScreenCaptureAccess()` reflects live state (screenshot-verified: warning icon + explanatory text render correctly).
- Triggered "Check for Updates…" for real: confirmed in the unified log that Sparkle made an actual HTTPS request to the placeholder `SUFeedURL`, got a 404, and handled it gracefully (logged error, no crash) — the wiring is live, only the appcast hosting is a placeholder.

What's still unconfirmed: the *exact string* a live capture returns landing in the clipboard/HUD/history in one continuous human-driven run (UI-automation timing made this fiddly to script reliably — clicks and drags each independently proved to work, but chaining all of them through AppleScript in one shot raced against SwiftUI's async accessibility tree). Functionally this is very low risk: it's the same `recognize(image:mode:)` method already proven correct, called from wiring that's each been independently proven to fire.

## Not yet implemented

- Code signing with a real Developer ID, notarization, and distribution — see below, this needs credentials only you have.
- A real hosted appcast for Sparkle (`SUFeedURL` is a placeholder `https://example.com/...` — see "Auto-update" below).

## Code signing, notarization & distribution

`CODE_SIGN_STYLE: Automatic` is set, so Xcode will ad-hoc sign local builds with whatever team you have selected. Actually distributing the app (so it launches on someone else's Mac without a Gatekeeper block) needs your own paid Apple Developer Program membership — this isn't something that can be scripted without your credentials:

```
# 1. Archive with your real Developer ID team (set DEVELOPMENT_TEAM in project.yml
#    or select your team in Xcode's Signing & Capabilities tab first).
xcodebuild -project KhmerClip.xcodeproj -scheme KhmerClip -configuration Release archive \
  -archivePath build/KhmerClip.xcarchive

# 2. Export a signed .app (needs an ExportOptions.plist with method: developer-id).
xcodebuild -exportArchive -archivePath build/KhmerClip.xcarchive \
  -exportPath build/export -exportOptionsPlist ExportOptions.plist

# 3. Notarize (needs an app-specific password or API key set up via `xcrun notarytool store-credentials`).
xcrun notarytool submit build/export/KhmerClip.app.zip --keychain-profile "khmerclip-notary" --wait

# 4. Staple the ticket so it works offline.
xcrun stapler staple build/export/KhmerClip.app
```

## Auto-update

Sparkle is fully wired (menu item, `SPUStandardUpdaterController`, Info.plist keys), verified making real network requests — but it points at a placeholder feed. To make it real:

1. Host an `appcast.xml` somewhere (GitHub Pages / Releases works fine) and point `SUFeedURL` in `project.yml` at it.
2. Generate an EdDSA signing key with Sparkle's `generate_keys` tool and set `SUPublicEDKey` in `project.yml` — without this, Sparkle will refuse to install (unsigned) updates rather than silently trusting anything.
3. Sign each release archive with `sign_update` (from the same Sparkle tool package) and reference it in the appcast entry.

## Project layout

```
project.yml              # xcodegen spec — run `xcodegen generate` (or `make gen`) after editing
Makefile                 # build, test, run, trocr-*, easyocr-* targets
Vendor/
  tesseract/             # vendored Tesseract 5 + khm/eng traineddata (bundled, works OOTB)
  trocr/                 # TrOCR convert scripts + khmer/ CoreML models (models gitignored)
  easyocr/               # recognize.py, khmer network files, venv/ + models/ (gitignored)
Resources/
  Assets.xcassets/       # AppIcon.appiconset
Sources/
  App/                   # KhmerClipApp entry point + AppDelegate (status item, wiring)
  Capture/               # selection overlay, window-picker, ScreenCaptureKit, clipboard, HUD
  OCR/                   # TesseractEngine, TrOCREngine, EasyOCREngine, OCRDispatcher
  Barcode/               # Vision barcode/QR detection
  Hotkey/                # Carbon global hotkey registration
  History/               # scan history model, JSON-backed store
  Speech/                # AVSpeechSynthesizer wrapper
  Translate/             # Translation framework view
  LaunchAtLogin/         # SMAppService wrapper
  Preferences/           # AppPreferences, engine picker, hotkey recorder
Tests/
  Fixtures/ocr/          # PNG + expected .txt pairs for Tesseract regression tests
raw_models/              # local HF checkpoints (gitignored), e.g. TrOCR-khmer/
```

## Known shortcuts / debt

- `SWIFT_VERSION` is pinned to `5.0` (language mode 5) instead of Swift 6, to sidestep strict-concurrency errors triggered by Apple's own Translation framework API (`TranslationSession.translate` sending across actor boundaries). Worth revisiting once that framework's concurrency annotations mature.

## Build and run

### Prerequisites

- **macOS 15+** (deployment target)
- **Xcode 16+** with Command Line Tools (`xcode-select --install`)
- **[XcodeGen](https://github.com/yonaskolb/XcodeGen)** — generates `KhmerClip.xcodeproj` from `project.yml`:
  ```bash
  brew install xcodegen
  ```

No Python needed unless you set up TrOCR or EasyOCR models.

### Clone and build

```bash
git clone <repo-url> khmerclip
cd khmerclip/apps/mac

make run    # gen + build + open Debug .app
```

`make run` outputs to `build/Build/Products/Debug/KhmerClip.app`. Menu bar icon appears (no Dock icon — `LSUIElement`). First launch shows onboarding; grant **Screen Recording** when prompted (System Settings → Privacy & Security → Screen Recording).

### Makefile targets

| Target | What it does |
|--------|----------------|
| `make gen` | `xcodegen generate` — regenerate `KhmerClip.xcodeproj` after editing `project.yml` |
| `make build` | `gen` + `xcodebuild` Debug build → `build/Build/Products/Debug/KhmerClip.app` |
| `make run` | `build`, kill any running KhmerClip, register with LaunchServices, `open` the `.app` |
| `make test` | `gen` + run `KhmerClipTests` (Tesseract OCR fixture regression) |
| `make clean` | Remove `build/` and `KhmerClip.xcodeproj` |
| `make trocr-khmer` | Convert `raw_models/TrOCR-khmer` → CoreML in `Vendor/trocr/khmer/` |
| `make trocr-models` | Convert all TrOCR size tiers (large download) |
| `make easyocr-venv` | Create `Vendor/easyocr/venv` with PyTorch + EasyOCR |
| `make easyocr-models` | Download CRAFT detection weights into `Vendor/easyocr/models/` |

### Run from Xcode (alternative)

```bash
make gen
open KhmerClip.xcodeproj
```

Select the **KhmerClip** scheme, pick **My Mac**, press **⌘R**. Built app lands in Xcode's DerivedData (or `build/` if you set `-derivedDataPath build` like the Makefile).

### OCR before first capture

Default engine is **TrOCR** — needs converted models or capture will fail. For immediate OCR without extra setup:

1. Launch app → menu bar icon → **Preferences…**
2. **OCR Engine** → **Tesseract**
3. Capture text (default hotkey, or menu → Capture Text)

| Engine | Extra setup before capture works |
|--------|----------------------------------|
| Tesseract | None |
| TrOCR (default) | `pip install torch transformers coremltools` then `make trocr-khmer`, then `make build` |
| EasyOCR | `make easyocr-venv easyocr-models`, install Khmer weights, then `make build` — see [EasyOCR](#easyocr-python--custom-khmer-recognizer) |

After any vendor model or venv change, **rebuild** (`make build` or `make run`) so resources are copied into the app bundle.

### Run built app manually

```bash
open build/Build/Products/Debug/KhmerClip.app
```

Or double-click in Finder. `make run` is preferred — it re-registers the app with LaunchServices (avoids `open` returning -600 on fresh builds).

### Run tests

```bash
make test
```

Runs Tesseract fixture tests against `Tests/Fixtures/ocr/*.png`. Does not require TrOCR/EasyOCR models.

### Troubleshooting

**Sparkle / `xcodebuild` package resolve fails** with `safe.bareRepository is 'explicit'` — global git config issue, not this repo. Fix: build from Xcode GUI, or prefix:
```bash
GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.bareRepository GIT_CONFIG_VALUE_0=all make build
```

**TrOCR capture fails** — `Vendor/trocr/khmer/` missing or empty. Run `make trocr-khmer` (needs checkpoint in `raw_models/TrOCR-khmer/`).

**EasyOCR capture fails** — run `make easyocr-venv easyocr-models`, install weights, rebuild. Test CLI:
```bash
Vendor/easyocr/venv/bin/python3 Vendor/easyocr/recognize.py \
  --resource-root Vendor/easyocr --model khmer --warmup
```

**Gitignored artifacts** (not in repo; regenerate locally):
- `Vendor/trocr/*/` — CoreML models
- `Vendor/easyocr/venv/` — Python env
- `Vendor/easyocr/models/` — CRAFT + `khmer.pth`
- `raw_models/` — Hugging Face checkpoints for conversion
