# Mac CoreML OCR gate

**Status:** FAIL (mixed-subset CER only) — recorded 2026-10-03 after Task 8
full-200 KV-cache FP16 bench on Apple Silicon.

This gate compares on-device CoreML TrOCR (`trocr-coreml` via `OCRBench`) against the vendored Tesseract baseline on the frozen 200-fixture set. Outcome drives the app default OCR engine policy in `AppPreferences.swift`.

## Prerequisites

- Mac with Apple Silicon, Xcode 16+, Python 3.12+
- Khmer checkpoint at `apps/mac/raw_models/TrOCR-khmer/` (KV-cache FP16 export in `Vendor/trocr/khmer/`)
- Eval harness: `cd training/eval && ../apps/mac/.trocr-venv/bin/pip install -e .`

```bash
export REPO=$(git rev-parse --show-toplevel)
export KHMERCLIP_VENDOR=$REPO/apps/mac/Vendor
```

---

## Step 1: Convert to CoreML and build

From `apps/mac`:

```bash
cd "$REPO/apps/mac"
make trocr-khmer KHMER_CKPT=raw_models/TrOCR-khmer
make ocrbench
```

Release `OCRBench` lands at `build/Build/Products/Release/OCRBench`. For a one-off Debug build:

```bash
xcodebuild -project KhmerClip.xcodeproj -scheme OCRBench -configuration Debug \
  -derivedDataPath build \
  CODE_SIGN_IDENTITY="-" CODE_SIGNING_REQUIRED=NO \
  build
```

Smoke-test TrOCR before the full bench:

```bash
export KHMERCLIP_VENDOR="$REPO/apps/mac/Vendor"
"$REPO/apps/mac/build/Build/Products/Release/OCRBench" \
  --engine trocr --preload \
  "$REPO/training/eval/fixtures/printed/printed-001.png"
```

---

## Step 2: Mac gate bench

Full fixture bench (`trocr-coreml` vs `tesseract`), same run for gate numbers:

```bash
cd "$REPO/training/eval"
export KHMERCLIP_VENDOR="$REPO/apps/mac/Vendor"
KHMERCLIP_TROCR_UNITS=gpu python scripts/bench.py \
  --engine tesseract \
  --engine trocr-coreml \
  --trocr-coreml-bin "$REPO/apps/mac/build/Build/Products/Release/OCRBench" \
  --out results/stage2_kv_gpu.csv \
  --candidate trocr-coreml \
  --compare-to tesseract
```

The script prints per-engine aggregates and a final `PASS` or `FAIL` from `passes_gate()`.
TrOCR is preloaded before the timed loop; each CSV `ms` is per-image `recognizeText` time.

### Gate criteria (`khmerclip_eval.summary.passes_gate`)

TrOCR CoreML **passes** only if all of the following hold vs Tesseract on the same run:

| Check | Rule |
| --- | --- |
| Overall CER | `trocr-coreml` mean CER **<** Tesseract mean CER |
| Latency | `trocr-coreml` median ms **≤ 2×** Tesseract median ms |
| Mixed-script CER | `trocr-coreml` mixed-subset mean CER **≤** Tesseract mixed-subset mean CER |

---

## Step 3: Record outcome and default policy

### Step 3a — PASS

- **App default:** keep `.trocr` in `AppPreferences.swift` (`?? .trocr`).
- Document aggregate stats and `PASS` reasons below.

### Step 3b — FAIL (latency only)

Re-export `microsoft/trocr-small-printed` on 3090, repeat bench, re-evaluate.

### Step 3c — FAIL (CER / mixed / after small fallback)

- **App default:** `?? .tesseract` in `AppPreferences.swift` (applied 2026-10-03).
- TrOCR remains selectable in Preferences for users with converted models.

```swift
self.ocrEngine = (defaults.string(forKey: Keys.ocrEngine)).flatMap(OCREngineKind.init) ?? .tesseract
```

---

## Results (Task 8 — 2026-10-03, chosen run `KHMERCLIP_TROCR_UNITS=gpu`)

| Engine | n | Mean CER | Median ms | Mixed CER |
| --- | ---: | ---: | ---: | ---: |
| tesseract | 200 | 0.2580 | 106.1 | 0.0894 |
| trocr-coreml | 200 | 0.0593 | 95.3 | 0.0950 |

**Compute-unit sweep (same checkpoint, full 200):**

| Units | trocr-coreml CER | Median ms | Mixed CER | Gate |
| --- | ---: | ---: | ---: | --- |
| all | 0.0593 | 105.9 | 0.0950 | FAIL (mixed) |
| gpu | 0.0593 | 95.3 | 0.0950 | FAIL (mixed) |
| ane | — | — | — | CoreML plan error -14 (decoder); not measured |

**Gate:** **FAIL**

- PASS overall CER: 0.0593 < 0.2580
- PASS latency: 95.3 ms ≤ 2× 106.1 ms
- FAIL mixed CER: 0.0950 > 0.0894 (Tesseract)

**Default policy applied:** `.tesseract` for new installs (`AppPreferences.swift`). TrOCR default compute units when selected: `.cpuAndGPU` (`TrOCREngine.ModelCache`).

**Training follow-up (mixed-only failure):** mixed Khmer+Latin synthetic data on the training side (per speed+accuracy plan; not `ui_strings.txt` content). See `docs/superpowers/plans/2026-10-03-trocr-finetune-new-data.md`.

**Skipped fallbacks:** encoder FP32 and int8 (stage-2 CER equals stage-1 parity 0.0593; latency already under 2× Tesseract).

**Checkpoint:** `apps/mac/raw_models/TrOCR-khmer/` → KV FP16 `Vendor/trocr/khmer/`

**Artifacts:** `results/stage2_kv_*.csv` / `.txt` (gitignored CSVs; summary in this file only)
