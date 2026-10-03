# Mac CoreML OCR gate

**Status:** PENDING — waiting for Task 8 Step 5 (3090 multi-dataset train) to produce `experiments/checkpoint-N/final/`.

This gate compares on-device CoreML TrOCR (`trocr-coreml` via `OCRBench`) against the vendored Tesseract baseline on the frozen 200-fixture set. Outcome drives the app default OCR engine policy in `AppPreferences.swift`.

## Prerequisites

- Mac with Apple Silicon, Xcode 16+, Python 3.12+
- Completed 3090 training run with checkpoint at `training/trocr/experiments/checkpoint-N/final/`
- Eval harness installed: `cd training/eval && python -m pip install -e .`
- CoreML conversion deps: `pip install torch transformers coremltools`

Set repo root once (adjust if your clone path differs):

```bash
export REPO=/path/to/khmerclip
export CHECKPOINT_N=1   # replace with actual checkpoint number from 3090
```

---

## Step 1: Copy checkpoint to Mac

From the 3090 host, copy the Hugging Face `final/` folder to the Mac app raw-models directory (gitignored):

```bash
mkdir -p "$REPO/apps/mac/raw_models/TrOCR-khmer"
rsync -av "3090:$REPO/training/trocr/experiments/checkpoint-${CHECKPOINT_N}/final/" \
  "$REPO/apps/mac/raw_models/TrOCR-khmer/"
```

Verify the folder contains model weights, config, and the extended tokenizer saved by training.

---

## Step 2: Convert to CoreML and build

From `apps/mac`:

```bash
cd "$REPO/apps/mac"
make trocr-khmer KHMER_CKPT=raw_models/TrOCR-khmer
make build
xcodebuild -project KhmerClip.xcodeproj -scheme OCRBench -configuration Debug \
  -derivedDataPath build \
  CODE_SIGN_STYLE=Manual CODE_SIGN_IDENTITY="" \
  PROVISIONING_PROFILE_SPECIFIER="" DEVELOPMENT_TEAM="" \
  build
```

This writes CoreML packages to `Vendor/trocr/khmer/` (gitignored) and builds `OCRBench` at `build/Build/Products/Debug/OCRBench`.

Smoke-test TrOCR before the full bench:

```bash
export KHMERCLIP_VENDOR="$REPO/apps/mac/Vendor"
"$REPO/apps/mac/build/Build/Products/Debug/OCRBench" \
  --engine trocr --preload \
  "$REPO/training/eval/fixtures/printed/printed-001.png"
```

---

## Step 3: Mac gate bench

Run the full fixture bench (`trocr-coreml` vs `tesseract`), writing `mac_gate.csv`:

```bash
cd "$REPO/training/eval"
export KHMERCLIP_VENDOR="$REPO/apps/mac/Vendor"
python scripts/bench.py \
  --engine tesseract \
  --engine trocr-coreml \
  --trocr-coreml-bin "$REPO/apps/mac/build/Build/Products/Debug/OCRBench" \
  --out results/mac_gate.csv \
  --candidate trocr-coreml \
  --compare-to tesseract
```

The script prints per-engine aggregates and a final `PASS` or `FAIL` from `passes_gate()`.
The CoreML runner sends all fixture paths to one `OCRBench` process. TrOCR is
preloaded before the timed loop, and each CSV `ms` value comes from the
per-image `recognizeText` timing rather than process startup or model loading.

### Gate criteria (`khmerclip_eval.summary.passes_gate`)

TrOCR CoreML **passes** only if all of the following hold vs Tesseract on the same run:

| Check | Rule |
| --- | --- |
| Overall CER | `trocr-coreml` mean CER **<** Tesseract mean CER |
| Latency | `trocr-coreml` median ms **≤ 2×** Tesseract median ms |
| Mixed-script CER | `trocr-coreml` mixed-subset mean CER **≤** Tesseract mixed-subset mean CER |
| UI CER | candidate UI-subset mean CER **≤** Tesseract UI-subset mean CER |

Reference baseline (2026-08-27): see [baseline.md](baseline.md).

---

## Step 4: Record outcome and default policy

Update **Status** at the top of this file to `PASS` or `FAIL`, fill in the results table, and apply the default-engine policy below.

### Step 4a — PASS

- **App default:** keep `.trocr` in `AppPreferences.swift` (`?? .trocr` — already the current default).
- **Ship:** TrOCR remains the default OCR engine; Tesseract and EasyOCR stay selectable in Preferences.
- Document aggregate stats and `PASS` reasons here.

### Step 4b — FAIL (latency only)

If CER and mixed CER pass but median ms exceeds 2× Tesseract:

1. Re-train or export from `microsoft/trocr-small-printed` on 3090.
2. Repeat Steps 1–3 with the small checkpoint.
3. Re-evaluate; if small model passes, follow Step 4a.

### Step 4c — FAIL (CER after small fallback)

If the small model still fails the gate (accuracy):

- **App default:** change `AppPreferences.swift` line 46 to `?? .tesseract`.
- **Ship:** Tesseract becomes the default; TrOCR (and EasyOCR) remain available in Preferences for users who convert models locally.
- Document `FAIL` reasons and both engine aggregates here.

```swift
// AppPreferences.swift — apply ONLY on Step 4c FAIL
self.ocrEngine = (defaults.string(forKey: Keys.ocrEngine)).flatMap(OCREngineKind.init) ?? .tesseract
```

---

## Results (fill after Step 3)

| Engine | n | Mean CER | Median ms | Mixed CER |
| --- | ---: | ---: | ---: | ---: |
| tesseract | — | — | — | — |
| trocr-coreml | — | — | — | — |

**Gate:** _(PASS / FAIL — pending)_

**Default policy applied:** _(none yet — pending gate)_

**Checkpoint:** `experiments/checkpoint-${CHECKPOINT_N}/final/` → `apps/mac/raw_models/TrOCR-khmer/`

**Artifacts:** `results/mac_gate.csv` (gitignored if large; commit summary in this file only)

---

## Step 5: Commit

After recording PASS/FAIL:

```bash
git add training/eval/results/mac_gate.md training/eval/README.md apps/mac/README.md
# If Step 4c FAIL only:
# git add apps/mac/Sources/Preferences/AppPreferences.swift
git commit -m "$(cat <<'EOF'
docs(eval): record Mac CoreML OCR gate outcome and default policy
EOF
)"
```

Never commit `.mlpackage` weights, `raw_models/`, or `Vendor/trocr/*/`.
