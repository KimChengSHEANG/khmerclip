# KhmerClip: Project Guide

This guide covers the whole repo: the macOS app, the three OCR models, how each model is trained, the datasets and fonts behind them, how to evaluate them, and how to build and run everything. The subproject READMEs go deeper on each piece. This page shows how the pieces fit together and flags the problems that aren't obvious from the code.

Last verified against the tree on 2026-10-10 (commit `d2f0d07`).

---

## 1. What the project is

KhmerClip is a menu-bar app for macOS that runs OCR on any region of the screen. It is a TextSniper-style tool built mainly for **Khmer** text and for mixed Khmer/English text. Apple's Vision framework doesn't support Khmer (`VNRecognizeTextRequest` doesn't list it), so the app ships its own OCR engines. This repo contains the app and the training code for those engines.

App features:
- drag-select screen capture
- capture a whole window (press Space while selecting)
- OCR an image file
- QR/barcode scanning (Vision)
- auto-copy to the clipboard
- a result HUD
- scan history
- Khmer text-to-speech
- on-device translation
- global hotkeys
- launch at login
- Sparkle auto-update (the update feed is still a placeholder)

### Repository layout

```
khmerclip/
├── README.md                    top-level pointer
├── .python-version              pyenv env name: "khmerclip"
├── apps/mac/                    macOS app (Swift 5, SwiftUI/AppKit, XcodeGen)
│   ├── project.yml              XcodeGen spec: targets KhmerClip, OCRBench, KhmerClipTests
│   ├── Makefile                 gen / build / run / test / trocr-* / easyocr-*
│   ├── Sources/
│   │   ├── App/                 entry point, AppDelegate (status item + wiring)
│   │   ├── Capture/             selection overlay, window picker, ScreenCaptureKit, HUD, clipboard
│   │   ├── OCR/                 OCREngine, TesseractEngine, TrOCREngine (+ tokenizer), EasyOCREngine, OCRTextFormatter
│   │   ├── Barcode/             Vision QR/barcode
│   │   ├── History/  Speech/  Translate/  Preferences/  Onboarding/  LaunchAtLogin/
│   │   └── Tools/OCRBenchMain.swift   CLI used by the eval harness (OCRBench target)
│   ├── Tests/                   XCTest + Fixtures/ocr/{1..6}.png/.txt
│   ├── Vendor/                  engines + models (only scripts/config in git, see §9.2)
│   └── raw_models/              Hugging Face checkpoints for CoreML conversion (gitignored)
├── training/
│   ├── trocr/                   TrOCR fine-tuning (HF Transformers, CUDA)
│   ├── easyocr/                 EasyOCR-compatible recognizer training (PyTorch, CUDA)
│   └── eval/                    frozen 200-image benchmark + CER/latency harness
└── docs/
    ├── PROJECT.md               this file
    └── superpowers/{specs,plans}/   design specs and implementation plans
```

---

## 2. The OCR models at a glance

| Engine | What runs in the app | Trained here? | Setup before it works | Status |
|---|---|---|---|---|
| **Tesseract** | Vendored `tesseract` 5.x binary, `-l khm+eng --psm 6` | No. Stock `tessdata_fast` models. A fine-tune is planned (§3.3) | Vendored binary + tessdata | Stable baseline, fastest (~75 ms median) |
| **TrOCR** | CoreML encoder + decoder converted from a Hugging Face fine-tune | Yes, in `training/trocr` | Checkpoint → `make trocr-khmer` → rebuild | App default. Mac gate **PENDING** |
| **EasyOCR** | Python subprocess: CRAFT detector + custom Khmer CRNN | Yes, in `training/easyocr` | `make easyocr-venv easyocr-models` + `khmer.pth` | Optional |

Pick the engine in **Preferences → OCR Engine**. The default is `.trocr` (`AppPreferences.swift`). If no converted TrOCR model is present, captures fail until you switch to Tesseract.

All three engines are wrapped behind `OCREngine` (`apps/mac/Sources/OCR/OCREngine.swift`). `OCRTextFormatter` post-processes the output.

---

## 3. Model 1: Tesseract

### 3.1 How the app uses it

`TesseractEngine.swift` runs `Vendor/tesseract/bin/tesseract` as a subprocess:

```
--tessdata-dir Vendor/tesseract/tessdata -l khm+eng --psm 6
```

- **`khm+eng`, not `khm` alone.** Stock `khm` can't read Latin script. With `khm` alone, mixed-subset CER rises to 0.557.
- **Self-contained binary.** About 14 dylibs are relocated with `dylibbundler` to `@executable_path/../lib/`, so no Homebrew install is needed at runtime.
- **Language models.** `khm.traineddata` and `eng.traineddata` come from [tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast). SHA-256 hashes are recorded in `training/eval/results/baseline.md`.

### 3.2 Training

None so far. The stock upstream LSTM models are used as-is.

### 3.3 Planned fine-tune (draft, not done)

Spec: `docs/superpowers/specs/2026-10-03-tesseract-khmer-finetune-design.md`. Plan: `docs/superpowers/plans/2026-10-03-tesseract-khmer-finetune.md`. Status: **Draft for review.**

| Setting | Planned value |
|---|---|
| Tooling | tesstrain pinned at `405346a3`, inside an `ubuntu:24.04` container with apt `tesseract-ocr` 5.3.x |
| Hardware | Operator server, capped at 8 CPUs (6 if TrOCR is training at the same time) |
| Start model | `START_MODEL=khm`, `MODEL_NAME=khm_ft` |
| Hyperparameters | `PSM=13`, `LEARNING_RATE=0.0001`, NORM_MODE 2 |
| Dictionary | Keep the stock `khm` word/punc/number dawgs, extracted with `dawg2wordlist` |
| Data | The cached Khmer line datasets (§6). Handwriting sets are excluded. SoyVitou is read by parquet file name, never `split="train"` |
| Label normalization | Strip outer whitespace and drop lines containing `\n`, `\r` or `\t`. No NFC |
| Checkpoint selection | Lowest BCER from `lstmeval` on the eval list. Fixtures are never used for selection |
| Ship gate | Mean CER lower, printed CER lower, mixed CER ≤, **UI CER ≤** (a new rule that applies to every candidate). Benchmarked with both `khm+eng` and `khm` |

Rendering synthetic lines with `text2image` is deferred, and gets added only if UI CER regresses.

### 3.4 Rebuilding the vendored Tesseract

```bash
brew install tesseract dylibbundler
cd apps/mac
mkdir -p Vendor/tesseract/{bin,lib,tessdata}
cp /opt/homebrew/bin/tesseract Vendor/tesseract/bin/tesseract
( cd Vendor/tesseract && dylibbundler -od -b -x ./bin/tesseract -d ./lib -p '@executable_path/../lib/' )
curl -L -o Vendor/tesseract/tessdata/khm.traineddata https://github.com/tesseract-ocr/tessdata_fast/raw/main/khm.traineddata
curl -L -o Vendor/tesseract/tessdata/eng.traineddata https://github.com/tesseract-ocr/tessdata_fast/raw/main/eng.traineddata
```

---

## 4. Model 2: TrOCR (Khmer fine-tune)

Code: `training/trocr/` (package `trocr_khmer`, scripts `train.py`, `evaluate_model.py`, `infer.py`, `diagnose.py`).

### 4.1 Architecture and tokenizer

- **Base model.** Starts from `microsoft/trocr-base-printed`, a ViT encoder with a RoBERTa-style decoder. Pass `--model_name microsoft/trocr-small-printed` to train the smaller, faster checkpoint.
- **Byte-level tokenizer.** The decoder always uses the byte-level BPE tokenizer from `trocr-base-printed`. On `base` this is a no-op. On `small` it replaces the XLM-R SentencePiece tokenizer, which maps Khmer to `<unk>` and doesn't round-trip, and the embedding table and LM head are resized to fit.
- **Per-character Khmer tokens.** One token is added for each Khmer character (U+1780–17FF, U+19E0–19FF, and the zero-width separators). This shortens sequences, which otherwise cost 3 byte tokens per character, and lets the decoder learn whole characters. See `khmer.py:add_khmer_tokens`.
- **Label length.** `MAX_LABEL_LEN = 128` tokens (`dataset.py`). The current Swift decoder caps output at `maxLength = 64` (`TrOCREngine.swift`). The speed/accuracy plan raises it to 128.

### 4.2 Training recipe (`scripts/train.py` defaults)

| Parameter | Value |
|---|---|
| Trainer | HF `Seq2SeqTrainer` |
| Epochs | 10 |
| Batch | 32 train × grad_accum 1, eval batch 64 |
| Learning rate | 5e-5, 500 warmup steps |
| Precision | bf16 (requires CUDA; the script asserts `torch.cuda.is_available()`) |
| Eval / save | every 2000 steps; mid-train CER is greedy and capped at 4096 samples |
| Best model | `metric_for_best_model="cer"`, `save_total_limit=3` |
| Validation split | 3% carved from the concatenated train data, `seed=42` |
| Augmentation (`--augment`) | Train split only. 50% of samples pass through clean; the rest get a random subset of Gaussian blur, contrast jitter, JPEG recompression and ±3° rotation |
| Target hardware | Ubuntu, i9-8950HK, 32 GB RAM, RTX 3090 24 GB (~18 GB at batch 32) |

Outputs go to `experiments/checkpoint-N/`. N auto-increments per run. The final model and the extended tokenizer are saved in `final/`.

### 4.3 Datasets

The default is `soyvitou`. Use `--datasets a b c` to concatenate several, or pass any HF hub id that exposes `image`/`text` columns. See §6 for details.

> **Known leak.** `load_ocr_datasets` loads `split="train"`. On the SoyVitou Hub card, `testset.parquet` is grouped under `train`, so TrOCR trains on SoyVitou's test set. That same test set is where the 100 printed eval fixtures come from. Printed-subset CER for TrOCR is therefore optimistic. The UI fixtures are clean. EasyOCR doesn't have this problem because it names parquet files explicitly.

### 4.4 Shipping to the app (CoreML)

1. Copy `experiments/checkpoint-N/final/` to `apps/mac/raw_models/TrOCR-khmer/`. A checkpoint is already present locally (`model.safetensors`, ~1.27 GB).
2. Convert from `apps/mac`:
   - Run `make trocr-khmer`, or `make trocr-khmer KHMER_CKPT=path`.
   - The first run creates `.trocr-venv` with `torch==2.7.0`, `transformers==4.57.*`, `coremltools` and `pillow`.
   - It then runs `Vendor/trocr/convert.py` and `export_tokenizer.py`, which write `Vendor/trocr/khmer/{encoder,decoder}.mlpackage` and `tokenizer.json`.
3. Rebuild with `make build`. `make build` and `make test` reconvert automatically when `model.safetensors` is newer than the converted `tokenizer.json`.

At runtime, `TrOCREngine` does the following:
- splits the image into line strips (`lineStrips`)
- resizes each strip to 384×384
- greedy-decodes with the CoreML decoder
- caches compiled `.mlmodelc` files
- preloads the model when a capture starts

### 4.5 Current measured state

Source: `docs/superpowers/plans/2026-10-03-trocr-speed-accuracy.md`. Measured on an M2 Max, on a 40-fixture sample (20 printed + 20 UI).

| Path | CER all | printed | UI | mixed | median ms |
|---|---:|---:|---:|---:|---:|
| Tesseract | 0.254 | 0.593 | 0.000 | 0.020 | 74 |
| CoreML app (current) | 0.339 | 0.179 | 0.510 | 0.100 | 1468 |
| HF PyTorch, beam 4 | 0.048 | 0.052 | 0.035 | 0.112 | 1034 |
| HF, greedy + repeat-cap 4 | 0.056 | 0.036 | 0.069 | 0.083 | — |

The checkpoint itself is good. The Swift/CoreML port loses accuracy and runs slowly. The plan fixes this without retraining:
- **Preprocessing parity:** no content crop, a PIL-exact resize, a 128-token cap, and a repeat cap.
- **Speed:** an FP16 encoder plus a stateful single-token decoder that keeps its KV cache (`ct.StateType`, macOS 15).
- **Tracked tooling:** the converter moves to a tracked `apps/mac/tools/trocr/`.

---

## 5. Model 3: EasyOCR custom Khmer recognizer

Code: `training/easyocr/` (package `khmer_ocr`). It vendors a port of the EasyOCR / clovaai *deep-text-recognition-benchmark* trainer.

### 5.1 Architecture

- **Recognizer.** `None-VGG-BiLSTM-CTC`, EasyOCR's "generation 2" network, with grayscale input `imgH=64`, `imgW=600`, padded rather than stretched. `batch_max_length=120`.
- **Charset.** 201 characters plus the CTC blank, for 202 output classes. `data/charset.txt` is generated from the dataset labels, includes the space character, and is the source of truth. The `character_list` in `khmer.yaml` must match it exactly.
- **Detection.** At inference, EasyOCR's stock CRAFT detector (`craft_mlt_25k.pth`) finds text regions before the recognizer runs.

### 5.2 Training recipe

| Parameter | `configs/khmer.yaml` | `configs/khmer_all.yaml` |
|---|---|---|
| Data | `soyvitou` | `soyvitou, sokheng, hanuman, km_en_line, gemma` |
| Batch ratio | `1` | `0.2` per source (equal sampling) |
| Iterations | 30,000 (~60 epochs of 62k at bs 128) | 100,000 |
| Val interval | 1000 | 2000 |

Shared settings:
- **Initialization.** Fine-tunes from `pretrained/english_g2.pth` with `FT: true` and `new_prediction: true`, so the CTC head is re-initialized for the Khmer charset. Training from scratch collapses to blank outputs.
- **Optimizer.** Adadelta (`adam: false`), `lr=1.0`, `rho=0.95`, `eps=1e-8`, grad clip 5.
- **Batch and seed.** Batch size 128, `manualSeed=1111`, 6 workers.
- **Decoding.** Greedy CTC.

Optional stronger variants: `FeatureExtraction: ResNet` and/or `Transformation: TPS`.

Outputs go to `experiments/model-N/`, which holds `best_accuracy.pth`, `best_norm_ED.pth`, `iter_*.pth` and the logs.

### 5.3 Pipeline

```bash
cd training/easyocr
pip install -e .
khmer-ocr-prepare                       # HF parquet -> LMDB + data/charset.txt (soyvitou)
khmer-ocr-prepare --datasets all        # or every source
mkdir -p pretrained && wget -O pretrained/english_g2.zip \
  https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip && unzip -o pretrained/english_g2.zip -d pretrained
khmer-ocr-train --config configs/khmer.yaml
khmer-ocr-eval --config configs/khmer.yaml --saved_model experiments/model-1/best_norm_ED.pth \
  --eval_data data/lmdb/test/soyvitou --save_preds preds_test.csv     # accuracy, CER, WER, norm-ED
khmer-ocr-package --config configs/khmer.yaml --saved_model experiments/model-1/best_norm_ED.pth --name khmer_g2
```

### 5.4 Shipping to the app

```bash
cd apps/mac
make easyocr-venv            # Vendor/easyocr/venv with PyTorch + EasyOCR (~1–2 GB)
make easyocr-models          # CRAFT weights -> Vendor/easyocr/models/
python3 Vendor/easyocr/install_khmer_model.py Vendor/easyocr/khmer   # khmer.pth -> models/, syncs yaml/py, rebuilds character_list
make build
```

Command-line check:

```bash
Vendor/easyocr/venv/bin/python3 Vendor/easyocr/recognize.py --resource-root Vendor/easyocr --model khmer path/to/image.png
```

The app uses `Vendor/easyocr/venv/bin/python3` when it is bundled, and otherwise falls back to `python3` on `PATH`.

---

## 6. Datasets

All training data comes from Hugging Face line-image datasets with `image` and `text` columns. The two training projects use **different aliases for the same repos**:

| HF repo | TrOCR alias | EasyOCR alias | Size | License | Content |
|---|---|---|---|---|---|
| [SoyVitou/62k-images-khmer-printed-dataset](https://huggingface.co/datasets/SoyVitou/62k-images-khmer-printed-dataset) | `soyvitou` (default) | `soyvitou` (default) | ~62.7k, with train/valid/test parquets | MIT | Printed Khmer + English lines |
| [Sokheng/khmer-synthetic-ocr-v1-100k](https://huggingface.co/datasets/Sokheng/khmer-synthetic-ocr-v1-100k) | `sokheng` | `sokheng` | 95k train / 5k test | CC-BY-4.0 | Synthetic receipts, menus, prices, signage. 70% single-line, **30% multi-line blocks with `\n` in labels** |
| [seanghay/khmer-hanuman-100k](https://huggingface.co/datasets/seanghay/khmer-hanuman-100k) | `hanuman` | `hanuman` | 100,001 train | not stated | Khmer lines |
| [mrrtmob/khmer_english_ocr_image_line](https://huggingface.co/datasets/mrrtmob/khmer_english_ocr_image_line) | `mrrtmob` | `km_en_line` | 12,138,214 (~40 GB). **Gated:** accept the terms and run `huggingface-cli login` | CC-BY-4.0 | Synthetic Khmer, English and mixed lines, 5–100 chars |
| [KiteAether/c_khmer_gemma_ocr_train](https://huggingface.co/datasets/KiteAether/c_khmer_gemma_ocr_train) | `kiteaether` | `gemma` | 7,468 train | not stated | Khmer OCR lines with Gemma-generated labels |

Candidates listed in `training/trocr/docs/datasets.txt` but **not registered** yet:
- `lkhapple/KhmerOCR-Dataset`
- `sopheakvoatei/khmer-post-surya-ocr`
- `sopheakvoatei/khmer-post-surya-skipped-blocks-ocr`

### Data handling notes
- **Validation splits.** TrOCR carves a 3% validation split with seed 42. EasyOCR uses each dataset's own splits where they exist, writing LMDBs under `data/lmdb/{train,val,test}/<name>/`.
- **Multi-line labels.** Sokheng's multi-line labels conflict with line-level recognizers. The Tesseract plan drops lines containing `\n`. TrOCR and EasyOCR currently keep them.
- **Normalization.** Don't apply Unicode normalization to labels. CER is computed on raw text.

---

## 7. Fonts

| Where | Font | Details |
|---|---|---|
| **Eval UI fixtures** (`training/eval/fixtures/ui/*`) | **Noto Sans Khmer** (Google Fonts, variable), stored at `training/eval/fonts/NotoSansKhmer-Regular.ttf` | OFL (`fonts/OFL.txt`). SHA-256 `f37a8431…28f2ca`. Rendered at 40 px by `scripts/build_ui_fixtures.py` from `ui_strings.txt`. Override with `KHMER_UI_FONT=/path/font.ttf` |
| **Sokheng dataset** | 49 Khmer TTF fonts (Battambang, Noto Sans Khmer, KhmerOS, Siemreap, …) | Per the dataset card: rendered with Pango + Cairo + HarfBuzz, 20–40 pt, paper/texture backgrounds, ±3° skew |
| **mrrtmob dataset** | "Multiple Khmer and English fonts" | The card names no fonts. Augmented with noise, blur and rotation |
| **SoyVitou, Hanuman, KiteAether** | Not documented upstream | The cards don't say which fonts or renderer were used |
| **Tesseract `khm`** | Upstream tessdata_fast model | Fonts are whatever upstream trained on. Not retrained here |

**Rendering caveat.** Khmer needs full OpenType shaping (coeng subscripts, vowel reordering). PIL with only basic layout produces garbage that doesn't look like real screen text. Make sure Pillow is built with **libraqm** before you regenerate the UI fixtures or render any training lines. The original Tesseract accuracy check used CoreText rendering for the same reason.

---

## 8. Evaluation harness (`training/eval`)

A frozen benchmark of 200 images is shared by all engines. `fixtures/labels.jsonl` lists each image with its expected text and subset, and `fixtures/SHA256SUMS` verifies the files.

| Subset | Count | Source |
|---|---:|---|
| printed | 89 | Evenly spaced rows of SoyVitou `testset.parquet` |
| ui | 88 | App-style strings (`ui_strings.txt`) rendered in Noto Sans Khmer |
| mixed | 23 | Any row whose text contains an ASCII letter or digit |

Metrics: per-fixture **CER** (via `jiwer`), averaged per subset, plus wall-clock **latency** per image.

**Runners**
- `tesseract`: runs the vendored binary.
- `trocr-hf`: runs a PyTorch checkpoint with beam 4 and the same generation settings as `infer.py`.
- `trocr-coreml`: runs the `OCRBench` Swift tool. The model is preloaded and per-image time comes from `recognizeText`.
- `easyocr`: runs the EasyOCR pipeline.

**Tesseract baseline (2026-08-27)**

| Subset | Mean CER | Median ms |
|---|---:|---:|
| All | 0.2580 | 76.3 |
| Printed | 0.5370 | 81.4 |
| UI | 0.0199 | 72.8 |
| Mixed | 0.0894 | 90.7 |

**Mac CoreML gate** (`khmerclip_eval.summary.passes_gate`). TrOCR CoreML must meet all three conditions:
- mean CER < Tesseract
- median latency ≤ 2× Tesseract
- mixed CER ≤ Tesseract

Status: **PENDING**. The checklist is in `training/eval/results/mac_gate.md`.

| Gate outcome | Action |
|---|---|
| PASS | Keep `.trocr` as the default |
| FAIL on latency | Retry with `trocr-small-printed` |
| FAIL on CER after the small-model fallback | Switch the default in `AppPreferences.swift` to `.tesseract` |

```bash
cd training/eval
python -m pip install -e '.[dev]'
pytest
python scripts/bench.py --engine tesseract --out results/baseline_tesseract.csv
# CoreML gate (after building OCRBench, see mac_gate.md):
KHMERCLIP_VENDOR=$PWD/../../apps/mac/Vendor python scripts/bench.py \
  --engine tesseract --engine trocr-coreml \
  --trocr-coreml-bin ../../apps/mac/build/Build/Products/Debug/OCRBench \
  --out results/mac_gate.csv --candidate trocr-coreml --compare-to tesseract
```

Regenerate the fixtures (requires network for the printed set):

```bash
python -m pip install -e . datasets
python scripts/build_printed_fixtures.py
python scripts/build_ui_fixtures.py
python scripts/write_manifest.py
```

---

## 9. Building and running the app

### 9.1 Prerequisites
- macOS 15+ (the deployment target) on Apple Silicon
- Xcode 16+ and the Command Line Tools (`xcode-select --install`)
- `brew install xcodegen`
- Python 3.12 only if you convert TrOCR models or use EasyOCR

### 9.2 What's in git under `Vendor/`

`apps/mac/.gitignore` tracks only the scripts and small config files in `Vendor/`:

- `Vendor/trocr/convert.py` and `export_tokenizer.py`
- `Vendor/easyocr/recognize.py`, `install_khmer_model.py` and `requirements.txt`
- `Vendor/easyocr/khmer/`: `charset.txt`, `khmer.yaml` and `user_network/`

Everything else under `Vendor/` must be generated locally:

| What | How to get it |
|---|---|
| Tesseract binary, libs and tessdata | §3.4 |
| TrOCR CoreML models (`Vendor/trocr/<size>/`) | `make trocr-khmer` (§4.4) |
| EasyOCR venv and CRAFT weights | `make easyocr-venv easyocr-models` |
| `khmer.pth` weights | Training (§5), then `install_khmer_model.py` |

Before 2026-10-10 the whole `Vendor/` directory was ignored by mistake, so older clones are missing these scripts.

### 9.3 Build and run

```bash
cd apps/mac
make run          # xcodegen generate + xcodebuild Debug + lsregister + open
```

- **Output.** The app is built to `build/Build/Products/Debug/KhmerClip.app`. It runs from the menu bar only (`LSUIElement`), with no Dock icon.
- **First launch.** The app shows onboarding. Grant **Screen Recording** in System Settings → Privacy & Security.
- **Engine.** For OCR with no further setup, switch to Preferences → OCR Engine → **Tesseract**. TrOCR, the default, needs the converted models from §4.4.

| Target | Purpose |
|---|---|
| `make gen` | Regenerate `KhmerClip.xcodeproj` from `project.yml`. The project file is generated, so edit `project.yml` instead |
| `make build` | Debug build, reconverting TrOCR first if the checkpoint is newer |
| `make run` | Build, kill any running instance, re-register with LaunchServices, and open the app |
| `make test` | Run `KhmerClipTests`: Tesseract fixture regression, TrOCR line-strip tests, formatter, geometry |
| `make clean` | Remove `build/` and the generated project |
| `make trocr-khmer [KHMER_CKPT=…]` | Convert the Khmer checkpoint to CoreML |
| `make trocr-models` | Convert small, base, large and khmer (large download, ~3.7 GB+) |
| `make easyocr-venv` / `make easyocr-models` | Set up the EasyOCR runtime and download CRAFT weights |

To build and run in Xcode instead: run `make gen && open KhmerClip.xcodeproj`, select the **KhmerClip** scheme, and press ⌘R. Set `SIGN_ID` in the Makefile to a stable signing identity so the Screen Recording/Accessibility grant survives rebuilds.

After any change to models or the venv, **rebuild**. `Vendor/` is copied into the app bundle as folder resources at build time.

### 9.4 Python environments

| Env | Where | Used for |
|---|---|---|
| pyenv `khmerclip` (Python 3.12) | repo root, via `.python-version` | EasyOCR training, eval harness |
| `training/trocr/venv` | Ubuntu + CUDA box | TrOCR training: `pip install uv && uv pip install torch torchvision -r requirements.txt -e .` |
| `apps/mac/.trocr-venv` | Mac, created automatically by the Makefile | CoreML conversion only (torch 2.7.0, transformers 4.57.*, coremltools) |
| `apps/mac/Vendor/easyocr/venv` | Mac, bundled into the app | EasyOCR runtime |

---

## 10. Training quick reference

### TrOCR (Ubuntu + RTX 3090)

```bash
cd training/trocr
python3 -m venv venv && source venv/bin/activate
pip install uv && uv pip install torch torchvision && uv pip install -r requirements.txt && uv pip install -e .
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python scripts/train.py --smoke_test --augment                         # 200 samples, 1 epoch
python scripts/train.py --datasets soyvitou sokheng hanuman kiteaether --augment --epochs 10
tensorboard --logdir experiments/checkpoint-1
python scripts/evaluate_model.py experiments/checkpoint-1/final --num_beams 4
python scripts/infer.py experiments/checkpoint-1/final line.png
pytest tests/
```

To resume a run, pass `--resume_from_checkpoint experiments/checkpoint-N/checkpoint-STEP`. On OOM, lower `--train_batch_size` / `--eval_batch_size` or raise `--grad_accum`.

### EasyOCR

See §5.3 for the full pipeline. For a quick smoke run, use `khmer-ocr-prepare --max-samples 2000` and then `khmer-ocr-train --num_iter 300 --batch_size 32 --valInterval 100`.

---

## 11. Known issues and technical debt

1. **Tesseract is not in git** (§9.2). A fresh clone has to rebuild the vendored binary (§3.4) before the out-of-the-box engine works.
2. **TrOCR trains on the SoyVitou test set** (§4.3). Printed fixture CER is inflated. Use `trocr_khmer.data_sources` with explicit parquet files, as EasyOCR does.
3. **The CoreML TrOCR port is slower and less accurate than the PyTorch checkpoint** (§4.5). A fix is planned, and the gate stays pending until then.
4. **Default engine is `.trocr`.** Captures fail on machines without converted models.
5. **Swift decoder `maxLength = 64` vs training `MAX_LABEL_LEN = 128`.** Long lines can be truncated.
6. **Sparkle.** `SUFeedURL` is a placeholder (`https://example.com/...`), and there is no `SUPublicEDKey`. To ship updates, host an appcast, generate EdDSA keys, and sign releases.
7. **Distribution.** There is no Developer ID signing or notarization yet. See `apps/mac/README.md` for the archive → export → `notarytool` → `stapler` steps.
8. **`SWIFT_VERSION` is pinned to 5.0** to avoid strict-concurrency errors from the Translation framework.
9. **`apps/mac/README.md` is out of date in places.** Its layout lists `Hotkey/` and `OCRDispatcher`, which don't exist. Hotkeys use the MASShortcut package.

---

## 12. Further reading

| Topic | File |
|---|---|
| App details, features, troubleshooting | `apps/mac/README.md` |
| TrOCR training details | `training/trocr/README.md` |
| EasyOCR training details | `training/easyocr/README.md` |
| Eval harness and fixtures | `training/eval/README.md` |
| Mac gate checklist | `training/eval/results/mac_gate.md` |
| Accuracy/speed spec | `docs/superpowers/specs/2026-08-27-khmer-ocr-accuracy-speed-design.md` |
| TrOCR CoreML speed plan | `docs/superpowers/plans/2026-10-03-trocr-speed-accuracy.md` |
| Tesseract fine-tune spec + plan | `docs/superpowers/specs/2026-10-03-tesseract-khmer-finetune-design.md`, `docs/superpowers/plans/2026-10-03-tesseract-khmer-finetune.md` |
