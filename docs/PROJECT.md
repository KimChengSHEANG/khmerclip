# KhmerClip: Project Guide

This guide covers the whole repo: the macOS app, the Windows/Linux desktop app, the three OCR models, how each model is trained, the datasets and fonts behind them, how to evaluate them, and how to build and run everything. The subproject READMEs go deeper on each piece. This page shows how the pieces fit together and flags the problems that aren't obvious from the code.

Last verified against the tree on 2026-10-10 (commit `21a117b`, after merging `feat/trocr-speed-accuracy`, `feat/trocr-finetune-data`, `feat/desktop-windows-linux` and `feat/tesseract-finetune`).

---

## 1. What the project is

KhmerClip is a menu-bar app for macOS that runs OCR on any region of the screen. It is a TextSniper-style tool built mainly for **Khmer** text and for mixed Khmer/English text. Apple's Vision framework doesn't support Khmer (`VNRecognizeTextRequest` doesn't list it), so the app ships its own OCR engines. This repo contains the app, the training code for those engines, and an in-progress Windows/Linux port (§10).

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
├── Makefile                     `make check`: every test suite in the repo (§9.5)
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
│   ├── Tests/                   XCTest + Fixtures/ocr/{1..6}.png/.txt (+ PIL resize golden)
│   ├── tools/trocr/             CoreML converter: convert.py, kv_decoder.py, export_tokenizer.py, tests/
│   ├── Vendor/                  engines + models (only scripts/config in git, see §9.2)
│   └── raw_models/              Hugging Face checkpoints for CoreML conversion (gitignored)
├── apps/desktop/                Windows/Linux app (Python 3.12, PySide6, onnxruntime), in progress (§10)
├── training/
│   ├── trocr/                   TrOCR fine-tuning (HF Transformers, CUDA) + synthetic renderer
│   ├── tesseract/               Tesseract khm fine-tune (tesstrain in Docker, CPU)
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
| **Tesseract** | Vendored `tesseract` 5.x binary, `-l khm+eng --psm 6` | Pipeline exists in `training/tesseract`; smoke run didn't improve on stock (§3.3). Ships stock `tessdata_fast` | Vendored binary + tessdata | **App default.** Stable baseline (~75–110 ms median) |
| **TrOCR** | CoreML FP16 encoder + stateful KV-cache decoder, converted from a Hugging Face fine-tune | Yes, in `training/trocr` | Checkpoint → `make trocr-khmer` → rebuild | Optional. Mac gate **FAIL** on mixed CER only (§8); fine-tune v1 not yet re-gated with the KV decoder |
| **EasyOCR** | Python subprocess: CRAFT detector + custom Khmer CRNN | Yes, in `training/easyocr` | `make easyocr-venv easyocr-models` + `khmer.pth` | Optional |

Pick the engine in **Preferences → OCR Engine**. Since the 2026-10-03 Mac gate the default is `.tesseract` (`AppPreferences.swift`), so a fresh build works with no model setup.

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

The app still ships the stock upstream LSTM models. A fine-tune pipeline now lives in `training/tesseract/` (§3.3), but no fine-tuned model has passed its gate yet.

### 3.3 Fine-tune pipeline (`training/tesseract`, smoke only)

Spec: `docs/superpowers/specs/2026-10-03-tesseract-khmer-finetune-design.md`. Plan: `docs/superpowers/plans/2026-10-03-tesseract-khmer-finetune.md`. README: `training/tesseract/README.md`.

What's built:
- **`khm_tess` package.** `khm-export-gt` exports the cached HF line datasets to tesstrain ground truth (`.png` + `.gt.txt`). `khm-split-lists` writes the train/eval lists, with SoyVitou-valid as the eval list. Default sources (`DEFAULT_SOURCES` in `export_gt.py`): `soyvitou-train`, `soyvitou-valid:2000`, `sokheng`, `hanuman`, `kiteaether`, `darayut`, `siemreap:100000`, `mrrtmob:200000`. Override with `SOURCES="…"`.
- **Pinned container + Makefile.** `Dockerfile` (ubuntu:24.04, tesseract 5.3.4, tesstrain `405346a`, `tessdata_best/khm`) and targets `image check download gt dicts lists train logs evaluate`. It runs on the training server (`192.168.1.25`), CPU only. All generated data goes to `~/khm-tesstrain` (`WORK`), never under `~/Sync`.
- **Eval engine `tesseract-ft`** in `training/eval` (`--tesseract-ft-root`, `--tesseract-ft-lang`) to bench a candidate `traineddata` against stock.

**Smoke result (2026-10-03, `training/tesseract/results/finetune.md`):** 8.54 it/s at 8 CPUs. Eval BCER stayed at the start value of 49.851, so the "BCER strictly below start" go rule failed. The training log shows many `Encoding of string failed` errors and ZWSP (U+200B) in labels, and train BCER rose to ~98.8% by iteration 2000. The full run (`mrrtmob:200000`, 3 epochs, ~64 h estimated) is on hold until that is fixed.

| Setting | Value |
|---|---|
| Tooling | tesstrain pinned at `405346a3`, inside an `ubuntu:24.04` container with apt `tesseract-ocr` 5.3.x |
| Hardware | Operator server, capped at 8 CPUs (6 if TrOCR is training at the same time) |
| Start model | `START_MODEL=khm`, `MODEL_NAME=khm_ft` |
| Hyperparameters | `PSM=13`, `LEARNING_RATE=0.0001`, NORM_MODE 2 |
| Dictionary | Keep the stock `khm` word/punc/number dawgs, extracted with `dawg2wordlist` |
| Data | The cached Khmer line datasets (§6). Handwriting sets are excluded. SoyVitou is read by parquet file name, never `split="train"` |
| Label normalization | Strip outer whitespace and drop lines containing `\n`, `\r` or `\t`. No NFC |
| Checkpoint selection | Lowest BCER from `lstmeval` on the eval list. Fixtures are never used for selection |
| Ship gate | Mean CER lower, printed CER lower, mixed CER ≤, **UI CER ≤** (now in `passes_gate` for every candidate, §8). Benchmarked with both `khm+eng` and `khm` |

Rendering synthetic lines with `text2image` is deferred, and gets added only if UI CER regresses. To ship: copy the best `~/khm-tesstrain/data/khm_ft/tessdata_fast/<checkpoint>.traineddata` to the Mac, run the gate, and replace `apps/mac/Vendor/tesseract/tessdata/khm.traineddata` only on PASS.

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

Code: `training/trocr/` (package `trocr_khmer`, scripts `train.py`, `evaluate_model.py`, `infer.py`, `diagnose.py`, plus `render_synth.py`, `fetch_fonts.py`, `data_stats.py` and `remote.sh`).

### 4.1 Architecture and tokenizer

- **Base model.** Starts from `microsoft/trocr-base-printed`, a ViT encoder with a RoBERTa-style decoder. Pass `--model_name microsoft/trocr-small-printed` to train the smaller, faster checkpoint.
- **Byte-level tokenizer.** The decoder always uses the byte-level BPE tokenizer from `trocr-base-printed`. On `base` this is a no-op. On `small` it replaces the XLM-R SentencePiece tokenizer, which maps Khmer to `<unk>` and doesn't round-trip, and the embedding table and LM head are resized to fit.
- **Per-character Khmer tokens.** One token is added for each Khmer character (U+1780–17FF, U+19E0–19FF, and the zero-width separators). This shortens sequences, which otherwise cost 3 byte tokens per character, and lets the decoder learn whole characters. See `khmer.py:add_khmer_tokens`.
- **Label length.** `MAX_LABEL_LEN = 128` tokens (`dataset.py`). Labels longer than that are dropped during training instead of truncated. The Swift decoder matches it with `maxLength = 128` (`TrOCREngine.swift`).

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

Newer `train.py` options:
- **Weighted mixing.** `--datasets name:N` takes N rows of a source per epoch: a shuffled subset, or repeated copies when N is larger than the source.
- **Per-source holdout.** Each source gets its own seeded 3% holdout *before* resampling, so repeated rows never leak into eval. `--max_eval_per_source 300` keeps every domain in the eval set. Holdouts differ from runs before 2026-10.
- **Filters.** `--max_aspect` drops images too wide to read after the 384×384 resize.
- **Early stopping.** `--early_stopping_patience N`.
- **Continue from a finished run.** Pass its `final/` as `--model_name`. That starts a fresh optimizer and LR schedule and reuses the saved Khmer tokenizer, refusing to load if the vocab would change. Don't use `--resume_from_checkpoint` for this: it restores the old global step, so a finished run stops immediately.
- **Remote runs.** `scripts/remote.sh '<cmd>'` rsyncs `training/trocr` and `training/eval` to the 3090 box (`TROCR_HOST`, default `kim@192.168.1.25`) under the same `~/Sync` path and runs the command there.

### 4.3 Datasets

The default is `soyvitou`. Use `--datasets a b c` to concatenate several, or pass any HF hub id that exposes `image`/`text` columns, or a local directory (`imagefolder` with `metadata.jsonl` holding `file_name`, `text`) such as `cache/collected/<name>/`. See §6 for the registered sources.

> **Test-set leak (fixed 2026-10).** The `soyvitou` alias used to load `split="train"`, which on the Hub includes `testset.parquet`, where the printed eval fixtures come from. It now reads only `trainset.parquet` + `validset.parquet`. Checkpoints trained before the fix (including `checkpoint-3`, the init for fine-tune v1) still have inflated printed CER.

**Synthetic data.** `scripts/fetch_fonts.py` downloads 33 OFL Google Fonts Khmer files into `cache/fonts/`. `scripts/render_synth.py` renders lines from existing dataset labels in those fonts, with every eval fixture and UI string removed first. It renders handwriting-style fonts, native and sheared italics, scanned-document degradation, and light/dark UI panels into parquet shards under `cache/synth/{train,val}`, loaded as `synth` / `synth_val`. Rendering fails closed if Pillow lacks libraqm. `scripts/data_stats.py` reports label-length and aspect outliers for a source before you mix it in.

### 4.3a Fine-tune v1 (`experiments/checkpoint-4`)

Results: `training/eval/results/finetune_v1.md`, baseline in `finetune_baseline.md`. Gate: **PASS** vs init `checkpoint-3/final` (step 72k).

```bash
python scripts/train.py --model_name experiments/checkpoint-3/final \
  --datasets handwriting:60000 synth:400000 darayut:100000 siemreap:100000 soyvitou:60000 \
             sokheng:80000 hanuman:80000 kiteaether:60000 mrrtmob:200000 \
  --augment --epochs 3 --lr 2e-5 --warmup_steps 1000 \
  --max_eval_per_source 300 --max_eval_samples 0 --eval_steps 2000 \
  --early_stopping_patience 5 --max_aspect 24
```

Early-stopped at step 40,000 (best weights at step 30,000, mid-train CER 0.0228).

| Fixtures (HF, beam 4, n=200) | all | printed | ui | mixed |
|---|---:|---:|---:|---:|
| init `checkpoint-3` | 0.0542 | 0.0425 | 0.0586 | 0.0828 |
| **fine-tune v1** | **0.0064** | 0.0064 | 0.0021 | 0.0226 |

New-domain CER also dropped sharply: handwriting test 0.318 → 0.014, synth val 0.056 → 0.008, siemreap 0.279 → 0.076. Printed and mixed deltas are informational because the init trained on the leaked test set.

### 4.4 Shipping to the app (CoreML)

1. Copy `experiments/checkpoint-N/final/` to `apps/mac/raw_models/TrOCR-khmer/`. Locally this currently holds fine-tune v1 (`checkpoint-4`). The previous model is backed up as `raw_models/TrOCR-khmer-v0-step72k`.
2. Convert from `apps/mac`:
   - Run `make trocr-khmer`, or `make trocr-khmer KHMER_CKPT=path`.
   - The first run creates `.trocr-venv` with `torch==2.7.0`, `transformers==4.57.*`, `coremltools`, `pillow`, `pytest` and `jiwer`.
   - It then runs `tools/trocr/convert.py` and `tools/trocr/export_tokenizer.py`. They write `Vendor/trocr/khmer/encoder.mlpackage` (FP16), `decoder.mlpackage` (stateful KV-cache decoder, one token per call, built from `kv_decoder.py`) and `tokenizer.json`.
   - `make trocr-test` runs the converter tests on tiny random models: stateful decoder vs eager PyTorch over several steps.
3. Rebuild with `make build`. `make build` and `make test` reconvert automatically when `model.safetensors` **or any `tools/trocr/*.py`** is newer than the converted `tokenizer.json`. An export from before the KV-cache change crashes the app with "doesn't support stateful predictions", and this rule rebuilds it.

At runtime, `TrOCREngine` does the following:
- splits the image into line strips (`lineStrips`). A single Khmer line with detached marks stays one strip
- resizes each strip to 384×384 with a PIL-exact bilinear resize, with no content crop, to match training preprocessing
- greedy-decodes up to 128 tokens per line, with a fresh zeroed KV state per line. A repeat-run cap of 4 stops one token from looping to max length
- runs on CoreML compute units `.cpuAndGPU` by default. `KHMERCLIP_TROCR_UNITS=all|gpu|ane|cpu` overrides this for benchmarks. ANE fails to build a plan for the stateful decoder (error -14)
- caches compiled `.mlmodelc` files and preloads the model when a capture starts

### 4.5 Current measured state

Source: `training/eval/results/trocr_speed_accuracy.md`. Full 200 fixtures, Release `OCRBench`, same `bench.py` run as Tesseract. Checkpoint: v0 (`checkpoint-3`, step 72k).

| Stage | CER | printed | ui | mixed | median ms | ms/step |
|---|---:|---:|---:|---:|---:|---:|
| Tesseract | 0.2580 | 0.5370 | 0.0199 | 0.0894 | 107.9 | — |
| 0: old port (FP32, no KV) | 0.5577 | 0.1982 | 1.0514 | 0.0598 | 1367.2 | 173.1 |
| 1: preprocessing parity | 0.0593 | 0.0433 | 0.0662 | 0.0950 | 1286.8 | 148.5 |
| 2: KV cache FP16, units=gpu (shipped) | 0.0593 | 0.0433 | 0.0662 | 0.0950 | 95.3 | 7.5 |

Preprocessing parity brought CoreML CER to the PyTorch reference. The KV-cache decoder made it ~14× faster, now faster than Tesseract. The remaining gate failure is mixed CER (0.0950 vs Tesseract 0.0894).

Fine-tune v1 was gated on the Mac on 2026-10-04 with the *old* stateless decoder: CER 0.1700, mixed 0.0226, but median 1441 ms, so it failed on latency. **It hasn't been benchmarked with the KV-cache decoder yet.** Given its HF mixed CER of 0.0226, a KV re-gate is the obvious next step and could flip the default back to TrOCR.

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
| [SoyVitou/khmer-handwritten-dataset-4.2k](https://huggingface.co/datasets/SoyVitou/khmer-handwritten-dataset-4.2k) | `handwriting` (train+val), `handwriting_test` (test) | — | ~3.8k / ~0.4k | not checked | Real handwritten Khmer lines and words |
| [Darayut/khmer-document-synthetic-low-res](https://huggingface.co/datasets/Darayut/khmer-document-synthetic-low-res) | `darayut` | — | 100k | not checked | Low-res synthetic document lines (`label` column) |
| [KimkosalYon/khmer-ocr-200k-siemreap-arial](https://huggingface.co/datasets/KimkosalYon/khmer-ocr-200k-siemreap-arial) | `siemreap` | — | 200k | not checked | Siemreap/Arial font lines |
| local `cache/synth/{train,val}` | `synth`, `synth_val` | — | 400k / 3k | own render, OFL fonts | Multi-font synthetic from `render_synth.py` (§4.3) |

The Tesseract fine-tune (§3.3) reads the same HF cache by parquet file name, with its own aliases (`soyvitou-train`, `soyvitou-valid`, `darayut`, `siemreap`, `mrrtmob`, …).

Candidates listed in `training/trocr/docs/datasets.txt` but **not registered** yet:
- `lkhapple/KhmerOCR-Dataset`
- `sopheakvoatei/khmer-post-surya-ocr`
- `sopheakvoatei/khmer-post-surya-skipped-blocks-ocr`

### Data handling notes
- **Validation splits.** TrOCR carves a seeded 3% holdout per source before resampling (§4.2). EasyOCR uses each dataset's own splits where they exist, writing LMDBs under `data/lmdb/{train,val,test}/<name>/`.
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
| **TrOCR synthetic (`synth`)** | 33 OFL Google Fonts Khmer files (`fetch_fonts.py` → `training/trocr/cache/fonts/`) | Includes handwriting-style faces. Rendered with raqm shaping by `render_synth.py` |
| **Tesseract `khm`** | Upstream tessdata_fast model | Fonts are whatever upstream trained on. The fine-tune (§3.3) hasn't shipped |
| **Desktop app UI** | Noto Sans Khmer, bundled at `apps/desktop/resources/fonts/` (OFL) | Set on every text widget so Khmer renders on stock Windows/Linux |

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
- `tesseract-ft`: runs a candidate fine-tuned `traineddata` (`--tesseract-ft-root`, `--tesseract-ft-lang`, default `khm+eng`).
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

**Ship gate** (`khmerclip_eval.summary.passes_gate`). A candidate must meet all four conditions against Tesseract on the same run:
- mean CER < Tesseract
- median latency ≤ 2× Tesseract
- mixed CER ≤ Tesseract
- **UI CER ≤ Tesseract** (added with the Tesseract fine-tune; applies to every candidate, and `bench.py` now prints `ui_cer`)

**Mac CoreML gate status: FAIL (mixed CER only), recorded 2026-10-03.** The checklist and both recorded runs are in `training/eval/results/mac_gate.md`.

| Run | Checkpoint / decoder | CER | median ms | mixed | UI | Result |
|---|---|---:|---:|---:|---:|---|
| 2026-10-03 | v0 step 72k, KV-cache FP16, gpu | 0.0593 | 95.3 | 0.0950 | 0.0662 | FAIL: mixed (Tesseract 0.0894) |
| 2026-10-04 | fine-tune v1, old stateless decoder | 0.1700 | 1441.4 | 0.0226 | — | FAIL: latency |

The four-rule gate was added after these runs, so neither was checked against UI CER. v0's UI CER (0.0662) is above Tesseract's (0.0199), so v0 would fail that rule too.

| Gate outcome | Action |
|---|---|
| PASS | Set the default to `.trocr` |
| FAIL on latency | Retry with `trocr-small-printed` |
| FAIL on CER / mixed | Keep `.tesseract` as the default (applied 2026-10-03). TrOCR stays selectable |

```bash
cd training/eval
python -m pip install -e '.[dev]'
pytest
python scripts/bench.py --engine tesseract --out results/baseline_tesseract.csv
# CoreML gate (after building OCRBench, see mac_gate.md):
KHMERCLIP_VENDOR=$PWD/../../apps/mac/Vendor KHMERCLIP_TROCR_UNITS=gpu python scripts/bench.py \
  --engine tesseract --engine trocr-coreml \
  --trocr-coreml-bin ../../apps/mac/build/Build/Products/Release/OCRBench \
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

- `Vendor/trocr/convert.py` and `export_tokenizer.py`. These are **stale copies**: the Makefile now uses `tools/trocr/` (§12)
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
- **Engine.** Tesseract is the default and works with no further setup. TrOCR needs the converted models from §4.4.

| Target | Purpose |
|---|---|
| `make gen` | Regenerate `KhmerClip.xcodeproj` from `project.yml`. The project file is generated, so edit `project.yml` instead |
| `make build` | Debug build, reconverting TrOCR first if the checkpoint is newer |
| `make run` | Build, kill any running instance, re-register with LaunchServices, and open the app |
| `make test` | Run `KhmerClipTests`: Tesseract fixture regression, TrOCR line-strip/preprocess/tokenizer/end-to-end tests, formatter, geometry |
| `make clean` | Remove `build/` and the generated project |
| `make trocr-khmer [KHMER_CKPT=…]` | Convert the Khmer checkpoint to CoreML |
| `make trocr-test` | Converter unit tests (`tools/trocr/tests`), no checkpoint or network needed |
| `make ocrbench` | Release `OCRBench` CLI for the eval harness |
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
| `~/khm-tesstrain/.venv` | training server, via `UV_PROJECT_ENVIRONMENT` in `training/tesseract/Makefile` | Tesseract ground-truth export |
| throwaway uv envs (Python 3.12, uv-managed) | created by `make check` | Running each Python test suite in isolation |

From the repo root, `make build` builds every app that can be built on this machine. Today that's only the macOS app (`build-mac`, the same Debug build as `make -C apps/mac build`). `make build-desktop` exits with an error until the Windows/Linux GUI and packaging exist (§10). Those artifacts will come from CI, because PyInstaller can't cross-compile.

### 9.5 `make check` (all tests)

From the repo root, `make check` runs every test suite. `make -k check` keeps going past failures.

| Target | Runs |
|---|---|
| `check-eval` | `training/eval` pytest |
| `check-tesseract` | `training/tesseract` pytest |
| `check-trocr` | `training/trocr` pytest (adds `torch`) |
| `check-desktop` | `apps/desktop` pytest with the `dev` extra |
| `check-mac-tools` | `make -C apps/mac trocr-test` |
| `check-mac` | `make -C apps/mac test`. Reconverts the Khmer TrOCR model first if it's stale |

Each Python suite runs through `uv run --isolated --no-project --managed-python --python 3.12`. A stale editable install, such as `khmerclip_eval` pointing at a worktree, or the active pyenv venv can't leak packages in. Running bare `pytest` in your pyenv env can give different results.

---

## 10. Desktop app for Windows + Linux (`apps/desktop`, in progress)

Spec: `docs/superpowers/specs/2026-10-03-khmerclip-desktop-windows-linux-design.md`. Plan: `docs/superpowers/plans/2026-10-03-khmerclip-desktop-windows-linux.md`. `apps/mac` is untouched by this work.

**Stack.** Python 3.12 (`<3.13`), PySide6-Essentials 6.8, onnxruntime (CPU), `platformdirs`, `pynput` (hotkeys on Windows/X11), `jeepney` (Wayland portal). One codebase for both OSes.

**Planned v1:** tray app, region capture, OCR, auto-copy, result popup, history, preferences, image-file OCR, launch at login, TrOCR (ONNX, downloaded on first use), daily update check against the public `KimChengSHEANG/khmerclip-releases` GitHub repo. Out of v1: QR/barcode, EasyOCR, TTS, translate, window picker.

**Implemented so far (first 4 plan tasks):**

| Module | What it does |
|---|---|
| `khmerclip/paths.py` | Per-user config/data dirs via `platformdirs` |
| `khmerclip/prefs.py` | JSON-backed preferences (`prefs.json`), atomic UTF-8 writes |
| `khmerclip/history.py` | Scan history (`history.json`, max 200, newest first). A corrupt file moves to `.bak` instead of crashing |
| `khmerclip/download.py` | Checksum-verified, cancellable downloader (`.part` → SHA-256 → atomic rename) |
| `khmerclip/ocr/tesseract.py` | Tesseract subprocess (PNG via stdin, `-l khm+eng --psm 6`) with pinned tessdata |
| `khmerclip/formatter.py`, `errors.py` | Output formatting, error types |
| `tools/fetch_tessdata.py` | Fetches the pinned `tessdata_fast` files |

**Not built yet:** the Qt GUI (overlay capture, Wayland portal, tray, popup, windows), the TrOCR ONNX engine and export tool, the updater, autostart, hotkeys, packaging (PyInstaller, Inno Setup, AppImage) and the GitHub Actions workflow.

```bash
cd apps/desktop
uv run --isolated --no-project --python 3.12 --with-editable '.[dev]' pytest -q   # or: make check-desktop from the root
```

---

## 11. Training quick reference

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

For a fine-tune from a previous run and the synthetic data steps, see §4.2–4.3a. From a Mac, run any of this on the box with `scripts/remote.sh '<cmd>'`.

### Tesseract (training server, Docker, CPU)

```bash
cd ~/Sync/workspace/khmerclip/training/tesseract
make image check          # build + verify toolchain pins
make download             # fetch datasets not yet in the HF cache
make gt                   # export ground truth (DEFAULT_SOURCES, ~650k lines)
make dicts lists          # khm dictionary + .lstmf + train/eval lists
make train                # detached; `make logs` to follow, re-run to resume
make evaluate             # BCER per checkpoint on SoyVitou-valid
```

Use `CPUS=6` while a TrOCR run shares the box, and `MODEL=` for a separate run such as a smoke test.

### EasyOCR

See §5.3 for the full pipeline. For a quick smoke run, use `khmer-ocr-prepare --max-samples 2000` and then `khmer-ocr-train --num_iter 300 --batch_size 32 --valInterval 100`.

---

## 12. Known issues and technical debt

1. **Tesseract is not in git** (§9.2). A fresh clone has to rebuild the vendored binary (§3.4) before the default engine works.
2. **`OCRTests` baselines `5.txt` / `6.txt` record known Tesseract misreads.** They were hand-corrected in July, before the Aug 26 tessdata swap, and failed afterwards. On 2026-10-10 they were regenerated from the vendored engine, as the test's doc comment instructs. They now lock in `ta nv SUT លី…` for `ផ្អែកបន្ថែមលើ…` and `meniga` for `ភាសាខ្មែរ`. The test is a regression check, not an accuracy check. If a better `khm.traineddata` ships (§3.3), regenerate the baselines and expect these lines to change.
3. **TrOCR fine-tune v1 not yet gated with the KV-cache decoder** (§4.5). The default stays `.tesseract` until it is. The `raw_models/TrOCR-khmer` → `Vendor/trocr/khmer` conversion now produces v1 in KV format.
4. **Mixed Khmer+Latin CER** is what failed the v0 gate. v1 trained with more mixed data (HF mixed 0.0226), but that hasn't been confirmed on-device.
5. **Tesseract fine-tune blocked** (§3.3): the smoke BCER didn't improve, and the labels hit `Encoding of string failed` and contain ZWSP. Clean the labels before a full run.
6. **Stale `apps/mac/Vendor/trocr/convert.py` / `export_tokenizer.py`** are still tracked even though `tools/trocr/` replaced them. Delete them along with their `.gitignore` exceptions.
7. **Checkpoints before 2026-10 trained on the SoyVitou test set** (§4.3). The loader is fixed, but their printed CER, and v1's printed/mixed deltas vs init, are optimistic.
8. **Desktop app is partial** (§10). There's no GUI or packaging yet.
9. **`docs/superpowers/` is tracked again.** The `feat/trocr-finetune-data` merge removed its `.gitignore` line (added in `151bbd6`), so plans and specs are committed. Restore the line if they should stay local.
10. **Sparkle.** `SUFeedURL` is a placeholder (`https://example.com/...`), and there is no `SUPublicEDKey`. To ship updates, host an appcast, generate EdDSA keys, and sign releases.
11. **Distribution.** There is no Developer ID signing or notarization yet. See `apps/mac/README.md` for the archive → export → `notarytool` → `stapler` steps.
12. **`SWIFT_VERSION` is pinned to 5.0** to avoid strict-concurrency errors from the Translation framework.
13. **`apps/mac/README.md` is out of date in places.** Its layout lists `Hotkey/` and `OCRDispatcher`, which don't exist. Hotkeys use the MASShortcut package.

---

## 13. Further reading

| Topic | File |
|---|---|
| App details, features, troubleshooting | `apps/mac/README.md` |
| TrOCR training details | `training/trocr/README.md` |
| EasyOCR training details | `training/easyocr/README.md` |
| Eval harness and fixtures | `training/eval/README.md` |
| Mac gate checklist | `training/eval/results/mac_gate.md` |
| Accuracy/speed spec | `docs/superpowers/specs/2026-08-27-khmer-ocr-accuracy-speed-design.md` |
| TrOCR CoreML speed plan + results | `docs/superpowers/plans/2026-10-03-trocr-speed-accuracy.md`, `training/eval/results/trocr_speed_accuracy.md` |
| TrOCR fine-tune on new data | `docs/superpowers/plans/2026-10-03-trocr-finetune-new-data.md`, `training/eval/results/finetune_baseline.md`, `finetune_v1.md` |
| Tesseract fine-tune pipeline + smoke result | `training/tesseract/README.md`, `training/tesseract/results/finetune.md` |
| Desktop (Windows/Linux) spec + plan | `docs/superpowers/specs/2026-10-03-khmerclip-desktop-windows-linux-design.md`, `docs/superpowers/plans/2026-10-03-khmerclip-desktop-windows-linux.md` |
| Tesseract fine-tune spec + plan | `docs/superpowers/specs/2026-10-03-tesseract-khmer-finetune-design.md`, `docs/superpowers/plans/2026-10-03-tesseract-khmer-finetune.md` |
