# Tesseract Khmer Fine-Tune Design

**Date:** 2026-10-03
**Status:** Draft for review

## Goal

Fine-tune Tesseract's Khmer LSTM model on the cached Khmer line datasets so the bundled `khm.traineddata` beats the current one on the frozen 200-fixture set. It must improve overall and printed CER without making the UI or mixed KM+EN subsets worse. Training runs on the operator's server `192.168.1.25`.

## Evidence (measured 2026-10-03 on the frozen fixtures)

| Config (vendored tessdata_fast `khm`, psm 6) | All | Printed | UI | Mixed |
| --- | ---: | ---: | ---: | ---: |
| `khm+eng` (current app) | 0.2580 | 0.5370 | 0.0199 | 0.0894 |
| `khm` only | 0.2434 | 0.3834 | 0.0199 | 0.5566 |
| `eng+khm` | 0.3194 | 0.6412 | 0.0540 | 0.0900 |

- Spacing isn't what drives printed errors: whitespace-insensitive printed CER is 0.5404, against 0.5370 raw.
- 24 of 89 printed predictions are Latin-only junk, because `eng` wins under `khm+eng`. Stock `khm` can't read Latin, which is why mixed CER is 0.5566 with `khm` alone.
- **Hypothesis:** a `khm` fine-tuned on Khmer + English line images reads both scripts. It can then win either alone or as `khm+eng`. The bench measures both settings.

## Constraints

- **Start model:** `tessdata_best/khm.traineddata` (float). SHA-256 `3d2d975e67a319bf1cd67626b12fdebc1e417a4ebef9c9afee01d9927b512815`, from tessdata_best commit `9ddc24e750eec0994223a9edc3fcb434a2244f3b`. The vendored model is tessdata_fast (SHA-256 `47f11057…`). It is integer-only and can't be fine-tuned.
- **Ship format:** convert the chosen checkpoint with `--convert_to_int`.
  - tessdata_fast models are usually a smaller network than tessdata_best, so an int-converted best network may be slower than the current model.
  - The smoke model has the final network, so the plan measures this on the Mac against the 2× latency gate before the long run.
- **Server:**
  - Ubuntu 22.04, i9-8950HK (12 threads), 31 GB RAM, 365 GB free.
  - Docker is available; sudo is not.
  - Tesseract training is CPU-only, so the RTX 3090 isn't used.
  - Training runs in a container (`ubuntu:24.04`, apt `tesseract-ocr` 5.3.x, which ships `lstmtraining`, `combine_tessdata`, `unicharset_extractor`, `merge_unicharsets` and `dawg2wordlist`). It is capped at 8 CPUs because the box also runs immich and open-webui. If the TrOCR fine-tune from `docs/superpowers/plans/2026-10-03-trocr-finetune-new-data.md` is running on the 3090 at the same time, use `CPUS=6` so its dataloaders keep enough cores.
- **Tooling:** tesstrain is pinned at commit `405346a3a67d8e4e049341d1da6a4b752e0b8351`. It uses NORM_MODE 2, the default for non-Indic, non-RTL scripts, which matches how upstream trained `khm`.
- **Work directory:** `~/khm-tesstrain` on the server, outside `~/Sync`. The repo there is Syncthing-synced, and hundreds of thousands of generated files would flood the sync.

## Data

All sources are line-level image + text parquet in `~/.cache/huggingface/hub` on the server. The first five are already cached there.

| Source | Files used | Rows | Use |
| --- | --- | ---: | --- |
| SoyVitou printed | `trainset.parquet` | all | train |
| SoyVitou printed | `validset.parquet` | 2000, stride-sampled | **eval list** (checkpoint selection) |
| SoyVitou printed | `testset.parquet` | — | **excluded**: the fixtures come from it |
| Sokheng synthetic | `data/train-*.parquet` | all | train |
| Sokheng synthetic | `data/test-*.parquet` | — | excluded |
| seanghay hanuman | `data/train-*.parquet` | all | train |
| KiteAether | `data/train-*.parquet` | all | train |
| mrrtmob KM+EN | `data/train-*.parquet` | 200,000, stride-sampled | train; the Latin source |
| Darayut low-res documents (`label` column) | `data/train-*.parquet` | all (100k) | train; downloaded by `make download` |
| KimkosalYon Siemreap/Arial KM+EN | `data/train-*.parquet` | 100,000, stride-sampled | train; downloaded by `make download` |

Excluded, with reasons:
- `SoyVitou/khmer-handwritten-dataset-4.2k`: handwriting. Tesseract here targets printed text and screen text, and handwriting lines would pull the model away from both.
- `lkhapple/KhmerOCR-Dataset`: image column only, no labels.
- `sopheakvoatei/khmer-post-surya-ocr`: 24 full pages with model-generated (Surya) pseudo-labels, not line ground truth.
- `sopheakvoatei/khmer-post-surya-skipped-blocks-ocr`: 76 picture blocks with mostly empty text.

**Leakage rules:**
- Drop any line whose stripped text exactly matches a `training/eval/fixtures/labels.jsonl` text or a `training/eval/ui_strings.txt` line.
- Never read SoyVitou through `load_dataset(split="train")`: the Hub card groups `testset.parquet` into `train`.

**Label normalization:** strip outer whitespace. Drop lines that are empty or contain `\n`, `\r` or `\t`. Leave everything else as is: no NFC and no collapsing of spaces, because CER is computed on raw text.

## Recipe

- tesstrain fine-tune: `START_MODEL=khm`, `MODEL_NAME=khm_ft`, `PSM=13`, `LEARNING_RATE=0.0001` (the tesstrain fine-tune default).
- Keep the start model's dictionary: extract `khm.lstm-word-dawg`, `khm.lstm-punc-dawg` and `khm.lstm-number-dawg` into tesstrain's wordlist, punc and numbers files with `dawg2wordlist`.
- Replace tesstrain's random 90/10 split with:
  - `list.eval` = the exported SoyVitou-valid lines (at most 2000)
  - `list.train` = everything else
- Iteration budget: set it from the smoke run's measured rate. The default is `EPOCHS=3`; reduce the mrrtmob sample if that would exceed 4 days.
- Checkpoint selection: run `lstmeval` on `list.eval` for every checkpoint and take the lowest BCER. Fixtures are never used to select.

## Gate (Mac, vendored binary, frozen fixtures)

The candidate is `tesseract-ft`: the vendored binary with the fine-tuned `khm`. It ships only if, against the current `tesseract`:

1. Mean CER is strictly lower.
2. Median latency is ≤ 2×.
3. Mixed-subset CER is ≤.
4. **UI-subset CER is ≤.** This is a new gate rule, and it applies to every candidate, including TrOCR.

The bench runs with both `--tesseract-ft-lang khm+eng` and `--tesseract-ft-lang khm`. If both pass, the language setting with the lower mean CER ships. If it is `khm`, the app's `-l` changes to `khm`.

**Fail behavior:** keep the current vendored `khm`, and record the numbers in `training/tesseract/results/finetune.md`.

## Out of scope

- Training from scratch, or changing the network spec.
- Synthetic rendering with `text2image`. Add it only if UI CER regresses.
- Retraining `eng`.
- Fixing the TrOCR SoyVitou train/test overlap in `training/trocr/src/trocr_khmer/data_sources.py`. It is noted for the TrOCR track.
