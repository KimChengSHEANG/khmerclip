# Khmer OCR Accuracy + Speed Design

**Date:** 2026-08-27  
**Status:** Approved for planning (pending user review of this file)

## Goal

Ship a Khmer clip-text OCR path for the Mac app that **beats Tesseract CER** on a fixed printed + UI eval set and stays **≤ ~2× Tesseract median latency** on Mac. Stack is free: whatever wins becomes the default engine.

## Constraints

- **Product:** Mac clip-text OCR (region crop → text). One model must handle **UI/screen text** and **printed docs**.
- **Train:** Linux machine with **NVIDIA RTX 3090** (CUDA). Mac is for convert, bench latency, and shipping.
- **Train-time budget:** Weeks OK for a serious multi-dataset fine-tune.
- **Success bar:** Strictly lower CER than Tesseract on the frozen fixture set; median Mac CoreML (or chosen engine) latency ≤ 2× median Tesseract on the same set; mixed KM+EN subset CER not worse than Tesseract.
- **Fail behavior:** Keep Tesseract as default; still ship the improved neural model as a Preferences option and publish bench numbers.

## Out of scope (v1)

- Training a brand-new architecture from scratch
- EasyOCR as the primary ship path (latency vs Tesseract unlikely to pass)
- Cloud GPU beyond the operator’s 3090 box
- Rewriting text detection (input is the user’s clip crop)

## Architecture

Three phases:

1. **Eval harness** — Fixed fixtures + runners for Tesseract, current TrOCR (CoreML on Mac), EasyOCR. Produce CER + ms/clip. Tesseract defines the bar.
2. **Train** — Fine-tune TrOCR on the 3090 (prefer `microsoft/trocr-base-printed`; fall back to `trocr-small-printed` only if Mac CoreML latency fails the 2× gate). Reuse existing Khmer token graft and generation fixes in `training/trocr`.
3. **Ship** — `make trocr-khmer` → CoreML into `Vendor/trocr/khmer/`. Flip app default only if pass criteria hold on Mac after convert.

```
fixtures (printed + UI)
    → bench (Tess / TrOCR / EasyOCR)
    → baseline report

3090: train TrOCR multi-dataset
    → HF checkpoint CER (Linux)
    → copy to Mac → CoreML convert
    → Mac bench (CER + latency)
    → pass? default TrOCR : keep Tess default
```

## Components

| Unit | Location | Responsibility |
|------|----------|----------------|
| Eval fixtures | `training/eval/fixtures/` | PNGs + `labels.jsonl`; printed + UI-style crops |
| Manifest | `training/eval/fixtures/SHA256SUMS` | Freeze set; detect accidental edits |
| Bench CLI | `training/eval/bench.py` | Run engines → CSV + summary (CER, ms/clip) |
| Tess runner | wraps `apps/mac/Vendor/tesseract` | Same binary the app uses |
| TrOCR runner (Mac) | CoreML via existing app path or thin wrapper | **Latency gate measured here** |
| TrOCR runner (Linux) | `training/trocr/scripts/infer.py` / HF eval | Train-side CER during/after training |
| EasyOCR runner | `apps/mac/Vendor/easyocr/recognize.py` | Baseline only |
| Train | `training/trocr/scripts/train.py` | CUDA 3090 fine-tune |
| Export | `apps/mac` `make trocr-khmer KHMER_CKPT=…` | HF → CoreML |
| App default | Preferences / `OCREngineKind` | Flip after Mac pass |

## Data

### Eval fixtures (frozen bar)

- Size: about **100–300** crops.
- Mix: ~half printed (held-out / soyvitou-test style), ~half UI-style (system fonts, anti-alias, mixed Khmer + English + numerals).
- Labels: exact UTF-8 in `labels.jsonl` (one record per image).
- **No train leakage:** fixtures must not come from the train split used in Phase 2. Prefer soyvitou **test** (or equivalent held-out) plus hand-built / script-generated UI crops.
- Commit small fixtures + SHA256 manifest in git when size allows; otherwise commit generator script + checksums of the frozen set.

### Training data (3090)

- Datasets (existing registry): soyvitou, sokheng, hanuman, kiteaether; add gated `mrrtmob` only if Hugging Face auth is available.
- Augmentations: light blur, JPEG noise, contrast jitter, slight rotation — bias toward UI realism without destroying printed sharpness.
- Keep existing TrOCR Khmer token extension and generation-config fixes (`trocr_khmer.processor`, `khmer.py`).

## Training recipe

- **Primary base:** `microsoft/trocr-base-printed`.
- **Latency fallback:** `microsoft/trocr-small-printed` if converted CoreML median latency exceeds 2× Tesseract.
- **Optimizer / loop:** Existing `Seq2SeqTrainer` setup; CER as `metric_for_best_model`; `load_best_model_at_end`.
- **Duration:** Full multi-dataset run; 10+ epochs or until validation CER plateaus. Weeks on 3090 acceptable.
- **Checkpoints:** Keep best CER + recent checkpoints so a flat run can stop early.

## Pass / fail criteria

Ship as **default** only if all hold on the **Mac CoreML** bench of the frozen fixtures:

1. Aggregate CER **strictly below** Tesseract.
2. Median latency **≤ 2×** median Tesseract.
3. Mixed KM+EN subset CER **not worse** than Tesseract.

Otherwise: Tesseract remains default; improved TrOCR stays selectable; publish numbers under `training/eval/`.

## Testing

- Unit: CER helper on known pred/gt pairs; fixture loader fails closed on missing labels.
- Smoke: `bench.py --engine tesseract --limit 5` exits 0 and writes CSV.
- Train smoke: existing `--smoke_test` on 3090 before long runs.
- **Export gate:** After CoreML convert, re-run Mac bench (HF CER does not guarantee CoreML CER or latency).
- App: keep existing Tesseract fixture tests; do not put multi‑GB weights in CI.

## Risks

| Risk | Mitigation |
|------|------------|
| CoreML worse/slower than HF | Measure only after convert; fall back to `trocr-small` for latency |
| UI domain gap | Explicit UI fixtures + UI-oriented augs; add more synthetic UI data before more epochs if needed |
| Train/eval leakage | Held-out + hand UI only; document fixture sources in `training/eval/README.md` |
| Long train with no gain | Intermediate eval vs Tesseract gap; stop if plateau |

## Ops

1. Build/freeze fixtures; run baseline bench; commit report sketch under `training/eval/` (e.g. `baseline.md` once numbers exist).
2. Train on 3090; evaluate CER with HF on the same fixture labels where possible.
3. Copy `experiments/checkpoint-N/final/` to Mac (`raw_models/TrOCR-khmer/` or `KHMER_CKPT=`).
4. `make trocr-khmer` → Mac bench → decide default.
5. Update `apps/mac/README.md` and `training/eval/README.md` when default flips.

## Docs to update when implementing

- Create: `training/eval/README.md`
- Update: `training/trocr/README.md` (3090 train → Mac export flow)
- Update: `apps/mac/README.md` (default engine only after pass)
