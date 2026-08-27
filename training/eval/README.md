# Khmer OCR evaluation

This harness measures character error rate (CER) and latency on a frozen set of
printed and app-style Khmer text images.

## Fixtures

The checked-in set contains 100 printed samples and 100 rendered UI samples.
Printed rows are evenly spaced across the 6,269 rows in
`SoyVitou/62k-images-khmer-printed-dataset/testset.parquet`. The Hub dataset
metadata currently groups every parquet under `train`, so the builder names
`testset.parquet` directly and loads it as `split="test"` to prevent train-set
leakage.

Each `fixtures/labels.jsonl` row has `id`, relative `image`, expected `text`,
and `subset`. Source metadata is included so the selected rows and UI rendering
can be audited. A row is `mixed` when its text contains an ASCII letter or
digit; otherwise it is `printed` or `ui`.

The UI fixtures use the OFL-licensed
[Google Fonts Noto Sans Khmer](https://github.com/google/fonts/tree/main/ofl/notosanskhmer)
variable font, stored as `fonts/NotoSansKhmer-Regular.ttf`:

```text
f37a8431a0c5d5ed2f81a767417546aca576a81fb7eff9c924d46aecf828f2ca
```

Set `KHMER_UI_FONT` to use another compatible font when regenerating locally.

## Regenerate

From `training/eval`, install the harness plus the fixture-only dependency and
then rebuild:

```bash
python -m pip install -e . datasets
python scripts/build_printed_fixtures.py
python scripts/build_ui_fixtures.py
python scripts/write_manifest.py
wc -l fixtures/labels.jsonl
(cd fixtures && shasum -a 256 -c SHA256SUMS)
```

The printed builder requires network access to Hugging Face. The output is
deterministic for the pinned source parquet content, Pillow version, font, and
rendering platform. Run the manifest writer after either builder.

## Benchmark

From `training/eval`, run the vendored Tesseract baseline:

```bash
python scripts/bench.py \
  --engine tesseract \
  --out results/baseline_tesseract.csv
```

The CSV contains per-fixture predictions, references, CER, and elapsed
milliseconds. Aggregate results and environment details are recorded in
`results/baseline.md`.

### TrOCR (Hugging Face)

Run TrOCR-HF benchmarks from the CUDA training venv so `trocr_khmer` and
PyTorch are available. Install the eval harness editable inside that venv:

```bash
cd training/trocr
pip install -e ../eval
python ../eval/scripts/bench.py \
  --engine tesseract \
  --engine trocr-hf \
  --trocr-model-dir experiments/checkpoint-N/final \
  --out ../eval/results/current_trocr_hf.csv \
  --candidate trocr-hf \
  --compare-to tesseract
```

Generation settings match `training/trocr/scripts/infer.py`: beam search with
`num_beams=4`, `no_repeat_ngram_size=0`, and `length_penalty=1.0`.

## Mac CoreML gate

After the 3090 training run (Task 8) produces a checkpoint, run the on-device
CoreML gate on a Mac: copy `final/` into the app, convert with `make
trocr-khmer`, bench `trocr-coreml` vs Tesseract on the full fixture set, and
record PASS/FAIL.

**Operator checklist:** [results/mac_gate.md](results/mac_gate.md) (currently
**PENDING** — no trained checkpoint yet).

The gate uses `OCRBench` (Task 7) and `passes_gate()` from
`khmerclip_eval.summary`: TrOCR must beat Tesseract on mean CER, stay within
2× median latency, and not regress on mixed-script CER. Outcome controls whether
the macOS app keeps TrOCR as the default engine or falls back to Tesseract.
