# Fine-tune baseline (starting checkpoints)

Recorded 2026-10-03 on the 3090 box, `bench.py --engine trocr-hf` (beam 4), frozen 200 fixtures.

| Model | all | printed | ui | mixed |
| --- | ---: | ---: | ---: | ---: |
| `checkpoint-3/final` (= step 72k, best mid-train CER 0.0381) | 0.0542 | 0.0425 | 0.0586 | 0.0828 |
| `last-step` (step 3,513,080, mid-train CER ~0.38) | 0.3015 | 0.1400 | 0.5132 | 0.1162 |
| Tesseract (baseline.md) | 0.2580 | 0.5370 | 0.0199 | 0.0894 |

Caveat: the old `soyvitou` alias trained on `testset.parquet`, which is where the printed fixtures
come from, so printed CER for both rows is optimistic.

**Init for fine-tuning:** `checkpoint-3/final` (lower `all` CER).
