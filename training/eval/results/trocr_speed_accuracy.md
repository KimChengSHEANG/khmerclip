# TrOCR CoreML speed + accuracy, by stage

Full frozen 200 fixtures, Mac CoreML via `OCRBench` (Release), same `bench.py`
run as Tesseract. Plan: `docs/superpowers/plans/2026-10-03-trocr-speed-accuracy.md`.
Printed fixtures leaked into this checkpoint's training (soyvitou testset), so
absolute printed CER is optimistic; ui fixtures are clean.

| Stage | CER | printed | ui | mixed | median ms | strips/img | enc ms | steps/img | ms/step | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Tesseract | 0.2580 | 0.5370 | 0.0199 | 0.0894 | 107.9 | — | — | — | — | — |
| 0 baseline (FP32, no KV) | 0.5577 | 0.1982 | 1.0514 | 0.0598 | 1367.2 | 1.1 | 36.6 | 17.2 | 173.1 | FAIL |
