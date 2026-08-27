# Tesseract baseline

Recorded 2026-08-27 using the frozen 200-fixture manifest.

| Subset | Count | Mean CER | Median latency |
| --- | ---: | ---: | ---: |
| All | 200 | 0.2580 | 76.3 ms |
| Printed | 89 | 0.5370 | 81.4 ms |
| UI | 88 | 0.0199 | 72.8 ms |
| Mixed | 23 | 0.0894 | 90.7 ms |

Command:

```bash
python scripts/bench.py \
  --engine tesseract \
  --out results/baseline_tesseract.csv
```

Environment:

- macOS 26.6.2 (25G83), Apple Silicon
- Python 3.12.7
- Tesseract 5.5.3, Leptonica 1.87.0
- OCR language order: `khm+eng`; page segmentation mode: `6`
- `khm.traineddata` SHA-256:
  `47f110575341b322052f3becbefee61a3ecf1ef549352b5d4d33d28afe30d099`
- `eng.traineddata` SHA-256:
  `7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2`

Latency is wall-clock time around each vendored Tesseract subprocess and is
machine-dependent. CER is computed per fixture and then averaged.
