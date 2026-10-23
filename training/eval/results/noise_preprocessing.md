# Noise-robust preprocessing: results

Spec: [`docs/superpowers/specs/2026-10-23-noise-robust-preprocessing-design.md`](../../../docs/superpowers/specs/2026-10-23-noise-robust-preprocessing-design.md).
Plan: [`docs/superpowers/plans/2026-10-23-noise-robust-preprocessing.md`](../../../docs/superpowers/plans/2026-10-23-noise-robust-preprocessing.md).
Every number is the PyTorch app pipeline (`bench.py --engine trocr-pipeline`, v3 step 16000, variant base `split_touching=1,split_wide_at=12,crop=1`) unless marked on-device. Error counts and CER are `scripts/compare.py`'s character errors (CER = errors / reference chars).

## Test sets

- `fixtures_noisy/` (local only): 300 Chrome-rendered clips from `build_screen_fixtures.py noisy`, 50 per noise kind. The text, fonts, sizes, 1x/2x and light/dark are as in the screen set. Labels are the text only.
- `fixtures_real/` (local only, never synced): the user's app captures, hand-verified. 4 of 6 verified (002-005). 001 (tiny, blurry) and 006 (uncertain heading) await the user's labels.

## Baseline (v3 + PR #5 preprocessing)

| Set | n | char errors | CER |
|---|---:|---:|---:|
| frozen 200 | 200 | 35 | 0.0122 |
| screen 450 | 450 | 716 | 0.0176 |
| slices 320 | 320 | 164 | 0.0107 |
| **noisy 300** | 300 | **4349** | **0.1541** |
| real 4 | 4 | 2 | 0.0072 (real-004, handwriting, only) |

| Noise kind | char errors | CER |
|---|---:|---:|
| edgecut | 1115 | 0.3948 |
| background | 1472 | 0.3213 |
| avatar | 976 | 0.1874 |
| icons | 355 | 0.0670 |
| chrome | 232 | 0.0469 |
| underline | 199 | 0.0371 |

Noise multiplies the error rate about 9x over clean screen clips (0.154 vs 0.018). Cut-off lines and photo/gradient backgrounds do the most damage.
