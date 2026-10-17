# Accuracy v3: screen-capture evaluation and candidate gate (2026-10-17)

Plan: [`docs/superpowers/plans/2026-10-17-accuracy-v3.md`](../../../docs/superpowers/plans/2026-10-17-accuracy-v3.md).
Each candidate below was gated on paired per-fixture character errors (`scripts/compare.py`) and kept only if it helped.

## New evaluation sets (local only, not committed)

| Set | Built by | Clips | What it is |
|---|---|---:|---|
| `fixtures_screen/` `screen` | `build_screen_fixtures.py render` | 300 | 1-5 wrapped lines, 20-90 chars, Chrome-rendered |
| `fixtures_screen/` `screen_long` | same | 150 | one 60-120 char line |
| `fixtures_slices/` `slice_*` | `build_screen_fixtures.py slices` | 320 | single lines, 40 per style: plain, small (10-13 px 1x), bold, low contrast, coloured, display font, numbers, mixed |

- Text: Khmer Wikipedia (`20231101.km`). 1,907 candidate clips were dropped because a line shared a 16-char shingle with a training source's labels; the training sources contain a lot of Wikipedia text.
- Fonts: Khmer Sangam MN, Khmer MN, Khmer OS System / Battambang / Khmer OS (slices add Khmer OS Muol / Bokor / Metal Chrieng). None is in training; all stay eval-only.
- 12-20 css px at 1x and 2x, light and dark, 15% bold, line height 1.4-1.8.
- Labels are exact (rendered from text). Lines joined by a space, as OCRBench prints them.

`fixtures_screen/` and `fixtures_slices/` are gitignored: the text is CC BY-SA and the fonts aren't redistributable. Rebuild with the commands in the script docstring.

## Pipeline runner

`khmerclip_eval.pipeline` runs the app pipeline in PyTorch (Mac `lineStrips`, plain 384x384 resize, greedy, repeat cap 4, max 128 ids). `bench.py --engine trocr-pipeline --variant k=v,...` toggles candidates. Parity: on the frozen 200 it reproduces the shipped v2 (`checkpoint-5/checkpoint-10000`) ui 0.0032 and mixed 0.0133 exactly.

## Baseline: v2 on the screen set

| Subset | v2 CER (mean per clip) |
|---|---:|
| frozen 200 | 0.0081 |
| screen (1-5 lines) | 0.1953 |
| screen_long | 0.0398 |
| 1-line clips | 0.0286 |
| 2 / 3 / 4 / 5-line clips | 0.282 / 0.366 / 0.483 / 0.535 |

Cause: `lineStrips` found too few strips for 93 of 153 multi-line clips. At Khmer line heights the subscripts of one line touch the above-marks of the next, so no blank row separates them, and the merge rule then folded real lines into the merged double line as "small marks". Two to five lines were squashed into one 384x384 input.

## Candidates

Character errors, paired against the previous kept configuration. Sets: frozen 200, screen 450.

| Candidate | frozen errs | screen errs | +/- (screen) | Verdict |
|---|---:|---:|---:|---|
| (e) split touching lines at the line pitch | 40 -> 40 | 9394 -> 1835 (-80.5%) | 87 / 5 | **keep** (Mac, desktop) |

(e) splits runs of touching lines at the row-profile autocorrelation pitch, cutting only sparse valleys (<= 0.2 x the run's densest row), before the mark merge. Clips segmented correctly: 357 -> 439 of 450; frozen fixtures stay one strip each. Swift and desktop strip counts match the Python reference on all 650 fixtures.
