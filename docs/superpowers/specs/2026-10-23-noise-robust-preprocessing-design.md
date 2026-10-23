# Noise-robust preprocessing: design

Date: 2026-10-23. Status: draft for approval.

## Why

People capture text straight off the screen, and a selection rarely holds only text. Icons, emoji, buttons, borders, underlines, table lines, avatars, photo or gradient backgrounds, and half-cut lines at the selection edge come along with it. The OCR model (TrOCR v3) reads clean lines well (0.34% CER on the frozen 200, 1.6% on Chrome-rendered screen clips after the PR #5 fixes). But the preprocessing in front of it treats everything that differs from the background as text. That noise then:

- **merges or mis-splits lines:** a border or icon bridges two lines, so the row-ink profile has no gap;
- **inflates the content crop:** an icon at the far end of a line stretches the crop, so the text is smaller in the 384×384 input;
- **gets read as characters:** a fragment of a cut-off line, or an icon inside a strip, comes out as junk text.

A real example from 2026-10-23: pale "Fri 23 Oct" on the macOS lock-screen photo came out as "Fri 23.06t".

Today's background estimate is the mean of the clip's four corner pixels (`ink_mask` in `apps/desktop/src/khmerclipper/ocr/preprocess.py`, `inkMask` in `apps/mac/Sources/OCR/TrOCREngine.swift`, `ink_mask` in `training/eval/src/khmerclipper_eval/pipeline.py`). Any pixel more than `THRESHOLD` away from it in any channel counts as ink. One icon in a corner, or a photo background, breaks this.

The fix is image processing in the apps, with no retraining. Retraining on more synthetic screen data (v4, 2026-10-23) didn't move the remaining errors (`training/eval/results/accuracy_v3.md`).

## Goals

1. A **noisy screen test set** that measures the problem. Without it, no change below can be shown to help.
2. Preprocessing that **ignores non-text marks** when it splits lines and crops, so clean-text accuracy holds and noisy-capture accuracy improves.
3. The **same behaviour in all three implementations**: Mac (Swift), desktop (Python), and the eval pipeline (Python), as for the line-split fixes. Strip counts must match on every fixture.

## Non-goals

- Retraining or changing the model.
- Layout analysis: reading order across columns, tables, or multi-block pages.
- OCR of text inside images or icons, such as logo lettering.
- Changing what the model sees inside a line strip beyond removing noise. No global binarization, sharpening, or forced black-on-white: the model was trained on coloured, anti-aliased text, so that is likely to hurt.

## Anti-goals: a failure even if it "works"

- **Any regression on clean text** beyond the bars below. Most captures are clean.
- **Dropping real text.** Khmer has detached marks above and below the base line (the reason `mergeMarks` exists), small vowel signs, and the ១–៩ digits. A filter that removes these, or a short last line, is worse than leaving the noise in.
- **The implementations drifting apart:** a rule present on Mac but not on desktop, or different thresholds.
- **Noticeable extra latency.** Median capture time should not grow by more than 10% on any set.

## Constraints

- **Ship check per change.** Each step below is kept only if, paired against the previous kept configuration:
  - noisy-set character errors drop by at least 5%, with more clips better than worse;
  - frozen-200 character errors rise by at most 3;
  - screen-450 and slices-320 character errors rise by at most 1%;
  - median latency on each set rises by at most 10%.

  This is the bar the v3 work used, plus the noisy set. Measure with the PyTorch pipeline runner (`bench.py --engine trocr-pipeline`, same variant flags as PR #5), then confirm the final configuration on device with `OCRBench` (CoreML) and the desktop ONNX path.
- **Parity.** Python eval pipeline = desktop = Swift: identical strip counts on all fixtures, including the noisy set, as checked for the two-line cut.
- **Test-set licensing.** The noisy set stays local and gitignored, like `fixtures_screen/`: it uses Wikipedia text (CC BY-SA) and non-redistributable macOS fonts. Background photos must be generated or freely licensed; never commit them.
- **Generated test images must look like real screens:** rendered by headless Chrome like `build_screen_fixtures.py`, not composited in PIL.

## Decided

### Step 1: noisy test set (`build_screen_fixtures.py noisy`)
A new subcommand of `training/eval/scripts/build_screen_fixtures.py` renders about 300 clips into `training/eval/fixtures_noisy/`. Text, fonts, sizes, 1×/2× and light/dark come from the screen set. Labels are the text only, with noise excluded. Six noise kinds, about 50 clips each, and one kind per clip so results can be broken down:

| Kind | What it is |
|---|---|
| `icons` | emoji or simple SVG icons before, after or between words (list bullets, app icons, a checkmark) |
| `chrome` | text inside buttons or input boxes, or with a 1–2 px border or divider line above or below |
| `underline` | underlined links, plus table rows with grid lines |
| `background` | text on a CSS gradient, or on a generated or free-licence photo-like texture, including low contrast |
| `edgecut` | the selection slices through the line above and/or below, leaving partial glyphs at the top and bottom edges |
| `avatar` | a chat-style row: a round avatar or image block left of 1–3 lines of text |

The CER of each noise kind under v3 + PR #5 is recorded as the baseline before any change.

### Steps 2–5, in this order, each through the ship check
1. **Robust background and ink estimate** (step 2). Replace the four-corner mean with a background estimate that survives a noisy corner and non-uniform backgrounds. One example: the most common colour on the clip's border; per-strip estimation is also allowed. Text darker or lighter than its background must both work.
2. **Remove non-text components before line splitting** (step 3). Label connected components of the ink mask and drop those that can't be text: long thin horizontal or vertical lines (borders, underlines, grid lines), large solid blocks (images, avatars, buttons), and isolated blobs far from every text row. Then build the row profile from what remains. Khmer marks, vowel signs and digits must survive. The frozen 200 and slices are the check for this.
3. **Drop cut-off fragments at the selection edges** (step 4). Strips that touch the top or bottom edge and are much shorter than the clip's real lines are discarded rather than decoded.
4. **Fill removed noise with background before decoding** (step 5). Where a removed component overlaps a strip's crop, paint it with the local background colour so the model never sees it.

A step that fails the ship check is dropped and recorded with its numbers. The next step is measured against the last kept configuration.

### How the ship check is applied (builder's calls, made at design time)
- **Enabling steps.** Step 2 may be kept even if it alone doesn't cut noisy-set errors by 5%, provided it causes no regression on any set. Steps 2 and 3 can then be checked together as a bundle. Constants of a kept step may be retuned while a later step is developed; the check then re-runs on the bundle.
- **Which clips the 5% bar counts.** The bar applies to the noise kinds a step targets, with no regression on the other kinds:

  | Step | Targeted kinds |
  |---|---|
  | 2 | `background` |
  | 3 | `chrome`, `underline`, `avatar`, `icons` |
  | 4 | `edgecut` |
  | 5 | `icons`, `chrome` |

  Whole-set numbers are reported too.
- **Binding measurement.** Keep/drop for accuracy uses the PyTorch pipeline runner. The Swift and desktop ports, and the on-device latency check (`OCRBench` and the desktop ONNX path), run once on the final bundle. If on-device latency misses the 10% bar, profile and optimise first (for example, run component labelling on the downscaled scan mask); drop the most expensive step only if that fails.
- **Parity.** Beyond strip counts, the final strip rectangles and content-crop boxes must match within 1 px across the eval pipeline, desktop and Swift on every fixture. A small dump mode, such as an `OCRBench` flag that prints boxes, provides the Swift side. Implement components with plain arrays in all three (no scipy or OpenCV); the desktop app has only numpy.
- **Latin fallback.** The Tesseract fallback for Latin-only lines gets the same cleaned strip the model gets. The noisy set includes some Latin-only lines in `background` and `chrome`, and the on-device check covers the fallback path, which the PyTorch runner leaves out.

### Result record
Results go in a new `training/eval/results/noise_preprocessing.md`, in the format of `accuracy_v3.md`: a table per step, per-kind breakdown on the noisy set, and keep/drop with reasons.

## Left to the builder

- Exact algorithms and thresholds inside each step, such as the background estimator, component size and aspect cutoffs, and fragment height ratio. They must be fixed constants, the same on every platform, and justified by the numbers.
- How the icons and textures are produced, within the licensing constraint.
- Exact counts per noise kind, roughly 50 each.
- Whether step 5 is worth doing at all if step 3 already captures most of the gain. Measure first.
- Code structure. Follow the existing pattern: Python functions in `preprocess.py` and `pipeline.py` mirror Swift static functions in `TrOCREngine.swift`, each with one unit test per platform on a synthetic profile or image.

## Real captures (user approved, 2026-10-23)

The user's saved captures (`~/Library/Application Support/KhmerClipper/captures`, 6 so far) are hand-labelled and copied into `training/eval/fixtures_real/`, the subset name `fixtures.py` already reserves for app captures. They stay on this Mac only: gitignored, never pushed, never synced to the 3090 box. `training/eval/.gitignore` already lists `fixtures_real/`. Add `fixtures_noisy/` to it, and add `--exclude fixtures_real` to the rsync in `training/trocr/scripts/remote.sh`, which today copies all of `training/eval` to the box. Six clips are too few for a percentage bar, so they are a sanity check, not a gate. Report per-clip results for every step, and treat a clear regression on a real clip as a reason to stop and look before keeping a step. Add more as the user captures more.
