# Noise-Robust Preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how screen noise (icons, borders, underlines, photo backgrounds, cut-off lines, avatars) hurts OCR, then add preprocessing steps that ignore it, keeping only steps that pass the ship check.

**Architecture:**
- Each step is first built in the eval pipeline (`training/eval/src/khmerclipper_eval/pipeline.py`) behind a `--variant` flag and checked with `bench.py --engine trocr-pipeline` plus `compare.py`.
- Only kept steps are then ported to the desktop app (`apps/desktop/src/khmerclipper/ocr/preprocess.py`) and the Mac app (`apps/mac/Sources/OCR/TrOCREngine.swift`).
- Parity is checked by dumping strip and crop boxes from all three.

**Tech Stack:** Python 3.12+, numpy and Pillow (no scipy or OpenCV); Swift 5 with CoreGraphics; headless Chrome for test images; PyTorch TrOCR runner; CoreML `OCRBench`; desktop ONNX.

**Spec:** `docs/superpowers/specs/2026-10-23-noise-robust-preprocessing-design.md`

## Global Constraints

- **Ship check per step, paired against the last kept configuration:**
  - the noisy kinds the step targets: character errors −5% or better, with improved > worsened;
  - the other noisy kinds: no increase;
  - frozen 200: at most +3 errors;
  - `fixtures_screen` and `fixtures_slices`: at most +1% errors;
  - median latency: at most +10% on every set.
- **Targeted kinds:** step 2 `noisy_background`; step 3 `noisy_chrome`, `noisy_underline`, `noisy_avatar`, `noisy_icons`; step 4 `noisy_edgecut`; step 5 `noisy_icons`, `noisy_chrome`.
- **Enabling step:** step 2 may be kept with no gain if nothing regresses. Steps 2 and 3 may then be checked as a bundle, and kept constants may be retuned (re-check the bundle).
- **Keep/drop is decided with the PyTorch pipeline runner.** Variant base for every run: `split_touching=1,split_wide_at=12,crop=1`. On-device checks (`OCRBench` and desktop ONNX) run once, on the final bundle. If latency is over, optimise before dropping a step.
- **Parity:** strip rectangles and content-crop boxes match within 1 px across eval pipeline, desktop and Swift on every fixture.
- **Thresholds:** fixed constants, the same on every platform, each with a comment naming the measurement that set it.
- **Never drop real text:** Khmer above/below marks, vowel signs and ១–៩ digits survive every filter. The frozen 200 and slices check this.
- **Not allowed:** global binarization, sharpening, or forced black-on-white of what the model sees.
- **Privacy:** `fixtures_noisy/` and `fixtures_real/` stay local. Both are gitignored, and `fixtures_real/` is excluded from `remote.sh`.
- **Photos:** background photos are generated or free-licence, and never committed.

## Review Focus

1. **A clip with no noise at all** (most real captures) must give byte-identical strips with every new flag on. Each step's tests include a clean-clip identity test.
2. **Light text on a dark or photo background:** the background estimate must not pick the text colour when glyphs touch a large share of the border, as in a tight crop of bold text. Task 3 tests a dark clip with light glyphs on 40% of its top and bottom rows.
3. **A capture that is only an icon, image or line, with no text:** the filters must not crash or return an empty strip list. Return `[image]` as today. Tasks 4 and 5 test an all-noise image.
4. **Khmer subscript or above-marks touching the selection edge:** the edge-fragment rule must not drop the only line or the real first/last line. Task 5 tests a single-line clip whose marks touch the top edge.
5. **Very small 1x text (10–11 px):** whole Khmer glyphs are only a few pixels, so size cutoffs scaled for 2x must not delete them. Task 4 tests a 1x synthetic line of 3×5-px blobs.

---

## File Structure

| File | Change |
|---|---|
| `training/eval/scripts/build_screen_fixtures.py` | `noisy` subcommand, `noisy_jobs()`, `noisy_page()` |
| `training/eval/src/khmerclipper_eval/fixtures.py` | allow `noisy_*` subsets |
| `training/eval/.gitignore`, `training/trocr/scripts/remote.sh` | ignore `fixtures_noisy/`; don't sync `fixtures_real/` |
| `training/eval/src/khmerclipper_eval/pipeline.py` | `background_color`, `text_mask`, `drop_edge_fragments`, `fill_removed`, plus flags on `line_strips` / `content_crop` / `recognize` |
| `training/eval/scripts/bench.py` | new boolean variant keys; `--boxes` dump |
| `training/eval/tests/test_noise.py` (new) | pipeline unit tests |
| `training/eval/tests/test_screen_fixtures.py` | `noisy_jobs` tests |
| `training/eval/results/noise_preprocessing.md` (new) | per-step results |
| `apps/desktop/src/khmerclipper/ocr/preprocess.py`, `trocr.py` | port of kept steps; Latin fallback gets the cleaned strip |
| `apps/desktop/tests/test_preprocess.py` | port tests |
| `apps/mac/Sources/OCR/TrOCREngine.swift` | port of kept steps |
| `apps/mac/Sources/Tools/OCRBenchMain.swift` | `--boxes` dump |
| `apps/mac/Tests/TrOCRTests.swift` | port tests |

---

### Task 1: Noisy test set generator

**Files:**
- Modify: `training/eval/scripts/build_screen_fixtures.py`, `training/eval/src/khmerclipper_eval/fixtures.py:39`, `training/eval/.gitignore`, `training/trocr/scripts/remote.sh:12-15`
- Test: `training/eval/tests/test_screen_fixtures.py`

**Interfaces:**
- Produces:
  - `NOISE_KINDS = ["icons", "chrome", "underline", "background", "edgecut", "avatar"]`
  - `noisy_jobs(pool: list[str], rng: random.Random, per_kind: int = 50) -> list[dict]`: each job has the screen-job keys (`id`, `image`, `subset`, `lines`, `font`, `size`, `scale`, `dark`, `bg`, `fg`, `line_height`, `bold`, `pad`) plus `kind` and `noise` (a dict of kind-specific parameters). `subset == f"noisy_{kind}"`, `id == f"noisy-{kind}-{n:03d}"`, `image == f"noisy_{kind}/{id}.png"`.
  - `noisy_page(job: dict) -> str`: the HTML for one clip.
  - CLI: `build_screen_fixtures.py noisy --text screen_text.jsonl --out fixtures_noisy [--per_kind 50]`.
  - `labels.jsonl` rows as for screen fixtures, plus `"kind"`.

- [ ] **Step 1: Write failing tests**

```python
def test_noisy_jobs_cover_every_kind_deterministically():
    pool = ["សួស្តី ពិភពលោក", "ការអប់រំ និងបច្ចេកវិទ្យា", "Settings ការកំណត់"] * 20
    a = bsf.noisy_jobs(pool, random.Random(1), per_kind=5)
    b = bsf.noisy_jobs(pool, random.Random(1), per_kind=5)
    assert a == b
    assert Counter(j["kind"] for j in a) == {k: 5 for k in bsf.NOISE_KINDS}
    assert all(j["subset"] == f"noisy_{j['kind']}" for j in a)

def test_noisy_labels_hold_only_the_text():
    for job in bsf.noisy_jobs(["សួស្តី ពិភពលោក"] * 40, random.Random(2), per_kind=3):
        assert bsf.label(job["lines"]) == " ".join(job["lines"])  # noise lives in job["noise"], never in lines

def test_noisy_page_draws_the_noise_but_label_text_is_unchanged():
    job = next(j for j in bsf.noisy_jobs(["សួស្តី"] * 40, random.Random(3), per_kind=2) if j["kind"] == "underline")
    html = bsf.noisy_page(job)
    assert "សួស្តី" in html and ("underline" in html or "border" in html)

def test_some_background_and_chrome_clips_are_latin_only():
    jobs = bsf.noisy_jobs(["សួស្តី ពិភពលោក", "Settings"] * 40, random.Random(4), per_kind=20)
    latin = [j for j in jobs if j["kind"] in ("background", "chrome") and j["lines"] == ["Settings"]]
    assert latin  # the on-device check must exercise the Tesseract fallback
```

- [ ] **Step 2: Run `cd training/eval && PYTHONPATH=src .venv/bin/python -m pytest tests/test_screen_fixtures.py -q`.** Expected: FAIL, `noisy_jobs` not defined.

- [ ] **Step 3: Implement `NOISE_KINDS`, `noisy_jobs`, `noisy_page` and the `noisy` subcommand**
  - **Text and style:** sample them like `render()` does: fonts, 12–20 px, 1x/2x, 30% dark, 1–3 lines from the pool.
  - **`noise` parameters per kind:**
    - `icons`: 1–3 emoji or inline SVG (bullet, checkmark, circle) at start, end or between words.
    - `chrome`: a bordered button or input box around the text, or a 1–2 px divider above or below.
    - `underline`: CSS `text-decoration: underline`, or a table row with 1 px grid lines.
    - `background`: a CSS linear or radial gradient, or a generated texture as an inline data URI (numpy noise plus blur, never a downloaded photo); 30% low contrast.
    - `edgecut`: render 3 lines, then crop so 20–60% of the line above and/or below remains.
    - `avatar`: a 32–48 px circle or block left of the text.
  - **Latin-only clips:** in `background` and `chrome`, 15% of clips use a Latin-only line taken from `LATIN`.
  - **Crop:** `edgecut` crops by pixel rows after render; the other kinds crop to ink with `crop_to_ink`, using bg plus noise as ink so the noise stays in the clip.
  - **Wiring:** add `"kind"` to the `labels.jsonl` keys. Allow `subset.startswith("noisy_")` in `fixtures.py`, as `slice_` is today.

- [ ] **Step 4: Run the tests.** Expected: PASS.

- [ ] **Step 5: Ignore the new set and keep real captures off the box**
  - Add `fixtures_noisy/` to `training/eval/.gitignore`.
  - Add `--exclude fixtures_real` to the rsync in `remote.sh`.
  - Run `git check-ignore training/eval/fixtures_noisy/x.png`. Expected: the path is printed.

- [ ] **Step 6: Render on the Mac**
  - Run `PYTHONPATH=src .venv/bin/python scripts/build_screen_fixtures.py noisy --text screen_text.jsonl --out fixtures_noisy`.
  - Expected: 300 PNGs and `fixtures_noisy/labels.jsonl` with 300 rows.
  - Open 2 clips per kind and confirm the noise is visible and the text isn't clipped.

- [ ] **Step 7: Commit:** `git add` the four modified files and the test, then `git commit -m "eval: noisy screen fixtures (six noise kinds)"`.

### Task 2: Real captures and baseline

**Files:**
- Create: `training/eval/results/noise_preprocessing.md`
- Local only: `training/eval/fixtures_real/`

**Interfaces:**
- Consumes: `fixtures_noisy/` (Task 1), `scripts/draft_real_fixtures.py`, `scripts/compare.py`
- Produces: baseline CSVs `results/noise_base_{fixtures,fixtures_screen,fixtures_slices,fixtures_noisy,fixtures_real}.csv`, which later tasks pair against.

- [ ] **Step 1: Draft the real-capture labels**
  - Run `PYTHONPATH=src .venv/bin/python scripts/draft_real_fixtures.py ~/Library/Application\ Support/KhmerClipper/captures`.
  - Expected: 6 rows in `fixtures_real/labels.jsonl`, `"verified": false`.
- [ ] **Step 2: Verify the labels by hand**
  - Correct each `text` from the image (`fixtures_real/review.md`) and set `"verified": true`.
  - Confirm with `git status --short training/eval/fixtures_real`. Expected: no output (ignored).
- [ ] **Step 3: Run the baseline on all five sets**
  - `bench.py --fixtures <set> --engine trocr-pipeline --trocr-model-dir ../../apps/mac/raw_models/TrOCR-khmer --variant split_touching=1,split_wide_at=12,crop=1 --out results/noise_base_<set>.csv`
  - Expected: 5 CSVs; the frozen-200 CER matches `accuracy_v3.md` (0.0034 ± 0.0005).
- [ ] **Step 4: Write the baseline section of `noise_preprocessing.md`:** per-kind CER on the noisy set, per-clip results on the real set, and the frozen, screen and slices totals.
- [ ] **Step 5: Commit:** `git add results/noise_preprocessing.md` (CSVs of local sets stay uncommitted), then `git commit -m "eval: noise baseline"`.

### Task 3: Step 2, robust background estimate (eval pipeline)

**Files:**
- Modify: `training/eval/src/khmerclipper_eval/pipeline.py:44-60` (`ink_mask`), `scripts/bench.py` (`parse_variant` bool keys)
- Test: `training/eval/tests/test_noise.py`

**Interfaces:**
- Produces:
  - `background_color(px: np.ndarray) -> np.ndarray`: `px` is an `(h, w, 3)` int array; returns int RGB of shape `(3,)`.
  - `ink_mask(image, scan_max=512, robust_bg: bool = False)`: the return type is unchanged.
  - `line_strips(..., robust_bg: bool = False)`, `content_crop(image, robust_bg: bool = False)` and `recognize(..., robust_bg: bool = False)` thread the flag through.
  - Variant key: `robust_bg`.

- [ ] **Step 1: Write failing tests in `tests/test_noise.py`**

```python
def test_background_ignores_an_icon_in_a_corner():
    px = np.full((40, 200, 3), 250); px[:12, :12] = (200, 30, 30)  # red icon top-left
    assert tuple(P.background_color(px)) == (250, 250, 250)

def test_background_when_light_text_touches_much_of_the_border():
    px = np.full((30, 100, 3), 20)                    # dark-mode background
    px[0, :40] = px[-1, 10:50] = 230                  # tight crop: glyph tops/bottoms on 40% of top and bottom rows
    assert tuple(P.background_color(px)) == (20, 20, 20)

def test_robust_bg_is_identical_on_a_clean_clip():
    img = draw_line("ការអប់រំ", fg=(0, 0, 0), bg=(255, 255, 255))      # helper in this test module
    assert np.array_equal(P.ink_mask(img, robust_bg=True)[0], P.ink_mask(img)[0])
```

- [ ] **Step 2: Run `PYTHONPATH=src .venv/bin/python -m pytest tests/test_noise.py -q`.** Expected: FAIL, `background_color` missing.
- [ ] **Step 3: Implement `background_color`**
  - Take the one-pixel border of `px`.
  - Quantise each channel to 16 levels (`// 16`), take the most common bin, and return the mean of the border pixels in that bin.
  - In `ink_mask`, use it in place of the four-corner mean when `robust_bg` is set.
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Run the ship check**
  - Run all 5 sets with `--variant split_touching=1,split_wide_at=12,crop=1,robust_bg=1`.
  - Pair each against its baseline with `scripts/compare.py results/noise_base_<set>.csv results/noise_s2_<set>.csv`.
  - **Keep** when `noisy_background` errors fall at least 5% (improved > worsened) and nothing regresses beyond the Global Constraints. **Provisionally keep**, as an enabling step, when there's no gain but no regression either.
- [ ] **Step 6: Record and commit**
  - Add a step 2 table to `noise_preprocessing.md`.
  - `git add pipeline.py scripts/bench.py tests/test_noise.py results/noise_preprocessing.md`, then `git commit -m "eval: robust background estimate (step 2)"`.

### Task 4: Step 3, remove non-text components (eval pipeline)

**Files:**
- Modify: `training/eval/src/khmerclipper_eval/pipeline.py`, `scripts/bench.py`
- Test: `training/eval/tests/test_noise.py`

**Interfaces:**
- Consumes: `ink_mask(..., robust_bg)` (Task 3)
- Produces:
  - `components(mask: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int, int, int]]]`: a label image plus a box per component, `(x0, y0, x1, y1, area)`, with 8-connectivity. Plain BFS or union-find, no scipy.
  - `text_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]`: returns `(kept, removed)` bool masks.
  - `line_strips` and `content_crop` gain `drop_noise: bool = False`; when set they build the row profile and crop box from `kept`.
  - Variant key: `drop_noise`.

- [ ] **Step 1: Write failing tests**

```python
def test_text_mask_drops_lines_and_blocks_but_keeps_glyphs():
    m = np.zeros((60, 300), bool)
    m[20:30, 10:15] = m[20:30, 20:26] = True       # two glyph-sized blobs
    m[16:18, 20:22] = True                          # a detached above-mark
    m[45, 0:300] = True                             # 1 px divider across the clip
    m[5:55, 250:290] = True                         # solid image block
    kept, removed = P.text_mask(m)
    assert kept[20:30, 10:15].all() and kept[16:18, 20:22].all()
    assert removed[45, :].all() and removed[5:55, 250:290].all()

def test_text_mask_keeps_tiny_1x_glyphs():
    m = np.zeros((12, 80), bool)
    for x in range(2, 78, 5): m[3:8, x:x + 3] = True  # 3x5 px glyphs of 10 px text at 1x
    kept, _ = P.text_mask(m)
    assert kept.sum() == m.sum()

def test_text_mask_on_all_noise_keeps_line_strips_working():
    img = Image.new("RGB", (200, 60), "white"); ImageDraw.Draw(img).rectangle((10, 10, 190, 50), fill="black")
    assert len(P.line_strips(img, split_touching=True, robust_bg=True, drop_noise=True)) == 1

def test_drop_noise_is_identical_on_a_clean_clip():
    img = draw_line("ការអប់រំ", fg=(0, 0, 0), bg=(255, 255, 255))
    a = P.line_strips(img, split_touching=True, robust_bg=True, drop_noise=True)
    b = P.line_strips(img, split_touching=True, robust_bg=True)
    assert [s.size for s in a] == [s.size for s in b]
```

- [ ] **Step 2: Run the tests.** Expected: FAIL, `text_mask` missing.
- [ ] **Step 3: Implement `components` and `text_mask`.** Remove a component when any of the following holds; tune the constants in Step 5:
  - **Line:** its box is at least 8× longer than its thickness, and the thickness is at most 3 px at the mask's scale.
  - **Block:** fill ratio above 0.6 and box height above 1.8× the median height of the remaining components.
  - **Isolated blob:** its rows overlap no row band held by at least 3 other components.

  Starting values come from the frozen-200 component statistics, measured once. Comment each constant with the measurement that set it.
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Run the ship check with `robust_bg=1,drop_noise=1`.** Targets: chrome, underline, avatar, icons. Pair against the last kept configuration. If step 2 was provisionally kept, judge steps 2 and 3 as a bundle against the baseline. Retuning is allowed; re-run the check after any change.
- [ ] **Step 6: Record and commit:** `git commit -m "eval: drop non-text components (step 3)"`, with the table in `noise_preprocessing.md`.

### Task 5: Step 4, drop cut-off edge fragments (eval pipeline)

**Files:**
- Modify: `pipeline.py` (`line_strips`), `scripts/bench.py`
- Test: `tests/test_noise.py`

**Interfaces:**
- Produces:
  - `drop_edge_fragments(runs: list[list[int]], height: int) -> list[list[int]]`: removes the first run if `lo == 0`, and the last if `hi == height - 1`, when it is shorter than `EDGE_FRAGMENT = 0.6` × the median height of the other runs. It never removes the only run. Applied after `merge_marks`.
  - Variant key: `drop_edge`.

- [ ] **Step 1: Write failing tests**

```python
def test_edge_fragment_dropped():
    assert P.drop_edge_fragments([[0, 4], [10, 30], [40, 60], [70, 79]], 80) == [[10, 30], [40, 60]]

def test_single_line_with_marks_touching_top_is_kept():
    assert P.drop_edge_fragments([[0, 22]], 23) == [[0, 22]]

def test_full_height_first_line_touching_edge_is_kept():
    assert P.drop_edge_fragments([[0, 20], [30, 50]], 51) == [[0, 20], [30, 50]]
```

- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement `drop_edge_fragments`** and call it in `line_strips` when `drop_edge` is set.
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Run the ship check with the kept flags plus `drop_edge=1`.** Target: `noisy_edgecut`; `fixtures_screen` must not regress, since screen clips start and end on real lines.
- [ ] **Step 6: Record and commit:** `git commit -m "eval: drop edge fragments (step 4)"`.

### Task 6: Step 5, paint removed noise over (eval pipeline)

Skip this task, and record why, if step 3 was dropped. Without step 3, nothing is removed.

**Files:**
- Modify: `pipeline.py` (`recognize`, `content_crop`), `scripts/bench.py`
- Test: `tests/test_noise.py`

**Interfaces:**
- Consumes: `text_mask` (Task 4), `background_color` (Task 3)
- Produces:
  - `fill_removed(image: Image.Image, removed: np.ndarray, scale: int, bg: np.ndarray) -> Image.Image`: paints the full-resolution pixels under `removed`, dilated by 1 mask pixel, with `bg`.
  - Variant key: `fill_noise`, applied to each strip before `content_crop` and decoding.

- [ ] **Step 1: Write failing tests**

```python
def test_fill_removed_paints_only_removed_pixels():
    img = Image.new("RGB", (40, 20), (255, 255, 255)); img.putpixel((5, 5), (0, 0, 0)); img.putpixel((30, 10), (0, 0, 0))
    removed = np.zeros((20, 40), bool); removed[10, 30] = True
    out = P.fill_removed(img, removed, 1, np.array([255, 255, 255]))
    assert out.getpixel((30, 10)) == (255, 255, 255) and out.getpixel((5, 5)) == (0, 0, 0)

def test_fill_noise_is_identical_on_a_clean_clip():
    img = draw_line("ការអប់រំ", fg=(0, 0, 0), bg=(255, 255, 255))
    flags = dict(split_touching=True, robust_bg=True, drop_noise=True)
    assert [s.tobytes() for s in P.line_strips(img, fill_noise=True, **flags)] == \
           [s.tobytes() for s in P.line_strips(img, **flags)]
```

- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement `fill_removed`** and apply it in `line_strips` when `fill_noise` is set, before strips are cropped.
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Run the ship check with the kept flags plus `fill_noise=1`.** Targets: icons, chrome.
- [ ] **Step 6: Record and commit:** `git commit -m "eval: paint removed noise (step 5)"`.

### Task 7: Port the kept steps to desktop and Mac, with parity

Port only the steps recorded as kept. With every eval flag at its kept setting, behaviour is unconditional in the apps; the apps get no flags.

**Files:**
- Modify: `apps/desktop/src/khmerclipper/ocr/preprocess.py`, `apps/desktop/src/khmerclipper/ocr/trocr.py:110-140`, `apps/mac/Sources/OCR/TrOCREngine.swift`, `apps/mac/Sources/Tools/OCRBenchMain.swift`, `training/eval/scripts/bench.py`
- Test: `apps/desktop/tests/test_preprocess.py`, `apps/mac/Tests/TrOCRTests.swift`

**Interfaces:**
- Consumes: the eval functions from Tasks 3–6 and their final constants.
- Produces:
  - **Desktop:** `background_color`, `text_mask`, `drop_edge_fragments`, `fill_removed`, with the same names and signatures as in eval (desktop `ink_mask` returns `InkMask`).
  - **Swift:** `static func backgroundColor(_ px: [UInt8], width: Int, height: Int) -> (Int, Int, Int)`, `static func textMask(_ ink: [Bool], width: Int, height: Int) -> (kept: [Bool], removed: [Bool])`, `static func dropEdgeFragments(_ runs: [(lo: Int, hi: Int)], height: Int) -> [(lo: Int, hi: Int)]`. Fill is folded into `lineStrips`.
  - **Box dump:** `OCRBench --boxes <images...>` prints `id strip k x y w h crop x y w h` per strip. `bench.py --engine trocr-pipeline --boxes` and a desktop equivalent (`python -m khmerclipper.ocr.preprocess --boxes <images...>`) print the same format.

- [ ] **Step 1: Write failing port tests.** Mirror every eval test from Tasks 3–6 for the kept steps, with the same inputs and assertions, in `test_preprocess.py` and `TrOCRTests.swift` (as the two-line-cut tests were).
- [ ] **Step 2: Run them.** Run `cd apps/desktop && .venv/bin/python -m pytest -q` and `cd apps/mac && make test`. Expected: the new tests fail.
- [ ] **Step 3: Implement the ports.** In `trocr.py` (desktop) and `recognizeText` (Swift), pass the cleaned strip, the one the model decoded, to the Latin/Tesseract fallback.
- [ ] **Step 4: Run both suites.** Expected: all pass.
- [ ] **Step 5: Check parity**
  - Dump boxes from all three implementations for all five sets, then diff.
  - Expected: every strip and crop box within 1 px, and identical strip counts. Fix any drift before going on.
- [ ] **Step 6: Commit:** `git commit -m "feat: noise-robust preprocessing on Mac and desktop (kept steps <list>)"`.

### Task 8: On-device check and results

**Files:**
- Modify: `training/eval/results/noise_preprocessing.md`

- [ ] **Step 1: Run the Mac check**
  - Run `make ocrbench`, then `bench.py --engine trocr-coreml` on all five sets.
  - Pair against the on-device baseline: the PR #5 CSVs for the frozen, screen and slices sets, plus a fresh run at the Task 2 baseline commit for the noisy and real sets.
  - Expected: Global Constraints met, including latency +10% at most.
- [ ] **Step 2: Run the desktop check.** Run the desktop ONNX path on all five sets (the `deskbench.py` approach from PR #5, or `bench.py --engine trocr-desktop`). Expected: the same accuracy direction as the pipeline runner, and latency +10% at most.
- [ ] **Step 3: If latency is over budget**
  - Profile first. Likely fix: run `components` on the downscaled scan mask, which `ink_mask` already produces.
  - Re-check.
  - Drop the most expensive step only if that fails, and record it.
- [ ] **Step 4: Write the final results.** Per-step and final tables, per-kind breakdown, real-capture per-clip results, kept/dropped steps with reasons.
- [ ] **Step 5: Commit, push, and open a PR** with the results table in the body.
  - `git commit -m "results: noise-robust preprocessing"`
  - `git push -u origin feat/noise-preprocessing`
  - `gh pr create`
  - Wait for CI to pass before asking to merge.
