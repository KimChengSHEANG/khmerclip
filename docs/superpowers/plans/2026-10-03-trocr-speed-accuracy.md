# TrOCR On-Device Speed + Accuracy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Mac app's CoreML TrOCR path pass the spec's Mac gate on the frozen 200 fixtures: CER below Tesseract, median latency ≤ 2× Tesseract, and mixed-subset CER ≤ Tesseract. Do it by fixing the Swift/CoreML port and rebuilding the decoder with a KV cache. No retraining.

**Architecture:** The fine-tuned checkpoint is already good: HF PyTorch beam-4 scores CER 0.048 on a 40-fixture sample. The app's CoreML path scores 0.339 on the same sample and takes 1468 ms per image. The work has two parts:

- **Accuracy parity.** Make Swift preprocessing and decoding match what the model was trained on: line splitting, no content crop, a PIL-exact resize, a 128-token cap, and a cap on how many times one token can repeat in a row.
- **Speed.** Re-export the model as an FP16 encoder plus a single-token decoder that keeps its self-attention KV cache in a CoreML state (`ct.StateType`, macOS 15). Swift then calls the decoder one token at a time.

The conversion code moves out of the gitignored `Vendor/` into a tracked `apps/mac/tools/trocr/`.

**Tech Stack:** Swift 5 / CoreML (macOS 15 `MLState`), XCTest, xcodegen. Python 3.12 in `apps/mac/.trocr-venv` (torch 2.7.0, transformers 4.57.6, coremltools 9.0, pytest). Eval harness `training/eval` (`bench.py`).

**Spec:** `docs/superpowers/specs/2026-08-27-khmer-ocr-accuracy-speed-design.md` defines the pass/fail criteria and fallbacks. The evidence for this plan is in "Measured findings" below.

**Related:** `docs/superpowers/plans/2026-10-03-trocr-finetune-new-data.md` is the training-side follow-up on the 3090. It is independent of this plan. Its finished checkpoints come back through this plan's converter (`make trocr-khmer KHMER_CKPT=…`). That plan also found that the old model **trained on the soyvitou testset, which is where the 100 printed fixtures come from**. Absolute printed CER is therefore optimistic, while the UI fixtures are clean. Judge port parity here as CoreML vs HF on the same checkpoint, not against absolute numbers.

## Measured findings (2026-10-03, M2 Max, macOS 27.0.1, 40-fixture sample: 20 printed + 20 ui)

| Path | CER all | printed | ui | mixed | median ms |
|---|---:|---:|---:|---:|---:|
| Tesseract | 0.254 | 0.593 | 0.000 | 0.020 | 74 |
| CoreML app (current) | 0.339 | 0.179 | 0.510 | 0.100 | 1468 |
| HF raw image, greedy | 0.172 | 0.310 | 0.069 | 0.083 | 599 (PyTorch CPU) |
| HF raw image, beam 4 | **0.048** | 0.052 | 0.035 | 0.112 | 1034 |
| HF content-crop (like Swift), greedy | 0.229 | 0.388 | 0.117 | 0.083 | — |
| **HF raw, greedy + run cap 4** | **0.056** | **0.036** | 0.069 | 0.083 | 13.4 steps/img (vs 17.7 greedy) |

Root causes found:
1. **Line over-splitting.** `lineStrips` splits 17 of 200 single-line fixtures into 2–3 strips. Khmer marks above the base line form their own ink run, 37–46% of the tallest run's height, which is above the 35% merge cutoff. Example: `ui-008` decodes as `'C ៛  ៛ បទពេប ព្រើ ដ្ឋើ'` from 3 strips. Raising the cutoff to 50% merges all 17.
2. **`contentCrop` hurts.** Training images are never cropped. Cropping adds about 0.06 CER with greedy decoding and about 0.07 with beam.
3. **Greedy repetition loops.** HF greedy fails on 2 printed rows by repeating token 1437 (`Ġ`, a space) until max length (128 and 89 tokens). These two rows make up almost all of the printed greedy–beam gap (0.31 vs 0.05). On `printed-012` the model recovers after the run and emits the real tail (`១៩៩២ ។`), so stopping at the run would lose text. The measured fix is a **run cap**: once the same token has appeared 4 times in a row, exclude it from the next argmax. This takes greedy decoding from 0.172 to **0.056**, beam-level accuracy at greedy cost.
4. **Swift resize is not PIL.** `CGContext .medium` produces pixels up to about 0.5 (on a [-1, 1] scale) away from the PIL bilinear resize used in training. The comment in `TrOCRTests.swift` already documents this.
5. **Token cap mismatch.** Swift uses `maxLength = 64`; training uses `MAX_LABEL_LEN = 128`. `printed-080` is 75 characters and cannot fit in 64 tokens.
6. **Slow decoder export.** The decoder is exported in FP32 with no KV cache and a dynamic `RangeDim` input. Every step re-runs the whole prefix plus cross-attention over 577 encoder tokens.

Gate numbers for the full 200 (from `training/eval/results/baseline.md`): Tesseract CER 0.2580; median 76.3 ms, so the latency limit is 152.6 ms; mixed CER 0.0894. Always recompute these from the same `bench.py` run; `passes_gate` does that.

## Global Constraints

- Work happens on branch `feat/trocr-speed-accuracy` in the worktree `.worktrees/trocr-speed-accuracy` (Task 0). Every relative path and `git` command in this plan runs from that worktree root, never from the main checkout. Nothing is committed to `main` until the branch is finished.
- Gate (from the spec): candidate CER **strictly below** Tesseract; median latency **≤ 2×** Tesseract; mixed-subset CER **not worse** than Tesseract. Measure all three on the **Mac CoreML** bench over the **full frozen 200 fixtures**. Never decide based on a subset.
- Deployment target stays `macOS 15.0` (`apps/mac/project.yml`). CoreML states need `ct.target.macOS15`.
- Python conversion env: `apps/mac/.trocr-venv`, pinned to `torch==2.7.0` and `transformers==4.57.*`, with coremltools (9.0 is installed).
- Checkpoint: `apps/mac/raw_models/TrOCR-khmer`. It is a fine-tuned `trocr-base-printed` with vocab 50414, `decoder_start_token_id=0`, `eos_token_id=2`, 384×384 input, mean = std = 0.5, and PIL resample 2 (bilinear).
- Decode cap: **128** total ids, including the start token. This matches training's `MAX_LABEL_LEN = 128` and `generation_config.max_length = 128`.
- Never commit `.mlpackage` files, `raw_models/`, `Vendor/trocr/*/`, venvs, or bench CSVs.
- Builds of the OCRBench tool target must pass `CODE_SIGN_IDENTITY="-"`. An empty identity fails to sign command-line tools.
- **Known risk; state it in the results:** the mixed criterion may still fail after parity. HF beam-4 scores 0.112 on the 3 sampled mixed rows, against Tesseract's 0.020. If the full-200 gate fails only on mixed, the fix belongs to the training side (`2026-10-03-trocr-finetune-new-data.md`), using synthetic Khmer+Latin data that excludes `training/eval/ui_strings.txt` to avoid fixture leakage.
- **Out of scope:** retraining, on-device beam search, and vocab pruning. Greedy plus the run cap reached 0.056 on the sample, against 0.048 for beam-4. Start a follow-up beam-search plan only if full-200 run-capped greedy CER is more than 0.01 above HF beam-4 CER.

## Review Focus

1. **Single-line Khmer with detached marks above or below** (e.g. `ui-008`, `ui-005`, `printed-024`): must produce **one** strip. Test: Task 2 `testSingleLineFixturesWithDetachedMarksStayOneStrip`.
2. **Dark-mode UI clip** (`ui-008` has background (23, 25, 28)): must split and preprocess the same as light mode. Test: Task 2 covers `ui-008` (dark) and `ui-005` (light).
3. **A line longer than 64 tokens** (`printed-080`, 75 chars): must not be cut at 64. Test: Task 4 `testLongLineIsNotTruncatedAt64Tokens`.
4. **Greedy repetition loop** (one token repeated): no token may repeat more than 4 times in a row, and text after the run must survive. Test: Task 4 `testBlockedTokenCapsRuns` and `testArgmaxSkipsBlockedToken`; Task 7 keeps the cap in the KV loop.
5. **Decoder state reused across lines or calls** (uninitialised or stale KV cache): must not leak between lines. Test: Task 7 `testRepeatedRecognitionIsDeterministic` recognizes A, then B, then A, and expects the same A.
6. **Multi-line clip:** must still split into one strip per line. The existing `testLineStripsSplitsMultiLineClipIntoOneStripPerLine` must stay green in Task 2.

## File structure

| Path | Responsibility | Task |
|---|---|---|
| `apps/mac/Sources/OCR/TrOCREngine.swift` | Line split, preprocessing, decode loop, model cache, run stats | 1, 2, 3, 4, 7 |
| `apps/mac/Sources/Tools/OCRBenchMain.swift` | Emits `stats …` line per image | 1 |
| `apps/mac/Makefile` | `ocrbench` target; converter paths → `tools/trocr`; pytest in venv | 1, 5 |
| `apps/mac/Tests/TrOCRTests.swift` | Line-split, resize-parity, run-cap, long-line, determinism tests | 2, 3, 4, 7 |
| `apps/mac/tools/trocr/convert.py` | HF → CoreML: FP16 encoder + stateful KV decoder | 5, 6 |
| `apps/mac/tools/trocr/kv_decoder.py` | `KVDecoderStep` single-token decoder module (PyTorch) | 5 |
| `apps/mac/tools/trocr/export_tokenizer.py` | Moved unchanged from `Vendor/trocr` | 5 |
| `apps/mac/tools/trocr/make_resize_golden.py` | Writes PIL-resized golden bytes for the Swift test | 3 |
| `apps/mac/tools/trocr/hf_reference.py` | HF PyTorch greedy + run cap over fixtures → bench-format CSV (parity reference) | 4 |
| `apps/mac/tools/trocr/tests/test_kv_decoder.py` | Eager parity: KV step logits == HF decoder logits | 5 |
| `apps/mac/tools/trocr/tests/test_convert.py` | CoreML stateful decoder == eager over 5 steps | 6 |
| `apps/mac/Tests/Fixtures/ocr/printed-001.pil384.rgb` | Golden 384×384 RGB bytes (PIL bilinear) | 3 |
| `training/eval/.gitignore` | Ignore `results/*.csv` | 1 |
| `training/eval/results/trocr_speed_accuracy.md` | Stage-by-stage full-200 numbers | 1, 4, 8 |
| `training/eval/results/mac_gate.md` | Gate outcome; fixed build command | 8 |

---

### Task 0: Create the worktree

**Files:**
- None. This plan is already committed on `main`, so the new branch contains it.

**Interfaces:**
- Produces: worktree `.worktrees/trocr-speed-accuracy` on branch `feat/trocr-speed-accuracy`, based on `main`. Every later task runs from the worktree root. `REPO=$(git rev-parse --show-toplevel)` resolves to the worktree, and all `cd apps/mac` / `cd training/eval` paths are relative to it.

- [ ] **Step 1: Create it (REQUIRED SUB-SKILL: superpowers:using-git-worktrees)**

The repo already uses `.worktrees/`, which is gitignored. Run from the main checkout:

```bash
cd /Users/kim/Sync/workspace/khmerclipper
git worktree add .worktrees/trocr-speed-accuracy -b feat/trocr-speed-accuracy main
```

Expected: `git worktree list` shows `.worktrees/trocr-speed-accuracy … [feat/trocr-speed-accuracy]`, and `.worktrees/trocr-speed-accuracy/docs/superpowers/plans/2026-10-03-trocr-speed-accuracy.md` exists.

- [ ] **Step 2: Bring in the gitignored artifacts the plan needs**

A fresh worktree is missing the checkpoint, the vendored engines, the converted models, and the conversion venv, because all of them are gitignored.

- The checkpoint and the venv are read-only inputs, so symlink them. Packages installed later (`pytest`, `jiwer`) go into main's venv; that is additive and harmless.
- Tesseract and EasyOCR are also only read, so symlink them.
- `Vendor/trocr` gets rewritten by this plan: Task 5 moves the converter out of it, and Task 6 reconverts the models. Make it a real directory and APFS-clone its contents (`cp -c`, instant and no extra disk), so main's models stay untouched.

```bash
MAIN=/Users/kim/Sync/workspace/khmerclipper/apps/mac
cd /Users/kim/Sync/workspace/khmerclipper/.worktrees/trocr-speed-accuracy/apps/mac
ln -s $MAIN/raw_models raw_models
ln -s $MAIN/.trocr-venv .trocr-venv
mkdir -p Vendor/trocr
ln -s $MAIN/Vendor/tesseract Vendor/tesseract
ln -s $MAIN/Vendor/easyocr Vendor/easyocr
cp -cR $MAIN/Vendor/trocr/khmer Vendor/trocr/khmer
cp -c $MAIN/Vendor/trocr/convert.py $MAIN/Vendor/trocr/export_tokenizer.py Vendor/trocr/
ls -la raw_models .trocr-venv Vendor Vendor/trocr
```

Expected: `raw_models` and `.trocr-venv` are symlinks. `Vendor/tesseract` and `Vendor/easyocr` are symlinks. `Vendor/trocr` is a real directory containing `khmer/` (with `encoder.mlpackage`, `decoder.mlpackage`, `tokenizer.json`), `convert.py`, and `export_tokenizer.py`. `git status --short` in the worktree is empty, because everything added here is gitignored.

**Compiled-model cache warning:** `TrOCREngine.ModelCache` compiles models into the shared `~/Library/Application Support/KhmerClipper/TrOCRCompiled/khmer-{encoder,decoder}.mlmodelc`. It decides whether to recompile by comparing timestamps. After Task 6 converts the KV models in the worktree, a main-checkout build would load the worktree's newer compiled models. While this branch is active, either don't run main's app, or delete that `TrOCRCompiled` folder before switching back.

- [ ] **Step 3: Smoke check — OCRBench builds and reads in the worktree**

```bash
cd apps/mac && xcodegen generate >/dev/null && xcodebuild -project KhmerClipper.xcodeproj -scheme OCRBench \
  -configuration Release -derivedDataPath build CODE_SIGN_IDENTITY="-" CODE_SIGNING_REQUIRED=NO build 2>&1 | grep -E "error:|BUILD"
KHMERCLIPPER_VENDOR=$PWD/Vendor build/Build/Products/Release/OCRBench --engine trocr --preload \
  ../../training/eval/fixtures/printed/printed-001.png
```

Expected: `** BUILD SUCCEEDED **`, then a line of Khmer text on stdout and `ms=…` on stderr.

When the branch is finished, merge or PR it with superpowers:finishing-a-development-branch. Remove the worktree with `git worktree remove .worktrees/trocr-speed-accuracy`. The symlinks and the cloned models are inside it, so they go with it. Main's artifacts are untouched.

---

### Task 1: Bench diagnostics + full-200 baseline

**Files:**
- Modify: `apps/mac/Sources/OCR/TrOCREngine.swift` (add `RunStats`, timing in `recognizeText` / `recognizeLine`)
- Modify: `apps/mac/Sources/Tools/OCRBenchMain.swift` (print stats line)
- Modify: `apps/mac/Makefile` (add `ocrbench` target)
- Modify: `training/eval/tests/test_trocr_coreml.py` (stats lines are ignored)
- Create: `training/eval/.gitignore`
- Create: `training/eval/results/trocr_speed_accuracy.md`

**Interfaces:**
- Consumes: nothing new
- Produces:
  - `TrOCREngine.RunStats { strips: Int, decoderSteps: Int, encoderMs: Double, decoderMs: Double }`
  - `static var TrOCREngine.lastRunStats: RunStats` (reset at the start of each `recognizeText`)
  - `static func TrOCREngine.milliseconds(since: ContinuousClock.Instant) -> Double`
  - One OCRBench stderr line per TrOCR image: `stats strips=<int> enc_ms=<f> steps=<int> dec_ms=<f>`
  - `make ocrbench` builds `apps/mac/build/Build/Products/Release/OCRBench`

- [ ] **Step 1: Write the Python test that pins the contract (stats lines must not be parsed as timings)**

Append to `training/eval/tests/test_trocr_coreml.py`:

```python
def test_recognize_many_ignores_stats_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        trocr_coreml.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(
            args[0],
            0,
            "one\n",
            "ms=7.5\nstats strips=1 enc_ms=3.0 steps=4 dec_ms=4.0\n",
        ),
    )

    assert trocr_coreml.recognize_many(
        [Path("/tmp/one.png")], Path("/tmp/OCRBench")
    ) == [("one", 7.5)]
```

- [ ] **Step 2: Run it**

Run: `cd training/eval && python -m pytest tests/test_trocr_coreml.py -v`
Expected: PASS. The regex already uses `fullmatch` on `ms=`. This test pins that contract before OCRBench starts printing stats lines. If it fails, fix `_TIMING_LINE` in `src/khmerclipper_eval/runners/trocr_coreml.py` before continuing.

- [ ] **Step 3: Add run stats to `TrOCREngine.swift`**

Insert after `private static let maxLength = 64`:

```swift
    /// Per-call timing breakdown for OCRBench diagnostics. ponytail: global,
    /// last-call-only -- fine for the single-threaded bench CLI; return it from
    /// recognizeText if the app ever needs it.
    struct RunStats {
        var strips = 0
        var decoderSteps = 0
        var encoderMs = 0.0
        var decoderMs = 0.0
    }
    static var lastRunStats = RunStats()

    static func milliseconds(since start: ContinuousClock.Instant) -> Double {
        let d = start.duration(to: .now)
        return Double(d.components.seconds) * 1_000 + Double(d.components.attoseconds) / 1e15
    }
```

In `recognizeText`, replace `for strip in lineStrips(image) {` with:

```swift
        lastRunStats = RunStats()
        let strips = lineStrips(image)
        lastRunStats.strips = strips.count
        for strip in strips {
```

In `recognizeLine`, wrap the encoder call:

```swift
        let pixelValues = try preprocess(image: image)
        let encodeStart = ContinuousClock.now
        let encoderOutput = try await encoder.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "pixel_values": MLFeatureValue(multiArray: pixelValues),
        ]))
        lastRunStats.encoderMs += milliseconds(since: encodeStart)
```

Put `let decodeStart = ContinuousClock.now` right before `var ids = [tokenizer.decoderStartId]`. Then put these lines right before `return tokenizer.decode(ids: ids)`:

```swift
        lastRunStats.decoderMs += milliseconds(since: decodeStart)
        lastRunStats.decoderSteps += ids.count - 1
```

- [ ] **Step 4: Print stats in `OCRBenchMain.swift`**

After the existing `fputs(String(format: "ms=%.3f\n", milliseconds), stderr)` line, add:

```swift
                if engine == "trocr" {
                    let s = TrOCREngine.lastRunStats
                    fputs(String(format: "stats strips=%d enc_ms=%.1f steps=%d dec_ms=%.1f\n",
                                 s.strips, s.encoderMs, s.decoderSteps, s.decoderMs), stderr)
                }
```

- [ ] **Step 5: Add the `ocrbench` Makefile target**

In `apps/mac/Makefile`, add `ocrbench` to `.PHONY`, then add:

```make
# Release OCRBench CLI for the eval harness (training/eval/scripts/bench.py
# --trocr-coreml-bin build/Build/Products/Release/OCRBench). Tool targets
# need an ad-hoc identity; an empty one fails to sign.
ocrbench: gen $(KHMER_MODEL)
	xcodebuild -project KhmerClipper.xcodeproj -scheme OCRBench -configuration Release \
		-derivedDataPath build CODE_SIGN_IDENTITY="-" CODE_SIGNING_REQUIRED=NO build
```

Run: `cd apps/mac && make ocrbench 2>&1 | grep -E "error:|BUILD"`
Expected: `** BUILD SUCCEEDED **`

- [ ] **Step 6: Full-200 baseline bench + stats**

```bash
REPO=$(git rev-parse --show-toplevel)
export KHMERCLIPPER_VENDOR=$REPO/apps/mac/Vendor
BIN=$REPO/apps/mac/build/Build/Products/Release/OCRBench
cd $REPO/training/eval
python scripts/bench.py --engine tesseract --engine trocr-coreml --trocr-coreml-bin $BIN \
  --out results/stage0_baseline.csv --candidate trocr-coreml --compare-to tesseract
$BIN --engine trocr $(python -c "import json;print(' '.join('fixtures/'+json.loads(l)['image'] for l in open('fixtures/labels.jsonl')))") \
  2>&1 >/dev/null | grep '^stats' | awk '{for(i=2;i<=NF;i++){split($i,a,"=");s[a[1]]+=a[2]}} END{for(k in s) printf "%s_mean=%.1f\n",k,s[k]/NR}'
```

Expected: the bench prints `FAIL` and per-engine aggregates. The awk command prints `strips_mean`, `enc_ms_mean`, `steps_mean`, and `dec_ms_mean`, which show where the time goes. ms/step = `dec_ms_mean / steps_mean`.

- [ ] **Step 7: Record results + ignore CSVs**

`training/eval/.gitignore`:

```gitignore
results/*.csv
```

`training/eval/results/trocr_speed_accuracy.md`:

```markdown
# TrOCR CoreML speed + accuracy, by stage

Full frozen 200 fixtures, Mac CoreML via `OCRBench` (Release), same `bench.py`
run as Tesseract. Plan: `docs/superpowers/plans/2026-10-03-trocr-speed-accuracy.md`.
Printed fixtures leaked into this checkpoint's training (soyvitou testset), so
absolute printed CER is optimistic; ui fixtures are clean.

| Stage | CER | printed | ui | mixed | median ms | strips/img | enc ms | steps/img | ms/step | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Tesseract | | | | | | — | — | — | — | — |
| 0 baseline (FP32, no KV) | | | | | | | | | | FAIL |
```

Fill both rows with the Step 6 numbers. Get per-subset CER with:

```bash
python -c "import csv,collections as c;d=c.defaultdict(list)
[d[(r['engine'],r['subset'])].append(float(r['cer'])) for r in csv.DictReader(open('results/stage0_baseline.csv'))]
[print(k,round(sum(v)/len(v),4)) for k,v in sorted(d.items())]"
```

- [ ] **Step 8: Commit**

```bash
git add apps/mac/Sources/OCR/TrOCREngine.swift apps/mac/Sources/Tools/OCRBenchMain.swift apps/mac/Makefile \
  training/eval/tests/test_trocr_coreml.py training/eval/.gitignore training/eval/results/trocr_speed_accuracy.md
git commit -m "feat(eval): per-image TrOCR timing stats and full-200 baseline"
```

---

### Task 2: Stop splitting single Khmer lines on detached marks

**Files:**
- Modify: `apps/mac/Sources/OCR/TrOCREngine.swift:192-196` (merge cutoff)
- Test: `apps/mac/Tests/TrOCRTests.swift` (`TrOCRLineSplitTests`)

**Interfaces:**
- Consumes: `TrOCREngine.lineStrips(_ image: CGImage) -> [CGImage]` (existing, internal)
- Produces: same signature. A run merges into its nearer neighbour when it is shorter than 35% of the tallest run (old rule), or when it is shorter than 50% of the tallest **and** its gap to the nearer neighbour is under 25% of the tallest. The gap condition keeps a short Latin line (such as "Hello world" next to tall Khmer) as its own line, because real line gaps are large. The over-split fixtures have gaps of 1–3 px against tallest runs of 22–24 px.

- [ ] **Step 1: Write the failing test**

Add inside `final class TrOCRLineSplitTests`:

```swift
    private static let monorepoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()   // Tests
        .deletingLastPathComponent()   // apps/mac
        .deletingLastPathComponent()   // apps
        .deletingLastPathComponent()   // repo root

    /// These frozen eval fixtures are each ONE text line whose above-base Khmer
    /// marks form a separate ink run 37-46% of the base line's height; they
    /// must not be read as extra lines. ui-008 is dark mode, ui-005 light.
    func testSingleLineFixturesWithDetachedMarksStayOneStrip() throws {
        for name in ["ui/ui-008.png", "ui/ui-005.png", "ui/ui-009.png", "printed/printed-024.png"] {
            let url = Self.monorepoRoot.appendingPathComponent("training/eval/fixtures/\(name)")
            let image = try XCTUnwrap(NSImage(contentsOf: url)?.cgImage(forProposedRect: nil, context: nil, hints: nil), name)
            XCTAssertEqual(TrOCREngine.lineStrips(image).count, 1, name)
        }
    }
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd apps/mac && xcodegen generate >/dev/null && xcodebuild -project KhmerClipper.xcodeproj -scheme KhmerClipperTests \
  -derivedDataPath build CODE_SIGN_IDENTITY="-" test -only-testing:KhmerClipperTests/TrOCRLineSplitTests 2>&1 | grep -E "error|passed|failed" | tail -8
```

Expected: FAIL with `XCTAssertEqual failed: ("3") is not equal to ("1") - ui/ui-008.png`, plus similar failures for the other files.

- [ ] **Step 3: Gap-aware merge rule**

In `lineStrips`, replace:

```swift
        // ponytail: fixed 35%-of-tallest merge cutoff; tune if headings next to
        // small body text get merged.
        while runs.count > 1 {
            let tallest = runs.map { $0.hi - $0.lo + 1 }.max()!
            guard let i = runs.firstIndex(where: { ($0.hi - $0.lo + 1) * 100 < tallest * 35 }) else { break }
```

with:

```swift
        // ponytail: fixed cutoffs. Merge a run < 35% of the tallest, or < 50%
        // when it sits within 25% of the tallest's height of a neighbour --
        // Khmer above-base marks form runs up to ~46% of the base line 1-3px
        // above it (measured: the old 35%-only rule split 17/200 single-line
        // eval fixtures), while real line gaps are far larger. Tune if
        // headings next to small body text get merged.
        while runs.count > 1 {
            let tallest = runs.map { $0.hi - $0.lo + 1 }.max()!
            func nearestGap(_ k: Int) -> Int {
                min(k > 0 ? runs[k].lo - runs[k - 1].hi : Int.max,
                    k < runs.count - 1 ? runs[k + 1].lo - runs[k].hi : Int.max)
            }
            guard let i = runs.indices.first(where: { k in
                let h = (runs[k].hi - runs[k].lo + 1) * 100
                return h < tallest * 35 || (h < tallest * 50 && nearestGap(k) * 4 < tallest)
            }) else { break }
```

- [ ] **Step 4: Run tests (new and existing 3-line test)**

Run the Step 2 command.
Expected: PASS for both `testSingleLineFixturesWithDetachedMarksStayOneStrip` and `testLineStripsSplitsMultiLineClipIntoOneStripPerLine`.

- [ ] **Step 5: Commit**

```bash
git add apps/mac/Sources/OCR/TrOCREngine.swift apps/mac/Tests/TrOCRTests.swift
git commit -m "fix(trocr): keep single Khmer lines with detached marks as one strip"
```

---

### Task 3: Training-identical preprocessing (no content crop, PIL-exact bilinear resize)

**Files:**
- Create: `apps/mac/tools/trocr/make_resize_golden.py`
- Create: `apps/mac/Tests/Fixtures/ocr/printed-001.pil384.rgb` (generated, 442,368 bytes)
- Modify: `apps/mac/Sources/OCR/TrOCREngine.swift` (delete `contentCrop`; add `resizedRGB`; rewrite `preprocess`)
- Test: `apps/mac/Tests/TrOCRTests.swift` (new `TrOCRPreprocessTests`; update `TrOCREndToEndTests` doc comment)

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces: `static func TrOCREngine.resizedRGB(_ image: CGImage, size: Int) -> [UInt8]?`. It returns `size*size*3` RGB bytes that match `PIL.Image.convert("RGB").resize((size, size), Image.BILINEAR)` to within ±1 per byte.

- [ ] **Step 1: Write the golden generator**

`apps/mac/tools/trocr/make_resize_golden.py`:

```python
"""Write the PIL-bilinear 384x384 RGB bytes that TrOCRPreprocessTests compares
Swift's resizedRGB against. Training (trocr_khmer.dataset.OCRDataset) resizes
with exactly this call, so matching it removes the train/inference pixel gap.

Usage: .trocr-venv/bin/python3 tools/trocr/make_resize_golden.py
"""
from pathlib import Path

from PIL import Image

MAC = Path(__file__).resolve().parents[2]
SRC = MAC.parents[1] / "training/eval/fixtures/printed/printed-001.png"
OUT = MAC / "Tests/Fixtures/ocr/printed-001.pil384.rgb"

img = Image.open(SRC).convert("RGB").resize((384, 384), Image.BILINEAR)
OUT.write_bytes(img.tobytes())
print("wrote", OUT, OUT.stat().st_size, "bytes")
```

Run: `cd apps/mac && .trocr-venv/bin/python3 tools/trocr/make_resize_golden.py`
Expected: `wrote …/printed-001.pil384.rgb 442368 bytes`

- [ ] **Step 2: Write the failing test**

Append to `apps/mac/Tests/TrOCRTests.swift`:

```swift
final class TrOCRPreprocessTests: XCTestCase {
    private static let macRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent()

    /// Training resizes with PIL bilinear (OCRDataset._preprocess_image);
    /// on-device pixels must match it or greedy decode drifts.
    func testResizedRGBMatchesPILBilinear() throws {
        let png = Self.macRoot.deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("training/eval/fixtures/printed/printed-001.png")
        let image = try XCTUnwrap(NSImage(contentsOf: png)?.cgImage(forProposedRect: nil, context: nil, hints: nil))
        let golden = try [UInt8](Data(contentsOf: Self.macRoot.appendingPathComponent("Tests/Fixtures/ocr/printed-001.pil384.rgb")))

        let swift = try XCTUnwrap(TrOCREngine.resizedRGB(image, size: 384))

        XCTAssertEqual(swift.count, golden.count)
        let maxDiff = zip(swift, golden).map { abs(Int($0) - Int($1)) }.max() ?? 0
        XCTAssertLessThanOrEqual(maxDiff, 1)
    }
}
```

- [ ] **Step 3: Run to verify it fails**

```bash
cd apps/mac && xcodegen generate >/dev/null && xcodebuild -project KhmerClipper.xcodeproj -scheme KhmerClipperTests \
  -derivedDataPath build CODE_SIGN_IDENTITY="-" test -only-testing:KhmerClipperTests/TrOCRPreprocessTests 2>&1 | grep -E "error|passed|failed" | tail -5
```

Expected: build error `type 'TrOCREngine' has no member 'resizedRGB'`.

- [ ] **Step 4: Implement `resizedRGB`, delete `contentCrop`, rewrite `preprocess`**

Delete the whole `contentCrop(_:)` function, doc comment included. Replace the whole `preprocess(image:size:)` function, doc comment included, with:

```swift
    /// RGB bytes resized exactly like PIL's `Image.resize(size, BILINEAR)` --
    /// the call training uses (trocr_khmer.dataset.OCRDataset). Separable
    /// triangle filter whose support widens by the scale factor when
    /// downsampling (antialias), horizontal pass then vertical, each rounded
    /// to UInt8 like PIL's 8-bit path. Mirrors Pillow's Resample.c
    /// precompute_coeffs.
    static func resizedRGB(_ image: CGImage, size: Int) -> [UInt8]? {
        let w = image.width, h = image.height
        // Draw in the image's own RGB space so values pass through unconverted,
        // like PIL (which ignores colour profiles); sRGB only for non-RGB images.
        guard w > 0, h > 0, let colorSpace = image.colorSpace.flatMap({ $0.model == .rgb ? $0 : nil })
                ?? CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        var src = [UInt8](repeating: 0, count: w * h * 4)
        guard let ctx = CGContext(data: &src, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: colorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))

        // (start index, normalized weights) per output pixel along one axis.
        func coeffs(inSize: Int, outSize: Int) -> [(Int, [Double])] {
            let scale = Double(inSize) / Double(outSize)
            let filterScale = max(scale, 1.0)
            let support = filterScale   // bilinear support 1.0 * filterScale
            return (0..<outSize).map { i in
                let center = (Double(i) + 0.5) * scale
                let lo = max(Int(center - support + 0.5), 0)
                let hi = min(Int(center + support + 0.5), inSize)
                var ws = (lo..<hi).map { x in max(0, 1 - abs((Double(x) - center + 0.5) / filterScale)) }
                let total = ws.reduce(0, +)
                if total > 0 { ws = ws.map { $0 / total } }
                return (lo, ws)
            }
        }
        func clip8(_ v: Double) -> UInt8 { UInt8(max(0, min(255, v.rounded()))) }

        let cx = coeffs(inSize: w, outSize: size), cy = coeffs(inSize: h, outSize: size)
        var tmp = [UInt8](repeating: 0, count: size * h * 3)   // horizontal pass
        for y in 0..<h {
            for (ox, (lo, ws)) in cx.enumerated() {
                for c in 0..<3 {
                    var acc = 0.0
                    for (k, wt) in ws.enumerated() { acc += wt * Double(src[(y * w + lo + k) * 4 + c]) }
                    tmp[(y * size + ox) * 3 + c] = clip8(acc)
                }
            }
        }
        var out = [UInt8](repeating: 0, count: size * size * 3)  // vertical pass
        for (oy, (lo, ws)) in cy.enumerated() {
            for x in 0..<size {
                for c in 0..<3 {
                    var acc = 0.0
                    for (k, wt) in ws.enumerated() { acc += wt * Double(tmp[((lo + k) * size + x) * 3 + c]) }
                    out[(oy * size + x) * 3 + c] = clip8(acc)
                }
            }
        }
        return out
    }

    /// ViTImageProcessor preprocessing, identical to training: PIL-bilinear
    /// resize to 384x384 (no aspect crop, no content crop -- training images are
    /// never cropped, and cropping measured +0.06 CER), then normalize with
    /// mean=std=0.5.
    private static func preprocess(image: CGImage, size: Int = 384) throws -> MLMultiArray {
        guard let rgb = resizedRGB(image, size: size) else { throw TrOCRError.preprocessingFailed }
        let array = try MLMultiArray(shape: [1, 3, NSNumber(value: size), NSNumber(value: size)], dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: size * size * 3)
        let plane = size * size
        for idx in 0..<plane {
            for c in 0..<3 {
                ptr[c * plane + idx] = Float32(rgb[idx * 3 + c]) / 127.5 - 1.0
            }
        }
        return array
    }
```

Update the `enum TrOCREngine` doc comment. Replace `reads printed Khmer when the text fills the frame -- preprocess() content-crops to make that so.` with `is fed exactly the training preprocessing (see preprocess()).`

In `TrOCREndToEndTests`, replace the whole doc comment (from `/// End-to-end smoke test` through `This test only proves the pipeline runs and produces text.`) with:

```swift
/// End-to-end smoke test for the CoreML pipeline (preprocessing -> encoder ->
/// greedy decode -> tokenizer). Does not assert exact text (that depends on
/// the checkpoint); pixel parity with PIL is pinned by TrOCRPreprocessTests and
/// accuracy is measured by training/eval/scripts/bench.py.
```

- [ ] **Step 5: Run tests**

Run the Step 3 command, then the `TrOCRLineSplitTests` command from Task 2.
Expected: PASS. If `maxDiff > 1`, the likely causes in order are:
1. Colour-space conversion: the draw must use the image's own RGB colour space, as in the code above.
2. Premultiplied alpha: the draw must use `noneSkipLast`.
3. Rounding: it must use `.rounded()` (half away from zero).

Do not relax the assertion above 1.

- [ ] **Step 6: Commit**

```bash
git add apps/mac/tools/trocr/make_resize_golden.py apps/mac/Tests/Fixtures/ocr/printed-001.pil384.rgb \
  apps/mac/Sources/OCR/TrOCREngine.swift apps/mac/Tests/TrOCRTests.swift
git commit -m "fix(trocr): match training preprocessing (PIL bilinear, no content crop)"
```

---

### Task 4: Decode cap 128 + repeat-run cap, HF parity reference, re-bench

**Files:**
- Modify: `apps/mac/Sources/OCR/TrOCREngine.swift` (`maxLength`, `maxRun`, `blockedToken`, `argmaxLastStep`, loop body)
- Test: `apps/mac/Tests/TrOCRTests.swift`
- Create: `apps/mac/tools/trocr/hf_reference.py`
- Modify: `training/eval/results/trocr_speed_accuracy.md`

**Interfaces:**
- Consumes: `RunStats` from Task 1
- Produces:
  - `static let TrOCREngine.maxLength = 128` (internal; was private 64)
  - `static let TrOCREngine.maxRun = 4`
  - `static func TrOCREngine.blockedToken(_ ids: [Int], maxRun: Int = maxRun) -> Int?`. It returns the token that already ends a run of `maxRun`, or nil.
  - `static func TrOCREngine.argmaxLastStep(_ logits: MLMultiArray, excluding: Int? = nil) -> Int` (internal; was private, without `excluding`)
  - `hf_reference.py` CSV with columns `engine,id,subset,cer,ms,pred,ref` (bench format; `engine` = `trocr-hf-greedy`)

- [ ] **Step 1: Write the failing tests**

In `TrOCREndToEndTests`, change `private static let repoRoot` to `static let repoRoot` so extensions can use it. Then append to `apps/mac/Tests/TrOCRTests.swift`:

```swift
final class TrOCRDecodeGuardTests: XCTestCase {
    func testBlockedTokenCapsRuns() {
        XCTAssertNil(TrOCREngine.blockedToken([0, 5, 5, 5]))           // 3 in a row: allowed
        XCTAssertEqual(TrOCREngine.blockedToken([0, 5, 5, 5, 5]), 5)   // 4 in a row: block a 5th
        XCTAssertNil(TrOCREngine.blockedToken([0, 5, 6, 5, 6]))        // alternating: not a run
        XCTAssertNil(TrOCREngine.blockedToken([0]))
    }

    func testArgmaxSkipsBlockedToken() throws {
        let logits = try MLMultiArray(shape: [1, 1, 4], dataType: .float32)
        for (i, v) in [0.1, 0.9, 0.5, 0.2].enumerated() { logits[i] = NSNumber(value: v) }
        XCTAssertEqual(TrOCREngine.argmaxLastStep(logits), 1)
        XCTAssertEqual(TrOCREngine.argmaxLastStep(logits, excluding: 1), 2)
    }

    func testDecodeCapMatchesTrainingMaxLabelLength() {
        XCTAssertEqual(TrOCREngine.maxLength, 128)   // training MAX_LABEL_LEN
    }
}

extension TrOCREndToEndTests {
    /// printed-080 is a 75-code-point line (~1 token per Khmer code point); the
    /// old 64-token cap cut it off. Count unicode scalars, not Characters:
    /// a Khmer cluster (consonant + coeng + subscript + vowel) is ONE Character.
    func testLongLineIsNotTruncatedAt64Tokens() async throws {
        let png = Self.repoRoot.deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("training/eval/fixtures/printed/printed-080.png")
        let image = try XCTUnwrap(NSImage(contentsOf: png)?.cgImage(forProposedRect: nil, context: nil, hints: nil))
        let text = try await TrOCREngine.recognizeText(in: image, model: .khmer)
        XCTAssertGreaterThan(text.unicodeScalars.count, 64)
    }
}
```

Add `import CoreML` at the top of `TrOCRTests.swift` if it is not already there.

- [ ] **Step 2: Run to verify failure**

```bash
cd apps/mac && xcodegen generate >/dev/null && xcodebuild -project KhmerClipper.xcodeproj -scheme KhmerClipperTests \
  -derivedDataPath build CODE_SIGN_IDENTITY="-" test -only-testing:KhmerClipperTests/TrOCRDecodeGuardTests \
  -only-testing:KhmerClipperTests/TrOCREndToEndTests 2>&1 | grep -E "error|passed|failed" | tail -8
```

Expected: build error `type 'TrOCREngine' has no member 'blockedToken'`.

- [ ] **Step 3: Implement**

Replace `private static let maxLength = 64` with:

```swift
    /// Total decoded ids incl. the start token; matches training
    /// MAX_LABEL_LEN / generation_config.max_length (128).
    static let maxLength = 128

    /// Greedy decode on this checkpoint can lock onto one token (measured: id
    /// 1437 "Ġ" repeated to max length on 2/20 printed fixtures) and sometimes
    /// recovers into real text afterwards. So instead of stopping, forbid a
    /// token from extending a run past maxRun: measured greedy CER 0.172 ->
    /// 0.056 (beam-4: 0.048). Labels never repeat one token more than 3x.
    /// ponytail: single-token runs only; 2-token cycles still end at maxLength.
    static let maxRun = 4

    static func blockedToken(_ ids: [Int], maxRun: Int = maxRun) -> Int? {
        guard ids.count >= maxRun, let last = ids.last,
              ids.suffix(maxRun).allSatisfy({ $0 == last }) else { return nil }
        return last
    }
```

Change `argmaxLastStep`'s signature from `private static func argmaxLastStep(_ logits: MLMultiArray) -> Int {` to `static func argmaxLastStep(_ logits: MLMultiArray, excluding: Int? = nil) -> Int {`. In its loop, change `for v in 0..<vocabSize {` to `for v in 0..<vocabSize where v != excluding {`.

In the `recognizeLine` loop, change `let nextId = argmaxLastStep(logits)` to:

```swift
            let nextId = argmaxLastStep(logits, excluding: blockedToken(ids))
```

- [ ] **Step 4: Run tests**

Run the Step 2 command.
Expected: all PASS. Like the existing end-to-end test, `testLongLineIsNotTruncatedAt64Tokens` needs the converted `Vendor/trocr/khmer` models.

- [ ] **Step 5: Write the HF parity reference**

`bench.py --engine trocr-hf` cannot serve as the parity reference. It decodes with beam 4, and it imports `trocr_khmer`, which is not installed on the Mac. This script runs in `.trocr-venv` and uses the same greedy decode, run cap, and 128 cap as Swift.

`apps/mac/tools/trocr/hf_reference.py`:

```python
"""HF PyTorch reference for CoreML parity: greedy decode with the same 128-id
cap and repeat-run cap (TrOCREngine.maxRun) the Swift engine uses, over the
frozen eval fixtures. Writes a bench.py-format CSV (engine,id,subset,cer,ms,
pred,ref) and prints mean CER per subset.

Usage (from apps/mac):
  .trocr-venv/bin/python3 tools/trocr/hf_reference.py raw_models/TrOCR-khmer \
      ../../training/eval/results/hf_greedy_runcap.csv
"""
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from transformers import LogitsProcessor, LogitsProcessorList, TrOCRProcessor, VisionEncoderDecoderModel

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "training/eval/src"))
from khmerclipper_eval.cer import character_error_rate  # noqa: E402

MAX_LEN = 128   # == TrOCREngine.maxLength
MAX_RUN = 4     # == TrOCREngine.maxRun


class RunCap(LogitsProcessor):
    """Forbid a token from extending a run of MAX_RUN identical tokens."""

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] >= MAX_RUN:
            tail = input_ids[:, -MAX_RUN:]
            for b in range(input_ids.shape[0]):
                if bool((tail[b] == tail[b, -1]).all()):
                    scores[b, tail[b, -1]] = -float("inf")
        return scores


def main() -> None:
    ckpt, out = Path(sys.argv[1]), Path(sys.argv[2])
    fixtures = REPO / "training/eval/fixtures"
    proc = TrOCRProcessor.from_pretrained(ckpt)
    model = VisionEncoderDecoderModel.from_pretrained(ckpt).eval()
    rows, by_subset = [], defaultdict(list)
    for line in (fixtures / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        pv = proc(images=Image.open(fixtures / r["image"]).convert("RGB"), return_tensors="pt").pixel_values
        t0 = time.perf_counter()
        with torch.no_grad():
            ids = model.generate(pv, num_beams=1, max_length=MAX_LEN, no_repeat_ngram_size=0,
                                 logits_processor=LogitsProcessorList([RunCap()]))
        ms = (time.perf_counter() - t0) * 1000
        pred = proc.batch_decode(ids, skip_special_tokens=True)[0]
        cer = character_error_rate(pred, r["text"])
        by_subset[r["subset"]].append(cer)
        rows.append({"engine": "trocr-hf-greedy", "id": r["id"], "subset": r["subset"],
                     "cer": cer, "ms": ms, "pred": pred, "ref": r["text"]})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["engine", "id", "subset", "cer", "ms", "pred", "ref"])
        w.writeheader()
        w.writerows(rows)
    allc = [c for v in by_subset.values() for c in v]
    print(f"all n={len(allc)} cer={sum(allc) / len(allc):.4f}")
    for k, v in sorted(by_subset.items()):
        print(f"{k} n={len(v)} cer={sum(v) / len(v):.4f}")


if __name__ == "__main__":
    main()
```

Run:

```bash
cd apps/mac && .trocr-venv/bin/python3 -m pip install -q jiwer && \
  .trocr-venv/bin/python3 tools/trocr/hf_reference.py raw_models/TrOCR-khmer ../../training/eval/results/hf_greedy_runcap.csv
```

Expected: prints `all n=200 cer=…` plus per-subset lines. On the 40-fixture sample this setup measured 0.056.

- [ ] **Step 6: Re-bench full 200 (parity stage, old FP32 model) and check parity**

```bash
cd apps/mac && make ocrbench 2>&1 | grep BUILD
REPO=$(git rev-parse --show-toplevel); export KHMERCLIPPER_VENDOR=$REPO/apps/mac/Vendor
cd $REPO/training/eval && python scripts/bench.py --engine tesseract --engine trocr-coreml \
  --trocr-coreml-bin $REPO/apps/mac/build/Build/Products/Release/OCRBench \
  --out results/stage1_parity.csv --candidate trocr-coreml --compare-to tesseract
python - <<'PY'
import csv
hf = {r["id"]: r for r in csv.DictReader(open("results/hf_greedy_runcap.csv"))}
cm = {r["id"]: r for r in csv.DictReader(open("results/stage1_parity.csv")) if r["engine"] == "trocr-coreml"}
mean = lambda rs: sum(float(r["cer"]) for r in rs) / len(rs)
print(f"HF greedy {mean(hf.values()):.4f}  CoreML {mean(cm.values()):.4f}")
diff = sorted(cm, key=lambda i: float(cm[i]["cer"]) - float(hf[i]["cer"]), reverse=True)[:10]
for i in diff:
    print(i, cm[i]["cer"], hf[i]["cer"], repr(cm[i]["pred"][:50]), "| HF", repr(hf[i]["pred"][:50]))
PY
```

Expected: CoreML CER is within 0.02 of HF greedy and below Tesseract. Latency is still over the gate at this stage. Add the row `1 parity (lines, PIL resize, no crop, cap 128, run cap 4)` to `trocr_speed_accuracy.md`, with stats from the Task 1 Step 6 awk command. Add an `HF greedy + run cap (PyTorch CPU)` reference row as well.

If CoreML is more than 0.02 worse than HF greedy, do not start Task 5. The diff listing shows which fixtures disagree. Fix the Swift side for those fixtures first; strips, preprocessing, and tokenizer are the likely causes.

- [ ] **Step 7: Commit**

```bash
git add apps/mac/Sources/OCR/TrOCREngine.swift apps/mac/Tests/TrOCRTests.swift apps/mac/tools/trocr/hf_reference.py \
  training/eval/results/trocr_speed_accuracy.md
git commit -m "fix(trocr): decode up to 128 tokens with a repeat-run cap; add HF parity reference"
```

---

### Task 5: Tracked converter home + KV-cache decoder module (eager parity)

**Files:**
- Create: `apps/mac/tools/trocr/kv_decoder.py`
- Create: `apps/mac/tools/trocr/tests/test_kv_decoder.py`
- Move with plain `mv`: `apps/mac/Vendor/trocr/convert.py` → `apps/mac/tools/trocr/convert.py`, and `apps/mac/Vendor/trocr/export_tokenizer.py` → `apps/mac/tools/trocr/export_tokenizer.py`. Both files are currently untracked because `apps/mac/Vendor/` is gitignored, and they get bundled into the .app.
- Modify: `apps/mac/Makefile` (paths; pytest in venv; `trocr-test` target)

**Interfaces:**
- Consumes: nothing from the Swift tasks
- Produces:
  - `class KVDecoderStep(nn.Module)` with `__init__(self, model: VisionEncoderDecoderModel, max_len: int, cache_dtype: torch.dtype = torch.float32)`
  - `forward(input_ids: int32[1,1], position: int32[1], attn_mask: float[1,1,1,max_len], write_mask: float[1,1,max_len,1], encoder_hidden_states: float[1,S,E]) -> logits float[1,1,V]`
  - Buffers `k_cache` and `v_cache`, shape `(num_layers, 1, num_heads, max_len, head_dim)`
  - Mask contract, used by every caller (Python tests and Swift): at step `pos`, `attn_mask[..., j] = 0` for `j <= pos` and `-1e4` otherwise; `write_mask[..., j, 0] = 1` only at `j == pos`
  - `make trocr-test` runs `pytest tools/trocr/tests`

- [ ] **Step 1: Move converter files and fix Makefile paths**

```bash
cd apps/mac && mkdir -p tools/trocr/tests
mv Vendor/trocr/convert.py Vendor/trocr/export_tokenizer.py tools/trocr/
```

In `apps/mac/Makefile`, replace every `Vendor/trocr/convert.py` with `tools/trocr/convert.py`, and every `Vendor/trocr/export_tokenizer.py` with `tools/trocr/export_tokenizer.py`. Change the venv install line to:

```make
	$(TROCR_PY) -m pip install torch==2.7.0 "transformers==4.57.*" coremltools pillow pytest jiwer
```

Add `trocr-test` to `.PHONY` and add:

```make
# Converter unit tests (tiny random models; no checkpoint or network needed).
trocr-test: | $(TROCR_PY)
	$(TROCR_PY) -m pytest tools/trocr/tests -v
```

Run: `.trocr-venv/bin/python3 -m pip install -q pytest`

- [ ] **Step 2: Write the failing test**

`apps/mac/tools/trocr/tests/test_kv_decoder.py`:

```python
import sys
from pathlib import Path

import pytest
import torch
from transformers import TrOCRConfig, ViTConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kv_decoder import KVDecoderStep  # noqa: E402

MAX_LEN = 16


def tiny_model(learned_pos: bool) -> VisionEncoderDecoderModel:
    enc = ViTConfig(image_size=32, patch_size=16, hidden_size=32, num_hidden_layers=2,
                    num_attention_heads=2, intermediate_size=64)
    dec = TrOCRConfig(vocab_size=50, d_model=48, decoder_layers=2, decoder_attention_heads=4,
                      decoder_ffn_dim=96, cross_attention_hidden_size=32, max_position_embeddings=64,
                      use_learned_position_embeddings=learned_pos, layernorm_embedding=True,
                      decoder_start_token_id=0, bos_token_id=0, eos_token_id=2, pad_token_id=1)
    torch.manual_seed(0)
    cfg = VisionEncoderDecoderConfig.from_encoder_decoder_configs(enc, dec)
    return VisionEncoderDecoderModel(config=cfg).eval()


def masks(pos: int, max_len: int = MAX_LEN):
    attn = torch.full((1, 1, 1, max_len), -1e4)
    attn[..., : pos + 1] = 0
    write = torch.zeros(1, 1, max_len, 1)
    write[:, :, pos, 0] = 1
    return attn, write


@pytest.mark.parametrize("learned_pos", [True, False])
@torch.no_grad()
def test_step_logits_match_full_decoder(learned_pos: bool) -> None:
    model = tiny_model(learned_pos)
    enc = model.encoder(pixel_values=torch.rand(1, 3, 32, 32)).last_hidden_state
    seq = torch.tensor([[0, 7, 9, 3, 11, 5]])
    want = model.decoder(input_ids=seq, encoder_hidden_states=enc).logits  # [1, T, V]

    step = KVDecoderStep(model, MAX_LEN).eval()
    for pos in range(seq.shape[1]):
        attn, write = masks(pos)
        got = step(seq[:, pos : pos + 1].int(), torch.tensor([pos], dtype=torch.int32), attn, write, enc)
        torch.testing.assert_close(got[0, 0], want[0, pos], atol=1e-4, rtol=1e-4)
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd apps/mac && make trocr-test`
Expected: FAIL with `ModuleNotFoundError: No module named 'kv_decoder'`.

- [ ] **Step 4: Implement `kv_decoder.py`**

`apps/mac/tools/trocr/kv_decoder.py`:

```python
"""Single-token TrOCR decoder step with a fixed-size self-attention KV cache.

Re-implements transformers' TrOCRDecoder forward (post-LN BART-style layers,
modeling_trocr.py 4.57) for ONE token per call, reusing the checkpoint's own
submodules so no weights are remapped. Self-attention keys/values live in the
k_cache/v_cache buffers, which coremltools converts to CoreML states
(ct.StateType). All shapes are static: attention always spans max_len
positions, with attn_mask hiding the unwritten ones, and the new K/V row is
blended in with a one-hot write_mask (no dynamic slicing).

ponytail: cross-attention K/V are recomputed from encoder_hidden_states every
step (~11 GFLOP/step for trocr-base). If ms/step is the bottleneck, move them
into a state filled once per line.
"""
import torch
from torch import nn


class KVDecoderStep(nn.Module):
    def __init__(self, model, max_len: int, cache_dtype: torch.dtype = torch.float32):
        super().__init__()
        causal_lm = model.decoder            # TrOCRForCausalLM
        dec = causal_lm.model.decoder        # TrOCRDecoder
        self.embed_tokens = dec.embed_tokens  # already applies embed_scale
        self.layernorm_embedding = dec.layernorm_embedding
        self.layers = dec.layers
        self.output_projection = causal_lm.output_projection
        if dec.config.use_learned_position_embeddings:
            pos_table = dec.embed_positions.weight            # index = position + 2
            self.pos_offset = dec.embed_positions.offset
        else:
            pos_table = dec.embed_positions.weights           # index = padding_idx + 1 + position
            self.pos_offset = dec.embed_positions.padding_idx + 1
        self.register_buffer("pos_table", pos_table.detach().clone(), persistent=False)
        attn = dec.layers[0].self_attn
        self.num_heads, self.head_dim = attn.num_heads, attn.head_dim
        shape = (len(dec.layers), 1, self.num_heads, max_len, self.head_dim)
        self.register_buffer("k_cache", torch.zeros(shape, dtype=cache_dtype))
        self.register_buffer("v_cache", torch.zeros(shape, dtype=cache_dtype))

    def _heads(self, x):  # [1, T, E] -> [1, H, T, D]
        return x.view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def _attend(self, q, k, v, mask):
        scores = q @ k.transpose(-1, -2)
        if mask is not None:
            scores = scores + mask
        out = scores.softmax(dim=-1) @ v  # [1, H, T, D]
        return out.transpose(1, 2).reshape(1, -1, self.num_heads * self.head_dim)

    def forward(self, input_ids, position, attn_mask, write_mask, encoder_hidden_states):
        h = self.embed_tokens(input_ids)
        h = h + self.pos_table.index_select(0, position + self.pos_offset).unsqueeze(0).to(h.dtype)
        if self.layernorm_embedding is not None:
            h = self.layernorm_embedding(h)
        keep = 1 - write_mask
        for i, layer in enumerate(self.layers):
            sa = layer.self_attn
            q = self._heads(sa.q_proj(h) * sa.scaling)
            k = self.k_cache[i : i + 1].squeeze(0).to(h.dtype) * keep + self._heads(sa.k_proj(h)) * write_mask
            v = self.v_cache[i : i + 1].squeeze(0).to(h.dtype) * keep + self._heads(sa.v_proj(h)) * write_mask
            self.k_cache[i : i + 1] = k.unsqueeze(0).to(self.k_cache.dtype)
            self.v_cache[i : i + 1] = v.unsqueeze(0).to(self.v_cache.dtype)
            h = layer.self_attn_layer_norm(h + sa.out_proj(self._attend(q, k, v, attn_mask)))

            ca = layer.encoder_attn
            q = self._heads(ca.q_proj(h) * ca.scaling)
            ck = self._heads(ca.k_proj(encoder_hidden_states))
            cv = self._heads(ca.v_proj(encoder_hidden_states))
            h = layer.encoder_attn_layer_norm(h + ca.out_proj(self._attend(q, ck, cv, None)))

            h = layer.final_layer_norm(h + layer.fc2(layer.activation_fn(layer.fc1(h))))
        return self.output_projection(h)
```

- [ ] **Step 5: Run tests**

Run: `cd apps/mac && make trocr-test`
Expected: 2 passed (learned and sinusoidal positions).

- [ ] **Step 6: Commit**

```bash
git add apps/mac/tools/trocr apps/mac/Makefile
git commit -m "feat(trocr): track converter under tools/ and add KV-cache decoder step"
```

---

### Task 6: Export FP16 encoder + stateful KV decoder to CoreML

**Files:**
- Modify: `apps/mac/tools/trocr/convert.py` (replace the export section)
- Create: `apps/mac/tools/trocr/tests/test_convert.py`

**Interfaces:**
- Consumes: `KVDecoderStep` (Task 5)
- Produces:
  - `convert.MAX_LEN = 128`
  - `def convert_encoder(model, img_size: int, precision) -> ct.models.MLModel`. Input `pixel_values` float32 `[1,3,S,S]`; output `encoder_hidden_states` float32.
  - `def convert_decoder(model, enc_shape: tuple, max_len: int, precision) -> ct.models.MLModel`. Inputs: `input_ids` int32 `[1,1]`, `position` int32 `[1]`, `attn_mask` float32 `[1,1,1,max_len]`, `write_mask` float32 `[1,1,max_len,1]`, `encoder_hidden_states` float32 `enc_shape`. States: `k_cache` and `v_cache`, float16. Output: `logits` float32 `[1,1,V]`.
  - CLI: `python3 tools/trocr/convert.py <size> <out_dir> [checkpoint] [--encoder-fp32] [--int8]`

- [ ] **Step 1: Write the failing test**

`apps/mac/tools/trocr/tests/test_convert.py`:

```python
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import coremltools as ct  # noqa: E402

from convert import convert_decoder, convert_encoder  # noqa: E402
from kv_decoder import KVDecoderStep  # noqa: E402
from test_kv_decoder import masks, tiny_model  # noqa: E402

MAX_LEN = 16
pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="CoreML predict needs macOS")


@torch.no_grad()
def test_stateful_decoder_matches_eager_over_steps() -> None:
    model = tiny_model(learned_pos=True)
    pixels = torch.rand(1, 3, 32, 32)
    enc = model.encoder(pixel_values=pixels).last_hidden_state

    enc_ml = convert_encoder(model, 32, ct.precision.FLOAT16)
    enc_out = enc_ml.predict({"pixel_values": pixels.numpy()})["encoder_hidden_states"]
    np.testing.assert_allclose(enc_out, enc.numpy(), atol=2e-2)

    dec_ml = convert_decoder(model, tuple(enc.shape), MAX_LEN, ct.precision.FLOAT16)
    state = dec_ml.make_state()
    eager = KVDecoderStep(model, MAX_LEN).eval()
    for pos, tok in enumerate([0, 7, 9, 3, 11]):
        attn, write = masks(pos, MAX_LEN)
        ids = torch.tensor([[tok]], dtype=torch.int32)
        p = torch.tensor([pos], dtype=torch.int32)
        want = eager(ids, p, attn, write, enc)[0, 0].numpy()
        got = dec_ml.predict(
            {"input_ids": ids.numpy(), "position": p.numpy(), "attn_mask": attn.numpy(),
             "write_mask": write.numpy(), "encoder_hidden_states": enc.numpy()},
            state=state,
        )["logits"][0, 0]
        np.testing.assert_allclose(got, want, atol=3e-2)
        assert int(got.argmax()) == int(want.argmax())
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd apps/mac && make trocr-test`
Expected: `test_convert.py` errors on import. The current `convert.py` runs its script body against `sys.argv` when imported, and it has no `convert_decoder`. Step 3 moves the body under `main()`.

- [ ] **Step 3: Rewrite `convert.py`**

Keep the header comment, changing `Usage:` to `tools/trocr/convert.py`, and keep `DEFAULT_CHECKPOINTS` unchanged. Replace everything from `size = sys.argv[1]` to the end of the file with:

```python
import argparse
import numpy as np
from kv_decoder import KVDecoderStep

MAX_LEN = 128  # total decoded ids incl. start; == training MAX_LABEL_LEN


def load(ckpt):
    model = VisionEncoderDecoderModel.from_pretrained(ckpt, low_cpu_mem_usage=False)
    # trocr-large-* sinusoidal position tables are left on the meta device by
    # HF fast init (plain attribute, not a buffer); recompute them for real.
    from transformers.models.trocr.modeling_trocr import TrOCRSinusoidalPositionalEmbedding
    for mod in model.modules():
        if isinstance(mod, TrOCRSinusoidalPositionalEmbedding) and mod.weights is not None and mod.weights.is_meta:
            mod.weights = mod.get_embedding(mod.weights.shape[0], mod.embedding_dim, mod.padding_idx)
    return model.eval()


class Encoder(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.encoder = m.encoder
        self.proj = getattr(m, "enc_to_dec_proj", None)  # only when hidden sizes differ

    def forward(self, pixel_values):
        h = self.encoder(pixel_values=pixel_values).last_hidden_state
        return self.proj(h) if self.proj is not None else h


def convert_encoder(model, img_size, precision):
    example = torch.rand(1, 3, img_size, img_size)
    traced = torch.jit.trace(Encoder(model).eval(), (example,))
    return ct.convert(
        traced,
        inputs=[ct.TensorType(name="pixel_values", shape=example.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="encoder_hidden_states", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=precision,
        minimum_deployment_target=ct.target.macOS15,
    )


def convert_decoder(model, enc_shape, max_len, precision):
    step = KVDecoderStep(model, max_len, cache_dtype=torch.float16).eval()
    attn = torch.full((1, 1, 1, max_len), -1e4)
    attn[..., 0] = 0
    write = torch.zeros(1, 1, max_len, 1)
    write[:, :, 0, 0] = 1
    example = (torch.zeros(1, 1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32),
               attn, write, torch.rand(*enc_shape))
    traced = torch.jit.trace(step, example)
    cache = ct.TensorType(shape=tuple(step.k_cache.shape), dtype=np.float16)
    return ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, 1), dtype=np.int32),
            ct.TensorType(name="position", shape=(1,), dtype=np.int32),
            ct.TensorType(name="attn_mask", shape=(1, 1, 1, max_len), dtype=np.float32),
            ct.TensorType(name="write_mask", shape=(1, 1, max_len, 1), dtype=np.float32),
            ct.TensorType(name="encoder_hidden_states", shape=tuple(enc_shape), dtype=np.float32),
        ],
        states=[ct.StateType(wrapped_type=cache, name="k_cache"),
                ct.StateType(wrapped_type=cache, name="v_cache")],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=precision,
        minimum_deployment_target=ct.target.macOS15,
    )


def int8(mlmodel):
    import coremltools.optimize.coreml as cto
    cfg = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric"))
    return cto.linear_quantize_weights(mlmodel, cfg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("size", choices=sorted(DEFAULT_CHECKPOINTS))
    p.add_argument("out_dir")
    p.add_argument("checkpoint", nargs="?")
    p.add_argument("--encoder-fp32", action="store_true", help="keep the encoder FP32 (FP16 CER fallback)")
    p.add_argument("--int8", action="store_true", help="int8 weight quantization (latency fallback)")
    a = p.parse_args()

    ckpt = a.checkpoint or DEFAULT_CHECKPOINTS[a.size]
    model = load(ckpt)
    sz = TrOCRProcessor.from_pretrained(ckpt).image_processor.size
    img_size = sz["height"] if isinstance(sz, dict) else getattr(sz, "height", sz)

    with torch.no_grad():
        enc_shape = tuple(Encoder(model)(torch.rand(1, 3, img_size, img_size)).shape)
    print("img_size", img_size, "encoder out", enc_shape)

    enc_prec = ct.precision.FLOAT32 if a.encoder_fp32 else ct.precision.FLOAT16
    enc_ml = convert_encoder(model, img_size, enc_prec)
    dec_ml = convert_decoder(model, enc_shape, MAX_LEN, ct.precision.FLOAT16)  # states must be fp16
    if a.int8:
        enc_ml, dec_ml = int8(enc_ml), int8(dec_ml)
    enc_ml.save(f"{a.out_dir}/encoder.mlpackage")
    dec_ml.save(f"{a.out_dir}/decoder.mlpackage")
    print("saved to", a.out_dir)


if __name__ == "__main__":
    main()
```

Keep the existing top imports (`torch`, `coremltools as ct`, `VisionEncoderDecoderModel`, `TrOCRProcessor`). Drop `sys` from the import line if nothing uses it any more.

If `ct.convert` rejects the in-place slice assignment to the state buffers, check the coremltools 9 "Stateful Models" guide. Its supported form is `self.buf[a:b] = value` on a registered buffer, which is what `KVDecoderStep` uses. Any change to `kv_decoder.py` must keep `test_kv_decoder.py` green.

- [ ] **Step 4: Run tests**

Run: `cd apps/mac && make trocr-test`
Expected: 3 passed (2 eager parity + 1 CoreML stateful).

- [ ] **Step 5: Convert the real checkpoint**

```bash
cd apps/mac && make trocr-khmer KHMER_CKPT=raw_models/TrOCR-khmer
ls Vendor/trocr/khmer   # decoder.mlpackage encoder.mlpackage tokenizer.json
```

Expected: the conversion completes and prints `encoder out (1, 577, 768)`.

- [ ] **Step 6: Commit**

```bash
git add apps/mac/tools/trocr/convert.py apps/mac/tools/trocr/tests/test_convert.py
git commit -m "feat(trocr): export FP16 encoder and stateful KV-cache decoder"
```

---

### Task 7: Swift KV-cache decode loop

**Files:**
- Modify: `apps/mac/Sources/OCR/TrOCREngine.swift` (`recognizeLine`, `ModelCache` compute units, type doc comment)
- Test: `apps/mac/Tests/TrOCRTests.swift`

**Interfaces:**
- Consumes: the decoder I/O contract from Task 6 (names, shapes, dtypes); the mask contract from Task 5; `blockedToken` / `argmaxLastStep(_:excluding:)` / `maxLength` (Task 4); `RunStats` (Task 1)
- Produces: same public API (`recognizeText`, `preload`). The env knob `KHMERCLIPPER_TROCR_UNITS` = `all` | `gpu` | `ane` | `cpu` (default `all`) picks the `MLComputeUnits` for both models.

- [ ] **Step 1: Write the failing test**

Add to the `extension TrOCREndToEndTests` block from Task 4:

```swift
    /// Each line gets a fresh, zeroed KV state: recognizing A, then B, then A
    /// again must give the same A (no stale cache leaking between calls).
    func testRepeatedRecognitionIsDeterministic() async throws {
        let fixtures = Self.repoRoot.deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("training/eval/fixtures")
        func load(_ name: String) throws -> CGImage {
            try XCTUnwrap(NSImage(contentsOf: fixtures.appendingPathComponent(name))?
                .cgImage(forProposedRect: nil, context: nil, hints: nil))
        }
        let a = try load("printed/printed-001.png"), b = try load("ui/ui-008.png")
        let first = try await TrOCREngine.recognizeText(in: a, model: .khmer)
        _ = try await TrOCREngine.recognizeText(in: b, model: .khmer)
        let again = try await TrOCREngine.recognizeText(in: a, model: .khmer)
        XCTAssertEqual(first, again)
        XCTAssertFalse(first.isEmpty)
    }
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd apps/mac && xcodegen generate >/dev/null && xcodebuild -project KhmerClipper.xcodeproj -scheme KhmerClipperTests \
  -derivedDataPath build CODE_SIGN_IDENTITY="-" test -only-testing:KhmerClipperTests/TrOCREndToEndTests 2>&1 | grep -E "error|passed|failed" | tail -8
```

Expected: FAIL. The Task 6 decoder needs `position`, `attn_mask`, `write_mask` and a state, so the old loop throws a CoreML missing-feature error.

- [ ] **Step 3: Replace the decode loop in `recognizeLine`**

Replace everything from `let decodeStart = ContinuousClock.now` through `return tokenizer.decode(ids: ids)` with:

```swift
        // Fresh KV state per line. Zero it: masked-out garbage could still be
        // NaN, and NaN * 0 survives both the mask and the one-hot blend.
        let state = decoder.makeState()
        for name in ["k_cache", "v_cache"] {
            state.withMultiArray(for: name) { cache in
                // Whole backing buffer incl. any stride padding -- one missed
                // element brings the NaN back.
                cache.withUnsafeMutableBytes { raw, _ in
                    if let base = raw.baseAddress { memset(base, 0, raw.count) }
                }
            }
        }

        // Mask contract (tools/trocr/kv_decoder.py): at step pos, attn_mask is
        // 0 for j <= pos and -1e4 after; write_mask is one-hot at pos.
        let n = NSNumber(value: maxLength)
        let inputIds = try MLMultiArray(shape: [1, 1], dataType: .int32)
        let position = try MLMultiArray(shape: [1], dataType: .int32)
        let attnMask = try MLMultiArray(shape: [1, 1, 1, n], dataType: .float32)
        let writeMask = try MLMultiArray(shape: [1, 1, n, 1], dataType: .float32)
        let attnPtr = attnMask.dataPointer.bindMemory(to: Float32.self, capacity: maxLength)
        let writePtr = writeMask.dataPointer.bindMemory(to: Float32.self, capacity: maxLength)
        for j in 0..<maxLength { attnPtr[j] = -1e4; writePtr[j] = 0 }
        let inputs = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputIds),
            "position": MLFeatureValue(multiArray: position),
            "attn_mask": MLFeatureValue(multiArray: attnMask),
            "write_mask": MLFeatureValue(multiArray: writeMask),
            "encoder_hidden_states": MLFeatureValue(multiArray: encoderHidden),
        ])

        let decodeStart = ContinuousClock.now
        var ids = [tokenizer.decoderStartId]
        for pos in 0..<(maxLength - 1) {
            try Task.checkCancellation()
            inputIds[0] = NSNumber(value: ids[pos])
            position[0] = NSNumber(value: pos)
            attnPtr[pos] = 0
            if pos > 0 { writePtr[pos - 1] = 0 }
            writePtr[pos] = 1

            let out = try await decoder.prediction(from: inputs, using: state)
            guard let logits = out.featureValue(for: "logits")?.multiArrayValue else {
                throw TrOCRError.predictionFailed
            }
            let nextId = argmaxLastStep(logits, excluding: blockedToken(ids))
            ids.append(nextId)
            await onStep(ids.count)
            if nextId == tokenizer.eosId { break }
        }

        lastRunStats.decoderMs += milliseconds(since: decodeStart)
        lastRunStats.decoderSteps += ids.count - 1
        return tokenizer.decode(ids: ids)
```

- [ ] **Step 4: Compute-units knob in `ModelCache.loadCompiled`**

Replace `return try MLModel(contentsOf: compiledURL)` with:

```swift
            let config = MLModelConfiguration()
            config.computeUnits = Self.computeUnits
            return try MLModel(contentsOf: compiledURL, configuration: config)
```

and add inside `ModelCache`:

```swift
        /// Bench knob: KHMERCLIPPER_TROCR_UNITS=all|gpu|ane|cpu. ponytail: env
        /// read, not a preference -- Task 8 picks the fastest and makes it the
        /// default here.
        private static let computeUnits: MLComputeUnits = {
            switch ProcessInfo.processInfo.environment["KHMERCLIPPER_TROCR_UNITS"] {
            case "gpu": return .cpuAndGPU
            case "ane": return .cpuAndNeuralEngine
            case "cpu": return .cpuOnly
            default: return .all
            }
        }()
```

In the `enum TrOCREngine` doc comment, replace `Greedy, no-KV-cache decode loop: OCR text regions are short enough that re-running the full decoder each step is simple and fast enough, without CoreML's more complex stateful-model support.` with `Greedy decode, one token per call, with the decoder's self-attention KV cache held in a CoreML state (macOS 15) -- see tools/trocr/kv_decoder.py for the mask contract.` In the same comment, replace `Vendor/trocr/convert.py` with `tools/trocr/convert.py`.

- [ ] **Step 5: Run all Swift tests**

```bash
cd apps/mac && make test 2>&1 | grep -E "error|Executed|failed" | tail -8
```

Expected: `Executed N tests, with 0 failures`.

- [ ] **Step 6: Commit**

```bash
git add apps/mac/Sources/OCR/TrOCREngine.swift apps/mac/Tests/TrOCRTests.swift
git commit -m "perf(trocr): KV-cache decode via CoreML state, one token per call"
```

---

### Task 8: Full-200 gate, compute-unit sweep, precision/latency fallbacks, default policy

**Files:**
- Modify: `apps/mac/Sources/OCR/TrOCREngine.swift` (default compute units, if not `.all`)
- Modify: `training/eval/results/trocr_speed_accuracy.md`
- Modify: `training/eval/results/mac_gate.md` (outcome; fix signing in build command)
- Modify: `apps/mac/Sources/Preferences/AppPreferences.swift` (**only** on the spec's FAIL path)
- Modify: `apps/mac/README.md` (default-engine note)

**Interfaces:**
- Consumes: everything above
- Produces: the recorded gate outcome and the chosen default compute units

- [ ] **Step 1: Bench each compute-unit setting on the full 200**

```bash
cd apps/mac && make ocrbench 2>&1 | grep BUILD
REPO=$(git rev-parse --show-toplevel); export KHMERCLIPPER_VENDOR=$REPO/apps/mac/Vendor
cd $REPO/training/eval
for u in all gpu ane; do
  KHMERCLIPPER_TROCR_UNITS=$u python scripts/bench.py --engine tesseract --engine trocr-coreml \
    --trocr-coreml-bin $REPO/apps/mac/build/Build/Products/Release/OCRBench \
    --out results/stage2_kv_$u.csv --candidate trocr-coreml --compare-to tesseract | tee results/stage2_kv_$u.txt
done
```

Expected: each run prints aggregates plus PASS/FAIL. In `trocr_speed_accuracy.md`, record one row per setting (`2 KV fp16 units=<u>`), with stats from the Task 1 awk command run under the same `KHMERCLIPPER_TROCR_UNITS`. The first `ane` run includes ANE compilation, but `OCRBench` preloads before timing, so the medians stay valid.

- [ ] **Step 2: Lock the fastest setting as default**

Pick the setting with the lowest median ms among those whose CER is within 0.005 of the best CER. If that is not `all`, change the `default:` branch of `ModelCache.computeUnits` to return it (e.g. `.cpuAndGPU`), and update its doc comment with the measured choice and date. Rebuild with `make ocrbench`.

- [ ] **Step 3: Precision fallback (only if FP16 lost accuracy)**

Compare the chosen stage-2 CER with stage 1 (the FP32 parity run). If stage 2 CER > stage 1 CER + 0.01:

```bash
cd apps/mac && mkdir -p Vendor/trocr/khmer && \
  .trocr-venv/bin/python3 tools/trocr/convert.py khmer Vendor/trocr/khmer raw_models/TrOCR-khmer --encoder-fp32 && \
  .trocr-venv/bin/python3 tools/trocr/export_tokenizer.py khmer Vendor/trocr/khmer/tokenizer.json raw_models/TrOCR-khmer && \
  make ocrbench
```

Re-run the Step 1 bench for the chosen setting only, and record it as row `3 encoder fp32`. If CER is still more than 0.01 worse, the cause is the FP16 decoder. In that case:

1. In `convert_decoder`, replace `compute_precision=precision` with `compute_precision=ct.transform.FP16ComputePrecision(op_selector=lambda op: op.op_type not in ("softmax", "layer_norm", "reduce_mean", "add"))`.
2. Re-run `make trocr-test`, reconvert, and re-bench.
3. Record the result as row `3b decoder mixed precision`.

- [ ] **Step 4: Latency fallback (only if median > 2× Tesseract)**

If the chosen setting's median ms is more than 2 × the Tesseract median from the same run, reconvert with `--int8` (the Step 3 commands with `--int8` added). Re-bench and record row `4 int8`. Keep int8 only if CER rises by no more than 0.01 over the best non-int8 stage. If latency still fails, the spec fallback is `microsoft/trocr-small-printed`, which needs a 3090 training run under the training-side plan. Record that as the next step in `mac_gate.md`.

- [ ] **Step 5: Record the gate and apply the spec's default policy**

Fill in the Results table, **Status**, and `PASS`/`FAIL` reasons in `training/eval/results/mac_gate.md` from the final chosen run. In that file's Step 2 build snippet, replace `CODE_SIGN_STYLE=Manual CODE_SIGN_IDENTITY="" PROVISIONING_PROFILE_SPECIFIER="" DEVELOPMENT_TEAM=""` with `CODE_SIGN_IDENTITY="-" CODE_SIGNING_REQUIRED=NO`, and add a line pointing to `make ocrbench`.

- **PASS:** keep `?? .trocr` in `AppPreferences.swift`. In `apps/mac/README.md`, state that TrOCR (KV-cache CoreML) is the default and that it passed the gate on <date>.
- **FAIL on mixed only:** apply the spec FAIL policy below. In `mac_gate.md`, record that the follow-up is mixed Khmer+Latin synthetic data on the training side (without `ui_strings.txt` content).
- **FAIL on CER or latency after Steps 3–4:** apply the spec FAIL policy below, and record the trocr-small fallback as the next step.

Spec FAIL policy (only on a FAIL outcome), in `apps/mac/Sources/Preferences/AppPreferences.swift`:

```swift
self.ocrEngine = (defaults.string(forKey: Keys.ocrEngine)).flatMap(OCREngineKind.init) ?? .tesseract
```

- [ ] **Step 6: Full regression**

```bash
cd apps/mac && make trocr-test && make test 2>&1 | grep -E "error|Executed|failed" | tail -5
cd ../../training/eval && python -m pytest -q
```

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add training/eval/results/trocr_speed_accuracy.md training/eval/results/mac_gate.md apps/mac/README.md \
  apps/mac/Sources/OCR/TrOCREngine.swift
# only on a FAIL outcome:
# git add apps/mac/Sources/Preferences/AppPreferences.swift
git commit -m "docs(eval): record KV-cache CoreML TrOCR gate outcome and default policy"
```

---

## Self-review notes

- **Spec coverage:**
  - Gate criteria: Task 8, Steps 1 and 5.
  - CoreML-only latency measurement: Tasks 1, 4, and 8.
  - Small-model latency fallback and FAIL default policy: Task 8, Steps 4 and 5.
  - Published numbers: Tasks 1, 4, and 8.
  - The spec's train-side items (datasets, augs, 3090) are covered by the earlier plan and by `2026-10-03-trocr-finetune-new-data.md`.
- **Names used across tasks:**
  - Task 1: `RunStats`, `lastRunStats`, `milliseconds(since:)`
  - Task 2: `lineStrips`
  - Task 3: `resizedRGB(_:size:)`
  - Task 4: `maxLength`, `maxRun`, `blockedToken(_:maxRun:)`, `argmaxLastStep(_:excluding:)`, `hf_reference.py` CSV
  - Task 5: `KVDecoderStep(model, max_len, cache_dtype)`, with inputs `input_ids, position, attn_mask, write_mask, encoder_hidden_states`
  - Task 6: `convert_encoder`, `convert_decoder`, `MAX_LEN = 128`, states `k_cache`/`v_cache`; the Swift loop in Task 7 consumes these
  - Tasks 7 and 8: `KHMERCLIPPER_TROCR_UNITS`
- **Ordering:** Task 4 Step 6 blocks Task 5 until CoreML greedy CER is within 0.02 of `hf_reference.py` (same decode rules), so a leftover port bug cannot get mixed into the speed numbers.
