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

## Step 2: robust background (`robust_bg`)

| Version | noisy (background kind) | frozen | screen | slices | Verdict |
|---|---|---|---|---|---|
| mode of the 1 px border | +4.7% (+14.0%) | 35 -> 46 | = | = | drop: a box border on the edge became "background" |
| mode of an edge band (min(h,w)//8 deep) | +0.6% (+1.9%) | 35 -> 42 | = | = | drop: crops moved on scanned paper (3 clips) |
| **band mode only when it differs from the corner mean by > 40** | -0.3% (-0.8%) | = | = | = | **keep (enabling step)**: no regression, latency within 10% |

On its own it barely moves the noisy set: gradients and textures still leak into the ink mask whichever single colour is chosen. It is kept because step 3 needs a background that survives a corner icon or a border on the edge.

## Step 3: drop non-text components (`drop_noise`), judged with step 2

First version: noisy -18.7% but frozen 35 -> 1106 errors. Cause: textured parchment scans are 82% "ink", so the whole clip was one block and got removed (the model then hallucinated); speckle dragged the median component height down, so real words counted as "isolated" and glyph strokes as "rules"; a 2 px CSS border is 4 px at 2x. Fixes, tuned with a proxy (share of ink removed per set: clean sets should lose ~0), then gated:

- typical glyph height = 90th percentile over components at least 3 px tall with 6+ px of ink (frozen clips losing > 2% ink: 50 -> 0);
- a mask over 30% ink, or a filter keeping < 30% of the ink, is a misread: filter nothing;
- rules are long as well as thin (keeps `=`, `-` and `។`); thickness limit max(3 px, 0.2 x glyph height);
- only components at least 0.8 x glyph height can be stray icons (keeps detached marks);
- components touching the clip edge are never blocks or stray icons (gradient patches leak in from the edge and fuse with glyphs);
- no noise pass inside `split_wide`: identical crops on 45/45 screen clips, preprocessing 63 -> 31 ms per clip.

## Step 4: drop cut-off edge fragments (`drop_edge`)

On top of steps 2+3: `noisy_edgecut` 1119 -> 880 errors (-21.4%, 4 better / 0 worse); every other set and kind unchanged. **Keep.**

## Steps 2+3+4 vs baseline

| Set | base errors | errors | change | better / worse | median ms |
|---|---:|---:|---:|---:|---|
| noisy 300 | 4349 | 3214 | **-26.1%** | 49 / 4 | 750 -> 771 |
| - avatar | 976 | 236 | -75.8% | 25 / 1 | |
| - underline | 199 | 134 | -32.7% | 6 / 0 | |
| - chrome | 232 | 165 | -28.9% | 12 / 2 | |
| - edgecut | 1115 | 880 | -21.1% | 4 / 0 | |
| - background | 1472 | 1444 | -1.9% | 2 / 1 | |
| - icons | 355 | 355 | 0 | 0 / 0 | |
| frozen 200 | 35 | 35 | 0 | 0 / 0 | 170 -> 189 |
| screen 450 | 716 | 716 | 0 | 0 / 0 | 691 -> 752 |
| slices 320 | 164 | 164 | 0 | 0 / 0 | 625 -> 641 |
| real 4 | 2 | 2 | 0 | 0 / 0 | |

Clean text is unchanged on every set. PyTorch-runner latency is +9% on screen clips and +11% (+19 ms) on the short frozen clips; the binding latency check is on device (final section). Icons inside a line are untouched by design: they sit on the text rows, close to glyphs.

## Step 5: paint removed noise with the background (`fill_noise`): dropped

On top of steps 2+3+4. Targets: icons, chrome.

| Version | icons + chrome | underline | avatar | noisy all | Verdict |
|---|---:|---:|---:|---:|---|
| paint everything removed (1 px growth) | 520 -> 490 (-5.8%, 9 / 7) | 134 -> 138 (+3.0%, 9 / 12) | -48.3% | -4.7% | drop: underline regressed |
| same, no growth | 520 -> 485 | 134 -> 141 (+5.2%) | -48.3% | -4.7% | drop: growth was not the cause |
| paint dropped components only (keep rules) | 520 -> 539 (+3.7%) | 134 -> 138 (+3.0%) | -48.3% | -2.8% | drop: chrome regressed |

Painting an underline out removes the descender pixels that cross it, and the model reads underlined text well anyway. Without painting frames, chrome gets worse. No version clears the bars, so the apps ship steps 2+3+4. `fill_noise` stays in the eval pipeline only, for the record. The avatar gain it would add (-48%) comes on top of step 3's -76%.

## On device: Mac (CoreML `OCRBench`, Latin fallback included), main vs this branch

| Set | main errors | errors | change | better / worse | median ms |
|---|---:|---:|---:|---:|---|
| frozen 200 | 11 | 11 | 0 | 0 / 0 | 101 -> 91 |
| screen 450 | 734 | 734 | 0 | 0 / 0 | 382 -> 401 (+5%) |
| slices 320 | 165 | 165 | 0 | 0 / 0 | 318 -> 285 |
| **noisy 300** | 4368 | **3239** | **-25.8%** | 47 / 5 | 452 -> 416 |
| - avatar | 966 | 240 | -75.2% | 26 / 1 | |
| - underline | 217 | 135 | -37.8% | 6 / 1 | |
| - chrome | 233 | 175 | -24.9% | 9 / 2 | |
| - edgecut | 1124 | 889 | -20.9% | 4 / 0 | |
| - background | 1473 | 1445 | -1.9% | 2 / 1 | |
| - icons | 355 | 355 | 0 | 0 / 0 | |
| real 4 | 12 | 12 | 0 | 0 / 0 | |

Every bar met on device: clean sets identical, latency within +10% on every set.

Parity: Swift and desktop give the same piece count on all 1276 fixtures. Piece and crop boxes are within 1 mask px on 1257, against 1256 on main. The rest is the existing CoreGraphics vs PIL downscaling of the 512 px crop mask, not this change.

## On device: desktop (ONNX on CPU, Latin fallback included), main vs this branch

CER here is a plain character edit distance (the desktop venv has no `jiwer`). It splits Khmer combining marks slightly differently, so compare within this table only.

| Set | main errors | errors | change | better / worse |
|---|---:|---:|---:|---:|
| frozen 200 | 39 | 30 | -23.1% | 2 / 0 (Tesseract now gets the noise-free crop) |
| screen 450 | 724 | 725 | +0.1% | 0 / 1 |
| slices 320 | 164 | 164 | 0 | 0 / 0 |
| **noisy 300** | 4349 | **3218** | **-26.0%** | 48 / 5 |
| real 4 | 2 | 2 | 0 | 0 / 0 |

Per kind on noisy: avatar -75.8%, underline -32.7%, chrome -27.2%, edgecut -21.1%, background -1.9%, icons 0. Latency is not worse on any set; main ran first and partly paid for warm-up, so its times are inflated.

## Outcome

Shipped in both apps: steps 2 (robust background), 3 (non-text components) and 4 (edge fragments). Dropped: step 5 (painting noise out). On device, errors on noisy screen captures fall about 26% (avatars -75%, underlines -33 to -38%, buttons/borders -25 to -27%, cut-off lines -21%). Clean text is unchanged, with one clip in 450 worse on desktop, and speed stays within budget.

Not solved: text on photo and gradient backgrounds (-2%, the largest remaining error at 31% CER), because a single background colour can't describe them. Icons on the text line are untouched (0%). Both need per-region background estimation or model-side training, which is outside this spec.
