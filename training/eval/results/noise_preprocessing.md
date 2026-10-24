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
