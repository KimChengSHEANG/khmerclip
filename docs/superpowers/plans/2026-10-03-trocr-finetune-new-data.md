# TrOCR Khmer Fine-tune on New Data (fonts, italic, handwriting, documents) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collect new Khmer OCR training data — real handwriting, scanned/low-res documents, more fonts, italic (native + synthetic), and user-collected crops — and continue training the last TrOCR model on the 3090 box with it, without regressing on printed/UI text — then copy the new model into the Mac app, convert, build and run it.

**Architecture:** Extend `trocr_khmer.data_sources` with pinned-file dataset specs (fixes eval leakage, adds new HF datasets, local parquet and image folders), per-source holdout + weighted resampling + long-row filtering. Add a raqm-shaped synthetic line renderer (`trocr_khmer.synth`) that pre-renders seeded parquet shards across 33 Google Fonts Khmer files. `train.py` learns to start a *fresh* Trainer from a previous run's `final/` (reusing its Khmer tokenizer unchanged), with fine-tune LR/warmup and early stopping.

**Tech Stack:** Python 3.12 (pyenv venv `khmerclipper` on the 3090 box), PyTorch 2.12 + CUDA, transformers 4.57 (<5), datasets 5.0, Pillow 12.3 with libraqm, fontTools (new), pytest 9.

**Spec:** [docs/superpowers/specs/2026-08-27-khmer-ocr-accuracy-speed-design.md](../specs/2026-08-27-khmer-ocr-accuracy-speed-design.md) (parent design; its pass/fail gate and Mac CoreML flow still apply). This plan's own goals/constraints are below — there is no separate spec for this follow-up.

## Context the engineer needs (read first)

- **Where training runs:** `ssh kim@192.168.1.25`, repo at `/home/kim/Sync/workspace/khmerclipper`, trainer at `training/trocr`. Python: `~/.pyenv/versions/khmerclipper/bin/python` (has torch, transformers, datasets, pytest, editable `trocr_khmer` and `khmerclipper_eval`). RTX 3090 24GB, 12 cores, 31GB RAM, ~366GB free disk.
- **Worktree:** all work happens on branch `feat/trocr-finetune-data` in the worktree `.worktrees/trocr-finetune-data` (Task 0). Every relative path and `git` command in this plan runs from that worktree root, never from the main checkout.
- **Code sync:** the repo lives under `~/Sync` on both machines (Syncthing). Syncthing **ignores** `experiments/`, `cache/`, `__pycache__`, `.venv`. Edit on the Mac; Task 1 adds `training/trocr/scripts/remote.sh`, which rsyncs this checkout's `training/trocr` + `training/eval` to the **same path on the box** (`/home/kim/Sync/workspace/khmerclipper/.worktrees/trocr-finetune-data/training`) and runs a command there with `PYTHONPATH=src:../eval/src`. Use it for every test/run so you never race Syncthing. The box's main checkout stays untouched; its `experiments/` (old checkpoints + the new run) is reached through a symlink (Task 1 Step 1b).
- **What "the last trained model" is:** a single long run `experiments/checkpoint-3` (resumed from checkpoint-1 → 2 → 3; all 5 datasets, 3,513,080 steps, 10 epochs, lr 5e-5). Its mid-train CER was **best (0.0381) at step 72,000**, then drifted to ~0.38 at the end. Because `load_best_model_at_end=True`, `experiments/checkpoint-3/final/model.safetensors` is byte-identical (sha256 `04c0ac11…`) to `experiments/checkpoint-1/checkpoint-72000`. The literal last step is `experiments/checkpoint-3/checkpoint-3513080` (sha256 `71c48269…`). Task 1 measures both; default init is `final/`.
- **Do not use `--resume_from_checkpoint`** to fine-tune: it restores global_step 3.5M of max_steps 3,513,080 plus the old optimizer/LR schedule, so the Trainer considers itself done. Fine-tuning = `--model_name <dir>` with a fresh Trainer (Task 4).
- **Leakage found:** `SoyVitou/62k-images-khmer-printed-dataset` exposes `trainset/validset/testset.parquet` but the hub card groups all 62,680 rows under `train`, so the old `soyvitou` alias trained on `testset.parquet` — the source of the 100 printed eval fixtures. Baseline printed-fixture CER for the old model is therefore optimistic. Task 2 pins soyvitou to train+valid. Keep fixtures frozen.
- **New data sources (verified on the Hub 2026-10-03):**

| alias | source | rows | content |
| --- | --- | ---: | --- |
| `handwriting` | `SoyVitou/khmer-handwritten-dataset-4.2k` trainset+valset parquet (MIT) | ~3.8k | real handwritten lines/words |
| `handwriting_test` | same, `testset.parquet` | ~0.4k | held-out handwriting eval only |
| `darayut` | `Darayut/khmer-document-synthetic-low-res` (`label` column) | 100k | low-res document lines, mixed digits/punct |
| `siemreap` | `KimkosalYon/khmer-ocr-200k-siemreap-arial` (MIT) | 200k | Siemreap/Arial font lines, some very long |
| `synth` / `synth_val` | local parquet rendered by Task 5–6 into `cache/synth/{train,val}` | 400k / 3k | 33 font files, italic, handwriting-style fonts, document + UI styles |
| any dir path | local `imagefolder` with `metadata.jsonl` (`file_name`, `text`) | — | user-collected crops (app screenshots, phone photos) |

  Rejected: `lkhapple/KhmerOCR-Dataset` (no labels), `sopheakvoatei/*surya*` (24 page-level rows), `Thareah/*` (empty), `Darachhat/khmer-document-corpus` (code repo, no data files).

## Global Constraints

- Remote host `kim@192.168.1.25`; remote repo `/home/kim/Sync/workspace/khmerclipper`; remote worktree mirror `/home/kim/Sync/workspace/khmerclipper/.worktrees/trocr-finetune-data`; remote python `~/.pyenv/versions/khmerclipper/bin/python`.
- Branch `feat/trocr-finetune-data` in worktree `.worktrees/trocr-finetune-data`; nothing is committed to `main` until the branch is finished (Task 9 Step 10).
- Remote commands must import the worktree's code: always `PYTHONPATH=src:../eval/src` (built into `remote.sh` and the tmux commands).
- `transformers>=4.40.0,<5` stays pinned (pyproject); do not upgrade torch/transformers/datasets.
- New dependency allowed: `fonttools>=4.40` only.
- Generated data, fonts and checkpoints live under `training/trocr/cache/` or `training/trocr/experiments/` (gitignored and Syncthing-ignored). Never commit fonts, parquet, or weights.
- Khmer rendering must use `ImageFont.Layout.RAQM`; fail closed if `PIL.features.check("raqm")` is False.
- Eval fixture texts (`training/eval/fixtures/labels.jsonl`) and `training/eval/ui_strings.txt` lines must never appear in synthetic training text.
- Fine-tuning starts a fresh Trainer from `experiments/checkpoint-3/final` (or the Task 1 winner); never `--resume_from_checkpoint` across runs.
- Loading a local fine-tuned checkpoint must add **0** tokens and keep decoder vocab **50414**.
- `MAX_LABEL_LEN = 128` tokens (unchanged).
- Long runs go in `tmux` on the box; do not keep an interactive session attached.

## Review Focus

1. **Unshaped Khmer:** if Pillow renders without raqm, COENG subscripts/vowel signs land in the wrong place and every synthetic label is wrong — expect a hard failure, not silent garbage. (Task 5 test `test_raqm_shapes_coeng_cluster` + Task 6 Step 2 eyeball check.)
2. **Fonts without Latin/digit glyphs on mixed text:** display fonts like Moul lack Latin; rendering "ឆ្នាំ 2024" would produce tofu boxes under a correct-looking label — expect such fonts to be skipped for that line. (Task 5 test `test_render_sample_skips_fonts_missing_glyphs`.)
3. **Upsampled duplicates leaking into the holdout:** `handwriting:60000` repeats ~3.8k rows ~16×; if the split happened after resampling, val CER would be fake — expect zero train/eval overlap. (Task 3 test `test_split_keeps_upsampled_rows_out_of_eval`.)
4. **Token-id shift when continuing from a fine-tuned dir:** re-grafting the base tokenizer or re-adding Khmer tokens in a different order would scramble the decoder — expect loading to refuse rather than resize. (Task 4 tests.)
5. **Eval leakage via data:** soyvitou `testset.parquet` and fixture strings inside the synthetic corpus — expect both excluded. (Task 2 test `test_soyvitou_excludes_testset`, Task 5 test `test_build_corpus_excludes_eval_text`.)

---

### Task 0: Create the worktree

**Files:**
- Move: `docs/superpowers/plans/2026-10-03-trocr-finetune-new-data.md` (untracked in main) → same path in the worktree

**Interfaces:**
- Produces: worktree `.worktrees/trocr-finetune-data` on branch `feat/trocr-finetune-data`, based on `main`. All later tasks run from its root.

- [ ] **Step 1: Create it (REQUIRED SUB-SKILL: superpowers:using-git-worktrees)**

The repo already uses `.worktrees/` (gitignored). From the main checkout:

```bash
cd /Users/kim/Sync/workspace/khmerclipper
git status --short   # expect only: ?? docs/superpowers/plans/2026-10-03-trocr-finetune-new-data.md
git worktree add .worktrees/trocr-finetune-data -b feat/trocr-finetune-data main
mkdir -p .worktrees/trocr-finetune-data/docs/superpowers/plans
mv docs/superpowers/plans/2026-10-03-trocr-finetune-new-data.md .worktrees/trocr-finetune-data/docs/superpowers/plans/
cd .worktrees/trocr-finetune-data
```

Expected: `git worktree list` shows `.worktrees/trocr-finetune-data … [feat/trocr-finetune-data]`; main's `git status` is clean again.

- [ ] **Step 2: Commit the plan on the branch**

```bash
git add docs/superpowers/plans/2026-10-03-trocr-finetune-new-data.md
git commit -m "docs: plan for TrOCR fine-tune on new Khmer data"
```

(Remote baseline tests run in Task 1 Step 1b, once `remote.sh` exists.)

---

### Task 1: Remote runner + baseline of the starting checkpoints

**Files:**
- Create: `training/trocr/scripts/remote.sh`
- Create: `training/eval/scripts/subset_cer.py`
- Create: `training/eval/results/finetune_baseline.md`
- Modify: `training/trocr/.gitignore` (add `cache/`)

**Interfaces:**
- Produces: `training/trocr/scripts/remote.sh '<command>'` — rsyncs code, runs `<command>` in remote `training/trocr` with the khmerclipper venv first on `PATH`. `python ../eval/scripts/subset_cer.py <csv>` prints per-subset CER. `experiments/last-step/` on the box: loadable model dir for the literal last step.

- [ ] **Step 1: Write `remote.sh`**

```bash
#!/usr/bin/env bash
# Push training/trocr + training/eval to the 3090 box, then run a command in training/trocr there.
#   scripts/remote.sh 'python -m pytest tests -q'
# Pass the remote command as ONE quoted string.
set -euo pipefail
HOST=${TROCR_HOST:-kim@192.168.1.25}
cd "$(dirname "$0")/../.."  # -> training/ of this checkout (main or a worktree)
# Mirror the same path under ~/Sync on the box, e.g. .../khmerclipper/.worktrees/trocr-finetune-data/training,
# so a worktree never overwrites the main checkout there (Syncthing would carry that back to main).
REMOTE_TRAINING=/home/kim/${PWD#"$HOME"/}
ssh "$HOST" "mkdir -p $REMOTE_TRAINING"
rsync -a \
  --exclude experiments --exclude cache --exclude __pycache__ --exclude .pytest_cache \
  --exclude venv --exclude .venv --exclude '*.egg-info' \
  trocr eval "$HOST:$REMOTE_TRAINING/"
# PYTHONPATH beats the venv's editable trocr_khmer install, which points at main's src.
ssh "$HOST" "cd $REMOTE_TRAINING/trocr && export PATH=\$HOME/.pyenv/versions/khmerclipper/bin:\$PATH && export PYTHONPATH=src:../eval/src && $1"
```

Run: `chmod +x training/trocr/scripts/remote.sh && training/trocr/scripts/remote.sh 'python -c "import torch, trocr_khmer; print(torch.cuda.is_available(), trocr_khmer.__file__)"'`
Expected: `True /home/kim/Sync/workspace/khmerclipper/.worktrees/trocr-finetune-data/training/trocr/src/trocr_khmer/__init__.py` — a path under `main`'s `training/trocr/src` means PYTHONPATH isn't taking effect; fix before continuing.

- [ ] **Step 1b: Link the existing checkpoints into the remote worktree**

`experiments/` (the old runs, and where the new run lands) stays in the main checkout on the box; symlink it so every relative `experiments/...` path in this plan works from the worktree. Syncthing ignores `experiments`, so the link stays box-local.

Run: `training/trocr/scripts/remote.sh 'ln -sfn /home/kim/Sync/workspace/khmerclipper/training/trocr/experiments experiments && ls experiments && python -m pytest tests -q'`
Expected: `checkpoint-1  checkpoint-2  checkpoint-3`, then the existing tests pass (clean baseline for the branch).

- [ ] **Step 2: Ignore generated data in git**

Append to `training/trocr/.gitignore` under `# Model artifacts`:

```gitignore
# Fonts + rendered synthetic data (scripts/fetch_fonts.py, scripts/render_synth.py)
cache/
```

- [ ] **Step 3: Write `subset_cer.py`**

```python
#!/usr/bin/env python3
"""Print mean CER per subset (printed / ui / mixed / all) for each engine in a bench.py CSV."""
import csv
import sys
from collections import defaultdict

by = defaultdict(list)
with open(sys.argv[1], encoding="utf-8") as f:
    for row in csv.DictReader(f):
        cer = float(row["cer"])
        by[(row["engine"], row["subset"])].append(cer)
        by[(row["engine"], "all")].append(cer)
for (engine, subset), cers in sorted(by.items()):
    print(f"{engine}\t{subset}\tn={len(cers)}\tcer={sum(cers) / len(cers):.4f}")
```

- [ ] **Step 4: Make the literal last step loadable**

Trainer `checkpoint-N/` dirs hold weights but no tokenizer/image processor. Symlink them together:

```bash
training/trocr/scripts/remote.sh 'mkdir -p experiments/last-step && cd experiments/last-step \
  && ln -sf ../checkpoint-3/checkpoint-3513080/model.safetensors ../checkpoint-3/checkpoint-3513080/config.json . \
  && (test -f ../checkpoint-3/checkpoint-3513080/generation_config.json && ln -sf ../checkpoint-3/checkpoint-3513080/generation_config.json . || true) \
  && for f in added_tokens.json merges.txt preprocessor_config.json special_tokens_map.json tokenizer_config.json tokenizer.json vocab.json; do ln -sf ../checkpoint-3/final/$f .; done && ls -L'
```

Expected: 9–10 file names listed by `ls -L`, no "No such file" errors.

- [ ] **Step 5: Bench both on the frozen fixtures (beam-4, HF)**

```bash
training/trocr/scripts/remote.sh 'mkdir -p experiments/bench && for m in checkpoint-3/final last-step; do \
  out=experiments/bench/baseline_$(echo $m | tr / _).csv; \
  python ../eval/scripts/bench.py --engine trocr-hf --trocr-model-dir experiments/$m --out $out \
  && python ../eval/scripts/subset_cer.py $out; done'
```

Expected: two blocks of `trocr-hf  all/mixed/printed/ui  n=…  cer=…` lines (n=200 for `all`).

- [ ] **Step 6: Record results and pick the init**

Create `training/eval/results/finetune_baseline.md` (fill every `…` from Step 5 output; date = today):

```markdown
# Fine-tune baseline (starting checkpoints)

Recorded <date> on the 3090 box, `bench.py --engine trocr-hf` (beam 4), frozen 200 fixtures.

| Model | all | printed | ui | mixed |
| --- | ---: | ---: | ---: | ---: |
| `checkpoint-3/final` (= step 72k, best mid-train CER 0.0381) | … | … | … | … |
| `last-step` (step 3,513,080, mid-train CER ~0.38) | … | … | … | … |
| Tesseract (baseline.md) | 0.2580 | 0.5370 | 0.0199 | 0.0894 |

Caveat: the old `soyvitou` alias trained on `testset.parquet`, which is where the printed fixtures
come from, so printed CER for both rows is optimistic.

**Init for fine-tuning:** `<winner>` (lower `all` CER).
```

`INIT` for later tasks = `experiments/checkpoint-3/final` unless `last-step` has the lower `all` CER, in which case `INIT=experiments/last-step`. Every later command that names `experiments/checkpoint-3/final` as the model means `INIT`.

- [ ] **Step 7: Commit**

```bash
git add training/trocr/scripts/remote.sh training/trocr/.gitignore training/eval/scripts/subset_cer.py training/eval/results/finetune_baseline.md
git commit -m "chore(trocr): remote runner and fine-tune baseline of starting checkpoints"
```

---

### Task 2: Dataset specs with pinned files + new sources

**Files:**
- Modify: `training/trocr/src/trocr_khmer/data_sources.py` (registry, `resolve_dataset`, loader)
- Modify: `training/trocr/src/trocr_khmer/__init__.py` (export `DatasetSpec`)
- Modify: `training/trocr/README.md` (Datasets table)
- Test: `training/trocr/tests/test_data_sources.py`

**Interfaces:**
- Produces:
  - `class DatasetSpec(NamedTuple)`: `path: str`, `image_col: str = "image"`, `text_col: str = "text"`, `data_files: list[str] | None = None`, `data_dir: str | None = None`; method `load_kwargs() -> dict` (only the non-None of `data_files`/`data_dir`).
  - `resolve_dataset(name: str) -> DatasetSpec` — registry alias → spec; existing directory → `DatasetSpec("imagefolder", data_dir=name)`; else raw hub id `DatasetSpec(name)`.
  - `_load_one(spec, smoke_test=False, smoke_samples=200) -> Dataset` — `load_dataset(spec.path, split="train", **spec.load_kwargs())`, columns normalized to `image`/`text`.
  - `_canonical(ds) -> Dataset` — casts to `Features({"image": Image(), "text": Value("string")})` only if features differ.
  - `load_ocr_datasets(names, smoke_test=False, smoke_samples=200) -> Dataset` (signature unchanged).
  - Aliases: `soyvitou` (pinned), `sokheng`, `hanuman`, `mrrtmob`, `kiteaether`, `handwriting`, `handwriting_test`, `darayut`, `siemreap`, `synth`, `synth_val`.

- [ ] **Step 1: Write the failing tests**

Create `training/trocr/tests/test_data_sources.py`:

```python
from datasets import Dataset, Features, Value
from datasets import Image as DSImage
from PIL import Image

from trocr_khmer import data_sources
from trocr_khmer.data_sources import DatasetSpec, load_ocr_datasets, resolve_dataset

CANONICAL = Features({"image": DSImage(), "text": Value("string")})


def _rows(texts, text_col="text", size=(64, 32), extra=None, text_type="string"):
    data = {"image": [Image.new("RGB", size, "white") for _ in texts], text_col: list(texts)}
    feats = {"image": DSImage(), text_col: Value(text_type)}
    if extra:
        data[extra] = ["x"] * len(texts)
        feats[extra] = Value("string")
    return Dataset.from_dict(data, features=Features(feats))


def test_soyvitou_excludes_testset():
    spec = resolve_dataset("soyvitou")
    assert spec.path == "parquet"
    assert spec.data_files and all("testset" not in f for f in spec.data_files)
    assert any("trainset.parquet" in f for f in spec.data_files)


def test_handwriting_train_and_test_are_disjoint_files():
    train = set(resolve_dataset("handwriting").data_files)
    test = set(resolve_dataset("handwriting_test").data_files)
    assert not train & test
    assert all(f.endswith("testset.parquet") for f in test)


def test_darayut_reads_label_column():
    assert resolve_dataset("darayut").text_col == "label"


def test_unknown_name_is_raw_hub_id():
    assert resolve_dataset("org/some-dataset") == DatasetSpec("org/some-dataset")


def test_existing_directory_is_imagefolder(tmp_path):
    spec = resolve_dataset(str(tmp_path))
    assert spec == DatasetSpec("imagefolder", data_dir=str(tmp_path))
    assert spec.load_kwargs() == {"data_dir": str(tmp_path)}


def test_loader_passes_pinned_files(monkeypatch):
    calls = []

    def fake_load(path, split, **kwargs):
        calls.append((path, split, kwargs))
        return _rows(["ក", "ខ"])

    monkeypatch.setattr(data_sources, "load_dataset", fake_load)
    load_ocr_datasets(["handwriting"])
    assert calls == [("parquet", "train", {"data_files": resolve_dataset("handwriting").data_files})]


def test_loader_normalizes_columns_and_features(monkeypatch):
    monkeypatch.setattr(
        data_sources,
        "load_dataset",
        lambda path, split, **kw: _rows(["ក", "ខ"], text_col="label", extra="source", text_type="large_string"),
    )
    ds = load_ocr_datasets(["darayut"])
    assert ds.column_names == ["image", "text"]
    assert ds.features == CANONICAL
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `training/trocr/scripts/remote.sh 'python -m pytest tests/test_data_sources.py -v'`
Expected: FAIL — `ImportError: cannot import name 'DatasetSpec'`.

- [ ] **Step 3: Implement the spec registry and loader**

In `training/trocr/src/trocr_khmer/data_sources.py`, replace everything from the `from datasets import …` line through the end of `load_ocr_datasets` (leave `load_ocr_split` as is for now) with:

```python
from pathlib import Path
from typing import NamedTuple

from datasets import Dataset, Features, Value, concatenate_datasets, load_dataset
from datasets import Image as DSImage

_SOYVITOU = "hf://datasets/SoyVitou/62k-images-khmer-printed-dataset"
_HANDWRITING = "hf://datasets/SoyVitou/khmer-handwritten-dataset-4.2k"

# Every source is normalized to exactly these columns/types so sources concatenate cleanly.
_FEATURES = Features({"image": DSImage(), "text": Value("string")})


class DatasetSpec(NamedTuple):
    """How to load one source. ``path`` is a hub id, or a builder name ("parquet",
    "imagefolder") when ``data_files`` / ``data_dir`` pin exactly which files to read."""

    path: str
    image_col: str = "image"
    text_col: str = "text"
    data_files: list[str] | None = None
    data_dir: str | None = None

    def load_kwargs(self) -> dict:
        return {k: v for k, v in (("data_files", self.data_files), ("data_dir", self.data_dir)) if v is not None}


DATASET_REGISTRY = {
    # The hub card lumps train/valid/test parquets into one "train" split. testset.parquet is
    # where training/eval/fixtures/printed comes from, so it must never be trained on.
    "soyvitou": DatasetSpec("parquet", data_files=[f"{_SOYVITOU}/trainset.parquet", f"{_SOYVITOU}/validset.parquet"]),
    "sokheng": DatasetSpec("Sokheng/khmer-synthetic-ocr-v1-100k"),
    "hanuman": DatasetSpec("seanghay/khmer-hanuman-100k"),
    "mrrtmob": DatasetSpec("mrrtmob/khmer_english_ocr_image_line"),  # gated: 12M Khmer+English lines
    "kiteaether": DatasetSpec("KiteAether/c_khmer_gemma_ocr_train"),
    # Real handwriting; testset.parquet is held out for evaluation (handwriting_test).
    "handwriting": DatasetSpec("parquet", data_files=[f"{_HANDWRITING}/trainset.parquet", f"{_HANDWRITING}/valset.parquet"]),
    "handwriting_test": DatasetSpec("parquet", data_files=[f"{_HANDWRITING}/testset.parquet"]),
    "darayut": DatasetSpec("Darayut/khmer-document-synthetic-low-res", text_col="label"),  # low-res document lines
    "siemreap": DatasetSpec("KimkosalYon/khmer-ocr-200k-siemreap-arial"),  # Siemreap/Arial font lines
    # Rendered locally by scripts/render_synth.py (paths relative to training/trocr).
    "synth": DatasetSpec("parquet", data_files=["cache/synth/train/*.parquet"]),
    "synth_val": DatasetSpec("parquet", data_files=["cache/synth/val/*.parquet"]),
}


# What the scripts train/eval on when --datasets isn't passed.
DEFAULT_DATASETS = ["soyvitou"]


def resolve_dataset(name: str) -> DatasetSpec:
    """Registry alias -> its spec; an existing directory -> a local imagefolder (images plus a
    ``metadata.jsonl`` with ``file_name`` and ``text``); anything else -> a raw hub id."""
    if name in DATASET_REGISTRY:
        return DATASET_REGISTRY[name]
    if Path(name).is_dir():
        return DatasetSpec("imagefolder", data_dir=name)
    return DatasetSpec(name)


def _normalize_columns(ds: Dataset, image_col: str, text_col: str) -> Dataset:
    """Rename the image/text columns to the canonical names and drop everything else, so
    datasets with differing schemas concatenate cleanly."""
    rename = {}
    if image_col != "image":
        rename[image_col] = "image"
    if text_col != "text":
        rename[text_col] = "text"
    if rename:
        ds = ds.rename_columns(rename)
    extra = [c for c in ds.column_names if c not in ("image", "text")]
    if extra:
        ds = ds.remove_columns(extra)
    return ds


def _canonical(ds: Dataset) -> Dataset:
    """Cast to the shared features only when they differ. Call this on already-subsampled data:
    a cast rewrites every row, which is ruinous on the 12M-row mrrtmob."""
    return ds if ds.features == _FEATURES else ds.cast(_FEATURES)


def _load_one(spec: DatasetSpec, smoke_test: bool = False, smoke_samples: int = 200) -> Dataset:
    if smoke_test:
        stream = load_dataset(spec.path, split="train", streaming=True, **spec.load_kwargs())
        ds = Dataset.from_list(list(stream.take(smoke_samples)))
    else:
        ds = load_dataset(spec.path, split="train", **spec.load_kwargs())
    return _normalize_columns(ds, spec.image_col, spec.text_col)


def load_ocr_datasets(names, smoke_test: bool = False, smoke_samples: int = 200) -> Dataset:
    """Load every row of each named source and concatenate (no holdout carved).

    ``smoke_test`` streams only the first ``smoke_samples`` rows of each (no full download).
    """
    parts = []
    for name in names:
        spec = resolve_dataset(name)
        print(f"Loading dataset {name} ({spec.path})" + (f" [smoke: {smoke_samples}]" if smoke_test else ""))
        parts.append(_canonical(_load_one(spec, smoke_test, smoke_samples)))
    if len(parts) == 1:
        return parts[0]
    combined = concatenate_datasets(parts)
    print(f"Combined {len(names)} datasets -> {len(combined)} samples")
    return combined
```

Update the module docstring: mention the new aliases (`handwriting`, `handwriting_test`, `darayut`, `siemreap`, `synth`, `synth_val`) and that a directory path loads as a local imagefolder.

In `training/trocr/src/trocr_khmer/__init__.py` add `DatasetSpec` to the `data_sources` import and to `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `training/trocr/scripts/remote.sh 'python -m pytest tests -v'`
Expected: all PASS (7 new + existing augment/runtime tests).

- [ ] **Step 5: Verify the pinned hub files actually load**

Run: `training/trocr/scripts/remote.sh 'python -c "from trocr_khmer import load_ocr_datasets as l; [print(n, len(l([n]))) for n in [\"soyvitou\", \"handwriting\", \"handwriting_test\", \"darayut\"]]"'`
Expected: soyvitou ≈ 56,4xx (62,680 minus the 6,269 test rows), handwriting + handwriting_test sum to 4,202, darayut 100000. No errors.

- [ ] **Step 6: Update README Datasets table**

In `training/trocr/README.md`, replace the Datasets table with the five original rows plus the six new rows from this plan's "New data sources" table, and add below it:

```markdown
`soyvitou` reads only `trainset.parquet` + `validset.parquet`; its `testset.parquet` holds the
printed eval fixtures. A directory path loads as a local `imagefolder` (images + `metadata.jsonl`
with `file_name`, `text`) — put your own collected crops (app screenshots, phone photos) in
`cache/collected/<name>/` and pass that path to `--datasets`.
```

- [ ] **Step 7: Commit**

```bash
git add training/trocr/src/trocr_khmer/data_sources.py training/trocr/src/trocr_khmer/__init__.py training/trocr/tests/test_data_sources.py training/trocr/README.md
git commit -m "feat(trocr): pinned dataset specs, handwriting/document/font sources, no soyvitou test leakage"
```

---

### Task 3: Weighted mixing, per-source holdout, long-row filter

**Files:**
- Modify: `training/trocr/src/trocr_khmer/data_sources.py` (`parse_source`, `resample`, `filter_ocr_rows`, `load_ocr_split`)
- Modify: `training/trocr/src/trocr_khmer/__init__.py` (export new names)
- Modify: `training/trocr/scripts/train.py` (new flags, split after tokenizer is ready)
- Modify: `training/trocr/scripts/evaluate_model.py` (`--full`, `--max_eval_per_source`)
- Modify: `training/trocr/README.md` (mixing syntax)
- Test: `training/trocr/tests/test_data_mixing.py`

**Interfaces:**
- Consumes: `resolve_dataset`, `_load_one`, `_canonical`, `load_ocr_datasets` (Task 2).
- Produces:
  - `parse_source(entry: str) -> tuple[str, int | None]` — `"handwriting:60000"` → `("handwriting", 60000)`.
  - `resample(ds, n: int, seed: int) -> Dataset` — deterministic shuffled subset (n < len) or repeated copies (n > len); `ValueError` on empty ds.
  - `filter_ocr_rows(ds, tokenizer=None, max_label_len=MAX_LABEL_LEN, max_aspect=None, num_proc=None) -> Dataset` — drops rows whose untruncated token count > `max_label_len` and/or image `width/height > max_aspect`; returns `ds` itself when both criteria are off.
  - `load_ocr_split(names, test_size=0.03, seed=42, smoke_test=False, smoke_samples=200, max_eval_per_source=None, tokenizer=None, max_label_len=MAX_LABEL_LEN, max_aspect=None, num_proc=None) -> DatasetDict` with `train`/`test`.
  - `train.py` flags: `--max_eval_per_source INT`, `--max_aspect FLOAT`, `--filter_num_proc INT` (default 8). `evaluate_model.py` flags: `--full`, `--max_eval_per_source INT`.

- [ ] **Step 1: Write the failing tests**

Create `training/trocr/tests/test_data_mixing.py`:

```python
from types import SimpleNamespace

import pytest
from datasets import Dataset, Features, Value
from datasets import Image as DSImage
from PIL import Image

from trocr_khmer import data_sources
from trocr_khmer.data_sources import filter_ocr_rows, load_ocr_split, parse_source, resample

FEATS = Features({"image": DSImage(), "text": Value("string")})


def _source(prefix, n):
    return Dataset.from_dict(
        {"image": [Image.new("RGB", (64, 32), "white") for _ in range(n)], "text": [f"{prefix}{i}" for i in range(n)]},
        features=FEATS,
    )


class _FakeTokenizer:
    """One token per character plus <s>/</s>, like the Khmer-extended tokenizer on Khmer text."""

    cls_token_id = 0

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=[0, *range(3, 3 + len(text)), 2])


@pytest.fixture
def fake_hub(monkeypatch):
    sources = {"fake/a": _source("a", 100), "fake/b": _source("b", 50)}
    monkeypatch.setattr(data_sources, "load_dataset", lambda path, split, **kw: sources[path])
    return sources


def test_parse_source():
    assert parse_source("handwriting:60000") == ("handwriting", 60000)
    assert parse_source("org/name") == ("org/name", None)
    assert parse_source("odd:name") == ("odd:name", None)


def test_resample_upsamples_with_every_row_and_subsamples_uniquely():
    ds = _source("x", 10)
    up = resample(ds, 25, seed=0)
    assert len(up) == 25 and set(up["text"]) == set(ds["text"])
    down = resample(ds, 4, seed=0)
    assert len(down) == 4 and len(set(down["text"])) == 4
    assert resample(ds, 4, seed=0)["text"] == down["text"]  # deterministic


def test_resample_empty_raises():
    with pytest.raises(ValueError):
        resample(_source("x", 0), 5, seed=0)


def test_split_keeps_upsampled_rows_out_of_eval(fake_hub):
    split = load_ocr_split(["fake/a:1000", "fake/b"], test_size=0.1, seed=42)
    train, test = set(split["train"]["text"]), set(split["test"]["text"])
    assert not train & test
    assert sum(t.startswith("a") for t in split["train"]["text"]) == 1000


def test_eval_is_capped_per_source(fake_hub):
    split = load_ocr_split(["fake/a", "fake/b"], test_size=0.2, seed=42, max_eval_per_source=3)
    texts = split["test"]["text"]
    assert len(texts) == 6
    assert sum(t.startswith("a") for t in texts) == 3


def test_filter_drops_long_labels_and_wide_images():
    ds = Dataset.from_dict(
        {
            "image": [Image.new("RGB", s, "white") for s in [(64, 32), (64, 32), (2000, 32)]],
            "text": ["ok", "x" * 200, "wide"],
        },
        features=FEATS,
    )
    kept = filter_ocr_rows(ds, tokenizer=_FakeTokenizer(), max_label_len=128, max_aspect=20)
    assert kept["text"] == ["ok"]


def test_filter_is_noop_without_criteria():
    ds = _source("x", 3)
    assert filter_ocr_rows(ds) is ds
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `training/trocr/scripts/remote.sh 'python -m pytest tests/test_data_mixing.py -v'`
Expected: FAIL — `ImportError: cannot import name 'filter_ocr_rows'`.

- [ ] **Step 3: Implement mixing, filter, and the new split**

In `data_sources.py`, add `DatasetDict` to the `from datasets import …` line and add `from trocr_khmer.dataset import MAX_LABEL_LEN`. Replace the existing `load_ocr_split` with:

```python
def parse_source(entry: str) -> tuple[str, int | None]:
    """``"name:N"`` -> ``(name, N)`` to train on exactly N rows of that source per epoch;
    a plain name -> ``(name, None)`` for all rows."""
    name, sep, rows = entry.rpartition(":")
    if sep and rows.isdigit():
        return name, int(rows)
    return entry, None


def resample(ds: Dataset, n: int, seed: int) -> Dataset:
    """Deterministically take ``n`` rows: a shuffled subset when ``n < len(ds)``, whole shuffled
    copies (plus a remainder) when ``n > len(ds)`` — upsampling small sources like handwriting."""
    if len(ds) == 0:
        raise ValueError("cannot resample an empty dataset")
    shuffled = ds.shuffle(seed=seed)
    return shuffled.select([i % len(shuffled) for i in range(n)])


def filter_ocr_rows(ds: Dataset, tokenizer=None, max_label_len: int = MAX_LABEL_LEN,
                    max_aspect: float | None = None, num_proc: int | None = None) -> Dataset:
    """Drop rows the model can't learn from: labels longer than ``max_label_len`` tokens (they
    would be silently truncated, teaching early stops) and images wider than ``max_aspect``
    (unreadable after the 384x384 squash)."""
    if tokenizer is None and max_aspect is None:
        return ds

    def keep(example) -> bool:
        if tokenizer is not None and len(tokenizer(example["text"]).input_ids) > max_label_len:
            return False
        if max_aspect is not None:
            width, height = example["image"].size
            if height == 0 or width / height > max_aspect:
                return False
        return True

    return ds.filter(keep, num_proc=num_proc)


def load_ocr_split(names, test_size: float = 0.03, seed: int = 42,
                   smoke_test: bool = False, smoke_samples: int = 200,
                   max_eval_per_source: int | None = None, tokenizer=None,
                   max_label_len: int = MAX_LABEL_LEN, max_aspect: float | None = None,
                   num_proc: int | None = None) -> DatasetDict:
    """Per source: carve a seeded holdout, cap it, resample the train part (``name:N``), filter
    both, then concatenate across sources.

    Splitting before resampling keeps upsampled duplicates out of the holdout; per-source caps
    stop one huge source (mrrtmob) from being the whole eval set. Same names + seed -> same
    holdout, so ``scripts/evaluate_model.py`` reproduces training's eval rows.
    """
    trains, tests = [], []
    for entry in names:
        name, rows = parse_source(entry)
        spec = resolve_dataset(name)
        print(f"Loading dataset {name} ({spec.path})" + (f" [smoke: {smoke_samples}]" if smoke_test else ""))
        split = _load_one(spec, smoke_test, smoke_samples).train_test_split(test_size=test_size, seed=seed)
        train, test = split["train"], split["test"]
        if max_eval_per_source is not None:
            test = test.select(range(min(max_eval_per_source, len(test))))
        if rows is not None and not smoke_test:
            train = resample(train, rows, seed)
        train = _canonical(filter_ocr_rows(train, tokenizer, max_label_len, max_aspect, num_proc))
        test = _canonical(filter_ocr_rows(test, tokenizer, max_label_len, max_aspect, num_proc))
        print(f"  {name}: train {len(train)} / eval {len(test)}")
        trains.append(train)
        tests.append(test)
    return DatasetDict(train=concatenate_datasets(trains), test=concatenate_datasets(tests))
```

Export `filter_ocr_rows`, `parse_source`, `resample` from `__init__.py` (import + `__all__`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `training/trocr/scripts/remote.sh 'python -m pytest tests -v'`
Expected: all PASS.

- [ ] **Step 5: Wire the flags into `train.py`**

Add after the `--augment` argument:

```python
    parser.add_argument(
        "--max_eval_per_source",
        type=int,
        default=None,
        help="Cap each source's holdout so the eval set covers every domain (e.g. 300)",
    )
    parser.add_argument(
        "--max_aspect",
        type=float,
        default=None,
        help="Drop rows whose image width/height exceeds this (unreadable after the 384x384 resize)",
    )
    parser.add_argument("--filter_num_proc", type=int, default=8, help="Processes for the length/aspect filter")
```

Append to the `--datasets` help string: `" Append :N to train on N rows of a source per epoch (upsample/subsample), e.g. handwriting:60000."`

Move the `split = load_ocr_split(...)` statement from before the processor load to directly after the `print(f"Added {num_added} Khmer tokens; …")` line (the length filter needs the Khmer-extended tokenizer), and change it to:

```python
    split = load_ocr_split(
        args.datasets,
        test_size=0.03,
        seed=42,
        smoke_test=args.smoke_test,
        max_eval_per_source=args.max_eval_per_source,
        tokenizer=processor.tokenizer,
        max_aspect=args.max_aspect,
        num_proc=args.filter_num_proc,
    )
```

- [ ] **Step 6: Wire `evaluate_model.py`**

Add `load_ocr_datasets` to its `trocr_khmer` import. Add arguments:

```python
    parser.add_argument(
        "--full",
        action="store_true",
        help="Evaluate every row of --datasets (for dedicated test sets like handwriting_test, synth_val)",
    )
    parser.add_argument("--max_eval_per_source", type=int, default=None)
```

Replace the two lines `split = load_ocr_split(args.datasets, test_size=0.03, seed=42)` and `split["test"] = limit_eval_dataset(split["test"], args.max_eval_samples)` with:

```python
    if args.full:
        eval_hf = load_ocr_datasets(args.datasets)
    else:
        eval_hf = load_ocr_split(
            args.datasets, test_size=0.03, seed=42, max_eval_per_source=args.max_eval_per_source
        )["test"]
    eval_hf = limit_eval_dataset(eval_hf, args.max_eval_samples)
```

and change `eval_dataset = OCRDataset(split["test"], processor)` to `eval_dataset = OCRDataset(eval_hf, processor)`.

- [ ] **Step 7: Smoke both scripts on the box**

Run: `training/trocr/scripts/remote.sh 'python scripts/train.py --smoke_test --datasets handwriting:999 darayut --max_eval_per_source 3 --max_aspect 20 --output_dir experiments/smoke-mix'`
Expected: per-source `train …/eval …` lines, 1 epoch completes, `Saved fine-tuned model to experiments/smoke-mix/final`.

Run: `training/trocr/scripts/remote.sh 'python scripts/evaluate_model.py experiments/checkpoint-3/final --datasets handwriting_test --full --max_eval_samples 64'`
Expected: `{'eval_cer': …, 'eval_samples': 64, …}`.

- [ ] **Step 8: README + commit**

Add to the Datasets section of `training/trocr/README.md`:

```markdown
Weight sources with `name:N` — N rows of that source per epoch (shuffled subset, or repeated
copies when N exceeds its size). Each source gets its own seeded 3% holdout *before* resampling,
so repeated rows never leak into eval; `--max_eval_per_source 300` keeps every domain in the eval
set. `--max_aspect` drops images too wide to read after the 384x384 resize, and labels longer
than `MAX_LABEL_LEN` tokens are always dropped during training (they would be truncated).

    python scripts/train.py --datasets handwriting:60000 darayut:100000 soyvitou --max_eval_per_source 300
    python scripts/evaluate_model.py experiments/checkpoint-4/final --datasets handwriting_test --full

Holdouts are carved per source, so they differ from runs before 2026-10.
```

```bash
git add training/trocr/src/trocr_khmer training/trocr/scripts/train.py training/trocr/scripts/evaluate_model.py training/trocr/tests/test_data_mixing.py training/trocr/README.md
git commit -m "feat(trocr): weighted source mixing, per-source holdout, long-row filter"
```

---

### Task 4: Fine-tune from a previous run (fresh Trainer, same tokenizer)

**Files:**
- Modify: `training/trocr/src/trocr_khmer/processor.py` (add `load_model_and_processor`)
- Modify: `training/trocr/src/trocr_khmer/__init__.py`
- Modify: `training/trocr/scripts/train.py` (use it; `--warmup_steps`, `--early_stopping_patience`)
- Modify: `training/trocr/README.md` ("Fine-tune from a previous run")
- Test: `training/trocr/tests/test_processor.py`

**Interfaces:**
- Consumes: `load_trocr_processor`, `add_khmer_tokens`, `BYTE_LEVEL_TOKENIZER` (existing); `train.py` as left by Task 3.
- Produces: `load_model_and_processor(model_name_or_path: str) -> tuple[VisionEncoderDecoderModel, TrOCRProcessor, int]` — hub id: graft + add Khmer tokens (returns count added); local dir: reuse saved tokenizer, `ValueError` if tokenizer length ≠ decoder vocab ("decoder vocab") or any Khmer token would be added ("missing"); returns 0. `train.py` flags `--warmup_steps INT` (default 500), `--early_stopping_patience INT` (default 0 = off).

- [ ] **Step 1: Write the failing tests**

Create `training/trocr/tests/test_processor.py` (uses the cached `microsoft/trocr-base-printed` tokenizer on the box):

```python
from types import SimpleNamespace

import pytest

from trocr_khmer import processor as processor_mod
from trocr_khmer.khmer import add_khmer_tokens
from trocr_khmer.processor import BYTE_LEVEL_TOKENIZER, load_model_and_processor, load_trocr_processor

KHMER_SAMPLE = ["ក", "ស", "្", "ា"]


def _fake_model(monkeypatch, vocab_size):
    model = SimpleNamespace(config=SimpleNamespace(decoder=SimpleNamespace(vocab_size=vocab_size)))
    monkeypatch.setattr(
        processor_mod.VisionEncoderDecoderModel, "from_pretrained", staticmethod(lambda path: model)
    )
    return model


@pytest.fixture
def extended_dir(tmp_path):
    proc = load_trocr_processor(BYTE_LEVEL_TOKENIZER, tokenizer_name=BYTE_LEVEL_TOKENIZER)
    add_khmer_tokens(proc)
    proc.save_pretrained(tmp_path)
    return tmp_path, proc


def test_local_checkpoint_reuses_saved_tokenizer(monkeypatch, extended_dir):
    path, saved = extended_dir
    model = _fake_model(monkeypatch, len(saved.tokenizer))
    got_model, proc, added = load_model_and_processor(str(path))
    assert got_model is model and added == 0
    assert len(proc.tokenizer) == len(saved.tokenizer)
    assert proc.tokenizer.convert_tokens_to_ids(KHMER_SAMPLE) == saved.tokenizer.convert_tokens_to_ids(KHMER_SAMPLE)


def test_local_checkpoint_vocab_mismatch_raises(monkeypatch, extended_dir):
    path, saved = extended_dir
    _fake_model(monkeypatch, len(saved.tokenizer) + 1)
    with pytest.raises(ValueError, match="decoder vocab"):
        load_model_and_processor(str(path))


def test_local_checkpoint_missing_khmer_tokens_raises(monkeypatch, tmp_path):
    proc = load_trocr_processor(BYTE_LEVEL_TOKENIZER, tokenizer_name=BYTE_LEVEL_TOKENIZER)
    proc.save_pretrained(tmp_path)
    _fake_model(monkeypatch, len(proc.tokenizer))
    with pytest.raises(ValueError, match="missing"):
        load_model_and_processor(str(tmp_path))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `training/trocr/scripts/remote.sh 'python -m pytest tests/test_processor.py -v'`
Expected: FAIL — `ImportError: cannot import name 'load_model_and_processor'`.

- [ ] **Step 3: Implement `load_model_and_processor`**

In `processor.py`: add `from pathlib import Path`, add `VisionEncoderDecoderModel` to the `transformers` import, add `from trocr_khmer.khmer import add_khmer_tokens`, then append:

```python
def load_model_and_processor(model_name_or_path: str):
    """Load a model + processor ready to train, from a hub base or a previous run's ``final/``.

    A hub base (e.g. ``microsoft/trocr-base-printed``) gets the byte-level tokenizer graft and
    the Khmer tokens, with the decoder resized to match. A local fine-tuned directory already
    carries that extended tokenizer: reuse it exactly as saved and refuse to change the vocab —
    shifted token ids would scramble everything the decoder learned.

    Returns ``(model, processor, num_added_tokens)``.
    """
    model = VisionEncoderDecoderModel.from_pretrained(model_name_or_path)
    if not Path(model_name_or_path).is_dir():
        processor = load_trocr_processor(model_name_or_path, tokenizer_name=BYTE_LEVEL_TOKENIZER)
        return model, processor, add_khmer_tokens(processor, model)

    processor = load_trocr_processor(model_name_or_path)
    vocab = model.config.decoder.vocab_size
    if len(processor.tokenizer) != vocab:
        raise ValueError(
            f"{model_name_or_path}: tokenizer has {len(processor.tokenizer)} tokens but decoder vocab is {vocab}"
        )
    num_added = add_khmer_tokens(processor)
    if num_added:
        raise ValueError(
            f"{model_name_or_path}: tokenizer is missing {num_added} Khmer tokens; "
            "refusing to resize a fine-tuned decoder"
        )
    return model, processor, 0
```

Export `load_model_and_processor` from `__init__.py` (import from `trocr_khmer.processor` + `__all__`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `training/trocr/scripts/remote.sh 'python -m pytest tests -v'`
Expected: all PASS.

- [ ] **Step 5: Use it in `train.py` + fine-tune flags**

Replace these lines:

```python
    processor = load_trocr_processor(args.model_name, tokenizer_name=BYTE_LEVEL_TOKENIZER)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name)
```

and the later `num_added = add_khmer_tokens(processor, model)` line (keep the `print` after it) with:

```python
    # Hub base -> graft byte-level tokenizer + add Khmer tokens; a previous run's final/ dir ->
    # reuse its saved Khmer tokenizer unchanged (fresh Trainer: new optimizer + LR schedule).
    model, processor, num_added = load_model_and_processor(args.model_name)
```

Delete the now-stale comment blocks above those lines (the graft rationale lives in `load_model_and_processor`/`khmer.py`). Remove now-unused imports (`VisionEncoderDecoderModel`, `BYTE_LEVEL_TOKENIZER`, `add_khmer_tokens`, `load_trocr_processor`), import `load_model_and_processor` from `trocr_khmer`, and add `EarlyStoppingCallback` to the `transformers` import.

Add arguments:

```python
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Stop after N evals without CER improvement (0 = off). Fine-tunes: 5.",
    )
```

Change the `--model_name` help to `"Base TrOCR checkpoint (hub id) or a previous run's final/ dir to continue fine-tuning"`. In `Seq2SeqTrainingArguments` change `warmup_steps=500` to `warmup_steps=args.warmup_steps`. In `Seq2SeqTrainer(...)` add:

```python
        callbacks=(
            [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
            if args.early_stopping_patience > 0
            else None
        ),
```

- [ ] **Step 6: Verify on the real checkpoint + smoke**

Run: `training/trocr/scripts/remote.sh 'python -c "from trocr_khmer import load_model_and_processor as f; m, p, n = f(\"experiments/checkpoint-3/final\"); print(n, len(p.tokenizer), m.config.decoder.vocab_size)"'`
Expected: `0 50414 50414`

Run: `training/trocr/scripts/remote.sh 'python scripts/train.py --smoke_test --model_name experiments/checkpoint-3/final --datasets handwriting darayut --lr 2e-5 --warmup_steps 2 --eval_steps 5 --early_stopping_patience 2 --output_dir experiments/smoke-ft'`
Expected: `Added 0 Khmer tokens; decoder vocab is now 50414`; the first logged `eval_cer` (step 5) is well below 1.0 — the model already reads Khmer. A value near 1.0 means the tokenizer or weights did not carry over: stop and debug.

- [ ] **Step 7: README + commit**

Add to `training/trocr/README.md` after "Train":

```markdown
## Fine-tune from a previous run

Pass a run's `final/` dir as `--model_name`. This starts a fresh Trainer (new optimizer + LR
schedule) and reuses the saved Khmer tokenizer unchanged — loading refuses if the vocab would
change. Do **not** use `--resume_from_checkpoint` for this: it restores the old global step and
schedule, so a finished run stops immediately.

    python scripts/train.py --model_name experiments/checkpoint-3/final \
      --datasets handwriting:60000 synth:400000 darayut:100000 soyvitou:60000 \
      --lr 2e-5 --warmup_steps 1000 --early_stopping_patience 5 --max_eval_per_source 300 --max_eval_samples 0
```

```bash
git add training/trocr/src/trocr_khmer training/trocr/scripts/train.py training/trocr/tests/test_processor.py training/trocr/README.md
git commit -m "feat(trocr): continue fine-tuning from a previous run's final/ with early stopping"
```

---

### Task 5: Synthetic line renderer (fonts, italic, handwriting-style, documents, UI)

**Files:**
- Create: `training/trocr/src/trocr_khmer/synth.py`
- Create: `training/trocr/scripts/fetch_fonts.py`
- Modify: `training/trocr/pyproject.toml`, `training/trocr/requirements.txt` (add `fonttools>=4.40`)
- Test: `training/trocr/tests/test_synth.py`

**Interfaces:**
- Produces (all in `trocr_khmer.synth`):
  - `GOOGLE_KHMER_FONTS: dict[str, str]` — google/fonts repo path → kind (`"text" | "display" | "handwriting" | "italic"`); `COENG = "្"`.
  - `@dataclass(frozen=True) FontInfo(path: Path, kind: str, codepoints: frozenset[int])`; `font_info(path, kind) -> FontInfo`; `load_fonts(font_dir: Path) -> list[FontInfo]`.
  - `covers(font: FontInfo, text: str) -> bool`; `require_raqm() -> None`.
  - `render_text(text, font_path, size, fg, bg, pad=(8, 6)) -> Image.Image`; `shear(image, k, bg) -> Image.Image`; `degrade_document(image, rng) -> Image.Image`.
  - `render_sample(text, fonts, rng) -> tuple[Image.Image, dict] | None` — meta keys exactly `font`, `kind`, `style`, `italic`.
  - `split_clusters(run) -> list[str]`; `chunk_line(text, max_chars) -> list[str]`; `build_corpus(texts, exclude, max_chars=32, min_chars=2) -> list[str]`; `corpus_part(text) -> "train" | "val"`.
  - `scripts/fetch_fonts.py --out cache/fonts`.

- [ ] **Step 1: Add the dependency and install it on the box**

Add `"fonttools>=4.40",  # glyph coverage checks for the synthetic renderer` to `pyproject.toml` `dependencies`, and `fonttools>=4.40` to `requirements.txt`.

Run: `training/trocr/scripts/remote.sh 'python -m pip install "fonttools>=4.40" && python -c "import fontTools; print(fontTools.version)"'`
Expected: a version ≥ 4.40.

- [ ] **Step 2: Write the failing tests**

Create `training/trocr/tests/test_synth.py`. It uses the repo's own OFL Noto Sans Khmer at `training/eval/fonts/NotoSansKhmer-Regular.ttf`:

```python
import random
from pathlib import Path

from PIL import Image, ImageFont, features

from trocr_khmer.synth import (
    COENG,
    build_corpus,
    chunk_line,
    corpus_part,
    covers,
    degrade_document,
    font_info,
    render_sample,
    render_text,
    shear,
)

NOTO = Path(__file__).resolve().parents[2] / "eval/fonts/NotoSansKhmer-Regular.ttf"
CLUSTER = "ស្ត្រ"  # s + coeng t + coeng r: subscripts stack only when shaped


def test_raqm_shapes_coeng_cluster():
    assert features.check("raqm"), "Pillow lacks libraqm; Khmer would render unshaped"
    raqm = ImageFont.truetype(str(NOTO), 48, layout_engine=ImageFont.Layout.RAQM).getbbox(CLUSTER)
    basic = ImageFont.truetype(str(NOTO), 48, layout_engine=ImageFont.Layout.BASIC).getbbox(CLUSTER)
    assert raqm != basic  # shaping repositioned the subscripts


def test_render_text_is_rgb_with_padding():
    image = render_text(CLUSTER, NOTO, 40, fg=(0, 0, 0), bg=(255, 255, 255), pad=(10, 5))
    assert image.mode == "RGB" and image.width > 20 and image.height > 10
    assert image.getpixel((0, 0)) == (255, 255, 255)


def test_shear_widens_canvas_and_keeps_corners_background():
    image = Image.new("RGB", (100, 40), (255, 255, 255))
    out = shear(image, 0.25, (255, 255, 255))
    assert out.size == (110, 40)
    assert out.getpixel((0, 0)) == (255, 255, 255)
    assert out.getpixel((109, 39)) == (255, 255, 255)


def test_degrade_document_keeps_size_and_mode():
    image = render_text(CLUSTER, NOTO, 40, fg=(0, 0, 0), bg=(250, 250, 245))
    out = degrade_document(image, random.Random(0))
    assert out.size == image.size and out.mode == "RGB"


def test_render_sample_is_deterministic():
    fonts = [font_info(NOTO, "text")]
    a, meta_a = render_sample("ភាសាខ្មែរ ២០២៤", fonts, random.Random(7))
    b, meta_b = render_sample("ភាសាខ្មែរ ២០២៤", fonts, random.Random(7))
    assert meta_a == meta_b and a.tobytes() == b.tobytes()
    assert set(meta_a) == {"font", "kind", "style", "italic"}


def test_render_sample_skips_fonts_missing_glyphs():
    fonts = [font_info(NOTO, "text")]
    assert not covers(fonts[0], "ក\U0001F600")
    assert render_sample("ក\U0001F600", fonts, random.Random(0)) is None


def test_chunk_line_respects_limit_and_keeps_coeng_attached():
    text = "ព្រះរាជាណាចក្រកម្ពុជាជាតិសាសនាព្រះមហាក្សត្រ​សួស្តី world"
    chunks = chunk_line(text, max_chars=12)
    assert chunks and all(len(c) <= 12 for c in chunks)
    assert all(not c.endswith(COENG) and not c.startswith(COENG) for c in chunks)
    assert "".join(chunks).replace(" ", "") == text.replace("​", "").replace(" ", "")


def test_build_corpus_excludes_eval_text():
    fixture = "រក្សាទុកជា"
    corpus = build_corpus(["សួស្តី", fixture, f"ចុច {fixture} ឥឡូវ", "សួស្តី"], exclude=[fixture], max_chars=32)
    assert corpus == ["សួស្តី"]


def test_corpus_part_is_stable_and_mostly_train():
    parts = [corpus_part(f"line {i}") for i in range(2000)]
    assert parts == [corpus_part(f"line {i}") for i in range(2000)]
    assert 0 < parts.count("val") < 100
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `training/trocr/scripts/remote.sh 'python -m pytest tests/test_synth.py -v'`
Expected: FAIL — `ModuleNotFoundError: No module named 'trocr_khmer.synth'`.

- [ ] **Step 4: Implement `synth.py`**

Create `training/trocr/src/trocr_khmer/synth.py`:

```python
"""Render synthetic Khmer text-line images across many fonts and styles.

Targets gaps in the existing training data: font variety (Google Fonts Khmer families), italic
(Kantumruy Pro Italic plus a synthetic shear, since almost no Khmer font ships an italic),
handwriting-like fonts (Fasthand, Freehand), scanned / low-res documents, and light/dark UI
panels.

Khmer needs complex shaping (COENG subscripts, reordered vowel signs). Pillow only shapes it with
the libraqm layout engine, so fonts are always opened with ``Layout.RAQM`` and ``require_raqm``
fails closed when raqm is missing — unshaped renders would silently produce wrong labels.
"""
from __future__ import annotations

import io
import math
import random
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

# google/fonts repo path -> kind. Kinds steer font sampling in render_sample.
GOOGLE_KHMER_FONTS = {
    "ofl/angkor/Angkor-Regular.ttf": "display",
    "ofl/battambang/Battambang-Regular.ttf": "text",
    "ofl/battambang/Battambang-Bold.ttf": "text",
    "ofl/battambang/Battambang-Light.ttf": "text",
    "ofl/bayon/Bayon-Regular.ttf": "display",
    "ofl/bokor/Bokor-Regular.ttf": "display",
    "ofl/chenla/Chenla.ttf": "display",
    "ofl/content/Content-Regular.ttf": "text",
    "ofl/content/Content-Bold.ttf": "text",
    "ofl/dangrek/Dangrek-Regular.ttf": "display",
    "ofl/fasthand/Fasthand-Regular.ttf": "handwriting",
    "ofl/freehand/Freehand-Regular.ttf": "handwriting",
    "ofl/hanuman/Hanuman[wght].ttf": "text",
    "ofl/kantumruypro/KantumruyPro[wght].ttf": "text",
    "ofl/kantumruypro/KantumruyPro-Italic[wght].ttf": "italic",
    "ofl/kdamthmorpro/KdamThmorPro-Regular.ttf": "text",
    "ofl/khmer/Khmer.ttf": "text",
    "ofl/kohsantepheap/KohSantepheap-Regular.ttf": "text",
    "ofl/kohsantepheap/KohSantepheap-Bold.ttf": "text",
    "ofl/konkhmersleokchher/KonkhmerSleokchher-Regular.ttf": "display",
    "ofl/koulen/Koulen-Regular.ttf": "display",
    "ofl/metal/Metal-Regular.ttf": "display",
    "ofl/moul/Moul-Regular.ttf": "display",
    "ofl/moulpali/Moulpali-Regular.ttf": "display",
    "ofl/nokora/Nokora[wght].ttf": "text",
    "ofl/notosanskhmer/NotoSansKhmer[wdth,wght].ttf": "text",
    "ofl/notoserifkhmer/NotoSerifKhmer[wdth,wght].ttf": "text",
    "ofl/odormeanchey/OdorMeanChey-Regular.ttf": "text",
    "ofl/preahvihear/Preahvihear-Regular.ttf": "text",
    "ofl/siemreap/Siemreap.ttf": "text",
    "ofl/suwannaphum/Suwannaphum-Regular.ttf": "text",
    "ofl/suwannaphum/Suwannaphum-Bold.ttf": "text",
    "ofl/taprom/Taprom-Regular.ttf": "display",
}

KIND_WEIGHTS = {"text": 0.55, "handwriting": 0.2, "display": 0.15, "italic": 0.1}
SYNTHETIC_ITALIC_P = 0.25  # extra shear on non-italic fonts
COENG = "្"
_IGNORABLE = {0x20, 0x200B, 0x200C, 0x200D}  # spaces / zero-width: need no glyph


@dataclass(frozen=True)
class FontInfo:
    path: Path
    kind: str
    codepoints: frozenset[int]


def font_info(path: Path, kind: str) -> FontInfo:
    from fontTools.ttLib import TTFont

    return FontInfo(Path(path), kind, frozenset(TTFont(str(path), lazy=True).getBestCmap()))


def load_fonts(font_dir: Path) -> list[FontInfo]:
    fonts = []
    for repo_path, kind in GOOGLE_KHMER_FONTS.items():
        path = Path(font_dir) / Path(repo_path).name
        if not path.is_file():
            raise FileNotFoundError(f"{path} missing; run scripts/fetch_fonts.py")
        fonts.append(font_info(path, kind))
    return fonts


def covers(font: FontInfo, text: str) -> bool:
    """True when the font has a glyph for every visible character (no tofu boxes)."""
    return all(ord(c) in font.codepoints or ord(c) in _IGNORABLE for c in text)


def require_raqm() -> None:
    if not features.check("raqm"):
        raise RuntimeError("Pillow lacks libraqm; Khmer would render unshaped. Install libraqm and reinstall Pillow.")


def render_text(text, font_path, size, fg, bg, pad=(8, 6)) -> Image.Image:
    """Shaped single-line render on a tight canvas plus padding."""
    font = ImageFont.truetype(str(font_path), size, layout_engine=ImageFont.Layout.RAQM)
    left, top, right, bottom = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), text, font=font)
    px, py = pad
    image = Image.new("RGB", (right - left + 2 * px, bottom - top + 2 * py), bg)
    ImageDraw.Draw(image).text((px - left, py - top), text, font=font, fill=fg)
    return image


def shear(image: Image.Image, k: float, bg) -> Image.Image:
    """Synthetic italic: lean right by ``k`` (dx per dy). The canvas widens by ``ceil(k*h)`` so
    nothing clips; the bottom row stays put and the top row shifts right."""
    w, h = image.size
    return image.transform(
        (w + math.ceil(k * h), h),
        Image.Transform.AFFINE,
        (1, k, -k * h, 0, 1, 0),
        resample=Image.Resampling.BICUBIC,
        fillcolor=bg,
    )


def degrade_document(image: Image.Image, rng: random.Random) -> Image.Image:
    """Scanned / photocopied look: low resolution, optional blur, sensor noise, JPEG artifacts.
    Noise is drawn from ``rng`` so renders are reproducible."""
    w, h = image.size
    scale = rng.uniform(0.35, 0.8)
    out = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)
    out = out.resize((w, h), Image.Resampling.BILINEAR)
    if rng.random() < 0.5:
        out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    noise = np.random.default_rng(rng.getrandbits(32)).normal(0, rng.uniform(4, 14), (h, w, 1))
    out = Image.fromarray(np.clip(np.asarray(out, dtype=np.float32) + noise, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=rng.randint(30, 75))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _colors(style: str, rng: random.Random):
    if style == "ui_dark":
        bg = rng.randint(15, 45)
        fg = rng.randint(205, 250)
        return (fg, fg, fg), (bg, bg, bg + rng.randint(0, 8))
    if style == "ui_light":
        bg = rng.randint(235, 255)
        fg = rng.randint(10, 60)
        return (fg, fg, fg), (bg, bg, bg)
    paper = rng.randint(225, 255)
    ink = rng.randint(0, 70)
    return (ink, ink, ink + rng.randint(0, 40)), (paper, paper, rng.randint(paper - 15, paper))


def render_sample(text: str, fonts: list[FontInfo], rng: random.Random):
    """Render one line in a random font/style. Returns ``(image, meta)``, or ``None`` when no
    font has glyphs for every character of ``text``."""
    usable = [f for f in fonts if covers(f, text)]
    if not usable:
        return None
    kinds = sorted({f.kind for f in usable})
    kind = rng.choices(kinds, weights=[KIND_WEIGHTS[k] for k in kinds])[0]
    font = rng.choice([f for f in usable if f.kind == kind])
    style = rng.choices(["plain", "document", "ui_dark", "ui_light"], weights=[0.35, 0.35, 0.15, 0.15])[0]
    fg, bg = _colors(style, rng)
    image = render_text(text, font.path, rng.randint(22, 64), fg, bg, pad=(rng.randint(4, 24), rng.randint(3, 16)))
    italic = kind == "italic"
    if not italic and rng.random() < SYNTHETIC_ITALIC_P:
        image = shear(image, rng.uniform(0.12, 0.3), bg)
        italic = True
    if style == "document":
        image = degrade_document(image, rng)
    return image, {"font": font.path.name, "kind": kind, "style": style, "italic": italic}


def _is_khmer(ch: str) -> bool:
    return "ក" <= ch <= "៿"


def _starts_cluster(ch: str, prev: str) -> bool:
    # A consonant / independent vowel starts a cluster unless a COENG just subscripted it;
    # any non-Khmer character starts its own unit.
    return not _is_khmer(ch) or ("ក" <= ch <= "ឳ" and prev != COENG)


def split_clusters(run: str) -> list[str]:
    """Split an unbroken run into Khmer orthographic clusters (safe cut points)."""
    clusters: list[str] = []
    for i, ch in enumerate(run):
        if clusters and not _starts_cluster(ch, run[i - 1]):
            clusters[-1] += ch
        else:
            clusters.append(ch)
    return clusters


def chunk_line(text: str, max_chars: int) -> list[str]:
    """Pack a label into renderable lines of at most ``max_chars``. Words split on spaces and
    ZWSP; words longer than the limit (common — Khmer rarely uses spaces) are cut at cluster
    boundaries. ZWSP is dropped: it is invisible, so it can't be learned from pixels."""
    chunks: list[str] = []
    cur = ""

    def push(piece: str, sep: str) -> None:
        nonlocal cur
        cand = f"{cur}{sep}{piece}" if cur else piece
        if len(cand) <= max_chars:
            cur = cand
        else:
            if cur:
                chunks.append(cur)
            cur = piece

    for word in text.replace("​", " ").split():
        if len(word) <= max_chars:
            push(word, " ")
        else:
            for i, cluster in enumerate(split_clusters(word)):
                push(cluster, " " if i == 0 else "")
    if cur:
        chunks.append(cur)
    return [c for c in chunks if len(c) <= max_chars]


def build_corpus(texts, exclude, max_chars: int = 32, min_chars: int = 2) -> list[str]:
    """Chunk labels into unique renderable lines, dropping any that equal, or contain, eval text
    (fixtures + UI strings) or one of its chunks."""
    banned = {e.strip() for e in exclude if e.strip()}
    for text in list(banned):
        banned.update(chunk_line(text, max_chars))
    contained = [b for b in banned if len(b) >= 4]
    seen: set[str] = set()
    out: list[str] = []
    for text in texts:
        for chunk in chunk_line(text, max_chars):
            if len(chunk) < min_chars or chunk in seen or chunk in banned:
                continue
            if any(b in chunk for b in contained):
                continue
            seen.add(chunk)
            out.append(chunk)
    return out


def corpus_part(text: str) -> str:
    """Stable ~2% val / 98% train assignment by content hash, so train and val renders never
    share a line across runs or seeds."""
    return "val" if zlib.crc32(text.encode("utf-8")) % 100 < 2 else "train"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `training/trocr/scripts/remote.sh 'python -m pytest tests -v'`
Expected: all PASS. If `test_chunk_line_respects_limit_and_keeps_coeng_attached` fails on the join assertion, the packer is dropping text — fix `chunk_line`, don't loosen the test.

- [ ] **Step 6: Write `fetch_fonts.py` and fetch on the box**

Create `training/trocr/scripts/fetch_fonts.py`:

```python
#!/usr/bin/env python3
"""Download the OFL Google Fonts Khmer files used by the synthetic renderer into cache/fonts.

    python scripts/fetch_fonts.py
"""
import argparse
import urllib.parse
import urllib.request
from pathlib import Path

from trocr_khmer.synth import GOOGLE_KHMER_FONTS

BASE = "https://raw.githubusercontent.com/google/fonts/main/"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("cache/fonts"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for repo_path in GOOGLE_KHMER_FONTS:
        dest = args.out / Path(repo_path).name
        if dest.is_file():
            continue
        urllib.request.urlretrieve(BASE + urllib.parse.quote(repo_path), dest)
        print(f"fetched {dest.name}")
    print(f"{len(GOOGLE_KHMER_FONTS)} fonts in {args.out}")


if __name__ == "__main__":
    main()
```

Run: `training/trocr/scripts/remote.sh 'python scripts/fetch_fonts.py && python -c "from pathlib import Path; from trocr_khmer.synth import load_fonts; print(len(load_fonts(Path(\"cache/fonts\"))))"'`
Expected: `33 fonts in cache/fonts` then `33`.

- [ ] **Step 7: Commit**

```bash
git add training/trocr/src/trocr_khmer/synth.py training/trocr/scripts/fetch_fonts.py training/trocr/tests/test_synth.py training/trocr/pyproject.toml training/trocr/requirements.txt
git commit -m "feat(trocr): raqm-shaped synthetic Khmer renderer with fonts, italic, handwriting and document styles"
```

---

### Task 6: Render synthetic shards, measure sources, baseline new domains

**Files:**
- Create: `training/trocr/scripts/render_synth.py`
- Create: `training/trocr/scripts/data_stats.py`
- Modify: `training/eval/results/finetune_baseline.md` (domain table, chosen `--max_aspect`)
- Modify: `training/trocr/README.md` ("Synthetic data")

**Interfaces:**
- Consumes: `resolve_dataset`, `DatasetSpec.load_kwargs`, `load_ocr_datasets` (Task 2); everything in `trocr_khmer.synth` (Task 5); `evaluate_model.py --full / --max_eval_per_source` (Task 3).
- Produces: `cache/synth/train/shard-*.parquet` (400k rows) and `cache/synth/val/shard-*.parquet` (3k rows) with columns `image, text, font, kind, style, italic` — loadable as `synth` / `synth_val`. A recorded `MAX_ASPECT` value used by Task 7.

- [ ] **Step 1: Write `render_synth.py`**

```python
#!/usr/bin/env python3
"""Pre-render synthetic Khmer line images to parquet shards (deterministic per --seed).

    python scripts/fetch_fonts.py
    python scripts/render_synth.py --part val --count 3000 --seed 2 --out cache/synth/val
    python scripts/render_synth.py --part train --count 400000 --seed 1 --out cache/synth/train

Text comes from existing datasets' labels, chunked to <= --max_chars, minus every eval fixture
and UI string. --part picks the train or val side of a stable content-hash split.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, Features, Value, load_dataset
from datasets import Image as DSImage

from trocr_khmer.data_sources import resolve_dataset
from trocr_khmer.synth import build_corpus, corpus_part, load_fonts, render_sample, require_raqm

EVAL = Path(__file__).resolve().parents[2] / "eval"
FEATURES = Features(
    {
        "image": DSImage(),
        "text": Value("string"),
        "font": Value("string"),
        "kind": Value("string"),
        "style": Value("string"),
        "italic": Value("bool"),
    }
)


def load_texts(name: str, k: int, seed: int) -> list[str]:
    """k random labels from a source's *training* side — text column only, images never decoded.

    Drops the same seeded 3% holdout that load_ocr_split carves (test_size=0.03, seed=42), so
    replay-holdout labels never reach training re-rendered in other fonts."""
    spec = resolve_dataset(name)
    ds = load_dataset(spec.path, split="train", **spec.load_kwargs()).select_columns([spec.text_col])
    ds = ds.train_test_split(test_size=0.03, seed=42)["train"]
    idx = random.Random(seed).sample(range(len(ds)), min(k, len(ds)))
    return ds.select(idx)[spec.text_col]


def eval_texts() -> list[str]:
    labels = (EVAL / "fixtures/labels.jsonl").read_text(encoding="utf-8").splitlines()
    texts = [json.loads(line)["text"] for line in labels if line.strip()]
    return texts + (EVAL / "ui_strings.txt").read_text(encoding="utf-8").splitlines()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--part", choices=["train", "val"], default="train")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--font_dir", type=Path, default=Path("cache/fonts"))
    p.add_argument("--text_sources", nargs="+", default=["hanuman", "sokheng", "soyvitou", "mrrtmob"])
    p.add_argument("--texts_per_source", type=int, default=200_000)
    p.add_argument("--max_chars", type=int, default=32)
    p.add_argument("--max_aspect", type=float, default=20.0)
    p.add_argument("--shard_size", type=int, default=10_000)
    args = p.parse_args()

    require_raqm()
    fonts = load_fonts(args.font_dir)
    texts = [t for name in args.text_sources for t in load_texts(name, args.texts_per_source, args.seed)]
    corpus = [c for c in build_corpus(texts, eval_texts(), args.max_chars) if corpus_part(c) == args.part]
    if not corpus:
        raise SystemExit("empty corpus")
    rng = random.Random(args.seed)
    rng.shuffle(corpus)
    print(f"{len(corpus)} {args.part} lines from {len(texts)} labels; rendering {args.count}")

    args.out.mkdir(parents=True, exist_ok=True)
    rows, shard, made, tried = defaultdict(list), 0, 0, 0
    while made < args.count:
        if tried > 20 * args.count:
            raise SystemExit(f"only {made} renderable lines after {tried} tries; check fonts/corpus")
        text = corpus[tried % len(corpus)]
        tried += 1
        sample = render_sample(text, fonts, rng)
        if sample is None:
            continue
        image, meta = sample
        if image.width / image.height > args.max_aspect:
            continue
        rows["image"].append(image)
        rows["text"].append(text)
        for key in ("font", "kind", "style", "italic"):
            rows[key].append(meta[key])
        made += 1
        if len(rows["text"]) == args.shard_size or made == args.count:
            Dataset.from_dict(rows, features=FEATURES).to_parquet(str(args.out / f"shard-{shard:05d}.parquet"))
            print(f"  shard {shard}: {made}/{args.count}")
            shard += 1
            rows = defaultdict(list)


if __name__ == "__main__":
    main()
```

- [ ] **Step 1b: Confirm the text-only split matches the training holdout**

`load_texts` relies on `train_test_split` depending only on row count + seed, not on which columns are loaded. Check once:

Run: `training/trocr/scripts/remote.sh 'python -c "
from datasets import load_dataset
from trocr_khmer import load_ocr_split
full = set(load_ocr_split([\"hanuman\"])[\"test\"][\"text\"])
txt = set(load_dataset(\"seanghay/khmer-hanuman-100k\", split=\"train\").select_columns([\"text\"]).train_test_split(test_size=0.03, seed=42)[\"test\"][\"text\"])
print(\"MATCH\" if full == txt else f\"MISMATCH {len(full ^ txt)}\")"'`
Expected: `MATCH`. On `MISMATCH`, stop — make `load_texts` build its holdout via `load_ocr_split([name])` text sets instead.

- [ ] **Step 2: Render a tiny preview and eyeball it**

Run:

```bash
training/trocr/scripts/remote.sh 'python scripts/render_synth.py --part train --count 40 --seed 9 --shard_size 40 --texts_per_source 2000 --out cache/synth/preview && python -c "
from datasets import load_dataset
ds = load_dataset(\"parquet\", data_files=\"cache/synth/preview/*.parquet\", split=\"train\")
for i, r in enumerate(ds): r[\"image\"].save(f\"cache/synth/preview/{i:02d}.png\")
print(ds.to_pandas()[[\"text\", \"font\", \"kind\", \"style\", \"italic\"]].to_string())"'
mkdir -p "$TMPDIR/synth-preview" && scp 'kim@192.168.1.25:/home/kim/Sync/workspace/khmerclipper/.worktrees/trocr-finetune-data/training/trocr/cache/synth/preview/*.png' "$TMPDIR/synth-preview/"
```

Then open/Read 10+ of the PNGs alongside the printed table.

Expected: 40 rows across several fonts/kinds/styles. **Check by eye**: subscripts (e.g. `្ត`, `្រ`) sit under the base consonant, italic rows lean right, no tofu boxes, each image's text matches its label. Stop and fix the renderer if any label/image disagree.

- [ ] **Step 3: Render the real shards (detached)**

```bash
ssh kim@192.168.1.25 "tmux new -d -s synth 'cd /home/kim/Sync/workspace/khmerclipper/.worktrees/trocr-finetune-data/training/trocr && export PATH=\$HOME/.pyenv/versions/khmerclipper/bin:\$PATH && export PYTHONPATH=src:../eval/src && rm -rf cache/synth/preview && python scripts/render_synth.py --part val --count 3000 --seed 2 --out cache/synth/val && python scripts/render_synth.py --part train --count 400000 --seed 1 --out cache/synth/train 2>&1 | tee cache/synth/render.log'"
```

Check later with `ssh kim@192.168.1.25 'tail -3 /home/kim/Sync/workspace/khmerclipper/.worktrees/trocr-finetune-data/training/trocr/cache/synth/render.log'` — expected to end with `shard 39: 400000/400000`. Continue with Steps 4–6 while it runs (they use the CPU lightly / the GPU only for eval).

- [ ] **Step 4: Write `data_stats.py`**

```python
#!/usr/bin/env python3
"""Report label-length and aspect-ratio outliers per source before mixing it into training.

    python scripts/data_stats.py soyvitou sokheng hanuman kiteaether mrrtmob handwriting darayut siemreap
"""
import argparse
import random

from trocr_khmer import MAX_LABEL_LEN, load_ocr_datasets, load_trocr_processor


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("datasets", nargs="+")
    p.add_argument("--model_dir", default="experiments/checkpoint-3/final", help="Source of the Khmer tokenizer")
    p.add_argument("--sample", type=int, default=2000)
    p.add_argument("--aspects", type=float, nargs="+", default=[16, 20, 24, 32])
    args = p.parse_args()

    tokenizer = load_trocr_processor(args.model_dir).tokenizer
    for name in args.datasets:
        ds = load_ocr_datasets([name])
        sub = ds.select(random.Random(0).sample(range(len(ds)), min(args.sample, len(ds))))
        too_long = sum(len(tokenizer(t).input_ids) > MAX_LABEL_LEN for t in sub["text"]) / len(sub)
        aspects = sorted(im.width / im.height for im in sub["image"])
        drops = " ".join(f">{a:g}:{sum(x > a for x in aspects) / len(aspects):.1%}" for a in args.aspects)
        print(
            f"{name}\tn={len(ds)}\ttoo_long={too_long:.1%}\t"
            f"aspect p50={aspects[len(aspects) // 2]:.1f} p95={aspects[int(len(aspects) * 0.95)]:.1f}\t{drops}"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Measure every source and choose `MAX_ASPECT`**

Run: `training/trocr/scripts/remote.sh 'python scripts/data_stats.py soyvitou sokheng hanuman kiteaether mrrtmob handwriting darayut siemreap'`
Expected: one line per source, e.g. `siemreap  n=200000  too_long=…%  aspect p50=… p95=…  >16:…% >20:…% >24:…% >32:…%`.

Decision rule: `MAX_ASPECT` = the smallest of 16 / 20 / 24 / 32 at which **every replay source** (soyvitou, sokheng, hanuman, kiteaether, mrrtmob) loses ≤ 10% of rows. If none qualifies, use no aspect filter (omit `--max_aspect` in Task 7) — the old model trained on those shapes, and the length filter still runs. siemreap/darayut/handwriting losing more is acceptable. If any source has `too_long` > 5%, note it in the results file: those rows were truncated in the old run and are a plausible cause of its CER drift.

- [ ] **Step 6: Baseline the init model on the new domains (greedy)**

Run (substitute `INIT` from Task 1 if it was `last-step`):

```bash
training/trocr/scripts/remote.sh 'INIT=experiments/checkpoint-3/final; \
  echo "== handwriting_test"; python scripts/evaluate_model.py $INIT --datasets handwriting_test --full; \
  for d in darayut siemreap handwriting soyvitou sokheng hanuman kiteaether mrrtmob; do echo "== $d"; \
    python scripts/evaluate_model.py $INIT --datasets $d --max_eval_per_source 1000; done'
```

After Step 3 finishes: `training/trocr/scripts/remote.sh 'python scripts/evaluate_model.py experiments/checkpoint-3/final --datasets synth_val --full'`.

Expected: one `{'eval_cer': …}` per eval set. handwriting_test / synth_val / darayut / siemreap should be clearly worse than the replay sources — that gap is what the fine-tune targets.

Append to `training/eval/results/finetune_baseline.md`:

```markdown
## New-domain baseline (greedy, `evaluate_model.py`)

| Eval set | rows | init CER |
| --- | ---: | ---: |
| handwriting_test (full) | … | … |
| synth_val (full) | 3000 | … |
| darayut holdout | 1000 | … |
| siemreap holdout | 1000 | … |
| handwriting holdout | … | … |
| soyvitou holdout* | 1000 | … |
| sokheng holdout* | 1000 | … |
| hanuman holdout* | 1000 | … |
| kiteaether holdout* | 1000 | … |
| mrrtmob holdout* | 1000 | … |

\* Per-source holdouts are new, so these rows were probably in the old run's training data; CER is optimistic.

## Source stats (`data_stats.py`, 2000-row sample)

(paste the data_stats.py output here as a code block)

**Chosen `--max_aspect`:** (value, or "none") — rule: smallest of 16/20/24/32 losing ≤10% of every replay source.
```

Fill every `…` and both parenthesized lines with measured values.

- [ ] **Step 7: README + commit**

Add to `training/trocr/README.md`:

```markdown
## Synthetic data

`scripts/fetch_fonts.py` downloads 33 OFL Google Fonts Khmer files into `cache/fonts/`.
`scripts/render_synth.py` renders lines from existing labels (minus every eval fixture / UI
string) in those fonts — handwriting-style fonts, native + sheared italic, scanned-document
degradation, light/dark UI panels — into parquet shards loaded as `synth` / `synth_val`.
Rendering requires Pillow built with libraqm (checked at start). `scripts/data_stats.py` reports
label-length / aspect outliers for any source before you mix it in.
```

```bash
git add training/trocr/scripts/render_synth.py training/trocr/scripts/data_stats.py training/eval/results/finetune_baseline.md training/trocr/README.md
git commit -m "feat(trocr): render synthetic shards, source stats, new-domain baseline"
```

---

### Task 7: Smoke + launch the fine-tune on the 3090

**Files:**
- None (operational task). Results are recorded in Task 8.

**Interfaces:**
- Consumes: `train.py` with Task 3/4 flags; `synth` shards (Task 6); `MAX_ASPECT` (Task 6) and `INIT` (Task 1).
- Produces: `experiments/checkpoint-N/final/` (next auto-numbered run dir, expected `checkpoint-4`; confirm from the `Writing this run to …` log line).

In every command below: replace `--max_aspect 20` with the Task 6 value (or delete the flag if Task 6 chose "none"), and replace `experiments/checkpoint-3/final` with `INIT` if Task 1 picked `last-step`.

- [ ] **Step 1: Confirm preconditions**

Run: `training/trocr/scripts/remote.sh 'tail -1 cache/synth/render.log; ls cache/synth/train | wc -l; nvidia-smi --query-gpu=memory.used --format=csv,noheader; python -m pytest tests -q'`
Expected: `shard 39: 400000/400000`, `40`, GPU memory a few hundred MiB (nothing else training), all tests pass.

- [ ] **Step 2: Full-mix smoke (catches feature mismatches across all 9 sources)**

Run: `training/trocr/scripts/remote.sh 'python scripts/train.py --smoke_test --model_name experiments/checkpoint-3/final --datasets handwriting:60000 synth:400000 darayut:100000 siemreap:100000 soyvitou:60000 sokheng:80000 hanuman:80000 kiteaether:60000 mrrtmob:200000 --augment --lr 2e-5 --warmup_steps 2 --eval_steps 20 --max_eval_per_source 3 --max_eval_samples 0 --max_aspect 20 --output_dir experiments/smoke-full'`

Expected: nine `train …/eval …` lines, `Added 0 Khmer tokens`, run completes and saves `final/`. A `concatenate_datasets` feature error means `_canonical` missed a source — fix it before Step 3.

- [ ] **Step 3: Launch the real run in tmux**

```bash
ssh kim@192.168.1.25 "tmux new -d -s finetune 'cd /home/kim/Sync/workspace/khmerclipper/.worktrees/trocr-finetune-data/training/trocr && export PATH=\$HOME/.pyenv/versions/khmerclipper/bin:\$PATH && export PYTHONPATH=src:../eval/src && rm -rf experiments/smoke-mix experiments/smoke-ft experiments/smoke-full && python scripts/train.py \
  --model_name experiments/checkpoint-3/final \
  --datasets handwriting:60000 synth:400000 darayut:100000 siemreap:100000 soyvitou:60000 sokheng:80000 hanuman:80000 kiteaether:60000 mrrtmob:200000 \
  --augment --epochs 3 --lr 2e-5 --warmup_steps 1000 \
  --max_eval_per_source 300 --max_eval_samples 0 --eval_steps 2000 \
  --early_stopping_patience 5 --max_aspect 20 2>&1 | tee experiments/finetune.log'"
```

Why these numbers: ~1.14M rows/epoch ≈ 35.6k steps at batch 32; 3 epochs ≈ 107k steps ≈ 15–16 h at the old run's ~1.9 steps/s. Handwriting is upsampled ~16× (augment on); replay sources are ~45% of each epoch to hold printed/UI accuracy. LR is 2.5× lower than the old run, whose CER drifted after step 72k at 5e-5; early stopping (5 evals = 10k steps without improvement) plus `load_best_model_at_end` guard against a repeat.

- [ ] **Step 4: Check the first evals, then leave it**

After ~30 min:

```bash
ssh kim@192.168.1.25 'cd /home/kim/Sync/workspace/khmerclipper/.worktrees/trocr-finetune-data/training/trocr && grep -o "Writing this run to .*" experiments/finetune.log && grep -o "eval_cer.\{0,12\}" experiments/finetune.log | tail -3'
```

Expected: the run dir (e.g. `experiments/checkpoint-4`) and finite eval_cer values not climbing toward 1.0. If the first eval_cer is ≥ 0.5, kill it (`ssh kim@192.168.1.25 tmux kill-session -t finetune`) and debug — the init weights did not carry over.

Optional live view: `ssh -L 6006:localhost:6006 kim@192.168.1.25 '~/.pyenv/versions/khmerclipper/bin/tensorboard --logdir /home/kim/Sync/workspace/khmerclipper/.worktrees/trocr-finetune-data/training/trocr/experiments/checkpoint-4'`, then browse `http://localhost:6006`.

- [ ] **Step 5: When done**

Done when `experiments/finetune.log` ends with `Saved fine-tuned model to experiments/checkpoint-4/final` (early-stopped or 3 epochs). No commit for this task.

---

### Task 8: Gate the fine-tuned model and hand off to the Mac flow

**Files:**
- Create: `training/eval/results/finetune_v1.md`

**Interfaces:**
- Consumes: Task 1 + Task 6 baseline numbers in `finetune_baseline.md`; `experiments/checkpoint-4/final` (Task 7; use the actual N).
- Produces: pass/fail decision; on pass, the checkpoint path for `training/eval/results/mac_gate.md`.

- [ ] **Step 1: Re-run every baseline measurement on the new model**

```bash
training/trocr/scripts/remote.sh 'NEW=experiments/checkpoint-4/final; \
  python ../eval/scripts/bench.py --engine trocr-hf --trocr-model-dir $NEW --out experiments/bench/finetune_v1.csv \
  && python ../eval/scripts/subset_cer.py experiments/bench/finetune_v1.csv; \
  echo "== handwriting_test"; python scripts/evaluate_model.py $NEW --datasets handwriting_test --full; \
  echo "== synth_val"; python scripts/evaluate_model.py $NEW --datasets synth_val --full; \
  for d in darayut siemreap handwriting soyvitou sokheng hanuman kiteaether mrrtmob; do echo "== $d"; \
    python scripts/evaluate_model.py $NEW --datasets $d --max_eval_per_source 1000; done; \
  grep -h best_metric experiments/checkpoint-4/checkpoint-*/trainer_state.json | tail -1'
```

Expected: fixture subset lines, one `eval_cer` per eval set, and the run's best mid-train CER.

- [ ] **Step 2: Write the comparison and decide**

Create `training/eval/results/finetune_v1.md` containing: the exact Task 7 train command, run dir, stopping step, best mid-train CER; a fixtures table (rows all/printed/ui/mixed) and an eval-set table (the Task 6 rows), each with columns `init | finetune_v1 | Δ`.

**PASS** only if all hold (gated on sets neither model trained on):
1. handwriting_test CER and synth_val CER are both lower than init.
2. Fixture `ui` CER ≤ init + 0.005; darayut and siemreap holdout CER ≤ init (they are new domains — they should improve).
3. Every replay holdout (soyvitou, sokheng, hanuman, kiteaether, mrrtmob) CER ≤ init + 0.01.

**Informational only (not gating):** fixture `printed` and `mixed` deltas. The init model trained on those exact soyvitou testset images; the fine-tune no longer does, so some rise is memorization fading, not real regression. The real ship bar for them is the parent spec's Tesseract comparison (printed 0.5370, mixed 0.0894) in the Mac gate.

Put `**Result: PASS**` or `**Result: FAIL — <criteria that failed>**` at the top. On FAIL, do not ship, and record the next lever: if replay regressed, raise the replay row counts; if handwriting didn't improve, the 4.2k real set is too small — collect real crops into `cache/collected/<name>/` (imagefolder + `metadata.jsonl`, see README) and add that path as a source.

- [ ] **Step 3: Commit**

```bash
git add training/eval/results/finetune_v1.md
git commit -m "docs(eval): fine-tune v1 results vs starting checkpoint"
```

Proceed to Task 9 either way: it puts the new model in the Mac app so you can try it, and its CoreML bench is the real ship decision. On FAIL, Task 9 still runs, but finish with its rollback step.

---

### Task 9: Copy the model to the Mac, build and run the app

Runs on **this Mac** in the **main checkout** `/Users/kim/Sync/workspace/khmerclipper/apps/mac` — not the worktree. `raw_models/`, `Vendor/` and `.trocr-venv/` are gitignored and Syncthing-ignored, so they exist only in main. The branch changes nothing under `apps/mac`, so building from main is correct. Result docs are edited and committed in the worktree.

**Files:**
- Replace (gitignored, main checkout): `apps/mac/raw_models/TrOCR-khmer/` ← box `experiments/checkpoint-4/final/`
- Regenerated (gitignored): `apps/mac/Vendor/trocr/khmer/{encoder.mlpackage,decoder.mlpackage,tokenizer.json}`
- Backup (gitignored): `apps/mac/raw_models/TrOCR-khmer-v0-step72k/`, `apps/mac/raw_models/khmer-coreml-v0-step72k/`
- Modify (worktree): `training/eval/results/mac_gate.md`, `training/eval/results/finetune_v1.md`

**Interfaces:**
- Consumes: `experiments/checkpoint-4/final` on the box (Task 7; use the real N). Main-checkout Makefile targets: `make trocr-khmer` (convert `KHMER_CKPT=raw_models/TrOCR-khmer` → `Vendor/trocr/khmer`), `make build`, `make test`, `make run`. Existing `.trocr-venv` (coremltools 9.0, transformers 4.57.6).
- Produces: `build/Build/Products/Debug/KhmerClipper.app` running the fine-tuned model; Mac CoreML CER/latency vs Tesseract.

- [ ] **Step 1: Back up the current model**

The current local checkpoint is the old step-72k model (sha256 `04c0ac11…`, same as `checkpoint-3/final`). Keep it and its CoreML conversion for rollback. Put the CoreML backup in `raw_models/`, not `Vendor/`: the whole `Vendor/trocr` folder is bundled into the app.

```bash
cd /Users/kim/Sync/workspace/khmerclipper/apps/mac
shasum -a 256 raw_models/TrOCR-khmer/model.safetensors | cut -c1-16   # expect 04c0ac11e85d3665
mv raw_models/TrOCR-khmer raw_models/TrOCR-khmer-v0-step72k
cp -R Vendor/trocr/khmer raw_models/khmer-coreml-v0-step72k
```

Expected: the hash prints `04c0ac11e85d3665`. If it doesn't, stop and ask what model is in there before moving it.

- [ ] **Step 2: Copy the fine-tuned checkpoint from the box**

```bash
cd /Users/kim/Sync/workspace/khmerclipper/apps/mac
rsync -avL kim@192.168.1.25:/home/kim/Sync/workspace/khmerclipper/training/trocr/experiments/checkpoint-4/final/ raw_models/TrOCR-khmer/
shasum -a 256 raw_models/TrOCR-khmer/model.safetensors | cut -d' ' -f1
ssh kim@192.168.1.25 'sha256sum /home/kim/Sync/workspace/khmerclipper/training/trocr/experiments/checkpoint-4/final/model.safetensors' | cut -d' ' -f1
```

Expected: `model.safetensors` (~1.27GB), `config.json`, `generation_config.json`, the tokenizer files and `preprocessor_config.json` are copied, and the two hashes are identical and **not** `04c0ac11…`.

- [ ] **Step 3: Convert to CoreML**

```bash
cd /Users/kim/Sync/workspace/khmerclipper/apps/mac
make trocr-khmer
ls -la Vendor/trocr/khmer
```

Expected: the conversion finishes without error, and `encoder.mlpackage`, `decoder.mlpackage` and `tokenizer.json` in `Vendor/trocr/khmer` have fresh timestamps. A failure here is in `Vendor/trocr/convert.py` / `export_tokenizer.py`, not the model; the vocab is still 50414, as the old model's was.

- [ ] **Step 4: Build and test**

```bash
cd /Users/kim/Sync/workspace/khmerclipper/apps/mac
make build
make test
```

Expected: `** BUILD SUCCEEDED **`, then `** TEST SUCCEEDED **`. The app is at `build/Build/Products/Debug/KhmerClipper.app`.

- [ ] **Step 5: Smoke the new model through OCRBench**

Build the bench tool, then OCR one printed and one UI fixture with the CoreML TrOCR path:

```bash
cd /Users/kim/Sync/workspace/khmerclipper/apps/mac
xcodebuild -project KhmerClipper.xcodeproj -scheme OCRBench -configuration Debug -derivedDataPath build \
  CODE_SIGN_STYLE=Manual CODE_SIGN_IDENTITY="" PROVISIONING_PROFILE_SPECIFIER="" DEVELOPMENT_TEAM="" build
export KHMERCLIPPER_VENDOR=/Users/kim/Sync/workspace/khmerclipper/apps/mac/Vendor
build/Build/Products/Debug/OCRBench --engine trocr --preload \
  ../../training/eval/fixtures/printed/printed-001.png ../../training/eval/fixtures/ui/ui-001.png
```

Expected: readable Khmer for both images, close to their `text` in `training/eval/fixtures/labels.jsonl` (`ui-001` is `បើកឯកសារ`). Empty output or the same few characters repeated means the conversion or tokenizer export is broken: roll back (Step 9) and debug.

- [ ] **Step 6: Run the app**

```bash
cd /Users/kim/Sync/workspace/khmerclipper/apps/mac
make run
pgrep -x KhmerClipper
```

Expected: `make run` quits any old instance and launches the new build; `pgrep` prints a PID. Then check by hand:
1. In Preferences, the OCR engine is **TrOCR** (the default; select it if a previous session changed it).
2. Clip a few Khmer regions on screen: an app's UI text, a printed PDF page, an italic or decorative line, a handwriting photo. The recognized text should match what you see.
3. Compare against the old model on the same regions if useful: Step 9 swaps it back in, and running Steps 2–6 again restores the new one.

- [ ] **Step 7: Mac CoreML gate bench (the ship decision)**

The eval harness must be installed in the Mac Python you use (`cd training/eval && python -m pip install -e .`, per `mac_gate.md` Prerequisites).

```bash
cd /Users/kim/Sync/workspace/khmerclipper/training/eval
export KHMERCLIPPER_VENDOR=/Users/kim/Sync/workspace/khmerclipper/apps/mac/Vendor
python scripts/bench.py \
  --engine tesseract --engine trocr-coreml \
  --trocr-coreml-bin /Users/kim/Sync/workspace/khmerclipper/apps/mac/build/Build/Products/Debug/OCRBench \
  --out "$TMPDIR/mac_gate_finetune_v1.csv" \
  --candidate trocr-coreml --compare-to tesseract
```

Expected: one aggregate line per engine (`n=200 cer=… median_ms=… mixed_cer=…`), then `PASS` or `FAIL` with reasons. Write the CSV to `$TMPDIR`, not main's `results/`: result docs are committed from the worktree.

- [ ] **Step 8: Record results (in the worktree)**

In the worktree (`/Users/kim/Sync/workspace/khmerclipper/.worktrees/trocr-finetune-data`):
- `training/eval/results/mac_gate.md`: set **Status** to the Step 7 outcome, fill the Results table from the Step 7 aggregates, and note the model ("TrOCR fine-tune v1, checkpoint-4, sha256 <first 16 of Step 2 hash>"). Apply that file's Step 4a/4b/4c default-engine policy. Step 4c edits `apps/mac/Sources/Preferences/AppPreferences.swift` in the worktree.
- `training/eval/results/finetune_v1.md`: add a "Mac CoreML" section with the same numbers and one line from Step 6's manual check.

```bash
git add training/eval/results/mac_gate.md training/eval/results/finetune_v1.md
git commit -m "docs(eval): Mac CoreML gate for TrOCR fine-tune v1"
```

(If Step 4c changed `AppPreferences.swift`, add it to the same commit.)

- [ ] **Step 9: Roll back if the new model is worse**

Do this if Task 8 failed, Step 5 output was garbage, or the Step 7 CER is worse than the old model's:

```bash
cd /Users/kim/Sync/workspace/khmerclipper/apps/mac
rm -rf raw_models/TrOCR-khmer-v1 && mv raw_models/TrOCR-khmer raw_models/TrOCR-khmer-v1
mv raw_models/TrOCR-khmer-v0-step72k raw_models/TrOCR-khmer
rm -rf Vendor/trocr/khmer && cp -R raw_models/khmer-coreml-v0-step72k Vendor/trocr/khmer
make build && make run
```

Expected: the app runs the old model again (`shasum` of `raw_models/TrOCR-khmer/model.safetensors` starts `04c0ac11`). The v1 checkpoint stays in `raw_models/TrOCR-khmer-v1` for later comparison.

- [ ] **Step 10: Finish the branch (REQUIRED SUB-SKILL: superpowers:finishing-a-development-branch)**

Pass or fail, the code and results are worth keeping: merge or PR `feat/trocr-finetune-data` per that skill. After the worktree is removed, delete its box-side mirror (code only — `experiments` there is a symlink, so the checkpoints in main's `experiments/` are untouched):

```bash
ssh kim@192.168.1.25 'rm -rf /home/kim/Sync/workspace/khmerclipper/.worktrees/trocr-finetune-data'
```

Run it only after confirming `ls -la …/trocr-finetune-data/training/trocr/experiments` on the box shows a symlink (`->`), and only once the worktree is gone from the Mac (otherwise Syncthing would push it back).
