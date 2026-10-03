# Tesseract Khmer Fine-Tune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune Tesseract's `khm` LSTM model on every usable Khmer line dataset, on the CPU of server `192.168.1.25`. Ship it into the Mac app only if it beats the current model on the frozen fixtures and makes neither UI nor mixed text worse.

**Architecture:**
- A small Python package (`training/tesseract/src/khm_tess`) exports Hugging Face hub-cache parquet files to tesstrain ground truth and rewrites the train/eval lists.
- A pinned Docker image (Ubuntu 24.04 + apt Tesseract 5.3 + tesstrain at a fixed commit) does the `.lstmf` generation and the `lstmtraining` fine-tune from `tessdata_best/khm`.
- A Makefile runs every server step.
- The eval harness gains a `tesseract-ft` engine and a UI no-regression rule, so the Mac bench compares old and new `khm` in one pass.

**Tech Stack:** Python 3.10+, pyarrow, Pillow, pytest, uv, huggingface_hub CLI, Docker, Tesseract 5.3 training tools, tesstrain, GNU make.

**Spec:** `docs/superpowers/specs/2026-10-03-tesseract-khmer-finetune-design.md`

## Global Constraints

- **Worktree:** all work happens on branch `feat/tesseract-finetune` in `.worktrees/tesseract-finetune` (Task 0).
  - On the Mac, every relative path and `git` command runs from that worktree root, never from the main checkout.
  - On the server, Syncthing delivers the worktree to `~/Sync/workspace/khmerclip/.worktrees/tesseract-finetune`, and every `make` runs from its `training/tesseract`.
- Start model: `tessdata_best/khm.traineddata`, SHA-256 `3d2d975e67a319bf1cd67626b12fdebc1e417a4ebef9c9afee01d9927b512815`, tessdata_best commit `9ddc24e750eec0994223a9edc3fcb434a2244f3b`.
- tesstrain commit: `405346a3a67d8e4e049341d1da6a4b752e0b8351`. Container base: `ubuntu:24.04`.
- Server work directory: `~/khm-tesstrain`, **outside** `~/Sync`, because the repo there is Syncthing-synced. Never write ground truth, `.lstmf` files, checkpoints or a venv under `~/Sync`.
- Training container: `--cpus 8` with `OMP_THREAD_LIMIT=8`. Use `CPUS=6` while the TrOCR fine-tune (`2026-10-03-trocr-finetune-new-data.md`) is running on the same box.
- SoyVitou `testset.parquet` and Sokheng `data/test-*.parquet` are never read for training.
- Drop any training line whose stripped text exactly matches a fixture text or a `ui_strings.txt` line.
- Labels: strip outer whitespace only. Drop a line if it is empty or contains `\n`, `\r` or `\t`. No NFC, no collapsing of spaces.
- Pick checkpoints by `lstmeval` BCER on the SoyVitou-valid eval list, never on the fixtures.
- Ship as integer (`tessdata_fast/…`) traineddata only.
- Gate (all must hold against `tesseract`):
  - mean CER strictly lower
  - median latency ≤ 2×
  - mixed CER ≤
  - UI CER ≤
- `apps/mac/Vendor/` is gitignored. Ship by replacing the file, and record its SHA-256 plus the source checkpoint in `training/tesseract/results/finetune.md`.

## Review Focus

1. **A UI regression hidden by a better overall CER.** A printed-heavy fine-tune could cut mean CER while UI CER gets 3–5× worse. The gate must FAIL in that case. → `test_passes_gate_rejects_ui_regression` (Task 1).
2. **Train/test leakage.** SoyVitou `testset.parquet`, Sokheng `test-*`, or a fixture/UI string ending up in the training ground truth. Expected: none are ever exported. → `test_source_files_excludes_test_splits` and `test_export_source_writes_pairs_and_stats` (blocklist), both in Task 2.
3. **Bad rows in a ~650k-row export.** Null or undecodable image bytes, tiny images, multi-line, tab, empty or None labels. Expected: the row is skipped and counted, and the export doesn't crash. → `test_export_source_writes_pairs_and_stats` (Task 2).
4. **Interrupted export.** A `.png` written but no `.gt.txt`, then the export is re-run. Expected: the pair is rebuilt, never left half-written, and the counts are stable. → `test_export_source_resumes_and_repairs_partial` (Task 2).
5. **Eval lines leaking into the train list, or an empty list.** Expected: SoyVitou-valid lines appear only in `list.eval`, and an empty side errors out. → `test_split_keeps_valid_out_of_train` and `test_split_errors_when_side_empty` (Task 3).

---

### Task 0: Create the worktree

**Files:**
- Copy (gitignored): `apps/mac/Vendor/tesseract/` → the worktree's `apps/mac/Vendor/tesseract/`. The bench and the app tests need the vendored binary, and `Vendor/` isn't in git.

The design and this plan are already committed on `main`, so the worktree inherits them.

**Interfaces:**
- Produces: worktree `.worktrees/tesseract-finetune` on branch `feat/tesseract-finetune`, based on `main`. All later tasks run from its root.

- [ ] **Step 1: Create it (REQUIRED SUB-SKILL: superpowers:using-git-worktrees)**

The repo already uses `.worktrees/` (gitignored). From the main checkout:

```bash
cd /Users/kim/Sync/workspace/khmerclip
git worktree add .worktrees/tesseract-finetune -b feat/tesseract-finetune main
mkdir -p .worktrees/tesseract-finetune/apps/mac/Vendor
cp -R apps/mac/Vendor/tesseract .worktrees/tesseract-finetune/apps/mac/Vendor/tesseract
cd .worktrees/tesseract-finetune
shasum -a 256 apps/mac/Vendor/tesseract/tessdata/khm.traineddata
```

Expected:
- `git worktree list` shows `.worktrees/tesseract-finetune … [feat/tesseract-finetune]`
- the SHA printed is `47f110575341b322052f3becbefee61a3ecf1ef549352b5d4d33d28afe30d099` (stock tessdata_fast)

- [ ] **Step 2: Confirm the server sees the worktree**

```bash
ssh 192.168.1.25 'ls ~/Sync/workspace/khmerclip/.worktrees/tesseract-finetune/docs/superpowers/plans'
```

Expected: `2026-10-03-tesseract-khmer-finetune.md` is listed. If it isn't, wait for Syncthing and check again. Don't copy files by hand.

---

### Task 1: Eval harness — `tesseract-ft` engine, `lang` param, UI gate

**Files:**
- Modify: `training/eval/src/khmerclip_eval/runners/tesseract.py`
- Modify: `training/eval/src/khmerclip_eval/summary.py`
- Modify: `training/eval/scripts/bench.py`
- Modify: `training/eval/results/mac_gate.md` (gate criteria table)
- Test: `training/eval/tests/test_summary.py`, and create `training/eval/tests/test_tesseract_runner.py`

Note: `passes_gate` is shared with the TrOCR plans. The new UI rule therefore also applies to TrOCR, which must keep UI CER ≤ Tesseract's 0.0199. Those plans edit `mac_gate.md` only to record outcomes. If one lands first, rebase and keep both changes.

**Interfaces:**
- Produces:
  - `recognize(image_path: Path, vendor_root: Path, lang: str = "khm+eng") -> str`
  - `aggregate(rows) -> dict[str, dict]`, where each value now also has `"ui_cer": float`
  - `passes_gate(candidate: dict, tesseract: dict) -> tuple[bool, list[str]]`, which also fails when `candidate.get("ui_cer", 0.0) > tesseract.get("ui_cer", 0.0)`
  - bench CLI: `--engine tesseract-ft --tesseract-ft-root PATH --tesseract-ft-lang LANG`

Run all eval tests from `training/eval` with:
`uv run --no-project --with jiwer --with pillow --with pytest --with-editable . pytest -q`

- [ ] **Step 1: Write the failing tests**

Append to `training/eval/tests/test_summary.py`:

```python
import pytest


def test_aggregate_ui_cer():
    rows = [
        {"engine": "t", "cer": 0.2, "ms": 1.0, "subset": "ui"},
        {"engine": "t", "cer": 0.4, "ms": 1.0, "subset": "ui"},
        {"engine": "t", "cer": 1.0, "ms": 1.0, "subset": "printed"},
    ]
    assert aggregate(rows)["t"]["ui_cer"] == pytest.approx(0.3)


def test_passes_gate_rejects_ui_regression():
    tess = {"cer": 0.25, "median_ms": 75.0, "mixed_cer": 0.09, "ui_cer": 0.02}
    cand = {"cer": 0.15, "median_ms": 75.0, "mixed_cer": 0.09, "ui_cer": 0.05}
    ok, reasons = passes_gate(cand, tess)
    assert ok is False
    assert any("ui_cer" in r for r in reasons)
```

Create `training/eval/tests/test_tesseract_runner.py`:

```python
import subprocess
from pathlib import Path

from khmerclip_eval.runners import tesseract


def _capture(monkeypatch):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="ក\n", stderr="")

    monkeypatch.setattr(tesseract.subprocess, "run", run)
    return calls


def test_recognize_defaults_to_khm_plus_eng(monkeypatch):
    calls = _capture(monkeypatch)
    assert tesseract.recognize(Path("x.png"), Path("/v")) == "ក"
    cmd = calls[0]
    assert cmd[cmd.index("-l") + 1] == "khm+eng"


def test_recognize_passes_lang(monkeypatch):
    calls = _capture(monkeypatch)
    tesseract.recognize(Path("x.png"), Path("/v"), lang="khm")
    cmd = calls[0]
    assert cmd[cmd.index("-l") + 1] == "khm"
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `cd training/eval && uv run --no-project --with jiwer --with pillow --with pytest --with-editable . pytest -q tests/test_summary.py tests/test_tesseract_runner.py`

Expected: FAIL.
- `KeyError: 'ui_cer'` in `test_aggregate_ui_cer`
- `assert True is False` in the UI-regression test
- `TypeError: recognize() got an unexpected keyword argument 'lang'`

- [ ] **Step 3: Implement**

In `runners/tesseract.py`, change the signature and the `-l` value:

```python
def recognize(image_path: Path, vendor_root: Path, lang: str = "khm+eng") -> str:
```

```python
            "-l",
            lang,
```

In `summary.py`, inside `aggregate`, after the `mixed = …` line:

```python
        ui = [float(r["cer"]) for r in items if r.get("subset") == "ui"]
```

and add this entry to the `out[engine]` dict:

```python
            "ui_cer": (sum(ui) / len(ui)) if ui else 0.0,
```

In `passes_gate`, before `return`:

```python
    if candidate.get("ui_cer", 0.0) > tesseract.get("ui_cer", 0.0):
        reasons.append(
            f"ui_cer {candidate.get('ui_cer', 0.0):.4f} > Tess {tesseract.get('ui_cer', 0.0):.4f}"
        )
```

In `scripts/bench.py`:
- add `"tesseract-ft"` to the `--engine` `choices` list
- add these arguments after `--tesseract-root`:

```python
    p.add_argument("--tesseract-ft-root", type=Path, default=None)
    p.add_argument("--tesseract-ft-lang", default="khm+eng")
```

- add this branch right after the `if engine == "tesseract":` branch in the per-item loop:

```python
            elif engine == "tesseract-ft":
                if not args.tesseract_ft_root:
                    raise SystemExit("--tesseract-ft-root required")
                pred = tess_runner.recognize(
                    item.image_path, args.tesseract_ft_root, args.tesseract_ft_lang
                )
```

- replace the per-engine summary `print` with:

```python
        print(
            f"{engine}: n={stats['n']} cer={stats['cer']:.4f} "
            f"median_ms={stats['median_ms']:.1f} mixed_cer={stats['mixed_cer']:.4f} "
            f"ui_cer={stats['ui_cer']:.4f}"
        )
```

In `training/eval/results/mac_gate.md`, add this row to the "Gate criteria" table, after the `Mixed-script CER` row:

```markdown
| UI CER | candidate UI-subset mean CER **≤** Tesseract UI-subset mean CER |
```
- [ ] **Step 4: Run all eval tests and a bench smoke**

Run: `cd training/eval && uv run --no-project --with jiwer --with pillow --with pytest --with-editable . pytest -q`
Expected: all pass, including the existing `test_passes_gate`, whose dicts have no `ui_cer`.

Run: `uv run --no-project --with jiwer --with pillow --with-editable . python scripts/bench.py --engine tesseract --engine tesseract-ft --tesseract-ft-root ../../apps/mac/Vendor/tesseract --limit 5 --out /tmp/ft_smoke.csv --candidate tesseract-ft`

Expected: two summary lines with identical `cer`, `mixed_cer` and `ui_cer` (same model), then `FAIL` with reason `CER … not < Tess …`. That proves a candidate that only ties is rejected.

- [ ] **Step 5: Commit**

```bash
git add training/eval/src/khmerclip_eval/runners/tesseract.py training/eval/src/khmerclip_eval/summary.py training/eval/scripts/bench.py training/eval/results/mac_gate.md training/eval/tests/test_summary.py training/eval/tests/test_tesseract_runner.py
git commit -m "feat(eval): add tesseract-ft engine and UI no-regression gate"
```

---

### Task 2: Ground-truth exporter (`khm_tess.export_gt`)

**Files:**
- Create: `training/tesseract/pyproject.toml`
- Create: `training/tesseract/.gitignore`
- Create: `training/tesseract/src/khm_tess/__init__.py` (empty)
- Create: `training/tesseract/src/khm_tess/export_gt.py`
- Test: `training/tesseract/tests/test_export_gt.py`

**Interfaces:**
- Produces:
  - `SOURCES: dict[str, tuple[str, str, str]]`, mapping alias → (hub cache dir, glob under the snapshot, text column)
  - `DEFAULT_SOURCES: list[str]`
  - `clean_label(text: str | None) -> str | None`
  - `load_blocklist(labels_jsonl: Path, ui_strings: Path) -> set[str]`
  - `source_files(hub: Path, alias: str) -> list[Path]`
  - `iter_rows(files: list[Path], limit: int | None, text_col: str = "text") -> Iterator[tuple[int, bytes | None, str | None]]`
  - `export_source(alias: str, files: list[Path], out_dir: Path, blocklist: set[str], limit: int | None = None) -> dict[str, int]`
  - `parse_source(spec: str) -> tuple[str, int | None]`
  - console script `khm-export-gt`
- Output naming, which Task 3 relies on: `<out>/<alias>-<index:08d>.png` plus `.gt.txt`. The eval rows therefore start with `soyvitou-valid-`.

Run tests from `training/tesseract` with: `uv run --extra dev pytest -q`

- [ ] **Step 1: Scaffold the package**

`training/tesseract/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "khm-tess"
version = "0.1.0"
description = "Export Khmer line datasets to tesstrain ground truth"
requires-python = ">=3.10"
dependencies = ["pyarrow>=14", "pillow>=10.0.0"]

[project.optional-dependencies]
dev = ["pytest>=7.0"]

[project.scripts]
khm-export-gt = "khm_tess.export_gt:main"
khm-split-lists = "khm_tess.split_lists:main"

[tool.setuptools.packages.find]
where = ["src"]
```

`training/tesseract/.gitignore`:

```gitignore
.venv/
out/
```

Create an empty `training/tesseract/src/khm_tess/__init__.py`.

- [ ] **Step 2: Write the failing tests**

`training/tesseract/tests/test_export_gt.py`:

```python
import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from khm_tess import export_gt as eg


def _png(w=40, h=20) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, "PNG")
    return buf.getvalue()


def _parquet(path, rows, text_col="text"):
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "image": pa.array(
                [{"bytes": b, "path": None} for b, _ in rows],
                type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
            ),
            text_col: pa.array([t for _, t in rows], type=pa.string()),
        }
    )
    pq.write_table(table, path)


def test_clean_label():
    assert eg.clean_label("  ក  ខ \n") == "ក  ខ"
    assert eg.clean_label("") is None
    assert eg.clean_label("   ") is None
    assert eg.clean_label(None) is None
    assert eg.clean_label("ក\nខ") is None
    assert eg.clean_label("ក\tខ") is None
    assert eg.clean_label("ក\rខ") is None


def test_load_blocklist(tmp_path):
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps({"id": "a", "image": "a.png", "text": " ក ខ ", "subset": "printed"})
        + "\n\n",
        encoding="utf-8",
    )
    ui = tmp_path / "ui.txt"
    ui.write_text("បើកឯកសារ\n\n", encoding="utf-8")
    assert eg.load_blocklist(labels, ui) == {"ក ខ", "បើកឯកសារ"}


def test_source_files_excludes_test_splits(tmp_path):
    hub = tmp_path / "hub"
    soy = hub / "datasets--SoyVitou--62k-images-khmer-printed-dataset/snapshots/s1"
    for name in ("trainset.parquet", "validset.parquet", "testset.parquet"):
        _parquet(soy / name, [(_png(), name)])
    sok = hub / "datasets--Sokheng--khmer-synthetic-ocr-v1-100k/snapshots/s2/data"
    _parquet(sok / "train-00000-of-00001.parquet", [(_png(), "t")])
    _parquet(sok / "test-00000-of-00001.parquet", [(_png(), "x")])

    assert [p.name for p in eg.source_files(hub, "soyvitou-train")] == ["trainset.parquet"]
    assert [p.name for p in eg.source_files(hub, "soyvitou-valid")] == ["validset.parquet"]
    assert [p.name for p in eg.source_files(hub, "sokheng")] == ["train-00000-of-00001.parquet"]
    with pytest.raises(FileNotFoundError):
        eg.source_files(hub, "hanuman")


def test_iter_rows_stride_sampling(tmp_path):
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    _parquet(a, [(_png(), f"r{i}") for i in range(4)])
    _parquet(b, [(_png(), f"r{i}") for i in range(4, 10)])
    assert [i for i, _, _ in eg.iter_rows([a, b], 3)] == [0, 3, 6]
    assert [t for _, _, t in eg.iter_rows([a, b], None)] == [f"r{i}" for i in range(10)]
    assert len(list(eg.iter_rows([a, b], 50))) == 10


def test_iter_rows_custom_text_column(tmp_path):
    f = tmp_path / "d.parquet"
    _parquet(f, [(_png(), "ក")], text_col="label")
    assert [t for _, _, t in eg.iter_rows([f], None, "label")] == ["ក"]
    assert eg.SOURCES["darayut"][2] == "label"


def test_export_source_writes_pairs_and_stats(tmp_path):
    src = tmp_path / "s.parquet"
    _parquet(
        src,
        [
            (_png(), "ក ខ"),          # kept, has space
            (_png(), "Hello ក"),      # kept, latin + space
            (_png(), "LEAK"),         # blocklisted
            (_png(), "   "),          # empty label
            (b"not an image", "ក"),   # undecodable image
            (_png(3, 3), "ខ"),        # too small
            (None, "គ"),              # null image bytes
        ],
    )
    out = tmp_path / "gt"
    out.mkdir()
    stats = eg.export_source("hanuman", [src], out, {"LEAK"})
    assert stats == {
        "kept": 2,
        "dropped_label": 1,
        "dropped_blocklist": 1,
        "dropped_image": 3,
        "with_space": 2,
        "with_latin": 1,
    }
    assert (out / "hanuman-00000000.gt.txt").read_text(encoding="utf-8") == "ក ខ\n"
    assert Image.open(out / "hanuman-00000001.png").mode == "L"
    assert sorted(p.name for p in out.iterdir()) == [
        "hanuman-00000000.gt.txt",
        "hanuman-00000000.png",
        "hanuman-00000001.gt.txt",
        "hanuman-00000001.png",
    ]


def test_export_flattens_transparent_background_to_white(tmp_path):
    im = Image.new("RGBA", (40, 20), (0, 0, 0, 0))  # fully transparent black
    im.putpixel((5, 5), (0, 0, 0, 255))               # one opaque black "ink" pixel
    buf = io.BytesIO()
    im.save(buf, "PNG")
    src = tmp_path / "s.parquet"
    _parquet(src, [(buf.getvalue(), "ក")])
    out = tmp_path / "gt"
    out.mkdir()
    eg.export_source("sokheng", [src], out, set())
    gray = Image.open(out / "sokheng-00000000.png")
    assert gray.getpixel((0, 0)) == 255   # background white, not black
    assert gray.getpixel((5, 5)) == 0     # ink preserved


def test_export_source_resumes_and_repairs_partial(tmp_path):
    src = tmp_path / "s.parquet"
    _parquet(src, [(_png(), "ក"), (_png(), "ខ")])
    out = tmp_path / "gt"
    out.mkdir()
    eg.export_source("kiteaether", [src], out, set())
    (out / "kiteaether-00000001.gt.txt").unlink()  # simulate crash before gt write
    stats = eg.export_source("kiteaether", [src], out, set())
    assert stats["kept"] == 2
    assert (out / "kiteaether-00000001.gt.txt").read_text(encoding="utf-8") == "ខ\n"


def test_parse_source():
    assert eg.parse_source("mrrtmob:200000") == ("mrrtmob", 200000)
    assert eg.parse_source("hanuman") == ("hanuman", None)
    with pytest.raises(SystemExit):
        eg.parse_source("bogus")
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `cd training/tesseract && uv run --extra dev pytest -q`
Expected: FAIL with `ImportError: cannot import name 'export_gt' from 'khm_tess'`.

- [ ] **Step 4: Implement `training/tesseract/src/khm_tess/export_gt.py`**

```python
"""Export cached Hugging Face Khmer line datasets to tesstrain ground truth.

Writes ``<out>/<alias>-<index:08d>.png`` + ``.gt.txt`` pairs and ``export_stats.json``.
Reads parquet straight from the hub cache so test splits can be excluded by file name:
SoyVitou's dataset card lumps testset.parquet (our eval fixtures' source) into ``train``.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import re
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

REPO = Path(__file__).resolve().parents[4]

# alias -> (hub cache dir, glob relative to the snapshot root, text column).
# Test splits are deliberately absent.
SOURCES = {
    "soyvitou-train": ("datasets--SoyVitou--62k-images-khmer-printed-dataset", "trainset.parquet", "text"),
    "soyvitou-valid": ("datasets--SoyVitou--62k-images-khmer-printed-dataset", "validset.parquet", "text"),
    "sokheng": ("datasets--Sokheng--khmer-synthetic-ocr-v1-100k", "data/train-*.parquet", "text"),
    "hanuman": ("datasets--seanghay--khmer-hanuman-100k", "data/train-*.parquet", "text"),
    "kiteaether": ("datasets--KiteAether--c_khmer_gemma_ocr_train", "data/train-*.parquet", "text"),
    "mrrtmob": ("datasets--mrrtmob--khmer_english_ocr_image_line", "data/train-*.parquet", "text"),
    "darayut": ("datasets--Darayut--khmer-document-synthetic-low-res", "data/train-*.parquet", "label"),
    "siemreap": ("datasets--KimkosalYon--khmer-ocr-200k-siemreap-arial", "data/train-*.parquet", "text"),
}
DEFAULT_SOURCES = [
    "soyvitou-train",
    "soyvitou-valid:2000",
    "sokheng",
    "hanuman",
    "kiteaether",
    "darayut",
    "siemreap:100000",
    "mrrtmob:200000",
]
LATIN_RE = re.compile(r"[A-Za-z]")
MIN_SIDE = 8


def clean_label(text: str | None) -> str | None:
    # ponytail: strip only; CER is on raw text, so no NFC / space collapsing.
    if text is None:
        return None
    label = text.strip()
    if not label or any(c in label for c in "\n\r\t"):
        return None
    return label


def load_blocklist(labels_jsonl: Path, ui_strings: Path) -> set[str]:
    block = set()
    for line in labels_jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            block.add(json.loads(line)["text"].strip())
    for line in ui_strings.read_text(encoding="utf-8").splitlines():
        if line.strip():
            block.add(line.strip())
    return block


def source_files(hub: Path, alias: str) -> list[Path]:
    repo, pattern, _ = SOURCES[alias]
    files = sorted(Path(p) for p in glob.glob(str(hub / repo / "snapshots" / "*" / pattern)))
    if not files:
        raise FileNotFoundError(f"no parquet for {alias} under {hub / repo}")
    return files


def iter_rows(
    files: list[Path], limit: int | None, text_col: str = "text"
) -> Iterator[tuple[int, bytes | None, str | None]]:
    """Yield (global_index, image_bytes, text); with ``limit``, every Nth row spread over all files."""
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    stride = 1 if limit is None else max(1, total // limit)
    index = taken = 0
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=256, columns=["image", text_col]):
            for image, text in zip(batch.column("image").to_pylist(), batch.column(text_col).to_pylist()):
                if index % stride == 0:
                    yield index, (image or {}).get("bytes"), text
                    taken += 1
                    if limit is not None and taken >= limit:
                        return
                index += 1


def _write_image(data: bytes | None, path: Path) -> bool:
    if not data:
        return False
    try:
        with Image.open(io.BytesIO(data)) as im:
            if min(im.size) < MIN_SIDE:
                return False
            if im.mode in ("RGBA", "LA", "P") or "transparency" in im.info:
                # Flatten onto white: a bare convert("L") turns transparent pixels black,
                # which would silently blank out black-on-transparent text lines.
                rgba = im.convert("RGBA")
                im = Image.new("RGBA", rgba.size, "white")
                im.alpha_composite(rgba)
            im.convert("L").save(path)
    except OSError:  # PIL.UnidentifiedImageError subclasses OSError
        path.unlink(missing_ok=True)
        return False
    return True


def export_source(
    alias: str, files: list[Path], out_dir: Path, blocklist: set[str], limit: int | None = None
) -> dict[str, int]:
    text_col = SOURCES[alias][2] if alias in SOURCES else "text"
    stats = dict(kept=0, dropped_label=0, dropped_blocklist=0, dropped_image=0, with_space=0, with_latin=0)
    for index, data, text in iter_rows(files, limit, text_col):
        label = clean_label(text)
        if label is None:
            stats["dropped_label"] += 1
            continue
        if label in blocklist:
            stats["dropped_blocklist"] += 1
            continue
        png = out_dir / f"{alias}-{index:08d}.png"
        gt = out_dir / f"{alias}-{index:08d}.gt.txt"
        if not (png.exists() and gt.exists()):  # gt is written last, so a pair implies a full write
            if not _write_image(data, png):
                stats["dropped_image"] += 1
                continue
            gt.write_text(label + "\n", encoding="utf-8")
        stats["kept"] += 1
        stats["with_space"] += " " in label
        stats["with_latin"] += bool(LATIN_RE.search(label))
    return stats


def parse_source(spec: str) -> tuple[str, int | None]:
    alias, _, limit = spec.partition(":")
    if alias not in SOURCES:
        raise SystemExit(f"unknown source {alias!r}; choose from {sorted(SOURCES)}")
    return alias, int(limit) if limit else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hub", type=Path, default=Path.home() / ".cache/huggingface/hub")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fixtures-labels", type=Path, default=REPO / "training/eval/fixtures/labels.jsonl")
    p.add_argument("--ui-strings", type=Path, default=REPO / "training/eval/ui_strings.txt")
    p.add_argument("--source", action="append", help="alias[:limit]; repeatable")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    blocklist = load_blocklist(args.fixtures_labels, args.ui_strings)
    all_stats = {}
    for spec in args.source or DEFAULT_SOURCES:
        alias, limit = parse_source(spec)
        all_stats[alias] = export_source(alias, source_files(args.hub, alias), args.out, blocklist, limit)
        print(alias, json.dumps(all_stats[alias]), flush=True)
    (args.out / "export_stats.json").write_text(json.dumps(all_stats, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `cd training/tesseract && uv run --extra dev pytest -q`
Expected: `9 passed`.

- [ ] **Step 6: Commit**

```bash
git add training/tesseract/pyproject.toml training/tesseract/.gitignore training/tesseract/src training/tesseract/tests/test_export_gt.py
git commit -m "feat(tesseract): export cached Khmer datasets to tesstrain ground truth"
```

---

### Task 3: Train/eval list splitter (`khm_tess.split_lists`)

**Files:**
- Create: `training/tesseract/src/khm_tess/split_lists.py`
- Test: `training/tesseract/tests/test_split_lists.py`

**Interfaces:**
- Consumes:
  - tesstrain's `<output_dir>/all-lstmf`, one `.lstmf` path per line, already shuffled
  - Task 2's naming: eval files start with `soyvitou-valid-`
- Produces:
  - `split(lstmf_paths: list[str], eval_prefix: str = "soyvitou-valid-") -> tuple[list[str], list[str]]`, returning `(train, eval)`
  - console script `khm-split-lists OUTPUT_DIR`, which overwrites `OUTPUT_DIR/list.train` and `OUTPUT_DIR/list.eval`

- [ ] **Step 1: Write the failing tests**

`training/tesseract/tests/test_split_lists.py`:

```python
import pytest

from khm_tess.split_lists import main, split


def test_split_keeps_valid_out_of_train():
    paths = [
        "/work/data/khm_ft-ground-truth/hanuman-00000001.lstmf",
        "/work/data/khm_ft-ground-truth/soyvitou-valid-00000003.lstmf",
        "/work/data/khm_ft-ground-truth/soyvitou-train-00000002.lstmf",
    ]
    train, eval_ = split(paths)
    assert train == [paths[0], paths[2]]
    assert eval_ == [paths[1]]


def test_split_errors_when_side_empty():
    with pytest.raises(ValueError, match="eval"):
        split(["/x/hanuman-00000001.lstmf"])
    with pytest.raises(ValueError, match="train"):
        split(["/x/soyvitou-valid-00000001.lstmf"])


def test_main_overwrites_lists(tmp_path, monkeypatch):
    (tmp_path / "all-lstmf").write_text("/g/hanuman-1.lstmf\n/g/soyvitou-valid-2.lstmf\n\n")
    (tmp_path / "list.train").write_text("stale\n")
    monkeypatch.setattr("sys.argv", ["khm-split-lists", str(tmp_path)])
    main()
    assert (tmp_path / "list.train").read_text() == "/g/hanuman-1.lstmf\n"
    assert (tmp_path / "list.eval").read_text() == "/g/soyvitou-valid-2.lstmf\n"
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `cd training/tesseract && uv run --extra dev pytest -q tests/test_split_lists.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'khm_tess.split_lists'`.

- [ ] **Step 3: Implement `training/tesseract/src/khm_tess/split_lists.py`**

```python
"""Replace tesstrain's random 90/10 split: SoyVitou-valid lines are the eval list, all else trains.

Run after ``make lists``; tesstrain won't regenerate list.* because they end up newer than all-lstmf.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def split(lstmf_paths: list[str], eval_prefix: str = "soyvitou-valid-") -> tuple[list[str], list[str]]:
    train = [p for p in lstmf_paths if not Path(p).name.startswith(eval_prefix)]
    eval_ = [p for p in lstmf_paths if Path(p).name.startswith(eval_prefix)]
    if not eval_:
        raise ValueError(f"no eval files starting with {eval_prefix!r}")
    if not train:
        raise ValueError("no train files")
    return train, eval_


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output_dir", type=Path)
    out = p.parse_args().output_dir
    paths = [line for line in (out / "all-lstmf").read_text().splitlines() if line.strip()]
    train, eval_ = split(paths)
    (out / "list.train").write_text("\n".join(train) + "\n")
    (out / "list.eval").write_text("\n".join(eval_) + "\n")
    print(f"train={len(train)} eval={len(eval_)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all package tests and confirm they pass**

Run: `cd training/tesseract && uv run --extra dev pytest -q`
Expected: `12 passed`.

- [ ] **Step 5: Commit**

```bash
git add training/tesseract/src/khm_tess/split_lists.py training/tesseract/tests/test_split_lists.py
git commit -m "feat(tesseract): split tesstrain lists with SoyVitou-valid as eval"
```

---

### Task 4: Pinned training container, Makefile and README; verify on the server

**Files:**
- Create: `training/tesseract/Dockerfile`
- Create: `training/tesseract/Makefile`
- Create: `training/tesseract/README.md`
- Modify: `README.md` (repo root) — add a layout row and a Training link

**Interfaces:**
- Consumes: the console scripts `khm-export-gt` and `khm-split-lists` (Tasks 2–3)
- Produces make targets, used on the server from `~/Sync/workspace/khmerclip/.worktrees/tesseract-finetune/training/tesseract`:
  - `image`
  - `check`
  - `download`
  - `gt [SOURCES="alias[:limit] …"]`
  - `dicts`
  - `lists`
  - `train [EPOCHS=n | MAX_ITERATIONS=n] [DETACH=]`
  - `logs`
  - `evaluate`
  - `test`
- Every target takes `MODEL=` (default `khm_ft`) and `CPUS=` (default 8).
- Outputs land under `~/khm-tesstrain/data/$(MODEL)/`; the int models go in `tessdata_fast/`.

- [ ] **Step 1: Write `training/tesseract/Dockerfile`**

```dockerfile
# Tesseract LSTM fine-tune toolchain (CPU only). Pins: tesstrain commit + tessdata_best khm SHA-256.
FROM ubuntu:24.04

ARG TESSTRAIN_REF=405346a3a67d8e4e049341d1da6a4b752e0b8351
ARG TESSDATA_BEST_REF=9ddc24e750eec0994223a9edc3fcb434a2244f3b
ARG KHM_BEST_SHA256=3d2d975e67a319bf1cd67626b12fdebc1e417a4ebef9c9afee01d9927b512815

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      tesseract-ocr make git python3 python3-pil wget ca-certificates bc \
 && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/tesseract-ocr/tesstrain /opt/tesstrain \
 && git -C /opt/tesstrain checkout "$TESSTRAIN_REF"

RUN mkdir -p /opt/tessdata_best \
 && wget -qO /opt/tessdata_best/khm.traineddata \
      "https://github.com/tesseract-ocr/tessdata_best/raw/${TESSDATA_BEST_REF}/khm.traineddata" \
 && echo "${KHM_BEST_SHA256}  /opt/tessdata_best/khm.traineddata" | sha256sum -c -
```

- [ ] **Step 2: Write `training/tesseract/Makefile`**

```make
# Tesseract khm fine-tune. Run on the training server; see README.md.
# All generated data lives in WORK (outside ~/Sync — Syncthing must not see it).
WORK    ?= $(HOME)/khm-tesstrain
HF_HUB  ?= $(HOME)/.cache/huggingface/hub
IMAGE   ?= khm-tesstrain
MODEL   ?= khm_ft
CPUS    ?= 8
JOBS    ?= 10
EPOCHS  ?= 3
MAX_ITERATIONS ?=
SOURCES ?=
DETACH  ?= -d

OUT   = $(WORK)/data/$(MODEL)
UV    = UV_PROJECT_ENVIRONMENT=$(WORK)/.venv uv run
DOCKER_ARGS = --cpus $(CPUS) --user $$(id -u):$$(id -g) -e HOME=/tmp -e OMP_THREAD_LIMIT=$(CPUS) -v $(WORK):/work
RUN   = docker run --rm $(DOCKER_ARGS) $(IMAGE)
TT    = make -C /opt/tesstrain DATA_DIR=/work/data TESSDATA=/opt/tessdata_best START_MODEL=khm MODEL_NAME=$(MODEL) PSM=13
ITERS = $(if $(MAX_ITERATIONS),MAX_ITERATIONS=$(MAX_ITERATIONS),EPOCHS=$(EPOCHS))

.PHONY: image check download gt dicts lists train logs evaluate test

image:
	docker build -t $(IMAGE) .

check:
	docker run --rm $(IMAGE) sh -c 'set -e; tesseract --version 2>&1 | head -1; tesseract --list-langs; \
	  ls /usr/share/tesseract-ocr/5/tessdata/configs/lstm.train; \
	  command -v lstmtraining lstmeval combine_tessdata combine_lang_model unicharset_extractor merge_unicharsets dawg2wordlist; \
	  git -C /opt/tesstrain rev-parse HEAD; sha256sum /opt/tessdata_best/khm.traineddata; \
	  combine_tessdata -d /opt/tessdata_best/khm.traineddata'

# Sources not yet in the hub cache (the other five are cached already).
download:
	$(UV) --with "huggingface_hub>=0.34" hf download Darayut/khmer-document-synthetic-low-res --repo-type dataset --include "data/*"
	$(UV) --with "huggingface_hub>=0.34" hf download KimkosalYon/khmer-ocr-200k-siemreap-arial --repo-type dataset --include "data/*"

gt:
	mkdir -p $(WORK)/data/$(MODEL)-ground-truth
	$(UV) khm-export-gt --hub $(HF_HUB) --out $(WORK)/data/$(MODEL)-ground-truth $(foreach s,$(SOURCES),--source $(s))

# Keep khm's dictionary: tesstrain builds the proto model from these three files if present.
dicts:
	mkdir -p $(OUT)
	$(RUN) sh -c 'set -e; mkdir -p /tmp/khm; combine_tessdata -u /opt/tessdata_best/khm.traineddata /tmp/khm/khm.; \
	  for d in word:wordlist punc:punc number:numbers; do \
	    dawg=/tmp/khm/khm.lstm-$${d%%:*}-dawg; \
	    if [ -f $$dawg ]; then dawg2wordlist /tmp/khm/khm.lstm-unicharset $$dawg /work/data/$(MODEL)/$(MODEL).$${d##*:}; \
	    else echo "skip: $$dawg not in start model"; fi; \
	  done; wc -l /work/data/$(MODEL)/$(MODEL).* || true'

lists:
	$(RUN) $(TT) -j $(JOBS) lists unicharset
	$(UV) khm-split-lists $(OUT)

# Re-running resumes: tesstrain's training target depends on phony targets, so lstmtraining always
# runs, and lstmtraining restores <model>_checkpoint when it exists.
train:
	-docker rm $(MODEL)-train
	docker run $(DETACH) --name $(MODEL)-train $(DOCKER_ARGS) $(IMAGE) $(TT) training $(ITERS)

logs:
	tail -f $(OUT)/training.log

# BCER on list.eval: start model first, then every checkpoint (lowest wins).
evaluate:
	$(RUN) sh -c 'lstmeval --model /opt/tessdata_best/khm.traineddata --eval_listfile /work/data/$(MODEL)/list.eval 2>&1 | grep "^BCER eval" | sed "s/^/START /"'
	$(RUN) $(TT) -j $(JOBS) traineddata evaluation
	grep -H "^BCER eval" $(OUT)/eval/*.eval.log | sort -t= -k2 -g | head -5

test:
	$(UV) --extra dev pytest -q
```

- [ ] **Step 3: Write `training/tesseract/README.md`**

````markdown
# Tesseract Khmer fine-tune

Fine-tunes `tessdata_best/khm` on Khmer line datasets and produces an integer
`khm.traineddata` for `apps/mac/Vendor/tesseract/tessdata/`. Design:
[`docs/superpowers/specs/2026-10-03-tesseract-khmer-finetune-design.md`](../../docs/superpowers/specs/2026-10-03-tesseract-khmer-finetune-design.md).

Training is CPU-only and runs in Docker on the training server (`192.168.1.25`), from this
directory in the Syncthing-synced clone. All generated data goes to `~/khm-tesstrain`
(`WORK`), never under `~/Sync`. Use `CPUS=6` while a TrOCR run shares the box.

## Run (server)

```bash
cd ~/Sync/workspace/khmerclip/training/tesseract   # or .worktrees/tesseract-finetune/... before merge
make image check          # build + verify toolchain pins
make download             # fetch the two datasets not already in the HF cache
make gt                   # export ground truth (DEFAULT_SOURCES, ~650k lines)
make dicts lists          # khm dictionary + .lstmf + train/eval lists
make train                # detached; `make logs` to follow, re-run to resume
make evaluate             # BCER per checkpoint on SoyVitou-valid
```

Sources and limits are in `src/khm_tess/export_gt.py` (`DEFAULT_SOURCES`). Override them with
`SOURCES="soyvitou-train mrrtmob:100000"`. Use `MODEL=` for a separate run such as a smoke test.

## Ship (Mac)

Copy the best `~/khm-tesstrain/data/khm_ft/tessdata_fast/<checkpoint>.traineddata` to the Mac,
run the gate described in [`results/finetune.md`](results/finetune.md), and replace
`apps/mac/Vendor/tesseract/tessdata/khm.traineddata` only on PASS.

## Tests

```bash
uv run --extra dev pytest -q
```
````

In the root `README.md`, add a row to the layout table after the TrOCR row:

```markdown
| [`training/tesseract`](training/tesseract) | Tesseract `khm` LSTM fine-tune (CPU, Docker) |
```

and add this bullet to the `## Training` list:

```markdown
- Tesseract: [`training/tesseract/README.md`](training/tesseract/README.md)
```

- [ ] **Step 4: Commit, then confirm Syncthing delivered the files to the server**

```bash
git add training/tesseract/Dockerfile training/tesseract/Makefile training/tesseract/README.md README.md
git commit -m "feat(tesseract): pinned tesstrain container and server Makefile"
ssh 192.168.1.25 'ls -a ~/Sync/workspace/khmerclip/.worktrees/tesseract-finetune/training/tesseract'
```

Expected: `.gitignore  Dockerfile  Makefile  README.md  pyproject.toml  src  tests`. If anything is missing, wait for Syncthing and re-check. Don't copy files into `~/Sync` by hand.

- [ ] **Step 5: Build and verify on the server**

```bash
ssh 192.168.1.25 'cd ~/Sync/workspace/khmerclip/.worktrees/tesseract-finetune/training/tesseract && mkdir -p ~/khm-tesstrain && make image check test'
```

Expected:
- the build log contains `/opt/tessdata_best/khm.traineddata: OK` (the SHA pin held)
- `check` prints `tesseract 5.3.x`
- `--list-langs` includes `eng` and `osd`
- the `lstm.train` config path is printed
- seven tool paths are printed
- `405346a3a67d8e4e049341d1da6a4b752e0b8351`
- the `3d2d975e…` hash
- the `combine_tessdata -d` component listing (note which of `lstm-word-dawg`, `lstm-punc-dawg` and `lstm-number-dawg` are present; `dicts` skips missing ones)
- `test` prints `12 passed`

If any tool or config is missing, change the Dockerfile base to `ubuntu:22.04`. Add `software-properties-common`, then `add-apt-repository -y ppa:alex-p/tesseract-ocr5`, before the `tesseract-ocr` install. Re-run, and record the base used in `results/finetune.md` (Task 5).

- [ ] **Step 6: Download the two uncached datasets**

```bash
ssh 192.168.1.25 'cd ~/Sync/workspace/khmerclip/.worktrees/tesseract-finetune/training/tesseract && make download && ls ~/.cache/huggingface/hub | grep -E "Darayut|KimkosalYon"'
```

Expected: both `datasets--Darayut--…` and `datasets--KimkosalYon--…` are listed (about 0.45 GB and 5.5 GB).

---

### Task 5: Smoke fine-tune on the server (rate + sanity)

No repo code changes. This task produces the iteration rate and the go/no-go for the full run.

**Interfaces:**
- Consumes: the Task 4 make targets
- Produces:
  - measured `RATE` (iterations/s)
  - start-model vs smoke-checkpoint BCER on `list.eval`
  - the full-run `mrrtmob` limit and `EPOCHS`
  
  All three are recorded in `training/tesseract/results/finetune.md`.

- [ ] **Step 1: Export a small ground-truth set**

```bash
ssh 192.168.1.25 'cd ~/Sync/workspace/khmerclip/.worktrees/tesseract-finetune/training/tesseract && make gt MODEL=khm_smoke SOURCES="soyvitou-train:3000 soyvitou-valid:300 darayut:1000 mrrtmob:2000"'
```

Expected: four JSON stat lines. `kept` should be close to each limit. `soyvitou-train` should have `with_space` > 0, and `mrrtmob` should have `with_latin` > 0.

- [ ] **Step 2: Build dicts, `.lstmf` files and lists**

```bash
ssh 192.168.1.25 'cd ~/Sync/workspace/khmerclip/.worktrees/tesseract-finetune/training/tesseract && make dicts lists MODEL=khm_smoke'
```

Expected:
- `wc -l` shows a non-empty wordlist
- tesstrain finishes without error
- last line `train=<≈6000> eval=<≈300>`

- [ ] **Step 3: Train in the foreground for 2000 iterations and time it**

```bash
ssh 192.168.1.25 'cd ~/Sync/workspace/khmerclip/.worktrees/tesseract-finetune/training/tesseract && time make train MODEL=khm_smoke DETACH= MAX_ITERATIONS=2000'
```

Expected:
- the log contains `Continuing from …khm_smoke.lstm`
- `At iteration …` lines appear
- the run ends with tesstrain's `--stop_training` step writing `khm_smoke.traineddata`

Compute `RATE = 2000 / real_seconds`. This slightly understates the rate, because setup time is included.

- [ ] **Step 4: Sanity-check that fine-tuning didn't damage the model**

```bash
ssh 192.168.1.25 'cd ~/Sync/workspace/khmerclip/.worktrees/tesseract-finetune/training/tesseract && make evaluate MODEL=khm_smoke'
```

Expected: a `START BCER eval=X` line, then up to five checkpoint lines sorted by BCER.

**Go rule:** the best smoke checkpoint BCER must be below the START BCER, because 2000 in-domain iterations should help on SoyVitou-valid. If it isn't, stop and use superpowers:systematic-debugging. Suspects, in order:
- unicharset merge or NORM_MODE (inspect `~/khm-tesstrain/data/khm_smoke/unicharset` and compare it with the start model's `combine_tessdata -u` output)
- `PSM=13` against the line images
- label/image mismatch (open a few `.png`/`.gt.txt` pairs)

- [ ] **Step 4b: Mac smoke gate (compatibility, latency, Khmer damage)**

The BCER check above can pass on Latin lines alone. It also says nothing about whether the model loads in the vendored 5.5.3 binary, or how fast an int-converted *best* network runs; tessdata_fast models are usually smaller networks. Check all three now, before spending days of CPU. This is a sanity check, not checkpoint selection.

Run from the worktree root on the Mac. Substitute `SMOKE_BEST` with the best smoke checkpoint name from Step 4.

```bash
mkdir -p training/tesseract/out
scp 192.168.1.25:khm-tesstrain/data/khm_smoke/tessdata_fast/SMOKE_BEST.traineddata training/tesseract/out/khm_smoke.traineddata
rm -rf training/tesseract/out/vendor-smoke
cp -R apps/mac/Vendor/tesseract training/tesseract/out/vendor-smoke
cp training/tesseract/out/khm_smoke.traineddata training/tesseract/out/vendor-smoke/tessdata/khm.traineddata
cd training/eval
uv run --no-project --with jiwer --with pillow --with-editable . python scripts/bench.py \
  --engine tesseract --engine tesseract-ft \
  --tesseract-ft-root ../tesseract/out/vendor-smoke --tesseract-ft-lang khm \
  --out ../tesseract/out/smoke_gate.csv
uv run --no-project python -c "import csv; r=[x for x in csv.DictReader(open('../tesseract/out/smoke_gate.csv')) if x['engine']=='tesseract-ft' and x['subset']=='printed']; print('printed', round(sum(float(x['cer']) for x in r)/len(r),4))"
cd ../..
```

Go only if all three hold. Otherwise stop and use superpowers:systematic-debugging.
- **Loads:** no Tesseract error, and `tesseract-ft` predictions are non-empty Khmer.
- **Latency:** `tesseract-ft` median_ms ≤ 2× the `tesseract` median_ms printed in the same run. If it fails, the full model will fail too. Record that the int-converted best network is too slow, and stop the plan.
- **No Khmer damage:** printed CER ≤ 0.42. That is stock `-l khm` at 0.3834 plus a 10% margin for a 2000-iteration model. Worse means the recoder, unicharset or NORM_MODE is wrong.

- [ ] **Step 5: Size the full run**

Fixed train lines at the defaults ≈ SoyVitou-train 50k + Sokheng 95k + hanuman 100k + KiteAether 7.4k + Darayut 100k + Siemreap 100k ≈ 452k. On top of that comes mrrtmob at 200k, for ≈ 652k in total. Run budget: 4 days = 345,600 s.
- `max_lines = RATE × 345600 / 3` (for `EPOCHS=3`)
- If `652000 ≤ max_lines`, keep the defaults.
- Otherwise set `MRRTMOB_LIMIT = max(50000, 200000 − (652000 − max_lines))`. If `452000 + MRRTMOB_LIMIT > max_lines` still holds, keep `MRRTMOB_LIMIT=50000` and set `EPOCHS = max(1, floor(RATE × 345600 / 502000))`.

- [ ] **Step 6: Record and commit**

Create `training/tesseract/results/finetune.md` with this content, filled from Task 4 Step 5 and Task 5 Steps 3–5:

```markdown
# Tesseract khm fine-tune

## Smoke (2026-10-03)

- Image: <ubuntu:24.04 or 22.04+PPA>, tesseract <version from check>, tesstrain 405346a
- Smoke set: soyvitou-train:3000, soyvitou-valid:300, darayut:1000, mrrtmob:2000
- Rate: <RATE> it/s at --cpus <CPUS>
- BCER on list.eval: start <X> → best smoke checkpoint <Y> (<checkpoint name>)
- Mac smoke gate (-l khm): loads <yes/no>; median <ms> vs Tess <ms>; printed CER <cer> (stock khm 0.3834)
- Full-run plan: mrrtmob:<MRRTMOB_LIMIT>, EPOCHS=<n>, est. <hours> h
```

```bash
git add training/tesseract/results/finetune.md
git commit -m "docs(tesseract): record smoke fine-tune rate and sanity check"
```

---

### Task 6: Full ground-truth export and fine-tune run (server)

No repo code changes. This task produces checkpoints under `~/khm-tesstrain/data/khm_ft/checkpoints/`.

**Interfaces:**
- Consumes: Task 5's `MRRTMOB_LIMIT` and `EPOCHS`
- Produces: `~/khm-tesstrain/data/khm_ft/{list.train,list.eval,training.log,checkpoints/}`

- [ ] **Step 1: Export the full ground truth**

If Task 5 kept the defaults, run it as written. Otherwise add `SOURCES="soyvitou-train soyvitou-valid:2000 sokheng hanuman kiteaether darayut siemreap:100000 mrrtmob:<MRRTMOB_LIMIT>"` to the `make gt` command.

```bash
ssh 192.168.1.25 'cd ~/Sync/workspace/khmerclip/.worktrees/tesseract-finetune/training/tesseract && nohup make gt > ~/khm-tesstrain/gt.log 2>&1 &'
```

Check progress with `ssh 192.168.1.25 tail -3 ~/khm-tesstrain/gt.log` until all eight sources are printed. `export_stats.json` is written last. Expected:
- `dropped_blocklist` is small (single-word UI strings can match real lines)
- `dropped_image` is < 1% per source

If any source drops > 5% of its rows, open some of its samples before continuing.

- [ ] **Step 2: Build `.lstmf` files and lists at full count**

```bash
ssh 192.168.1.25 'cd ~/Sync/workspace/khmerclip/.worktrees/tesseract-finetune/training/tesseract && make dicts && nohup make lists > ~/khm-tesstrain/lists.log 2>&1 &'
```

When `lists.log` ends with `train=… eval=…`, verify the counts:

```bash
ssh 192.168.1.25 'cd ~/khm-tesstrain && ls data/khm_ft-ground-truth | grep -c "\.gt\.txt$"; ls data/khm_ft-ground-truth | grep -c "\.lstmf$"; wc -l data/khm_ft/list.train data/khm_ft/list.eval; grep -ci error lists.log'
```

Expected:
- the `.lstmf` count is within 0.5% of the `.gt.txt` count (each missing one should appear as an error in `lists.log`)
- `list.eval` ≈ 2000
- `list.train` + `list.eval` = the `.lstmf` count

If the gap is larger than 0.5%, read `lists.log` before training.

- [ ] **Step 3: Launch the detached training**

```bash
ssh 192.168.1.25 'cd ~/Sync/workspace/khmerclip/.worktrees/tesseract-finetune/training/tesseract && make train EPOCHS=<n from Task 5>'
```

Add `CPUS=6` if the TrOCR run is active (`nvidia-smi` shows a python training process). Expected: a container ID is printed, and `docker ps` shows `khm_ft-train`.

- [ ] **Step 4: Monitor until done**

Check every few hours. Don't poll in a tight loop.

```bash
ssh 192.168.1.25 'grep -E "^At iteration|New best|Finished" ~/khm-tesstrain/data/khm_ft/training.log | tail -5; docker ps -a --filter name=khm_ft-train --format "{{.Status}}"'
```

Expected: the iteration count climbs, the BCER in `At iteration` lines trends down, and the run ends with `Finished!` and the container status `Exited (0)`.

If the box reboots or the container dies with a non-zero exit, re-run Step 3 with the same args. It resumes from `khm_ft_checkpoint`; the log shows `Successfully restored trainer from …`.

**Early stop:** if no `New best` line appears for 20% of the planned iterations, stop with `docker stop khm_ft-train` and go to Task 7.

---

### Task 7: Select checkpoint, Mac gate, record results

**Files:**
- Modify: `training/tesseract/results/finetune.md`

**Interfaces:**
- Consumes:
  - the Task 6 checkpoints
  - the Task 1 bench flags `--engine tesseract-ft --tesseract-ft-root --tesseract-ft-lang`
- Produces:
  - `training/tesseract/out/khm.traineddata` (gitignored)
  - its SHA-256
  - a gate verdict for `khm+eng` and for `khm`

- [ ] **Step 1: Evaluate every checkpoint on the server**

```bash
ssh 192.168.1.25 'cd ~/Sync/workspace/khmerclip/.worktrees/tesseract-finetune/training/tesseract && make evaluate'
```

Expected: a `START BCER eval=X` line, then the five best `…/eval/<name>.eval.log:BCER eval=…` lines. Record the lowest one's `<name>` as `BEST`.

- [ ] **Step 2: Copy the int model to the Mac and build a candidate vendor root**

Run from the worktree root on the Mac, with `BEST` substituted:

```bash
mkdir -p training/tesseract/out
scp 192.168.1.25:khm-tesstrain/data/khm_ft/tessdata_fast/BEST.traineddata training/tesseract/out/khm.traineddata
shasum -a 256 training/tesseract/out/khm.traineddata
rm -rf training/tesseract/out/vendor-ft
cp -R apps/mac/Vendor/tesseract training/tesseract/out/vendor-ft
cp training/tesseract/out/khm.traineddata training/tesseract/out/vendor-ft/tessdata/khm.traineddata
training/tesseract/out/vendor-ft/bin/tesseract training/eval/fixtures/printed/printed-002.png stdout \
  --tessdata-dir training/tesseract/out/vendor-ft/tessdata -l khm --psm 6
```

Expected: the SHA is printed, and the last command prints Khmer text with exit 0. That proves the 5.3-trained model loads in the vendored 5.5.3 binary.

- [ ] **Step 3: Run the gate for both language settings**

```bash
cd training/eval
for LANG_SET in khm+eng khm; do
  uv run --no-project --with jiwer --with pillow --with-editable . python scripts/bench.py \
    --engine tesseract --engine tesseract-ft \
    --tesseract-ft-root ../tesseract/out/vendor-ft --tesseract-ft-lang "$LANG_SET" \
    --candidate tesseract-ft --compare-to tesseract \
    --out "../tesseract/out/gate_${LANG_SET/+/_}.csv"
done
```

Expected: each run prints `tesseract:` and `tesseract-ft:` lines with `ui_cer`, then `PASS` or `FAIL` with reasons. The `tesseract` line must show cer 0.2580, mixed_cer 0.0894 and ui_cer 0.0199. If it doesn't, the vendored model changed, so stop.

- [ ] **Step 4: Get the per-subset breakdown**

```bash
uv run --no-project python - <<'EOF'
import csv, collections
for name in ("khm_eng", "khm"):
    rows = list(csv.DictReader(open(f"../tesseract/out/gate_{name}.csv", encoding="utf-8")))
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["engine"], r["subset"])].append(float(r["cer"]))
    for k in sorted(agg):
        print(name, k, round(sum(agg[k]) / len(agg[k]), 4))
EOF
```

- [ ] **Step 5: Decide**

- **Both settings PASS:** ship the one with lower `tesseract-ft` mean CER.
- **One PASSES:** ship that one.
- **Neither:** verdict FAIL; keep the current model.

Never ship on a FAIL. If the only failing reason is `ui_cer`, record it. The follow-up (not in this plan) is synthetic UI-style lines via `text2image`, per the spec.

- [ ] **Step 6: Record and commit**

Append to `training/tesseract/results/finetune.md`, filled from Steps 1–5 and Task 6:

```markdown
## Full run

- Sources: <the eight source specs used>; kept counts from export_stats.json: <alias=count, …>
- Lines: train <n>, eval <n>; EPOCHS <n>; iterations <n>; wall time <h>
- Checkpoint selection (lstmeval BCER on SoyVitou-valid): start <X>, best <BEST> <Y>
- Candidate file: `khm.traineddata` (tessdata_fast int), SHA-256 `<sha>`

## Mac gate (frozen 200 fixtures)

| Engine | -l | Mean CER | Printed | UI | Mixed | Median ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| tesseract (current) | khm+eng | 0.2580 | 0.5370 | 0.0199 | 0.0894 | <ms> |
| tesseract-ft | khm+eng | <cer> | <printed> | <ui> | <mixed> | <ms> |
| tesseract-ft | khm | <cer> | <printed> | <ui> | <mixed> | <ms> |

**Verdict:** <PASS (lang) | FAIL (reasons)>
```

```bash
git add training/tesseract/results/finetune.md
git commit -m "docs(tesseract): record fine-tune run and Mac gate verdict"
```

---

### Task 8: Ship into the app (only if Task 7 verdict is PASS)

Skip this task entirely on FAIL.

**Files:**
- Replace (gitignored): `apps/mac/Vendor/tesseract/tessdata/khm.traineddata`
- Modify: `apps/mac/README.md:44-46` (the Tesseract section)
- Possibly modify: `apps/mac/Tests/Fixtures/ocr/*.txt` (baselines, Step 3)
- Only if the winning setting is `khm`, also modify:
  - `apps/mac/Sources/OCR/TesseractEngine.swift` (doc comment and `"-l", "khm+eng"`)
  - `apps/mac/Tests/OCRTests.swift:10` (regen comment)
  - `training/eval/src/khmerclip_eval/runners/tesseract.py` (default `lang`)
  - `training/eval/tests/test_tesseract_runner.py` (default assertion)

**Interfaces:**
- Consumes: `training/tesseract/out/khm.traineddata` and the Task 7 verdict

- [ ] **Step 1: Back up and replace the model**

```bash
cp apps/mac/Vendor/tesseract/tessdata/khm.traineddata training/tesseract/out/khm.fast-orig.traineddata
cp training/tesseract/out/khm.traineddata apps/mac/Vendor/tesseract/tessdata/khm.traineddata
shasum -a 256 apps/mac/Vendor/tesseract/tessdata/khm.traineddata
```

Expected: the SHA equals the candidate SHA in `results/finetune.md`. Rollback is the reverse `cp` from `khm.fast-orig.traineddata`.

- [ ] **Step 2 (only if the winner is `khm`): switch the language**

In `TesseractEngine.swift`, change `"-l", "khm+eng",` to `"-l", "khm",`. Replace the two doc-comment lines that start "recognize Khmer script…" and "/// `khm+eng` so Khmer text mixed…" (through "…instead of just the Khmer runs.") with:

```swift
/// recognize Khmer script, which Vision cannot recognize at all. Runs `khm` alone: the bundled
/// model is fine-tuned on mixed Khmer + English lines (training/tesseract), and adding `eng`
/// let the English model hijack Khmer lines.
```

In `OCRTests.swift` line 10, change `-l khm+eng` to `-l khm`.

In `runners/tesseract.py`, change the default to `lang: str = "khm"`. In `test_tesseract_runner.py`, rename `test_recognize_defaults_to_khm_plus_eng` to `test_recognize_defaults_to_khm` and make it assert `== "khm"`.

Run: `cd training/eval && uv run --no-project --with jiwer --with pillow --with pytest --with-editable . pytest -q`
Expected: all pass.

- [ ] **Step 3: Run the app tests and review OCR baseline diffs**

Run: `cd apps/mac && make test`

`OCRTests.testFixturesRecognizeToExpectedText` compares against `.txt` baselines produced by the old model, so a new model may change them. For each mismatch, open the PNG and compare:
- If the new output is the correct reading, regenerate that `.txt` with the command in the `OCRTests.swift` header comment.
- If it is a worse reading, stop: don't ship. Restore with Step 1's rollback and add the regression to `results/finetune.md` as a FAIL reason.

Re-run `make test`. Expected: all tests pass.

- [ ] **Step 4: Update the docs**

Replace the `khm.traineddata` / `eng.traineddata` sentence at `apps/mac/README.md:46` with:

```markdown
`eng.traineddata` is from [tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast). `khm.traineddata` is fine-tuned from tessdata_best `khm` on Khmer + English line data (int-converted). Its SHA-256, the run details and the regeneration steps are in [`training/tesseract/results/finetune.md`](../../training/tesseract/results/finetune.md). Before 2026-10 the bundled `khm` was stock tessdata_fast (SHA-256 `47f110575341b322052f3becbefee61a3ecf1ef549352b5d4d33d28afe30d099`).
```

If the language switched, also change `-l khm+eng` on line 44 to `-l khm`, and replace the sentence "`khm+eng` (not `khm` alone) handles mixed-script lines." with "`khm` alone: the fine-tuned model reads mixed Khmer + English lines."

Append `**Shipped:** <date>, -l <lang>` to `results/finetune.md`.

- [ ] **Step 5: Commit**

```bash
git add apps/mac/README.md apps/mac/Tests/Fixtures training/tesseract/results/finetune.md
# only if the language switched:
# git add apps/mac/Sources/OCR/TesseractEngine.swift apps/mac/Tests/OCRTests.swift training/eval/src/khmerclip_eval/runners/tesseract.py training/eval/tests/test_tesseract_runner.py
git commit -m "feat(mac): ship fine-tuned Tesseract khm model"
```

- [ ] **Step 6: Finish the branch and install the model in the main checkout**

Use superpowers:finishing-a-development-branch to merge or PR `feat/tesseract-finetune`. `Vendor/` is gitignored, so the merge doesn't carry the model. After the merge, from the worktree root:

```bash
MAIN=/Users/kim/Sync/workspace/khmerclip
cp $MAIN/apps/mac/Vendor/tesseract/tessdata/khm.traineddata training/tesseract/out/khm.main-orig.traineddata
cp training/tesseract/out/khm.traineddata $MAIN/apps/mac/Vendor/tesseract/tessdata/khm.traineddata
shasum -a 256 $MAIN/apps/mac/Vendor/tesseract/tessdata/khm.traineddata
mkdir -p $MAIN/training/tesseract/out
cp training/tesseract/out/khm.traineddata training/tesseract/out/khm.fast-orig.traineddata $MAIN/training/tesseract/out/
```

Expected: the SHA matches `results/finetune.md`. Copying the model and its rollback file into main's gitignored `training/tesseract/out/` keeps them after the worktree is removed. Then run `cd $MAIN/apps/mac && make test` once more.
