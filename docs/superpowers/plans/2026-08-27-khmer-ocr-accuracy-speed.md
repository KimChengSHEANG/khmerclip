# Khmer OCR Accuracy + Speed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a frozen printed+UI eval harness that measures CER and latency against Tesseract, fine-tune TrOCR on a Linux RTX 3090 to beat that bar, export CoreML, and flip the Mac app default only if Mac CoreML pass criteria hold.

**Architecture:** New `training/eval/` package owns fixtures + `bench.py` runners (Tesseract, EasyOCR, TrOCR-HF, TrOCR-CoreML via a thin Swift CLI). Training stays in `training/trocr/` with multi-dataset runs + light UI-oriented augs. Shipping uses existing `apps/mac` `make trocr-khmer` + Preferences default gate.

**Tech Stack:** Python 3.10+, pytest, datasets/transformers/evaluate/jiwer, Pillow, vendored Tesseract, EasyOCR Vendor script, Swift/CoreML (`TrOCREngine`), Makefile export

**Spec:** `docs/superpowers/specs/2026-08-27-khmer-ocr-accuracy-speed-design.md`

## Global Constraints

- Beat Tesseract **CER** on frozen fixtures; Mac median latency **≤ 2×** Tesseract; mixed KM+EN subset CER not worse than Tesseract
- Train on **Linux RTX 3090** (CUDA); Mac = convert + CoreML latency/CER gate + ship
- Primary model: `microsoft/trocr-base-printed`; fallback `microsoft/trocr-small-printed` only if CoreML latency fails
- No train leakage into fixtures (held-out / soyvitou test + generated UI)
- EasyOCR is baseline only — not primary ship path
- Do not commit multi‑GB weights, venvs, or `raw_models/`
- On Mac CoreML **PASS**: default engine stays/set `.trocr`. On **FAIL**: set new-install default to `.tesseract`; keep neural selectable in Preferences; publish gate numbers

## File structure (create / modify

| Path | Responsibility |
|------|----------------|
| `training/eval/pyproject.toml` | Package `khmerclip-eval` + pytest |
| `training/eval/src/khmerclip_eval/cer.py` | CER helper (jiwer) |
| `training/eval/src/khmerclip_eval/fixtures.py` | Load `labels.jsonl` + images; fail closed |
| `training/eval/src/khmerclip_eval/summary.py` | Aggregate CER + median ms; pass/fail vs Tess |
| `training/eval/src/khmerclip_eval/runners/*.py` | Engine adapters |
| `training/eval/scripts/bench.py` | CLI: run engines → CSV + summary |
| `training/eval/scripts/build_printed_fixtures.py` | Sample soyvitou **test** → fixtures |
| `training/eval/scripts/build_ui_fixtures.py` | Render UI-style Khmer/EN lines → fixtures |
| `training/eval/scripts/write_manifest.py` | SHA256SUMS for fixtures |
| `training/eval/fixtures/` | PNGs + `labels.jsonl` + `SHA256SUMS` |
| `training/eval/tests/` | Unit tests for cer/fixtures/summary |
| `training/eval/README.md` | How to bench + interpret pass/fail |
| `apps/mac/Sources/Tools/OCRBenchMain.swift` | CLI: CoreML TrOCR (+ Tess) for Mac gate |
| `apps/mac/project.yml` | Add `OCRBench` command-line target |
| `training/trocr/src/trocr_khmer/augment.py` | Light UI/print augs |
| `training/trocr/src/trocr_khmer/dataset.py` | Optional aug hook |
| `training/trocr/scripts/train.py` | `--augment` flag |
| `training/trocr/README.md` | 3090 → Mac export flow |
| `apps/mac/README.md` | Default-engine policy after gate |
| `apps/mac/Sources/Preferences/AppPreferences.swift` | Default engine on fail path |

---

### Task 1: Eval package + CER helper (TDD)

**Files:**
- Create: `training/eval/pyproject.toml`
- Create: `training/eval/src/khmerclip_eval/__init__.py`
- Create: `training/eval/src/khmerclip_eval/cer.py`
- Create: `training/eval/tests/test_cer.py`

**Interfaces:**
- Consumes: nothing
- Produces: `def character_error_rate(prediction: str, reference: str) -> float`

- [ ] **Step 1: Write the failing test**

```python
# training/eval/tests/test_cer.py
from khmerclip_eval.cer import character_error_rate


def test_identical_is_zero():
    assert character_error_rate("សួស្តី", "សួស្តី") == 0.0


def test_total_mismatch_is_one():
    assert character_error_rate("abc", "xyz") == 1.0


def test_empty_reference_empty_pred():
    assert character_error_rate("", "") == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd training/eval && pip install -e ".[dev]" && pytest tests/test_cer.py -v`

Expected: FAIL (`ModuleNotFoundError` or missing `cer`)

- [ ] **Step 3: Write minimal implementation**

```toml
# training/eval/pyproject.toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "khmerclip-eval"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["jiwer>=3.0.0", "pillow>=10.0.0"]

[project.optional-dependencies]
dev = ["pytest>=7.0"]

[tool.setuptools.packages.find]
where = ["src"]
```

```python
# training/eval/src/khmerclip_eval/__init__.py
"""Frozen-fixture OCR bench for KhmerClip (CER + latency vs Tesseract)."""
```

```python
# training/eval/src/khmerclip_eval/cer.py
from jiwer import cer


def character_error_rate(prediction: str, reference: str) -> float:
    if reference == "" and prediction == "":
        return 0.0
    if reference == "":
        return 1.0
    return float(cer(reference, prediction))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd training/eval && pytest tests/test_cer.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/eval/pyproject.toml training/eval/src/khmerclip_eval/__init__.py   training/eval/src/khmerclip_eval/cer.py training/eval/tests/test_cer.py
git commit -m "$(cat <<EOF
feat(eval): add khmerclip-eval package with CER helper

EOF
)"
```

---

### Task 2: Fixture loader (TDD)

**Files:**
- Create: `training/eval/src/khmerclip_eval/fixtures.py`
- Create: `training/eval/tests/test_fixtures.py`

**Interfaces:**
- Consumes: nothing from Task 1
- Produces:
  - `@dataclass FixtureItem` with `id: str`, `image_path: Path`, `text: str`, `subset: str`
  - `def load_fixtures(root: Path) -> list[FixtureItem]`

`labels.jsonl` schema (one JSON object per line):

```json
{"id":"printed-001","image":"printed/printed-001.png","text":"សួស្តី","subset":"printed"}
{"id":"ui-001","image":"ui/ui-001.png","text":"ដក language ២០២៦","subset":"mixed"}
```

Allowed `subset` values: `printed`, `ui`, `mixed`.

- [ ] **Step 1: Write the failing test**

```python
# training/eval/tests/test_fixtures.py
import json
from pathlib import Path

import pytest
from PIL import Image

from khmerclip_eval.fixtures import load_fixtures


@pytest.fixture()
def fixture_root(tmp_path: Path) -> Path:
    (tmp_path / "printed").mkdir()
    img = tmp_path / "printed" / "a.png"
    Image.new("RGB", (8, 8), "white").save(img)
    (tmp_path / "labels.jsonl").write_text(
        json.dumps(
            {"id": "a", "image": "printed/a.png", "text": "ឆ", "subset": "printed"},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return tmp_path


def test_load_fixtures_ok(fixture_root: Path):
    items = load_fixtures(fixture_root)
    assert len(items) == 1
    assert items[0].text == "ឆ"
    assert items[0].image_path.exists()


def test_missing_image_raises(fixture_root: Path):
    (fixture_root / "labels.jsonl").write_text(
        '{"id":"x","image":"printed/missing.png","text":"a","subset":"printed"}\n',
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError):
        load_fixtures(fixture_root)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd training/eval && pytest tests/test_fixtures.py -v`

Expected: FAIL import `khmerclip_eval.fixtures`

- [ ] **Step 3: Write minimal implementation**

```python
# training/eval/src/khmerclip_eval/fixtures.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ALLOWED_SUBSETS = frozenset({"printed", "ui", "mixed"})


@dataclass(frozen=True)
class FixtureItem:
    id: str
    image_path: Path
    text: str
    subset: str


def load_fixtures(root: Path) -> list[FixtureItem]:
    root = Path(root)
    labels_path = root / "labels.jsonl"
    if not labels_path.is_file():
        raise FileNotFoundError(f"missing labels.jsonl under {root}")
    items: list[FixtureItem] = []
    with labels_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in ("id", "image", "text", "subset"):
                if key not in row:
                    raise ValueError(f"labels.jsonl:{line_no} missing {key}")
            if row["subset"] not in ALLOWED_SUBSETS:
                raise ValueError(f"labels.jsonl:{line_no} bad subset {row['subset']!r}")
            image_path = root / row["image"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            items.append(
                FixtureItem(
                    id=str(row["id"]),
                    image_path=image_path,
                    text=str(row["text"]),
                    subset=str(row["subset"]),
                )
            )
    if not items:
        raise ValueError(f"no fixtures in {labels_path}")
    return items
```

- [ ] **Step 4: Run tests**

Run: `cd training/eval && pytest tests/test_fixtures.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/eval/src/khmerclip_eval/fixtures.py training/eval/tests/test_fixtures.py
git commit -m "$(cat <<'EOF'
feat(eval): load frozen OCR fixtures from labels.jsonl

EOF
)"
```

---

### Task 3: Summary + pass/fail (TDD)

**Files:**
- Create: `training/eval/src/khmerclip_eval/summary.py`
- Create: `training/eval/tests/test_summary.py`

**Interfaces:**
- Consumes: row dicts with keys `engine`, `cer`, `ms`, `subset`
- Produces:
  - `def aggregate(rows: list[dict]) -> dict[str, dict]`
  - `def passes_gate(candidate: dict, tesseract: dict) -> tuple[bool, list[str]]`

Gate rules: `candidate["cer"] < tesseract["cer"]`; `candidate["median_ms"] <= 2 * tesseract["median_ms"]`; `candidate["mixed_cer"] <= tesseract["mixed_cer"]`.

- [ ] **Step 1: Write the failing test**

```python
# training/eval/tests/test_summary.py
from khmerclip_eval.summary import aggregate, passes_gate


def test_aggregate_median_and_cer():
    rows = [
        {"engine": "tesseract", "cer": 0.2, "ms": 10.0, "subset": "printed"},
        {"engine": "tesseract", "cer": 0.0, "ms": 30.0, "subset": "mixed"},
        {"engine": "trocr", "cer": 0.1, "ms": 20.0, "subset": "printed"},
        {"engine": "trocr", "cer": 0.0, "ms": 40.0, "subset": "mixed"},
    ]
    by = aggregate(rows)
    assert by["tesseract"]["n"] == 2
    assert by["tesseract"]["cer"] == 0.1
    assert by["tesseract"]["median_ms"] == 20.0
    assert by["tesseract"]["mixed_cer"] == 0.0


def test_passes_gate():
    tess = {"cer": 0.2, "median_ms": 50.0, "mixed_cer": 0.1}
    ok = {"cer": 0.1, "median_ms": 90.0, "mixed_cer": 0.1}
    bad_latency = {"cer": 0.1, "median_ms": 120.0, "mixed_cer": 0.1}
    assert passes_gate(ok, tess)[0] is True
    assert passes_gate(bad_latency, tess)[0] is False
```

- [ ] **Step 2: Run — expect FAIL**

Run: `cd training/eval && pytest tests/test_summary.py -v`

- [ ] **Step 3: Implement**

```python
# training/eval/src/khmerclip_eval/summary.py
from __future__ import annotations

import statistics
from collections import defaultdict


def aggregate(rows: list[dict]) -> dict[str, dict]:
    by_engine: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_engine[row["engine"]].append(row)
    out: dict[str, dict] = {}
    for engine, items in by_engine.items():
        cers = [float(r["cer"]) for r in items]
        mss = [float(r["ms"]) for r in items]
        mixed = [float(r["cer"]) for r in items if r.get("subset") == "mixed"]
        out[engine] = {
            "n": len(items),
            "cer": sum(cers) / len(cers),
            "median_ms": statistics.median(mss),
            "mixed_cer": (sum(mixed) / len(mixed)) if mixed else 0.0,
        }
    return out


def passes_gate(candidate: dict, tesseract: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not (candidate["cer"] < tesseract["cer"]):
        reasons.append(
            f"CER {candidate['cer']:.4f} not < Tess {tesseract['cer']:.4f}"
        )
    if not (candidate["median_ms"] <= 2.0 * tesseract["median_ms"]):
        reasons.append(
            f"median_ms {candidate['median_ms']:.1f} > 2× Tess {tesseract['median_ms']:.1f}"
        )
    if not (candidate["mixed_cer"] <= tesseract["mixed_cer"]):
        reasons.append(
            f"mixed_cer {candidate['mixed_cer']:.4f} > Tess {tesseract['mixed_cer']:.4f}"
        )
    return (len(reasons) == 0, reasons)
```

- [ ] **Step 4: pytest PASS**

Run: `cd training/eval && pytest tests/test_summary.py -v`

- [ ] **Step 5: Commit**

```bash
git add training/eval/src/khmerclip_eval/summary.py training/eval/tests/test_summary.py
git commit -m "$(cat <<'EOF'
feat(eval): aggregate CER/latency and encode ship pass gate

EOF
)"
```

---

### Task 4: Tesseract + EasyOCR runners + bench CLI

**Files:**
- Create: `training/eval/src/khmerclip_eval/runners/__init__.py`
- Create: `training/eval/src/khmerclip_eval/runners/tesseract.py`
- Create: `training/eval/src/khmerclip_eval/runners/easyocr.py`
- Create: `training/eval/scripts/bench.py`

**Interfaces:**
- `tesseract.recognize(image_path: Path, vendor_root: Path) -> str`
- `easyocr.recognize(image_path: Path, resource_root: Path, python_bin: Path) -> str`
- Bench engines: `tesseract`, `easyocr`, `trocr-hf`, `trocr-coreml`

Tesseract args must match app: `-l khm+eng`, `--psm 6`, `--tessdata-dir`.

- [ ] **Step 1: Add runners** (`tesseract.py` / `easyocr.py` subprocess wrappers matching app CLI)

Tesseract invocation:

```
Vendor/tesseract/bin/tesseract IMAGE stdout --tessdata-dir Vendor/tesseract/tessdata -l khm+eng --psm 6
```

EasyOCR invocation:

```
venv/bin/python3 recognize.py --resource-root ROOT --model khmer IMAGE
```

- [ ] **Step 2: Add `scripts/bench.py`**

CLI must:
1. `load_fixtures(--fixtures)`
2. For each `--engine` and item: time recognize; compute CER
3. Write CSV: `engine,id,subset,cer,ms,pred,ref`
4. Print `aggregate()` lines
5. If `--candidate` set: print PASS/FAIL from `passes_gate`

Flags: `--fixtures`, `--engine` (append), `--limit`, `--out`, `--tesseract-root`, `--easyocr-root`, `--easyocr-python`, `--trocr-model-dir`, `--trocr-coreml-bin`, `--compare-to`, `--candidate`

Defaults point at `apps/mac/Vendor/{tesseract,easyocr}`. Lazily import `trocr_hf` / `trocr_coreml` only when selected.

Include this complete `bench.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from khmerclip_eval.cer import character_error_rate
from khmerclip_eval.fixtures import load_fixtures
from khmerclip_eval.runners import easyocr as easyocr_runner
from khmerclip_eval.runners import tesseract as tess_runner
from khmerclip_eval.summary import aggregate, passes_gate

REPO = ROOT.parents[1]
DEFAULT_FIXTURES = ROOT / "fixtures"
DEFAULT_TESS = REPO / "apps/mac/Vendor/tesseract"
DEFAULT_EASYOCR = REPO / "apps/mac/Vendor/easyocr"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    p.add_argument("--engine", action="append", required=True,
                   choices=["tesseract", "easyocr", "trocr-hf", "trocr-coreml"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, default=Path("bench_results.csv"))
    p.add_argument("--tesseract-root", type=Path, default=DEFAULT_TESS)
    p.add_argument("--easyocr-root", type=Path, default=DEFAULT_EASYOCR)
    p.add_argument("--easyocr-python", type=Path,
                   default=DEFAULT_EASYOCR / "venv/bin/python3")
    p.add_argument("--trocr-model-dir", type=Path, default=None)
    p.add_argument("--trocr-coreml-bin", type=Path, default=None)
    p.add_argument("--compare-to", default="tesseract")
    p.add_argument("--candidate", default=None)
    args = p.parse_args()

    items = load_fixtures(args.fixtures)
    if args.limit is not None:
        items = items[: args.limit]

    rows: list[dict] = []
    for engine in args.engine:
        for item in items:
            t0 = time.perf_counter()
            if engine == "tesseract":
                pred = tess_runner.recognize(item.image_path, args.tesseract_root)
            elif engine == "easyocr":
                pred = easyocr_runner.recognize(
                    item.image_path, args.easyocr_root, args.easyocr_python)
            elif engine == "trocr-hf":
                from khmerclip_eval.runners import trocr_hf
                if not args.trocr_model_dir:
                    raise SystemExit("--trocr-model-dir required")
                pred = trocr_hf.recognize(item.image_path, args.trocr_model_dir)
            else:
                from khmerclip_eval.runners import trocr_coreml
                if not args.trocr_coreml_bin:
                    raise SystemExit("--trocr-coreml-bin required")
                pred = trocr_coreml.recognize(item.image_path, args.trocr_coreml_bin)
            ms = (time.perf_counter() - t0) * 1000.0
            rows.append({
                "engine": engine, "id": item.id, "subset": item.subset,
                "cer": character_error_rate(pred, item.text), "ms": ms,
                "pred": pred, "ref": item.text,
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["engine","id","subset","cer","ms","pred","ref"])
        w.writeheader(); w.writerows(rows)

    by = aggregate(rows)
    for engine, stats in by.items():
        print(f"{engine}: n={stats["n"]} cer={stats["cer"]:.4f} "
              f"median_ms={stats["median_ms"]:.1f} mixed_cer={stats["mixed_cer"]:.4f}")
    if args.candidate and args.compare_to in by and args.candidate in by:
        ok, reasons = passes_gate(by[args.candidate], by[args.compare_to])
        print("PASS" if ok else "FAIL")
        for r in reasons:
            print(f"  {r}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke Tesseract on a 1-image temp fixture; expect exit 0 and CSV**

- [ ] **Step 4: Commit runners + bench.py** with message `feat(eval): add Tesseract/EasyOCR runners and bench CLI`

---

### Task 5: Build + freeze fixtures (printed + UI)

**Files:**
- Create: `training/eval/scripts/build_printed_fixtures.py`
- Create: `training/eval/scripts/build_ui_fixtures.py`
- Create: `training/eval/scripts/write_manifest.py`
- Create: `training/eval/ui_strings.txt` (100 lines)
- Create: `training/eval/fixtures/` (generated)
- Create: `training/eval/README.md`
- Create: `training/eval/results/baseline.md`

**Rule:** ~100 printed from soyvitou **test** + ~100 UI renders = ~200 fixtures. No train-split leakage.

- [ ] **Step 1: Implement printed builder**

Load `SoyVitou/62k-images-khmer-printed-dataset` split=`test`. Take N=100 evenly spaced indices. Save `fixtures/printed/printed-XXX.png`. Write `labels.jsonl` (overwrite). Set `subset` to `mixed` if label has `[A-Za-z0-9]`, else `printed`.

- [ ] **Step 2: Implement UI builder**

`ui_strings.txt` = 100 Khmer/mixed lines. Render with Pillow + Noto Sans Khmer (`KHMER_UI_FONT` or `training/eval/fonts/NotoSansKhmer-Regular.ttf`). Alternate light/dark canvases. Append `ui/ui-XXX.png` rows to `labels.jsonl` with subset `ui` or `mixed`.

- [ ] **Step 3: Manifest**

`write_manifest.py` writes `fixtures/SHA256SUMS` (sha256 + relative path for every file except itself).

- [ ] **Step 4: Generate + baseline Tess**

```bash
cd training/eval
python scripts/build_printed_fixtures.py
python scripts/build_ui_fixtures.py
python scripts/write_manifest.py
wc -l fixtures/labels.jsonl
python scripts/bench.py --engine tesseract --out results/baseline_tesseract.csv
```

Paste aggregates into `results/baseline.md`. Document regenerate + bench in `README.md`.

- [ ] **Step 5: Commit** scripts, README, `ui_strings.txt`, SHA256SUMS, baseline.md; PNGs only if under ~50MB

Commit message: `feat(eval): freeze printed+UI OCR fixtures and Tesseract baseline`

---

### Task 6: TrOCR-HF runner

**Files:**
- Create: `training/eval/src/khmerclip_eval/runners/trocr_hf.py`
- Modify: `training/eval/README.md`

**Interfaces:**
- `recognize(image_path: Path, model_dir: Path) -> str`

Must match training infer: `num_beams=4`, `no_repeat_ngram_size=0`, `length_penalty=1.0`. Use `trocr_khmer.load_trocr_processor`, `configure_generation`, `VisionEncoderDecoderModel`. Cache model load with `lru_cache`.

- [ ] **Step 1: Add runner; README: install eval editable inside trocr CUDA venv**

- [ ] **Step 2: On 3090, bench checkpoint vs Tess on fixtures**

```bash
cd training/trocr
pip install -e ../eval
python ../eval/scripts/bench.py --engine tesseract --engine trocr-hf \
  --trocr-model-dir experiments/checkpoint-N/final \
  --out ../eval/results/current_trocr_hf.csv \
  --candidate trocr-hf --compare-to tesseract
```

- [ ] **Step 3: Commit** message `feat(eval): add Hugging Face TrOCR fixture runner`

---

### Task 7: Mac CoreML OCRBench CLI + runner

**Files:**
- Create: `apps/mac/Sources/Tools/OCRBenchMain.swift`
- Modify: `apps/mac/project.yml` (new `OCRBench` macOS CLI target sharing OCR sources)
- Create: `training/eval/src/khmerclip_eval/runners/trocr_coreml.py`

**Interfaces:**
- CLI: `OCRBench --engine trocr|tesseract <image.png>` → text on stdout; `ms=<float>` on stderr
- Env: `KHMERCLIP_VENDOR` → directory containing `tesseract/` and `trocr/` (default `Vendor`)
- Set `TesseractEngine.resourceRootOverride` / `TrOCREngine.resourceRootOverride` before recognize

Swift skeleton:

```swift
import AppKit
import Foundation

@main
struct OCRBenchMain {
    static func main() async {
        // parse --engine and image path
        let vendorEnv = ProcessInfo.processInfo.environment["KHMERCLIP_VENDOR"]
        let vendor = URL(fileURLWithPath: vendorEnv ?? "Vendor", isDirectory: true)
        TesseractEngine.resourceRootOverride = vendor.appendingPathComponent("tesseract", isDirectory: true)
        TrOCREngine.resourceRootOverride = vendor.appendingPathComponent("trocr", isDirectory: true)
        // load CGImage; time recognize; fputs ms= to stderr; print text
    }
}
```

Wire `project.yml` `OCRBench` tool target (OCR sources, no Sparkle). `xcodegen generate` then build.

Python runner: subprocess `[binary, "--engine", "trocr", image]` → stdout strip.

- [ ] **Step 1: Add Swift + project.yml; build OCRBench**
- [ ] **Step 2: Smoke Tess + TrOCR with `KHMERCLIP_VENDOR=.../apps/mac/Vendor`**
- [ ] **Step 3: Full Mac baseline bench with `--candidate trocr-coreml --compare-to tesseract`**
- [ ] **Step 4: Commit** message `feat(eval): add OCRBench CLI for Mac CoreML latency/CER gate`

---

### Task 8: Training augs + 3090 multi-dataset run

**Files:**
- Create: `training/trocr/src/trocr_khmer/augment.py`
- Create: `training/trocr/tests/test_augment.py`
- Modify: `training/trocr/src/trocr_khmer/dataset.py`
- Modify: `training/trocr/scripts/train.py`
- Modify: `training/trocr/README.md`

**Interfaces:**
- `def maybe_augment(image: Image.Image, rng: random.Random | None = None) -> Image.Image`
- `OCRDataset(..., augment: bool = False)`
- `train.py --augment`

Aug ops (each optional after 50% keep-identity): Gaussian blur, contrast jitter, JPEG recompress, small rotate (±3°).

Wire: in `__getitem__`, after RGB load, if `self.augment`: `image = maybe_augment(image)`.

- [ ] **Step 1: Failing test** `maybe_augment` keeps size/mode with fixed RNG
- [ ] **Step 2: Implement + wire; pytest**
- [ ] **Step 3: Smoke on 3090:** `python scripts/train.py --smoke_test --augment`
- [ ] **Step 4: Commit** message `feat(trocr): add light train-time image augmentations`
- [ ] **Step 5: Long train (operator; days/weeks)**

```bash
python scripts/train.py \
  --model_name microsoft/trocr-base-printed \
  --datasets soyvitou sokheng hanuman kiteaether \
  --augment --epochs 10
```

Periodic fixture CER via `bench.py` trocr-hf vs tesseract. Stop if val CER plateaus and Tess gap not closing. Add `mrrtmob` only with HF auth.

---

### Task 9: Export CoreML + Mac gate + default policy

**Files:**
- Modify: `apps/mac/Sources/Preferences/AppPreferences.swift` (only on FAIL path)
- Modify: `apps/mac/README.md`
- Modify: `training/eval/README.md`
- Create: `training/eval/results/mac_gate.md`

- [ ] **Step 1: Copy `final/` to Mac `apps/mac/raw_models/TrOCR-khmer/`**
- [ ] **Step 2: `make trocr-khmer KHMER_CKPT=raw_models/TrOCR-khmer && make build`**
- [ ] **Step 3: Mac gate bench** (`trocr-coreml` vs `tesseract`, write `mac_gate.csv`)
- [ ] **Step 4a PASS:** keep default `.trocr`; document PASS in `mac_gate.md`
- [ ] **Step 4b FAIL latency only:** train/export `microsoft/trocr-small-printed`; re-gate
- [ ] **Step 4c FAIL CER after fallback:** set default `?? .tesseract` in `AppPreferences`; ship neural as Preferences option; document FAIL
- [ ] **Step 5: Commit docs + default change (never commit `.mlpackage` weights)

Commit message: `docs(eval): record Mac CoreML OCR gate outcome and default policy`

---

## Spec coverage

| Spec requirement | Task |
|------------------|------|
| Eval harness CER + ms | 1–5 |
| Tess / TrOCR / EasyOCR runners | 4, 6, 7 |
| Frozen printed + UI fixtures | 5 |
| Pass gate | 3, 9 |
| 3090 train base + augs + multi-dataset | 8 |
| Small fallback | 9 |
| CoreML export + Mac re-bench | 9 |
| Default flip only on pass | 9 |

## Placeholder / consistency check

- No TBD left; Task 7 Vendor override uses `KHMERCLIP_VENDOR` + `resourceRootOverride`
- Names stable: `FixtureItem`, `load_fixtures`, `aggregate`, `passes_gate`, engines `tesseract` / `easyocr` / `trocr-hf` / `trocr-coreml`
