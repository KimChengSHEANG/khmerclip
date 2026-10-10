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
