**Result: PASS**

Fine-tune v1 vs init (`experiments/checkpoint-3/final`). Evaluated 2026-10-03 on the 3090 box.

## Run metadata

**Train command** (Task 7, tmux `finetune`):

```bash
python scripts/train.py \
  --model_name experiments/checkpoint-3/final \
  --datasets handwriting:60000 synth:400000 darayut:100000 siemreap:100000 soyvitou:60000 sokheng:80000 hanuman:80000 kiteaether:60000 mrrtmob:200000 \
  --augment --epochs 3 --lr 2e-5 --warmup_steps 1000 \
  --max_eval_per_source 300 --max_eval_samples 0 --eval_steps 2000 \
  --early_stopping_patience 5 --max_aspect 24
```

| Field | Value |
| --- | --- |
| Run dir | `experiments/checkpoint-4` |
| Checkpoint for Mac handoff | `experiments/checkpoint-4/final` |
| Stopping step | 40 000 (early stop; best weights at step 30 000) |
| Best mid-train CER | 0.02283142167629274 |

## Fixtures (`bench.py` trocr-hf, beam 4, n=200)

| Subset | init | finetune_v1 | Δ |
| --- | ---: | ---: | ---: |
| all | 0.0542 | 0.0064 | −0.0478 |
| printed | 0.0425 | 0.0064 | −0.0361 |
| ui | 0.0586 | 0.0021 | −0.0565 |
| mixed | 0.0828 | 0.0226 | −0.0602 |

Printed/mixed deltas are informational only (init trained on soyvitou testset overlap). Ship bar for those subsets is Tesseract comparison in `mac_gate.md`.

## Eval sets (`evaluate_model.py`, greedy/beam-4 per trainer defaults)

| Eval set | rows | init | finetune_v1 | Δ |
| --- | ---: | ---: | ---: | ---: |
| handwriting_test (full) | 307 | 0.3180 | 0.0142 | −0.3038 |
| synth_val (full) | 3000 | 0.0561 | 0.0082 | −0.0479 |
| darayut holdout | 1000 | 0.0797 | 0.0089 | −0.0708 |
| siemreap holdout | 1000 | 0.2785 | 0.0756 | −0.2029 |
| handwriting holdout | 117 | 0.3669 | 0.0289 | −0.3380 |
| soyvitou holdout* | 1000 | 0.0871 | 0.0182 | −0.0689 |
| sokheng holdout* | 1000 | 0.3067 | 0.0429 | −0.2638 |
| hanuman holdout* | 1000 | 0.1074 | 0.0146 | −0.0928 |
| kiteaether holdout* | 225 | 0.3405 | 0.0551 | −0.2854 |
| mrrtmob holdout* | 1000 | 0.0261 | 0.0144 | −0.0117 |

\* Replay holdouts; init CER optimistic if rows were in the old training mix.

## Gate criteria

1. **handwriting_test & synth_val below init** — PASS (0.0142 & 0.0082 vs 0.3180 & 0.0561).
2. **ui ≤ init + 0.005; darayut & siemreap ≤ init** — PASS (ui 0.0021 ≤ 0.0636; darayut 0.0089; siemreap 0.0756).
3. **Replay holdouts ≤ init + 0.01** — PASS (soyvitou 0.0182 ≤ 0.0971; sokheng 0.0429 ≤ 0.3167; hanuman 0.0146 ≤ 0.1174; kiteaether 0.0551 ≤ 0.3505; mrrtmob 0.0144 ≤ 0.0361).

## Mac CoreML (Task 9, 2026-10-04)

On-device gate (`bench.py` vs Tesseract, n=200, checkpoint-4 CoreML, sha256 `004a0e4bc6f687ac…`):

| Engine | Mean CER | Median ms | Mixed CER |
| --- | ---: | ---: | ---: |
| tesseract | 0.2580 | 110.0 | 0.0894 |
| trocr-coreml | 0.1700 | 1441.4 | 0.0226 |

**Mac gate:** FAIL (latency only — median > 2× Tesseract); accuracy passes. Default engine policy: Step 4b in `mac_gate.md` (keep TrOCR default).

**Manual app check:** KhmerClipper launched with new build; automated smoke only — operator should confirm TrOCR clips on UI/PDF/handwriting match on-screen text.

