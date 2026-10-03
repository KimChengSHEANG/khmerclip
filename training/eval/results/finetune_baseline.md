# Fine-tune baseline (starting checkpoints)

Recorded 2026-10-03 on the 3090 box, `bench.py --engine trocr-hf` (beam 4), frozen 200 fixtures.

| Model | all | printed | ui | mixed |
| --- | ---: | ---: | ---: | ---: |
| `checkpoint-3/final` (= step 72k, best mid-train CER 0.0381) | 0.0542 | 0.0425 | 0.0586 | 0.0828 |
| `last-step` (step 3,513,080, mid-train CER ~0.38) | 0.3015 | 0.1400 | 0.5132 | 0.1162 |
| Tesseract (baseline.md) | 0.2580 | 0.5370 | 0.0199 | 0.0894 |

Caveat: the old `soyvitou` alias trained on `testset.parquet`, which is where the printed fixtures
come from, so printed CER for both rows is optimistic.

**Init for fine-tuning:** `checkpoint-3/final` (lower `all` CER).

## New-domain baseline (greedy, `evaluate_model.py`)

| Eval set | rows | init CER |
| --- | ---: | ---: |
| handwriting_test (full) | 307 | 0.3180 |
| synth_val (full) | 3000 | 0.0561 |
| darayut holdout | 1000 | 0.0797 |
| siemreap holdout | 1000 | 0.2785 |
| handwriting holdout | 117 | 0.3669 |
| soyvitou holdout* | 1000 | 0.0871 |
| sokheng holdout* | 1000 | 0.3067 |
| hanuman holdout* | 1000 | 0.1074 |
| kiteaether holdout* | 225 | 0.3405 |
| mrrtmob holdout* | 1000 | 0.0261 |

\* Per-source holdouts are new, so these rows were probably in the old run's training data; CER is optimistic.

## Source stats (`data_stats.py`, 2000-row sample)

```
soyvitou	n=56411	too_long=0.0%	aspect p50=3.7 p95=15.2	>16:4.3% >20:0.9% >24:0.1% >32:0.0%
sokheng	n=95000	too_long=0.0%	aspect p50=2.8 p95=6.3	>16:0.0% >20:0.0% >24:0.0% >32:0.0%
hanuman	n=100001	too_long=0.1%	aspect p50=11.3 p95=23.7	>16:23.4% >20:11.2% >24:4.7% >32:0.5%
kiteaether	n=7468	too_long=0.0%	aspect p50=3.2 p95=6.7	>16:0.0% >20:0.0% >24:0.0% >32:0.0%
mrrtmob	n=11324362	too_long=0.0%	aspect p50=16.0 p95=16.0	>16:0.0% >20:0.0% >24:0.0% >32:0.0%
handwriting	n=3895	too_long=0.0%	aspect p50=2.4 p95=5.2	>16:0.0% >20:0.0% >24:0.0% >32:0.0%
darayut	n=100000	too_long=0.5%	aspect p50=9.4 p95=17.3	>16:8.2% >20:2.1% >24:0.4% >32:0.0%
siemreap	n=200000	too_long=7.7%	aspect p50=10.0 p95=29.0	>16:25.2% >20:16.7% >24:10.7% >32:2.2%
```

**Chosen `--max_aspect`:** 24 — rule: smallest of 16/20/24/32 losing ≤10% of every replay source. At 20, hanuman still drops 11.2%; at 24 all five replay sources are ≤10% (hanuman 4.7%, soyvitou 0.1%). siemreap `too_long=7.7%` (>5%): those labels would truncate at train time and may contribute to siemreap CER drift.
