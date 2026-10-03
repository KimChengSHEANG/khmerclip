# Tesseract khm fine-tune

## Smoke (2026-10-03)

- Image: ubuntu:24.04, tesseract 5.3.4, tesstrain 405346a
- Smoke set: soyvitou-train:3000, soyvitou-valid:300, darayut:1000, mrrtmob:2000
- Rate: 8.54 it/s at --cpus 8 (2000 iters in 234.1 s wall time)
- BCER on list.eval: start 49.851 → best smoke checkpoint 49.851 (khm_smoke.lstm, training-selected at iter ~100; go rule not met — not strictly below start). Final khm_smoke_checkpoint eval=98.782. make evaluate printed no tessdata checkpoint lines (only khm_smoke_checkpoint on disk; no *.checkpoint for tesstrain traineddata/evaluation).
- Mac smoke gate (-l khm): skipped (blocked at BCER go rule)
- Full-run plan (sized only; do not start until BCER go rule passes): mrrtmob:200000, EPOCHS=3, est. 64 h

## Notes

- list.train=5990, list.eval=300
- Training log: many Encoding of string failed / ZWSP (U+200B) in labels; train BCER rose to ~98.8% by iter 2000 while eval on start weights unchanged.
