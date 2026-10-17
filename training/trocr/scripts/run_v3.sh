#!/bin/sh
# v3 fine-tune (accuracy v3 plan, Task 4): v2 step 10k + screen strips + LazyGreed over a v2 replay mix.
# Waits for the screen-synth renders and any eval sweep holding the GPU.
#   nohup sh scripts/run_v3.sh > experiments/finetune_v3.log 2>&1 &
while pgrep -f "[r]ender_synth.py --screen" >/dev/null || pgrep -f "[s]weep.sh" >/dev/null; do sleep 60; done
[ "$(ls cache/synth_screen/train/*/*.parquet | wc -l)" -ge 24 ] || { echo "screen synth incomplete"; exit 1; }
exec python scripts/train.py --model_name experiments/v2_10k \
  --datasets synth_screen:150000 lazygreed:40000 synth_en:40000 synth:60000 handwriting:10000 darayut:20000 \
             siemreap:20000 soyvitou:30000 sokheng:20000 hanuman:20000 kiteaether:20000 mrrtmob:60000 \
  --augment --epochs 1 --lr 1e-5 --warmup_steps 500 \
  --max_eval_per_source 300 --max_eval_samples 0 --eval_steps 2000 --early_stopping_patience 4 --max_aspect 32
