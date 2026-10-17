#!/usr/bin/env bash
# Run bench.py trocr-pipeline for each variant on the frozen and screen sets.
#   scripts/remote.sh 'bash ../eval/scripts/sweep.sh experiments/v2_10k v2 "st=split_touching=1" "st_w12=split_touching=1,split_wide_at=12"'
set -euo pipefail
MODEL=$1; TAG=$2; shift 2
for spec in "$@"; do
  name=${spec%%=*}; variant=${spec#*=}
  for f in ${FIXSETS:-fixtures fixtures_screen}; do
    out=../eval/results/${TAG}_${name}_${f}.csv
    [ -s "$out" ] && continue
    python ../eval/scripts/bench.py --engine trocr-pipeline --trocr-model-dir "$MODEL" --fixtures ../eval/$f \
      --variant "$variant" --out "$out" 2>&1 | grep '^trocr-pipeline' | sed "s/^/$name $f /"
  done
done
