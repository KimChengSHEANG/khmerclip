#!/usr/bin/env bash
# Render a synthetic set on this Mac (more cores than the 3090 box's CPU), then push it to the
# box's training/trocr/cache, where train.py's DatasetSpec globs pick it up. Training stays on
# the box: scripts/remote.sh excludes cache/, so renders never travel with the code.
#   scripts/render_push.sh synth_screen_v5 8 20000 -- --screen --screen_sizes 10 14
# = 8 parallel processes x 20000 rows into cache/synth_screen_v5/train/p0..p7, then rsync.
# Args after -- go to render_synth.py. SEED_BASE (default 1000) seeds process i with SEED_BASE + i.
# First run on a Mac downloads the text sources from Hugging Face (needs `hf auth login` for the
# gated mrrtmob set) and needs Homebrew fribidi for Pillow's raqm shaping (brew install fribidi).
set -euo pipefail
NAME=$1; PROCS=$2; COUNT=$3; shift 3
[ "${1:-}" = "--" ] && shift
HOST=${TROCR_HOST:-kim@192.168.1.25}
cd "$(dirname "$0")/.."  # -> training/trocr
export DYLD_LIBRARY_PATH=/opt/homebrew/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}
export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}
PY=${PY:-.venv/bin/python}
"$PY" -c 'from PIL import features; assert features.check("raqm"), "Pillow lacks raqm: brew install fribidi"'
pids=()
for ((i = 0; i < PROCS; i++)); do
  "$PY" scripts/render_synth.py --out "cache/$NAME/train/p$i" --count "$COUNT" --seed $((${SEED_BASE:-1000} + i)) "$@" \
    > "cache/$NAME.p$i.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || { echo "a render process failed; see cache/$NAME.p*.log"; exit 1; }; done
REMOTE=/home/kim/${PWD#"$HOME"/}
ssh "$HOST" "mkdir -p $REMOTE/cache"
rsync -a "cache/$NAME" "$HOST:$REMOTE/cache/"
echo "pushed cache/$NAME to $HOST:$REMOTE/cache/"
