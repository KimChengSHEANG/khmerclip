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
