"""Experiment directory allocation for training runs.

Each new training session writes checkpoints and logs under
``experiments/model-{N}/`` where *N* increments automatically. Resuming with
``--saved_model experiments/model-2/best_norm_ED.pth`` continues in that same
directory instead of creating a new one.
"""
import os
import re

EXPERIMENTS_ROOT = "experiments"
_MODEL_DIR_RE = re.compile(r"^model-(\d+)$")


def next_experiment_index(root=EXPERIMENTS_ROOT):
    """Return the next unused model index under *root*."""
    if not os.path.isdir(root):
        return 1
    max_idx = 0
    for name in os.listdir(root):
        match = _MODEL_DIR_RE.match(name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1


def next_experiment_dir(root=EXPERIMENTS_ROOT):
    """Allocate a new experiment directory, e.g. ``experiments/model-3``."""
    idx = next_experiment_index(root)
    return os.path.join(root, f"model-{idx}")


def resolve_experiment_dir(saved_model=""):
    """Use the checkpoint's directory when resuming, else allocate a new run."""
    if saved_model:
        return os.path.dirname(os.path.abspath(saved_model))
    return next_experiment_dir()
