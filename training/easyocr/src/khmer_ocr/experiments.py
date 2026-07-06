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


def _is_under_root(path, root):
    """True if *path* is *root* or a subdirectory of *root*."""
    path_abs = os.path.abspath(path)
    root_abs = os.path.abspath(root)
    try:
        return os.path.commonpath([path_abs, root_abs]) == root_abs
    except ValueError:
        return False


def resolve_experiment_dir(saved_model="", root=EXPERIMENTS_ROOT):
    """Use the checkpoint's directory when resuming under *root*, else allocate."""
    if saved_model:
        checkpoint_dir = os.path.dirname(os.path.abspath(saved_model))
        if _is_under_root(checkpoint_dir, root):
            return checkpoint_dir
    return next_experiment_dir(root)
