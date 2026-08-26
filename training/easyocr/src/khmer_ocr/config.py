"""Shared helpers: config loading and charset (de)serialisation."""
import argparse

import yaml


def load_charset(path):
    """Read a charset file (one character per line) into a single string.

    Storing one char per line lets us represent the space character (which is a
    real, frequent character in Khmer sentences) without it being stripped.
    """
    chars = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # Drop only the trailing newline, keep everything else (incl. spaces).
            if line.endswith("\n"):
                line = line[:-1]
            if line == "":
                continue
            chars.append(line)
    return "".join(chars)


def save_charset(chars, path):
    """Write a set/iterable of characters, one per line, sorted by codepoint."""
    with open(path, "w", encoding="utf-8") as f:
        for c in sorted(set(chars)):
            if c == "\n":
                continue
            f.write(c + "\n")


def normalize_label(text):
    """Collapse runs of whitespace/newlines into single spaces."""
    return " ".join(text.split())


def label_in_charset(text, charset):
    """Return True when every character in ``text`` is in ``charset``."""
    allowed = set(charset)
    return all(c in allowed for c in text)


def filter_label_to_charset(text, charset):
    """Keep only characters present in ``charset``."""
    allowed = set(charset)
    return "".join(c for c in text if c in allowed)


def load_config(yaml_path, overrides=None):
    """Load a YAML config into an argparse.Namespace, applying CLI overrides."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return argparse.Namespace(**cfg)
