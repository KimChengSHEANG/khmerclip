"""Replace tesstrain's random 90/10 split: SoyVitou-valid lines are the eval list, all else trains.

Run after ``make lists``; tesstrain won't regenerate list.* because they end up newer than all-lstmf.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def split(lstmf_paths: list[str], eval_prefix: str = "soyvitou-valid-") -> tuple[list[str], list[str]]:
    train = [p for p in lstmf_paths if not Path(p).name.startswith(eval_prefix)]
    eval_ = [p for p in lstmf_paths if Path(p).name.startswith(eval_prefix)]
    if not eval_:
        raise ValueError(f"no eval files starting with {eval_prefix!r}")
    if not train:
        raise ValueError("no train files")
    return train, eval_


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output_dir", type=Path)
    out = p.parse_args().output_dir
    paths = [line for line in (out / "all-lstmf").read_text().splitlines() if line.strip()]
    train, eval_ = split(paths)
    (out / "list.train").write_text("\n".join(train) + "\n")
    (out / "list.eval").write_text("\n".join(eval_) + "\n")
    print(f"train={len(train)} eval={len(eval_)}")


if __name__ == "__main__":
    main()
