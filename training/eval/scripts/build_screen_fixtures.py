#!/usr/bin/env python3
"""Auto-labelled screen-capture fixtures: Khmer Wikipedia lines rendered by headless Chrome in
macOS / Khmer OS system fonts (never used in training), 1x and 2x, light and dark.

    # 1. candidate text (needs `datasets`; --dedup also needs trocr_khmer + training sources, i.e. the 3090 box)
    python scripts/build_screen_fixtures.py text --out screen_text.jsonl [--dedup]
    # 2. render on a Mac with Chrome and the fonts installed
    python scripts/build_screen_fixtures.py render --text screen_text.jsonl --out fixtures_screen

Subsets: `screen` = clips of 1-5 wrapped lines (20-90 chars), `screen_long` = one 60-120 char
line. Label = lines joined by one space (OCRBench and the pipeline runner space-join lines).
Output stays local: Wikipedia text is CC BY-SA and the fonts aren't redistributable.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
import time
from html import escape
from pathlib import Path

ZWSP = "​"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
FONTS = ["Khmer Sangam MN", "Khmer MN", "Khmer OS System", "Khmer OS Battambang", "Khmer OS"]
# Latin tokens spliced into ~10% of lines: product names, numbers, units seen on Khmer screens.
LATIN = ["Facebook", "Telegram", "iPhone 15", "USB-C", "COVID-19", "2024", "ABA Bank", "Wi-Fi",
         "PDF", "Google Maps", "10%", "$25", "Email", "YouTube", "Windows 11", "Bluetooth", "OK",
         "Download", "Settings", "5G", "TikTok", "08:30", "Zoom", "Excel"]


def tokens(text: str) -> list[tuple[str, str]]:
    """(separator before, word) pairs. Cut points are spaces and ZWSPs; a ZWSP joins invisibly."""
    out: list[tuple[str, str]] = []
    for chunk in text.split():
        for j, word in enumerate(p for p in chunk.split(ZWSP) if p):
            out.append((" " if j == 0 else "", word))
    return out


def wrap_lines(text: str, rng: random.Random, lo: int, hi: int) -> list[str]:
    """Greedy-pack words into lines of a random target width in [lo, hi]; drop lines outside it."""
    lines, cur, target = [], "", rng.randint(lo, hi)
    for sep, word in tokens(text):
        cand = cur + sep + word if cur else word
        if len(cand) <= target:
            cur = cand
            continue
        if lo <= len(cur) <= hi:
            lines.append(cur)
        cur, target = word, rng.randint(lo, hi)
    if lo <= len(cur) <= hi:
        lines.append(cur)
    return lines


def label(lines: list[str]) -> str:
    return " ".join(lines)


def splice_latin(line: str, rng: random.Random) -> str:
    words = line.split(" ")
    words.insert(rng.randint(0, len(words)), rng.choice(LATIN))
    return " ".join(words)


def is_khmer_para(p: str) -> bool:
    k = sum("ក" <= c <= "៿" for c in p)
    return len(p) > 150 and k > 0.7 * len(p.replace(" ", ""))


def shingles(s: str, n: int = 16) -> set[str]:
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}


def training_shingles(n: int = 16) -> set[str]:
    """16-char shingles of every training source's labels (mrrtmob: first 2M rows)."""
    from datasets import load_dataset
    from trocr_khmer.data_sources import resolve_dataset

    out: set[str] = set()
    for name in ["hanuman", "sokheng", "soyvitou", "kiteaether", "darayut", "siemreap", "handwriting", "mrrtmob"]:
        spec = resolve_dataset(name)
        ds = load_dataset(spec.path, split="train", **spec.load_kwargs()).select_columns([spec.text_col])
        if name == "mrrtmob":
            ds = ds.select(range(min(2_000_000, len(ds))))
        for t in ds[spec.text_col]:
            out |= shingles(t.replace(ZWSP, ""), n)
        print(f"  {name}: {len(out)} shingles", flush=True)
    return out


def make_text(out: Path, seed: int, n_screen: int, n_long: int, dedup: bool) -> None:
    from datasets import load_dataset

    rng = random.Random(seed)
    paras = [p.strip() for a in load_dataset("wikimedia/wikipedia", "20231101.km", split="train")
             for p in a["text"].split("\n") if is_khmer_para(p.strip())]
    rng.shuffle(paras)
    banned = training_shingles() if dedup else set()
    rows, dropped = [], 0
    for p in paras:
        want_long = sum(r["subset"] == "screen_long" for r in rows) < n_long
        want_screen = sum(r["subset"] == "screen" for r in rows) < n_screen
        if not (want_long or want_screen):
            break
        subset = "screen_long" if want_long and (not want_screen or rng.random() < 0.35) else "screen"
        lines = wrap_lines(p, rng, 60, 120) if subset == "screen_long" else wrap_lines(p, rng, 20, 90)
        if not lines:
            continue
        k = 1 if subset == "screen_long" else rng.randint(1, min(5, len(lines)))
        start = rng.randint(0, len(lines) - k)
        lines = [splice_latin(l, rng) if rng.random() < 0.1 else l for l in lines[start:start + k]]
        if banned and any(shingles(l) & banned for l in lines):
            dropped += 1
            continue
        rows.append({"subset": subset, "lines": lines})
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} clips to {out}; dropped {dropped} overlapping training text")


def chrome_png(html: str, png: Path, scale: int, width: int, height: int) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".html", encoding="utf-8", delete=False) as f:
        f.write(html)
    # Own profile dir: without it headless Chrome hands off to a running Chrome and never exits.
    with tempfile.TemporaryDirectory() as profile:
        cmd = [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
               f"--user-data-dir={profile}", f"--force-device-scale-factor={scale}",
               f"--window-size={width},{height}", f"--screenshot={png}", f"file://{f.name}"]
        for attempt in range(5):  # Chrome sometimes exits 2 when several start at once
            run = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if run.returncode == 0 and png.is_file():
                break
            time.sleep(1 + attempt)
        else:
            raise RuntimeError(f"Chrome failed rendering {png}: rc={run.returncode} {run.stderr[-500:]}")
    Path(f.name).unlink()


def page(lines: list[str], font: str, size: int, line_height: float, fg: str, bg: str, bold: bool) -> str:
    divs = "".join(f'<div style="white-space:nowrap">{escape(l)}</div>' for l in lines)
    return (f'<html><body style="margin:0;background:{bg}"><div style="display:inline-block;padding:40px;'
            f"font-family:'{font}',serif;font-size:{size}px;line-height:{line_height};color:{fg};"
            f'font-weight:{700 if bold else 400}">{divs}</div></body></html>')


def crop_to_ink(png: Path, bg: tuple[int, int, int], pad: int) -> None:
    import numpy as np
    from PIL import Image

    img = Image.open(png).convert("RGB")
    ink = (np.abs(np.asarray(img, dtype=np.int16) - np.array(bg)) > 40).any(axis=2)
    ys, xs = np.nonzero(ink)
    if xs.size == 0:
        raise RuntimeError(f"blank render {png}")
    if xs.max() >= img.width - 2 or ys.max() >= img.height - 2:
        raise RuntimeError(f"text clipped by window in {png}")
    img.crop((max(0, xs.min() - pad), max(0, ys.min() - pad),
              min(img.width, xs.max() + 1 + pad), min(img.height, ys.max() + 1 + pad))).save(png)


def check_fonts() -> None:
    """Chrome silently falls back for a missing font; refuse to label fallback renders as that font."""
    import hashlib

    def digest(font: str) -> str:
        with tempfile.TemporaryDirectory() as d:
            png = Path(d) / "f.png"
            chrome_png(page(["ភាសាខ្មែរ សួស្ដី"], font, 30, 1.5, "#000", "#fff", False), png, 1, 600, 200)
            return hashlib.sha256(png.read_bytes()).hexdigest()

    fallback = digest("No Such Font 123")
    missing = [f for f in FONTS if digest(f) == fallback]
    if missing:
        raise SystemExit(f"fonts not installed for Chrome: {missing}")


def render_one(out: Path, job: dict) -> None:
    html = page(job["lines"], job["font"], job["size"], job["line_height"], job["fg"], f"rgb{tuple(job['bg'])}",
                job["bold"])
    width = 120 + int(max(len(l) for l in job["lines"]) * job["size"] * 1.1)
    height = 120 + int(len(job["lines"]) * job["size"] * job["line_height"] * 1.3)
    png = out / job["image"]
    chrome_png(html, png, job["scale"], width, height)
    crop_to_ink(png, tuple(job["bg"]), job["pad"])


def render(text: Path, out: Path, seed: int, workers: int = 4) -> None:
    """Sample every clip's style up front (deterministic), then render in parallel."""
    from concurrent.futures import ThreadPoolExecutor

    rng = random.Random(seed)
    check_fonts()
    rows = [json.loads(l) for l in text.read_text(encoding="utf-8").splitlines() if l.strip()]
    jobs, counts = [], {"screen": 0, "screen_long": 0}
    for row in rows:
        subset = row["subset"]
        counts[subset] += 1
        fid = f"{subset.replace('_', '-')}-{counts[subset]:03d}"
        font, size, scale = rng.choice(FONTS), rng.randint(12, 20), rng.choice([1, 2])
        dark = rng.random() < 0.3
        g_bg, g_fg = (rng.randint(18, 40), rng.randint(200, 240)) if dark else (rng.randint(238, 255), rng.randint(0, 60))
        jobs.append({"id": fid, "image": f"{subset}/{fid}.png", "subset": subset, "lines": row["lines"],
                     "font": font, "size": size, "scale": scale, "dark": dark, "bg": [g_bg] * 3,
                     "fg": f"rgb({g_fg},{g_fg},{g_fg})", "line_height": rng.choice([1.4, 1.6, 1.8]),
                     "bold": rng.random() < 0.15, "pad": rng.randint(3, 16) * scale})
        (out / subset).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as pool:
        for job, _ in zip(jobs, pool.map(lambda j: render_one(out, j), jobs)):
            print(job["id"], job["font"], job["size"], f"{job['scale']}x", flush=True)
    keys = ("id", "image", "subset", "font", "size", "scale", "dark", "bold")
    (out / "labels.jsonl").write_text(
        "".join(json.dumps({**{k: j[k] for k in keys}, "text": label(j["lines"]), "lines": len(j["lines"])},
                           ensure_ascii=False) + "\n" for j in jobs), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=7)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("text")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--screen", type=int, default=300)
    t.add_argument("--long", type=int, default=150)
    t.add_argument("--dedup", action="store_true")
    r = sub.add_parser("render")
    r.add_argument("--text", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.cmd == "text":
        make_text(args.out, args.seed, args.screen, args.long, args.dedup)
    else:
        render(args.text, args.out, args.seed)


if __name__ == "__main__":
    main()
