#!/usr/bin/env python3
"""Auto-labelled screen-capture fixtures: Khmer Wikipedia lines rendered by headless Chrome in
macOS / Khmer OS system fonts (never used in training), 1x and 2x, light and dark.

    # 1. candidate text (needs `datasets`; --dedup also needs trocr_khmer + training sources, i.e. the 3090 box)
    python scripts/build_screen_fixtures.py text --out screen_text.jsonl [--dedup]
    # 2. render on a Mac with Chrome and the fonts installed
    python scripts/build_screen_fixtures.py render --text screen_text.jsonl --out fixtures_screen
    # 3. style slices (single lines: small, bold, low contrast, coloured, display font, numbers, mixed)
    python scripts/build_screen_fixtures.py slices --text screen_text.jsonl --out fixtures_slices

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
# Task 3b slices: single lines in the styles a plain 12-20 px render doesn't cover.
SLICES = ["plain", "small", "bold", "lowcontrast", "colored", "display", "numbers", "mixed"]
DISPLAY_FONTS = ["Khmer OS Muol Light", "Khmer OS Muol Pali", "Khmer OS Bokor", "Khmer OS Metal Chrieng"]
NUMBERS = ["១៥/០៨/២០២៤", "$12.50", "012 345 678", "២៥%", "10:45", "៛50,000", "2024-03-01", "៣០០០", "+855 12 345",
           "1,250", "ឆ្នាំ២០២៣", "No. 47", "៧៨៩", "3.14"]


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


def check_fonts(fonts: list[str] = FONTS) -> None:
    """Chrome silently falls back for a missing font; refuse to label fallback renders as that font."""
    import hashlib

    def digest(font: str) -> str:
        with tempfile.TemporaryDirectory() as d:
            png = Path(d) / "f.png"
            chrome_png(page(["ភាសាខ្មែរ សួស្ដី"], font, 30, 1.5, "#000", "#fff", False), png, 1, 600, 200)
            return hashlib.sha256(png.read_bytes()).hexdigest()

    fallback = digest("No Such Font 123")
    missing = [f for f in fonts if digest(f) == fallback]
    if missing:
        raise SystemExit(f"fonts not installed for Chrome: {missing}")


def insert_tokens(line: str, tokens: list[str], rng: random.Random, k: int) -> str:
    words = line.split(" ")
    for t in rng.sample(tokens, k):
        words.insert(rng.randint(0, len(words)), t)
    return " ".join(words)


def slice_jobs(pool: list[str], rng: random.Random, per_slice: int = 40) -> list[dict]:
    """Single-line clips, `per_slice` per style in SLICES; subset = "slice_<style>"."""
    pool = [l for l in pool if 20 <= len(l) <= 70]
    jobs = []
    for name in SLICES:
        for k in range(1, per_slice + 1):
            line = rng.choice(pool)
            font, size, scale = rng.choice(FONTS), rng.randint(12, 20), rng.choice([1, 2])
            g_bg, g_fg = rng.randint(238, 255), rng.randint(0, 60)
            bg, fg, bold = [g_bg] * 3, f"rgb({g_fg},{g_fg},{g_fg})", False
            if name == "small":
                size, scale = rng.randint(10, 13), 1
            elif name == "bold":
                bold = True
            elif name == "lowcontrast":
                bg = [rng.randint(200, 235), rng.randint(215, 245), rng.randint(185, 240)]
                g = rng.randint(95, 130)
                fg = f"rgb({g},{g},{g + rng.randint(0, 20)})"
            elif name == "colored":
                if rng.random() < 0.5:
                    fg = rng.choice(["rgb(26,115,232)", "rgb(217,48,37)", "rgb(24,128,56)", "rgb(142,36,170)"])
                else:
                    g = rng.randint(20, 40)
                    bg, fg = [g, g, g + 6], rng.choice(["rgb(138,180,248)", "rgb(242,139,130)", "rgb(129,201,149)"])
            elif name == "display":
                font, size = rng.choice(DISPLAY_FONTS), rng.randint(16, 24)
                line = "".join(c for c in line if not ("A" <= c <= "z")).strip() or line
            elif name == "numbers":
                line = insert_tokens(line, NUMBERS, rng, 2)
            elif name == "mixed":
                line = insert_tokens(line, LATIN, rng, 2)
            fid = f"slice-{name}-{k:03d}"
            jobs.append({"id": fid, "image": f"slices/{fid}.png", "subset": f"slice_{name}", "lines": [line],
                         "font": font, "size": size, "scale": scale, "dark": bg[0] < 128, "bg": bg, "fg": fg,
                         "line_height": 1.6, "bold": bold, "pad": rng.randint(3, 16) * scale})
    return jobs


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
    render_jobs(jobs, out, workers)


def render_jobs(jobs: list[dict], out: Path, workers: int = 4) -> None:
    from concurrent.futures import ThreadPoolExecutor

    for j in jobs:
        (out / j["image"]).parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as pool:
        for job, _ in zip(jobs, pool.map(lambda j: render_one(out, j), jobs)):
            print(job["id"], job["font"], job["size"], f"{job['scale']}x", flush=True)
    keys = ("id", "image", "subset", "font", "size", "scale", "dark", "bold")
    (out / "labels.jsonl").write_text(
        "".join(json.dumps({**{k: j[k] for k in keys}, "text": label(j["lines"]), "lines": len(j["lines"])},
                           ensure_ascii=False) + "\n" for j in jobs), encoding="utf-8")


def render_slices(text: Path, out: Path, seed: int, per_slice: int) -> None:
    check_fonts(FONTS + DISPLAY_FONTS)
    pool = [l for row in text.read_text(encoding="utf-8").splitlines() if row for l in json.loads(row)["lines"]]
    render_jobs(slice_jobs(pool, random.Random(seed), per_slice), out)


# Noisy clips (noise-robust preprocessing spec, 2026-10-23): screen text with the clutter a real
# selection picks up. One noise kind per clip, subset "noisy_<kind>"; the label is the text only.
NOISE_KINDS = ["icons", "chrome", "underline", "background", "edgecut", "avatar"]
ICONS = ["●", "✔", "★", "📌", "⚙️", "🔔", "✉️", "▶",
         '<svg width="1em" height="1em" viewBox="0 0 10 10" style="vertical-align:middle">'
         '<circle cx="5" cy="5" r="4" fill="#1a73e8"/></svg>']
SENTINEL = (255, 0, 255)  # page colour outside the clip; the render is cropped to everything else


def noisy_jobs(pool: list[str], rng: random.Random, per_kind: int = 50) -> list[dict]:
    pool = [l for l in pool if l.strip()]
    jobs = []
    for kind in NOISE_KINDS:
        for k in range(1, per_kind + 1):
            lines = [rng.choice(pool) for _ in range(1 if kind == "edgecut" else rng.randint(1, 3))]
            if kind in ("background", "chrome") and rng.random() < 0.15:
                lines = [rng.choice(LATIN)]  # Latin-only: exercises the Tesseract fallback on device
            font, size, scale = rng.choice(FONTS), rng.randint(12, 20), rng.choice([1, 2])
            dark = rng.random() < 0.3
            g_bg, g_fg = (rng.randint(18, 40), rng.randint(200, 240)) if dark else (rng.randint(238, 255), rng.randint(0, 60))
            noise: dict = {}
            if kind == "icons":
                noise = {"icons": [[rng.choice(ICONS), rng.choice(["start", "end"])] for _ in range(rng.randint(1, 3))]}
            elif kind == "chrome":
                noise = {"style": rng.choice(["button", "input", "divider_above", "divider_below"]),
                         "width": rng.choice([1, 2])}
            elif kind == "underline":
                noise = {"style": rng.choice(["link", "table"])}
            elif kind == "background":
                c = [[rng.randint(0, 255) for _ in range(3)] for _ in range(2)]
                noise = {"style": rng.choice(["linear", "radial", "texture"]), "colors": c,
                         "angle": rng.randint(0, 359), "seed": rng.randrange(10**9), "low": rng.random() < 0.3}
            elif kind == "edgecut":
                top, bottom = rng.choice([(1, 0), (0, 1), (1, 1)])
                noise = {"above": rng.choice(pool), "below": rng.choice(pool),
                         "keep_top": round(rng.uniform(0.2, 0.6), 2) if top else 0.0,
                         "keep_bottom": round(rng.uniform(0.2, 0.6), 2) if bottom else 0.0}
            elif kind == "avatar":
                noise = {"size": rng.randint(32, 48), "round": rng.random() < 0.7,
                         "color": [rng.randint(40, 220) for _ in range(3)]}
            fid = f"noisy-{kind}-{k:03d}"
            jobs.append({"id": fid, "image": f"noisy_{kind}/{fid}.png", "subset": f"noisy_{kind}", "kind": kind,
                         "lines": lines, "font": font, "size": size, "scale": scale, "dark": dark,
                         "bg": [g_bg] * 3, "fg": f"rgb({g_fg},{g_fg},{g_fg})",
                         "line_height": rng.choice([1.4, 1.6, 1.8]), "bold": rng.random() < 0.15,
                         "pad": rng.randint(3, 16), "noise": noise})
    return jobs


def texture_uri(seed: int, colors: list[list[int]]) -> str:
    """A generated photo-like texture (blurred noise between two colours), never a real photo."""
    import base64
    import io

    import numpy as np
    from PIL import Image, ImageFilter

    rng = np.random.default_rng(seed)
    t = Image.fromarray((rng.random((96, 96)) * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(6))
    a = np.asarray(t, dtype=float)[..., None] / 255
    a = (a - a.min()) / max(1e-6, a.max() - a.min())
    rgb = (np.array(colors[0]) * (1 - a) + np.array(colors[1]) * a).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(rgb).resize((384, 384), Image.BICUBIC).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def noisy_page(job: dict) -> str:
    n, kind = job["noise"], job["kind"]
    fg, bg = job["fg"], f"rgb{tuple(job['bg'])}"
    lines = [escape(l) for l in job["lines"]]
    if kind == "icons":
        first = lines[0]
        for icon, where in n["icons"]:
            first = f"{icon} {first}" if where == "start" else f"{first} {icon}"
        lines[0] = first
    box = f"padding:{job['pad']}px;background:{bg};"
    if kind == "background":
        a, b = (f"rgb{tuple(c)}" for c in n["colors"])
        if n["style"] == "linear":
            box = f"padding:{job['pad']}px;background:linear-gradient({n['angle']}deg,{a},{b});"
        elif n["style"] == "radial":
            box = f"padding:{job['pad']}px;background:radial-gradient(circle,{a},{b});"
        else:
            box = f"padding:{job['pad']}px;background:url({texture_uri(n['seed'], n['colors'])});background-size:cover;"
        lum = sum(sum(c) for c in n["colors"]) / 6
        fg = ("rgb(150,150,150)" if n["low"] else "rgb(255,255,255)") if lum < 128 else \
             ("rgb(110,110,110)" if n["low"] else "rgb(0,0,0)")
    text_style = (f"font-family:'{job['font']}',serif;font-size:{job['size']}px;line-height:{job['line_height']};"
                  f"color:{fg};font-weight:{700 if job['bold'] else 400};white-space:nowrap")
    if kind == "underline" and n["style"] == "link":
        body = "".join(f'<div><span style="text-decoration:underline;color:rgb(26,115,232)">{l}</span></div>'
                       for l in lines)
    elif kind == "underline":
        body = ('<table style="border-collapse:collapse">' + "".join(
            f'<tr><td style="border:1px solid rgb(150,150,150);padding:2px 6px">{l}</td></tr>' for l in lines)
            + "</table>")
    elif kind == "edgecut":
        body = "".join(f"<div>{escape(t)}</div>" for t in (n["above"], job["lines"][0], n["below"]))
        box = f"padding:0 {job['pad']}px;background:{bg};"
    else:
        body = "".join(f"<div>{l}</div>" for l in lines)
    if kind == "chrome":
        w, edge = n["width"], "rgb(140,140,140)"
        if n["style"] == "button":
            body = f'<div style="border:{w}px solid {edge};border-radius:6px;padding:4px 12px">{body}</div>'
        elif n["style"] == "input":
            body = f'<div style="border:{w}px solid {edge};padding:3px 6px">{body}</div>'
        elif n["style"] == "divider_above":
            body = f'<div style="border-top:{w}px solid {edge};padding-top:6px">{body}</div>'
        else:
            body = f'<div style="border-bottom:{w}px solid {edge};padding-bottom:6px">{body}</div>'
    if kind == "avatar":
        c = f"rgb{tuple(n['color'])}"
        radius = "50%" if n["round"] else "4px"
        body = (f'<div style="display:flex;align-items:center;gap:10px"><div style="flex:none;width:{n["size"]}px;'
                f'height:{n["size"]}px;border-radius:{radius};background:{c}"></div><div>{body}</div></div>')
    return (f'<!doctype html><html><body style="margin:20px;background:rgb{SENTINEL}"><div style="display:inline-block;{box}">'
            f'<div style="{text_style}">{body}</div></div></body></html>')


def crop_to_clip(png: Path) -> None:
    import numpy as np
    from PIL import Image

    img = Image.open(png).convert("RGB")
    clip = (np.abs(np.asarray(img, dtype=np.int16) - np.array(SENTINEL)) > 12).any(axis=2)
    ys, xs = np.nonzero(clip)
    if xs.size == 0:
        raise RuntimeError(f"blank render {png}")
    if xs.max() >= img.width - 2 or ys.max() >= img.height - 2:
        raise RuntimeError(f"clip cut by window in {png}")
    img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)).save(png)


def render_noisy_one(out: Path, job: dict) -> None:
    n = job["noise"]
    texts = job["lines"] + ([n["above"], n["below"]] if job["kind"] == "edgecut" else [])
    rows = len(job["lines"]) + (2 if job["kind"] == "edgecut" else 0)
    width = 260 + int(max(len(t) for t in texts) * job["size"] * 1.1) + 60
    height = 160 + int(rows * job["size"] * job["line_height"] * 1.3)
    png = out / job["image"]
    chrome_png(noisy_page(job), png, job["scale"], width, height)
    crop_to_clip(png)
    if job["kind"] == "edgecut":
        from PIL import Image

        img = Image.open(png)
        lh = job["size"] * job["line_height"] * job["scale"]  # three equal line boxes, no vertical padding
        top = round(lh * (1 - n["keep_top"]))
        bottom = round(2 * lh + lh * n["keep_bottom"]) if n["keep_bottom"] else round(2 * lh)
        img.crop((0, top, img.width, min(img.height, bottom))).save(png)


def render_noisy(text: Path, out: Path, seed: int, per_kind: int, workers: int = 4) -> None:
    from concurrent.futures import ThreadPoolExecutor

    check_fonts()
    pool = [l for row in text.read_text(encoding="utf-8").splitlines() if row for l in json.loads(row)["lines"]]
    jobs = noisy_jobs(pool, random.Random(seed), per_kind)
    for j in jobs:
        (out / j["image"]).parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as pool_:
        for job, _ in zip(jobs, pool_.map(lambda j: render_noisy_one(out, j), jobs)):
            print(job["id"], job["font"], job["size"], f"{job['scale']}x", flush=True)
    keys = ("id", "image", "subset", "kind", "font", "size", "scale", "dark", "bold")
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
    sl = sub.add_parser("slices", help="Task 3b: single-line style slices (subset slice_<style>)")
    sl.add_argument("--text", type=Path, required=True)
    sl.add_argument("--out", type=Path, required=True)
    sl.add_argument("--per_slice", type=int, default=40)
    nz = sub.add_parser("noisy", help="noise-robust spec: screen text with icons, borders, backgrounds, cut lines")
    nz.add_argument("--text", type=Path, required=True)
    nz.add_argument("--out", type=Path, required=True)
    nz.add_argument("--per_kind", type=int, default=50)
    args = p.parse_args()
    if args.cmd == "text":
        make_text(args.out, args.seed, args.screen, args.long, args.dedup)
    elif args.cmd == "slices":
        render_slices(args.text, args.out, args.seed, args.per_slice)
    elif args.cmd == "noisy":
        render_noisy(args.text, args.out, args.seed, args.per_kind)
    else:
        render(args.text, args.out, args.seed)


if __name__ == "__main__":
    main()
