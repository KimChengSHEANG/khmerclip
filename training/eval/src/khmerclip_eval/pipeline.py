"""End-to-end TrOCR pipeline as the apps run it, in PyTorch: Mac `TrOCREngine.lineStrips` split,
plain 384x384 bilinear resize (training preprocessing, no content crop), greedy decode capped at
128 ids with the repeat-run cap, lines joined by a space (as OCRBench prints them).

Candidate improvements are opt-in keyword flags on `recognize` so each can be gated alone:
  split_wide_at=R  cut strips wider than R x their height at blank-column gaps
  crop=True        desktop's content_crop before resizing
  beam_gate=T      re-decode a line with beam 4 when its least likely greedy token has p < T
  normalize=True   post-decode cleanup (see normalize_text)
  split_touching=True  cut runs of touching lines at the line pitch (line_strips)
  cycle_cap=True   also block a token continuing a 2-token cycle repeated 4 times
"""
from __future__ import annotations

import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 384
MAX_LENGTH = 128
MAX_RUN = 4
THRESHOLD = 40


def blocked_token(ids: list[int], max_run: int = MAX_RUN) -> int | None:
    """Token that may not extend a run of `max_run` identical ids (mac TrOCREngine.blockedToken)."""
    tail = ids[-max_run:]
    return tail[-1] if len(ids) >= max_run and len(set(tail)) == 1 else None


def cycle_blocked_token(ids: list[int], cycles: int = 4) -> int | None:
    """Token that would continue a 2-token cycle already repeated `cycles` times (greedy decode
    loops on " ។ ។ ។" past a line's end; the single-token repeat cap can't see it)."""
    tail = ids[-2 * cycles:]
    if len(tail) < 2 * cycles or tail[0] == tail[1]:
        return None
    return tail[0] if all(t == tail[i % 2] for i, t in enumerate(tail)) else None


def ink_mask(image: Image.Image, scan_max: int | None = 512) -> tuple[np.ndarray, int] | None:
    """Bool ink mask (any channel > 40 from the mean corner colour) at <= scan_max long edge."""
    w, h = image.size
    if w <= 1 or h <= 1:
        return None
    scale = 1 if scan_max is None else max(1, -(-max(w, h) // scan_max))
    sw, sh = w // scale, h // scale
    if sw <= 1 or sh <= 1:
        return None
    rgb = image.convert("RGB")
    px = np.asarray(rgb if scale == 1 else rgb.resize((sw, sh), Image.BILINEAR), dtype=np.int32)
    bg = np.stack([px[0, 0], px[0, -1], px[-1, 0], px[-1, -1]]).sum(axis=0) // 4
    return (np.abs(px - bg) > THRESHOLD).any(axis=2), scale


PITCH_STRENGTH = 0.1  # autocorrelation peak a line pitch needs
VALLEY = 0.2          # a cut between touching lines must be this sparse vs the run's densest row


def line_pitch(profile: np.ndarray) -> int | None:
    """Line pitch in rows: first autocorrelation peak of the row-ink profile above PITCH_STRENGTH,
    searched between n/12 and n/1.5 rows. None when the clip shows no repeating line structure."""
    x = profile - profile.mean()
    n = len(x)
    ac = np.correlate(x, x, "full")[n - 1:]
    if n < 8 or ac[0] <= 0:
        return None
    ac = ac / ac[0]
    for k in range(max(4, n // 12), min(n - 1, int(n / 1.5) + 1)):
        if ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1] and ac[k] > PITCH_STRENGTH:
            return k
    return None


def touching_cuts(profile: np.ndarray, pitch: int) -> list[int]:
    """Rows to cut a run holding several touching lines: one valley per expected line boundary,
    kept only when it is sparse (<= VALLEY x the run's densest row)."""
    h = len(profile)
    n = int((h - 0.5 * pitch) // pitch) + 1
    if n < 2:
        return []
    line_h = h - (n - 1) * pitch
    cuts = []
    for i in range(1, n):
        centre = (i - 1) * pitch + (pitch + line_h) / 2
        a, b = int(max(1, centre - pitch / 3)), int(min(h - 1, centre + pitch / 3 + 1))
        if a < b:
            c = a + int(np.argmin(profile[a:b]))
            if profile[c] <= VALLEY * profile.max():
                cuts.append(c)
    return cuts


def merge_marks(runs: list[list[int]]) -> list[list[int]]:
    """Swift lineStrips merge rule: a run < 35% of the tallest, or < 50% within a quarter of the
    tallest's height of a neighbour, is a detached mark and joins its nearer neighbour."""
    while len(runs) > 1:
        tallest = max(hi - lo + 1 for lo, hi in runs)

        def gap(k: int) -> float:
            return min(runs[k][0] - runs[k - 1][1] if k > 0 else math.inf,
                       runs[k + 1][0] - runs[k][1] if k < len(runs) - 1 else math.inf)

        i = next((k for k, (lo, hi) in enumerate(runs)
                  if (hi - lo + 1) * 100 < tallest * 35
                  or ((hi - lo + 1) * 100 < tallest * 50 and gap(k) * 4 < tallest)), None)
        if i is None:
            break
        above = runs[i][0] - runs[i - 1][1] if i > 0 else math.inf
        below = runs[i + 1][0] - runs[i][1] if i < len(runs) - 1 else math.inf
        j = i - 1 if above <= below else i + 1
        runs[min(i, j)] = [min(runs[i][0], runs[j][0]), max(runs[i][1], runs[j][1])]
        del runs[max(i, j)]
    return runs


def line_strips(image: Image.Image, split_touching: bool = False) -> list[Image.Image]:
    """Port of Swift `TrOCREngine.lineStrips`: runs of inked rows, detached marks merged in,
    strips extended halfway into the gaps.

    split_touching: first cut runs holding several lines whose marks touch (no blank row between
    them, common at Khmer line heights) at the line pitch, on a full-resolution mask. Splitting
    before the mark merge matters: a merged double line otherwise makes real lines look small."""
    m = ink_mask(image, scan_max=None if split_touching else 512)
    if m is None:
        return [image]
    mask, s = m
    runs: list[list[int]] = []
    for y, inked in enumerate(mask.any(axis=1)):
        if inked:
            if runs and runs[-1][1] == y - 1:
                runs[-1][1] = y
            else:
                runs.append([y, y])
    if split_touching and runs:
        profile = np.convolve(mask.sum(axis=1).astype(float), np.ones(3) / 3, "same")
        pitch = line_pitch(profile)
        if pitch:
            split = []
            for lo, hi in runs:
                cuts = touching_cuts(profile[lo:hi + 1], pitch) if hi - lo + 1 >= 1.5 * pitch else []
                edges = [lo] + [lo + c for c in cuts] + [hi + 1]
                split += [[edges[k], edges[k + 1] - 1] for k in range(len(edges) - 1)]
            runs = split
    runs = merge_marks(runs)
    if len(runs) <= 1:
        return [image]
    mh = mask.shape[0]
    out = []
    for k in range(len(runs)):
        top = 0 if k == 0 else (runs[k - 1][1] + runs[k][0] + 1) // 2
        bottom = mh if k == len(runs) - 1 else (runs[k][1] + runs[k + 1][0] + 1) // 2
        out.append(image.crop((0, top * s, image.width, min(image.height, bottom * s))))
    return out


def split_wide(strip: Image.Image, max_aspect: float, space_gap: float = 0.25,
               window: float = 0.25) -> list[tuple[str, Image.Image]]:
    """Cut a line strip whose *line* is wider than `max_aspect` x the strip height into ~equal
    pieces at blank-column runs.

    The line's horizontal extent comes from ink in the strip's central band (rows 30-70%): a strip
    spans the whole clip width and its top / bottom edges hold the neighbour lines' marks, and
    a piece cut from that blank tail makes TrOCR hallucinate text. Each ideal cut point snaps to
    the centre of the nearest blank-column run within +-`window` of a piece width; with no blank
    run there, that cut is skipped (never cut ink). Returns (separator, piece): " " when the gap
    is >= `space_gap` x text height, else "".
    """
    w, h = strip.size
    if w <= max_aspect * h:
        return [("", strip)]
    m = ink_mask(strip, scan_max=None)
    if m is None:
        return [("", strip)]
    mask = m[0]
    core = np.nonzero(mask[int(0.3 * h): max(int(0.3 * h) + 1, int(0.7 * h))].any(axis=0))[0]
    if core.size == 0:
        return [("", strip)]
    pad = max(2, h // 5)
    x0, x1 = max(0, int(core[0]) - pad), min(w, int(core[-1]) + 1 + pad)
    if x1 - x0 <= max_aspect * h:
        return [("", strip)]
    rows = np.nonzero(mask.any(axis=1))[0]
    text_h = int(rows[-1] - rows[0] + 1)
    blank = ~mask.any(axis=0)
    gaps, x = [], x0  # (start, end_exclusive) of blank-column runs inside the line
    while x < x1:
        if blank[x]:
            start = x
            while x < x1 and blank[x]:
                x += 1
            if start > x0 and x < x1:
                gaps.append((start, x))
        else:
            x += 1
    n = math.ceil((x1 - x0) / (max_aspect * h))
    piece_w = (x1 - x0) / n
    cuts: list[tuple[int, str]] = []
    for k in range(1, n):
        ideal = x0 + k * piece_w
        near = [g for g in gaps if abs((g[0] + g[1]) / 2 - ideal) <= window * piece_w
                and (not cuts or g[0] > cuts[-1][0])]
        if not near:
            continue
        g = min(near, key=lambda g: abs((g[0] + g[1]) / 2 - ideal))
        cuts.append(((g[0] + g[1]) // 2, " " if g[1] - g[0] >= space_gap * text_h else ""))
    if not cuts:
        return [("", strip)]
    pieces, left, sep = [], x0, ""
    for cx, next_sep in cuts:
        pieces.append((sep, strip.crop((left, 0, cx, h))))
        left, sep = cx, next_sep
    pieces.append((sep, strip.crop((left, 0, x1, h))))
    return pieces


def content_crop(image: Image.Image) -> Image.Image:
    """Port of desktop `preprocess.content_crop`: trim background margin, 4% padding."""
    m = ink_mask(image)
    if m is None:
        return image
    mask, s = m
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return image
    sh, sw = mask.shape
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    if (x1 - x0 + 1) * 20 < sw or (y1 - y0 + 1) * 20 < sh:
        return image
    px, py = max(1, (x1 - x0 + 1) * 4 // 100), max(1, (y1 - y0 + 1) * 4 // 100)
    return image.crop((max(0, (x0 - px) * s), max(0, (y0 - py) * s),
                       min(image.width, (x1 + 1 + px) * s), min(image.height, (y1 + 1 + py) * s)))


def normalize_text(text: str) -> str:
    """Post-decode cleanup; rules are added only when references follow them (accuracy_v3.md)."""
    return " ".join(text.replace("​", "").split())


def pixel_values(images: list[Image.Image]):
    import torch

    arr = np.stack([np.asarray(i.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32)
                    for i in images])
    return torch.from_numpy((arr / 255.0 - 0.5) / 0.5).permute(0, 3, 1, 2).contiguous()


@lru_cache(maxsize=2)
def load(model_dir: str, device: str):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "trocr/src"))
    from transformers import VisionEncoderDecoderModel
    from trocr_khmer import load_trocr_processor

    processor = load_trocr_processor(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device).eval()
    return processor, model


def default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def decode(images: list[Image.Image], model_dir: Path, device: str, beams: int = 1,
           cycle_cap: bool = False) -> list[tuple[str, float]]:
    """(text, min greedy token probability) per image."""
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList

    class RepeatCap(LogitsProcessor):
        def __call__(self, input_ids, scores):
            for row, ids in enumerate(input_ids.tolist()):
                for b in (blocked_token(ids), cycle_blocked_token(ids) if cycle_cap else None):
                    if b is not None:
                        scores[row, b] = -math.inf
            return scores

    processor, model = load(str(model_dir), device)
    tok = processor.tokenizer
    with torch.no_grad():
        out = model.generate(pixel_values(images).to(device), max_length=MAX_LENGTH, num_beams=beams,
                             do_sample=False, no_repeat_ngram_size=0, length_penalty=1.0, early_stopping=True,
                             logits_processor=LogitsProcessorList([RepeatCap()]),
                             decoder_start_token_id=model.config.decoder_start_token_id,
                             eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                             output_scores=True, return_dict_in_generate=True)
    texts = processor.batch_decode(out.sequences, skip_special_tokens=True)
    if beams > 1:
        return [(t.strip(), 1.0) for t in texts]
    probs = torch.stack([s.float().softmax(-1).max(-1).values for s in out.scores], 1)  # (batch, steps)
    done = (out.sequences[:, 1:] == tok.pad_token_id) & (out.sequences[:, :-1] != model.config.decoder_start_token_id)
    probs = probs.masked_fill(done[:, : probs.shape[1]], 1.0)
    return [(t.strip(), float(p.min())) for t, p in zip(texts, probs)]


def recognize(image: Image.Image, model_dir: Path, device: str | None = None, *, split_wide_at: float | None = None,
              crop: bool = False, beam_gate: float | None = None, normalize: bool = False,
              split_touching: bool = False, cycle_cap: bool = False) -> str:
    device = device or default_device()
    units: list[tuple[str, Image.Image]] = []  # (separator before, image to decode)
    for k, strip in enumerate(line_strips(image, split_touching)):
        pieces = split_wide(strip, split_wide_at) if split_wide_at else [("", strip)]
        units += [((" " if k else "") if i == 0 else sep, p) for i, (sep, p) in enumerate(pieces)]
    images = [content_crop(p) if crop else p for _, p in units]
    results = decode(images, model_dir, device, cycle_cap=cycle_cap)
    if beam_gate is not None:
        redo = [i for i, (_, p) in enumerate(results) if p < beam_gate]
        if redo:
            for i, r in zip(redo, decode([images[i] for i in redo], model_dir, device, beams=4, cycle_cap=cycle_cap)):
                results[i] = r
    text = "".join(sep + t for (sep, _), (t, _) in zip(units, results) if t or sep)
    return normalize_text(text) if normalize else " ".join(text.split())
