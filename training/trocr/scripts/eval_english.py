#!/usr/bin/env python3
"""Clean-English gate: render training/eval/en_ui_strings.txt in macOS system fonts (held out
from training's Google Latin fonts) as full-width app-style strips, light and dark, and decode
them the way the app does (greedy, 128 ids, repeat-run cap 4). Prints line accuracy and CER.
Runs on a Mac (needs /System/Library/Fonts); CPU is fine.

    python scripts/eval_english.py ../../apps/mac/raw_models/TrOCR-khmer
"""
import random
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import LogitsProcessor, LogitsProcessorList, TrOCRProcessor, VisionEncoderDecoderModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval/src"))
from khmerclip_eval.cer import character_error_rate  # noqa: E402

STRINGS = Path(__file__).resolve().parents[2] / "eval/en_ui_strings.txt"
FONTS = {
    "Georgia": "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "Arial": "/System/Library/Fonts/Supplemental/Arial.ttf",
    "Verdana": "/System/Library/Fonts/Supplemental/Verdana.ttf",
    "NewYork": "/System/Library/Fonts/NewYork.ttf",
    "SFNS": "/System/Library/Fonts/SFNS.ttf",
    "Tahoma": "/System/Library/Fonts/Supplemental/Tahoma.ttf",
}


class RunCap(LogitsProcessor):
    """== TrOCREngine.maxRun: forbid a token from extending a run of 4 identical ids."""

    def __call__(self, ids, scores):
        for b in range(ids.shape[0]):
            if ids.shape[1] >= 4 and bool((ids[b, -4:] == ids[b, -1]).all()):
                scores[b, ids[b, -1]] = -float("inf")
        return scores


def render(text: str, font: str, px: int, dark: bool) -> Image.Image:
    bg, fg = ((30, 30, 32), (235, 235, 235)) if dark else ((255, 255, 255), (34, 31, 32))
    im = Image.new("RGB", (int(px * 11.5), int(px * 2.0)), bg)
    ImageDraw.Draw(im).text((int(px * 0.8), int(px * 0.4)), text, font=ImageFont.truetype(font, px), fill=fg)
    return im


def main() -> None:
    proc = TrOCRProcessor.from_pretrained(sys.argv[1])
    model = VisionEncoderDecoderModel.from_pretrained(sys.argv[1]).eval()
    lines = [s for s in STRINGS.read_text(encoding="utf-8").splitlines() if s.strip()]
    rng = random.Random(0)
    cases = [(t, name, font, rng.choice([34, 36, 70]), rng.random() < 0.3) for t in lines for name, font in FONTS.items()]
    exact, cers, misses = 0, [], []
    for i in range(0, len(cases), 16):
        batch = cases[i:i + 16]
        pv = proc(images=[render(t, f, px, dark) for t, _, f, px, dark in batch], return_tensors="pt").pixel_values
        with torch.no_grad():
            ids = model.generate(pv, num_beams=1, max_length=128, no_repeat_ngram_size=0,
                                 logits_processor=LogitsProcessorList([RunCap()]))
        for (t, name, *_), out in zip(batch, proc.batch_decode(ids, skip_special_tokens=True)):
            out = out.strip()
            exact += out == t
            cers.append(character_error_rate(out, t))
            if out != t:
                misses.append(f"  {name:8} {t!r} -> {out!r}")
    print("\n".join(misses[:25]))
    print(f"en_ui: n={len(cases)} line_acc={exact / len(cases):.3f} cer={sum(cers) / len(cers):.4f}")


if __name__ == "__main__":
    main()
