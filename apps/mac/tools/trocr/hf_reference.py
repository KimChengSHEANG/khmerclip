"""HF PyTorch reference for CoreML parity: greedy decode with the same 128-id
cap and repeat-run cap (TrOCREngine.maxRun) the Swift engine uses, over the
frozen eval fixtures. Writes a bench.py-format CSV (engine,id,subset,cer,ms,
pred,ref) and prints mean CER per subset.

Usage (from apps/mac):
  .trocr-venv/bin/python3 tools/trocr/hf_reference.py raw_models/TrOCR-khmer \
      ../../training/eval/results/hf_greedy_runcap.csv
"""
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from transformers import LogitsProcessor, LogitsProcessorList, TrOCRProcessor, VisionEncoderDecoderModel

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "training/eval/src"))
from khmerclipper_eval.cer import character_error_rate  # noqa: E402

MAX_LEN = 128   # == TrOCREngine.maxLength
MAX_RUN = 4     # == TrOCREngine.maxRun


class RunCap(LogitsProcessor):
    """Forbid a token from extending a run of MAX_RUN identical tokens."""

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] >= MAX_RUN:
            tail = input_ids[:, -MAX_RUN:]
            for b in range(input_ids.shape[0]):
                if bool((tail[b] == tail[b, -1]).all()):
                    scores[b, tail[b, -1]] = -float("inf")
        return scores


def main() -> None:
    ckpt, out = Path(sys.argv[1]), Path(sys.argv[2])
    fixtures = REPO / "training/eval/fixtures"
    proc = TrOCRProcessor.from_pretrained(ckpt)
    model = VisionEncoderDecoderModel.from_pretrained(ckpt).eval()
    rows, by_subset = [], defaultdict(list)
    for line in (fixtures / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        pv = proc(images=Image.open(fixtures / r["image"]).convert("RGB"), return_tensors="pt").pixel_values
        t0 = time.perf_counter()
        with torch.no_grad():
            ids = model.generate(pv, num_beams=1, max_length=MAX_LEN, no_repeat_ngram_size=0,
                                 logits_processor=LogitsProcessorList([RunCap()]))
        ms = (time.perf_counter() - t0) * 1000
        pred = proc.batch_decode(ids, skip_special_tokens=True)[0]
        cer = character_error_rate(pred, r["text"])
        by_subset[r["subset"]].append(cer)
        rows.append({"engine": "trocr-hf-greedy", "id": r["id"], "subset": r["subset"],
                     "cer": cer, "ms": ms, "pred": pred, "ref": r["text"]})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["engine", "id", "subset", "cer", "ms", "pred", "ref"])
        w.writeheader()
        w.writerows(rows)
    allc = [c for v in by_subset.values() for c in v]
    print(f"all n={len(allc)} cer={sum(allc) / len(allc):.4f}")
    for k, v in sorted(by_subset.items()):
        print(f"{k} n={len(v)} cer={sum(v) / len(v):.4f}")


if __name__ == "__main__":
    main()
