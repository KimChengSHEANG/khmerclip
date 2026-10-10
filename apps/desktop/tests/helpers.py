from pathlib import Path

# Shared OCR fixtures from the mac app (PNG + expected .txt pairs).
FIXTURES = Path(__file__).resolve().parents[2] / "mac" / "Tests" / "Fixtures" / "ocr"


def normalize(s: str) -> str:
    return " ".join(s.split())


def cer(hyp: str, ref: str) -> float:
    """Character error rate: Levenshtein distance / len(ref)."""
    prev = list(range(len(ref) + 1))
    for i, h in enumerate(hyp, 1):
        cur = [i]
        for j, r in enumerate(ref, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (h != r)))
        prev = cur
    return prev[-1] / max(1, len(ref))
