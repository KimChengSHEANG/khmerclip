def apply(text: str, preserve_linebreaks: bool) -> str:
    """Port of OCRTextFormatter.swift: optionally flatten visual wrapping into one paragraph."""
    if preserve_linebreaks:
        return text
    return " ".join(s for line in text.splitlines() if (s := line.strip()))
