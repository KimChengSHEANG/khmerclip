from khmerclip.formatter import apply


def test_preserve_returns_text_unchanged():
    text = "ជួរទី១\nline two\n"
    assert apply(text, preserve_linebreaks=True) == text


def test_flatten_joins_trimmed_nonempty_lines_with_spaces():
    text = "  ជួរទី១  \r\n\n line two \nបី"
    assert apply(text, preserve_linebreaks=False) == "ជួរទី១ line two បី"


def test_flatten_of_blank_text_is_empty():
    assert apply("\n \n", preserve_linebreaks=False) == ""
